import sys
from threading import Thread

import numexpr as ne
import numpy as np
import scipy.interpolate
import scipy.fft
import scipy.ndimage

from progressbar import progressbar, ProgressBar  

from . import color
from ..tracer import misc as misc
from ..tracer.misc import PropertyChecker as pc
from ..tracer.r_image import RImage



def _color_variation(img: np.ndarray, th: float = 1e-3) -> bool:
    """
    Check if an image has color variation. 
    If any channel pixel is above th over the channel mean this function returns True.

    :param img: img to check
    :param th: sRGB value threshold
    :return:  if image has color variation
    """
    i0, i1, i2 = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    im0, im1, im2 = np.mean(img, axis=(0, 1))

    return np.any(ne.evaluate("abs(i0-im0) > th")) or np.any(ne.evaluate("abs(i1-im1) > th"))\
            or np.any(ne.evaluate("abs(i2-im2) > th"))


def _safe_normalize(img: np.ndarray) -> np.ndarray:
    """normalize array by its maximum and handle the case of a zero array"""
    img_max = np.max(img)
    return img / img_max if img_max else img.copy()


def _threaded(img:      np.ndarray, 
              psf:      np.ndarray, 
              i:        int, 
              img_out:  np.ndarray, 
              scx:      float, 
              pnx:      int, 
              scy:      float, 
              pny:      int)\
        -> None:

    # psf too small -> dirac pulse
    if scy*pny < 1 and scx*pnx < 1:
        psf2 = np.array([[1.0]])

    # zoom (=rescale) PSF
    else:
        psf2 = scipy.ndimage.zoom(psf, [scy, scx], order=3, mode="grid-constant", grid_mode=True, prefilter=True)

    # adapt values such that power mean after and before zooming stays the same
    # .. . the reason we need to do this is because some color components are lost in higher frequency components
    # but would nonetheless color the image
    if (mean_ := np.mean(psf2)):
        psf2 *= np.mean(psf) / mean_

    # pad psf
    psf2 = np.pad(psf2, ((4, 4), (4, 4)), mode="constant", constant_values=0)

    # do convolution
    img_out[:, :, i] = scipy.signal.fftconvolve(img, psf2, mode="full")


def convolve(img:                np.ndarray | str, 
             s_img:              list[float, float], 
             psf:                np.ndarray | RImage, 
             s_psf:              list[float, float],
             m:                  float = 1,
             rendering_intent:   str =  "Absolute",
             normalize:          bool = True,
             silent:             bool = False, 
             threading:          bool = True)\
        -> tuple[np.ndarray, list[float]]:
    """
    Convolve an image with a point spread function.
    Sizes of both are provided as lengths in parameters s_img and s_psf.
    The system magnification factor m scales the image.

    Image data should be a sRGB array or a path, in the latter case the image is loaded from file.
    The PSF is either an intensity image (when provided as array) or a colored on (when the type is RImage).

    Note that in the case that both are colored the convolution is only one possible solution of many.

    :param img: image. 3D sRGB numpy array (value range 0-1) or path string
    :param s_img: side lengths image in mm (x-length and y-length)
    :param psf: point spread function, 2D numpy intensity array or RImage
    :param s_psf: side lengths psf in mm(x-length and y-length)
    :param m: magnification factor, abs(m) > 1 means enlargement, abs(m) < 1 size reduction,
              m < 0 image flipping, m > 0 an upright image
    :param rendering_intent: rendering intent used for sRGB conversion of output
    :param normalize: if output image should be normalized to range [0-1] (input images are normalized by default)
    :param silent: if text output should be silenced
    :param threading: turn multithreading on and off
    :return: convoled image (3D sRGB numpy array) and new image side lengths list
    """
    # Init
    ###################################################################################################################
    ###################################################################################################################

    pc.check_type("img", img, np.ndarray | str)
    pc.check_type("psf", psf, np.ndarray | RImage)
    pc.check_type("s_img", s_img, tuple | list)
    pc.check_type("s_psf", s_psf, tuple | list)
    pc.check_above("s_psf[0]", s_psf[0], 0)
    pc.check_above("s_psf[1]", s_psf[1], 0)
    pc.check_above("s_img[0]", s_img[0], 0)
    pc.check_above("s_img[1]", s_img[1], 0)
    pc.check_type("m", m, int | float)
    pc.check_above("abs(m)", abs(m), 0)
    pc.check_type("rendering_intent", rendering_intent, str)
    pc.check_type("silent", silent, bool)
    pc.check_type("threading", threading, bool)

    # function for raising the exception and finishing the progress bar
    def raise_(excp):
        if not silent:
            bar.finish()
        raise excp
    
    # create progress bar
    bar = ProgressBar(fd=sys.stdout, prefix="Convolving: ", 
                      max_value=5).start() if not silent else None

    # Load image
    ###################################################################################################################
    ###################################################################################################################

    # load from files
    if isinstance(img, str):
        img_srgb = misc.load_image(img)

    # numpy array
    else:
        # normalize
        img_srgb = _safe_normalize(img)
    
        if img_srgb.ndim != 3 or img_srgb.shape[2] != 3:
            raise_(ValueError(f"Image needs to be a three dimensional array with three sRGB channels,"
                              f" but has shape {img_srgb.shape}"))

    img_lin = color.srgb_to_srgb_linear(img_srgb)  # linear sRGB version

    if not silent:
        bar.update(1)

    # Load PSF
    ###################################################################################################################
    ###################################################################################################################

    # rendered image with color information
    if isinstance(psf, RImage):

        img_color = _color_variation(img_lin)  # if it has color information
        if img_color and not silent:
            print("WARNING: Using an image with color variation in combination with a colored PSF. "
                  "As there many possible results depending on the spectral and local distributions on the object, "
                  "the scene will be rendered as one seen on a sRGB monitor.")

        # use srgb with negative values
        psf_lin = color.xyz_to_srgb_linear(psf.xyz(), rendering_intent="Ignore")
        
        # normalize
        psf_lin = _safe_normalize(psf_lin)

    # intensity array
    else:
        if psf.ndim != 2:
            raise_(ValueError(f"PSF needs to be a two dimensional array, but has shape {psf.shape}"))
       
        # normalize
        psf_lin = _safe_normalize(psf)
        
        # repeat intensities
        psf_lin = np.repeat(psf_lin[..., np.newaxis], 3, axis=2)

    if not silent:
        bar.update(2)

    # Shape Checks
    ###################################################################################################################
    ###################################################################################################################

    # grid coordinates:
    # image with side coordinates -5, 0, 5 has n = three pixels, 
    # but a side length of s=10, as we measure from the center of the pixels
    # the pixel size p is therefore s/(n-1)
    # from a known pixel size the length s is then (n-1)*p

    # image and psf properties
    iny, inx = img_lin.shape[:2]  # image pixel count
    isx, isy = s_img[0]*abs(m), s_img[1]*abs(m)  # image side lengths scaled by magnification magnitude
    ipx, ipy = isx/(inx-1), isy/(iny-1)  # image pixel sizes
    pny, pnx = psf_lin.shape[:2]  # psf pixel counts
    psx, psy = s_psf  # psf side lengths
    ppx, ppy = psx/(pnx-1), psy/(pny-1)  # psf pixel sizes
      
    if psx > 2*isx or psy > 2*isy:
        raise_(ValueError(f"m-scaled image size [{isx:.5g}, {isy:.5g}] is more than two times " 
                          f"smaller than PSF size [{psx:.5g}, {psy:.5g}]."))

    if pnx*pny > 4e6:
        raise_(ValueError("PSF needs to be smaller than 4MP"))

    if inx*iny > 4e6:
        raise_(ValueError("Image needs to be smaller than 4MP"))

    if ppx > ipx or ppy > ipy:
        if not silent:
            print(f"WARNING: PSF pixel sizes [{ppx:.5g}, {ppy:.5g}] larger than image pixel sizes"
                  f" [{ipx:.5g}, {ipy:.5g}], generally you want a PSF in a higher resolution")

    if pnx < 50 or pny < 50:
        raise_(ValueError(f"PSF too small with shape {psf.shape}, "
                          "needs to have at least 50 values in each dimension."))
    
    if inx < 50 or iny < 50:
        raise_(ValueError(f"Image too small with shape {img.shape}, "
                          "needs to have at least 50 values in each dimension."))
    
    if inx*iny < 2e4:
        if not silent:
            print(f"WARNING: Low resolution image.")

    if pnx*pny < 2e4:
        if not silent:
            print(f"WARNING: Low resolution PSF.")

    if not (0.2 < ppx/ppy < 5):
        if not silent:
            print(f"WARNING: Pixels of PSF are strongly non-square with side lengths [{ppx}mm, {ppy}mm]")
    
    if not (0.2 < ipx/ipy < 5):
        if not silent:
            print(f"WARNING: Pixels of image are strongly non-square with side lengths [{ipx}mm, {ipy}mm]")

    # Convolution
    ###################################################################################################################
    ###################################################################################################################

    # psf rescale factors
    scx = ppx / ipx
    scy = ppy / ipy
    
    # new psf pixel sizes, 8 for padding and rescaled size because of ndimage.zoom()
    pnx_ = 8 + max(round(pnx*scx), 1)
    pny_ = 8 + max(round(pny*scy), 1)

    # output image
    img2 = np.zeros((iny + pny_ - 1, inx + pnx_ - 1, 3), dtype=np.float64)

    if not silent:
        bar.update(3)

    if not threading:
        for i in range(3):
            _threaded(img_lin[:, :, i], psf_lin[:, :, i], i, img2, scx, pnx, scy, pny)
   
    else:
        # create threads
        thread_list = [Thread(target=_threaded, 
                              args=(img_lin[:, :, i], psf_lin[:, :, i], i, img2, scx, pnx, scy, pny))
                       for i in range(3)]

        # execute threads
        [thread.start() for thread in thread_list]
        [thread.join() for thread in thread_list]
    
    if not silent:
        bar.update(4)

    # Output Conversion
    ###################################################################################################################
    ###################################################################################################################

    # normalize image
    if normalize:
        img2 = _safe_normalize(img2)

    # map color into sRGB gamut by converting from sRGB linear to XYZ to sRGB
    img2 = color.srgb_linear_to_xyz(img2)
    img2 = color.xyz_to_srgb(img2, rendering_intent=rendering_intent, normalize=normalize) 
    
    # new image side lengths
    s2 = [(img2.shape[1]-1)*ipx, (img2.shape[0]-1)*ipy]

    # rotate image by 180deg when m is negative
    if m < 0:
        img2 = np.fliplr(np.flipud(img2))

    if not silent:
        bar.finish()

    return img2, s2

