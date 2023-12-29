import sys
from threading import Thread

import numpy as np
import scipy.signal
import cv2

from progressbar import progressbar, ProgressBar  

from . import color

from ..tracer.misc import PropertyChecker as pc
from ..tracer.r_image import RImage
from ..tracer.image import Image

from .. import global_options
from ..warnings import warning


def _color_variation(img: np.ndarray, th: float = 1e-3) -> bool:
    """
    Check if an image has color variation. 
    If any channel pixel is above th over the channel mean this function returns True.

    :param img: img to check
    :param th: sRGB value threshold
    :return:  if image has color variation
    """
    i0, i1, i2 = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    return np.any(np.abs(i0-i1) > th) or np.any(np.abs(i1-i2) > th) or np.any(np.abs(i2-i0) > th)


def _safe_normalize(img: np.ndarray) -> np.ndarray:
    """normalize array by its maximum and handle the case of a zero array"""
    img_max = np.max(img)
    return img / img_max if img_max else img.copy()


def convolve(img:                Image, 
             psf:                Image | RImage, 
             m:                  float = 1,
             slice_:             bool = False,
             padding_mode:       str = "constant",
             padding_value:      list = [0, 0, 0],
             cargs:              dict = {})\
        -> Image:
    """
    Convolve an image with a point spread function.
    The system magnification factor m scales the image.

    Image data should be a sRGB array (therefore non-linear, gamma corrected values). 
    The PSF is treated as either a linear intensity image (when provided as array) or a colored one (when the type is RImage).

    Note that in the case that both are colored the convolution is only one possible solution of many.

    TODO explain better
    With normalize=True in dictionary cargs (the default setting) the image colors are normalized so the brightest pixel has maximum brightness.
    Otherwise
    By default, the image brightness is normalized after convolution.
    With normalize=False the psf is set to a power sum of 1, so theis provided in argument cargs, the normalization is turned off and the psf

    :param img: image. 3D sRGB numpy array (value range 0-1) or path string
    :param psf: point spread function, Image or RImage.
    :param m: magnification factor, abs(m) > 1 means enlargement, abs(m) < 1 size reduction,
              m < 0 image flipping, m > 0 an upright image
    :param padding_mode: padding mode (from numpy.pad) for image padding before convolution.
    :param padding_value: padding value for padding_mode='constant'. 
     Must be a three element list (as it should be a sRGB color).
    :param cargs: overwrite for parameters for color.xyz_to_srgb. By default these parameters are:
     dict(rendering_intent="Absolute", normalize=True, clip=True, L_th=0, sat_scale=None).
     For instance, specify cargs=dict(normalize=False) to turn off normalization, but all other parameters unchanged
    :return: convolved image (3D sRGB numpy array) and new image side lengths list
    """
    # Init
    ###################################################################################################################
    ###################################################################################################################

    pc.check_type("img", img, Image)
    pc.check_type("psf", psf, Image | RImage)
    pc.check_type("m", m, int | float)
    pc.check_type("cargs", cargs, dict)
    pc.check_above("abs(m)", abs(m), 0)
    pc.check_type("padding_value", padding_value, list | np.ndarray)
    pc.check_type("slice_", slice_, bool)

    pval = np.asarray(padding_value, dtype=np.float64)

    # function for raising the exception and finishing the progress bar
    def raise_(excp):
        if global_options.show_progressbar:
            bar.finish()
        raise excp
    
    # create progress bar
    bar = ProgressBar(fd=sys.stdout, redirect_stderr=True, prefix="Convolving: ", 
                      max_value=5).start() if global_options.show_progressbar else None
    
    # Load PSF
    ###################################################################################################################
    ###################################################################################################################

    # rendered image with color information
    if isinstance(psf, RImage):

        psf_color = True

        # use srgb with negative values
        psf_lin = color.xyz_to_srgb_linear(psf.xyz(), rendering_intent="Ignore")

        # normalize
        psf_lin = _safe_normalize(psf_lin)

    # intensity array
    else:
        psf_lin = psf.data
        # TODO document that values are treated linearly

        psf_color = _color_variation(psf_lin)  # if it has color information
        if psf_color:
            raise ValueError("Colored PSF are only supported when provided as RImage.")
      
    # psf_lin /= np.sum(psf_lin)

    if global_options.show_progressbar:
        bar.update(1)

    # Load image
    ###################################################################################################################
    ###################################################################################################################

    img_srgb = img.data

    img_color = _color_variation(img_srgb)  # if image has color information
    if img_color and psf_color:
        warning("Using an image with color variation in combination with a colored PSF. "
                "As there many possible results depending on the spectral and local distributions on the object, "
                "the scene will be rendered as one seen on a sRGB monitor.")

    # Shape Checks
    ###################################################################################################################
    ###################################################################################################################

    # grid coordinates:
    # image with side coordinates -5, 0, 5 has n = three pixels, 
    # but a side length of s=10, as we measure from the center of the pixels
    # the pixel size p is therefore s/(n-1)
    # from a known pixel size the length s is then (n-1)*p

    # image and psf properties
    iny, inx = img.shape[:2]  # image pixel count
    isx, isy = img.s[0]*abs(m), img.s[1]*abs(m)  # image side lengths scaled by magnification magnitude
    ipx, ipy = isx/(inx-1), isy/(iny-1)  # image pixel sizes
    pny, pnx = psf_lin.shape[:2]  # psf pixel counts
    psx, psy = psf.s  # psf side lengths
    ppx, ppy = psx/(pnx-1), psy/(pny-1)  # psf pixel sizes
      
    if psx > 2*isx or psy > 2*isy:
        raise_(ValueError(f"m-scaled image size [{isx:.5g}, {isy:.5g}] is more than two times " 
                          f"smaller than PSF size [{psx:.5g}, {psy:.5g}]."))

    if pnx*pny > 4e6:
        raise_(ValueError("PSF needs to be smaller than 4MP"))

    if inx*iny > 4e6:
        raise_(ValueError("Image needs to be smaller than 4MP"))

    if ppx > ipx or ppy > ipy:
        warning(f"WARNING: PSF pixel sizes [{ppx:.5g}, {ppy:.5g}] larger than image pixel sizes"
                      f" [{ipx:.5g}, {ipy:.5g}], generally you want a PSF in a higher resolution")

    if pnx < 50 or pny < 50:
        raise_(ValueError(f"PSF too small with shape {psf.shape}, "
                          "needs to have at least 50 values in each dimension."))
    
    if inx < 50 or iny < 50:
        raise_(ValueError(f"Image too small with shape {img.shape}, "
                          "needs to have at least 50 values in each dimension."))
    
    if inx*iny < 2e4:
        warning(f"WARNING: Low resolution image.")

    if pnx*pny < 2e4:
        warning(f"WARNING: Low resolution PSF.")

    if not (0.2 < ppx/ppy < 5):
        warning(f"WARNING: Pixels of PSF are strongly non-square with side lengths [{ppx}mm, {ppy}mm]")
    
    if not (0.2 < ipx/ipy < 5):
        warning(f"WARNING: Pixels of image are strongly non-square with side lengths [{ipx}mm, {ipy}mm]")

    if pval.ndim != 1 or pval.shape[0] != 3:
        raise_(ValueError(f"padding_value must be a 3 element array/list, but has shape {pval.shape}"))

    # Convolution Preparation
    ###################################################################################################################
    ###################################################################################################################

    # psf rescale factors
    scx = ppx / ipx
    scy = ppy / ipy
    
    # new psf pixel sizes, 8 for padding and rescaled size rounded
    pnx_ = 8 + max(round(pnx*scx), 1)
    pny_ = 8 + max(round(pny*scy), 1)

    padding_needed = not (padding_mode == "constant" and padding_value == [0, 0, 0])
    pad_x = pnx_ if padding_needed else 0
    pad_y = pny_ if padding_needed else 0

    if padding_needed:
        if padding_mode == "constant":
            imgp = np.tile(pval, (img_srgb.shape[0]+2*pad_y, img_srgb.shape[1]+2*pad_x, 1))
            imgp[pad_y:-pad_y, pad_x:-pad_x] = img_srgb
        else:
            imgp = np.pad(img_srgb, ((pad_y, pad_y), (pad_x, pad_x), (0, 0)), mode=padding_mode)
    else:
        imgp = img_srgb

    img_lin = color.srgb_to_srgb_linear(imgp)  # linear sRGB version

    iny_, inx_ = img_lin.shape[:2]  # image pixel count
    
    # output image
    img2 = np.zeros((iny_ + pny_ - 1, inx_ + pnx_ - 1, 3), dtype=np.float64)

    if global_options.show_progressbar:
        bar.update(2)
    
    # INTER_AREA downscaling leaves the power sum unchanged (therefore the color and channel ratios should stay the same). 
    # Weighs pixels according to their area.
    # since no polynomial interpolation of higher degree takes place, the value range is limited
    # (as wouldn't be the case for instance with polynomials of degree 3 or higher)
    psf2 = cv2.resize(psf_lin, [max(round(scx*pnx), 1), max(round(scy*pny), 1)], interpolation=cv2.INTER_AREA)

    # normalize each channel by its sum. Doing this, the image power and power ratios stay the same
    # psf2 /= np.sum(psf2, axis=(0, 1))[np.newaxis, np.newaxis, :]
    if np.sum(psf2):
        psf2 /= np.sum(psf2) / 3 #[np.newaxis, np.newaxis, :]

    # pad psf
    psf2 = np.pad(psf2, ((4, 4), (4, 4), (0, 0)), mode="constant", constant_values=0)
  
    # update progress
    if global_options.show_progressbar:
        bar.update(3)
    
    # Convolution
    ###################################################################################################################
    ###################################################################################################################

    if not global_options.multithreading:
        img2 = scipy.signal.fftconvolve(img_lin, psf2, mode="full", axes=(0, 1))

    else:
        def threaded(img, psf, img_out, i):
            img_out[:, :, i] = scipy.signal.fftconvolve(img, psf, mode="full")

        # create threads
        thread_list = [Thread(target=threaded, args=(img_lin[:, :, i], psf2[:, :, i], img2, i)) for i in range(3)]
        [thread.start() for thread in thread_list]
        [thread.join() for thread in thread_list]
    
    if global_options.show_progressbar:
        bar.update(4)

    # Output Conversion
    ###################################################################################################################
    ###################################################################################################################
    
    # remove padding
    if padding_needed:
        img2 = img2[pad_y:-pad_y, pad_x:-pad_x]
    
    if slice_:
        iy0 = (pny_ - 1) // 2
        ix0 = (pnx_ - 1) // 2
        img2 = img2[iy0:iy0+iny, ix0:ix0+inx]

    # normalize image
    if not ("normalize" in cargs and not cargs["normalize"]):
        img2 = _safe_normalize(img2)

    # map color into sRGB gamut by converting from sRGB linear to XYZ to sRGB
    img2 = color.srgb_linear_to_xyz(img2)
    cargs0 = dict(rendering_intent="Absolute", normalize=True, clip=True, L_th=0, sat_scale=None)
    img2 = color.xyz_to_srgb(img2, **(cargs0 | cargs))
   
    # new image side lengths
    s2 = [(img2.shape[1]-1)*ipx, (img2.shape[0]-1)*ipy]

    # rotate image by 180deg when m is negative
    if m < 0:
        img2 = np.fliplr(np.flipud(img2))

    if global_options.show_progressbar:
        bar.finish()

    return Image(img2, s2)

