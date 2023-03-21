import sys
from threading import Thread  # threading

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


def _color_check(img: np.ndarray, th: float = 1e-3) -> bool:
    """
    Check if an image has color information. 
    If any channel difference is above th this function returns True.

    :param img: img to check
    :param th: sRGB value threshold
    :return:  if image has color information 
    """
    i0, i1, i2 = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    return np.any(ne.evaluate("abs(i0-i1) > th")) or np.any(ne.evaluate("abs(i1-i2) > th"))


def _safe_normalize(img: np.ndarray) -> np.ndarray:
    """normalize array by its maximum and handle the case of a zero array"""
    img_max = np.max(img)
    return img / img_max if img_max else img.copy()


def _threaded(img:      np.ndarray, 
              psf:      np.ndarray, 
              i:        int, 
              img_out:  np.ndarray, 
              psfm:     float, 
              scx:      float, 
              pnx:      int, 
              scy:      float, 
              pny:      int)\
        -> None:

    # psf too small -> dirac pulse with correct color
    if scy*pny < 1 and scx*pnx < 1:
        psf2 = np.array([[psfm]])

    # zoom (=rescale) PSF
    else:
        psf2 = scipy.ndimage.zoom(psf, [scy, scx], order=3, mode="constant", grid_mode=False, prefilter=True)

    # pad psf
    psf2 = np.pad(psf2, ((4, 4), (4, 4)), mode="constant", constant_values=0)

    # do convolution
    img_out[:, :, i] = scipy.signal.fftconvolve(img, psf2, mode="same")


def convolve(img:                np.ndarray | str, 
             s_img:              list[float, float], 
             psf:                np.ndarray | RImage, 
             s_psf:              list[float, float],
             m:                  float = 1,
             rendering_intent:   str =  "Absolute",
             silent:             bool = False, 
             threading:          bool = True)\
        -> tuple[np.ndarray, list[float]]:
    """

    :param img:
    :param s_img:
    :param psf:
    :param s_psf:
    :param m: magnification factor, abs(m) > 1 means enlargement, abs(m) < 1 size reduction,
              m < 0 image flipping, m > 0 an upright image
    :param rendering_intent:
    :param silent:
    :param threading:
    :return:
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
    img_color = _color_check(img_lin)  # if it has color information

    if not silent:
        bar.update(1)

    # Load PSF
    ###################################################################################################################
    ###################################################################################################################

    # rendered image with color information
    if isinstance(psf, RImage):
        psf_srgb = psf.get("sRGB (Absolute RI)")

        if img_color:
            # use irradiance and stack in channels
            psf_lin = np.repeat(psf.get("Irradiance")[:, :, np.newaxis], 3, axis=2)
            psf_color = False

            if not silent:
                print("WARNING: Image has color information, using irradiance of the PSF")
        else:
            psf_lin = psf.xyz()  # use XYZ color space as it includes all human visible colors
            psf_color = True
        
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
        psf_srgb = color.srgb_linear_to_srgb(psf_lin)  # srgb image for debug dictionary
        psf_color = False

    if not silent:
        bar.update(2)

    # Shape Checks
    ###################################################################################################################
    ###################################################################################################################

    # image and psf properties
    iny, inx = img_lin.shape[:2]  # image pixel count
    isx, isy = s_img[0]*abs(m), s_img[1]*abs(m)  # image side lengths scaled by magnification magnitude
    ipx, ipy = isx/(inx-1), isy/(iny-1)  # image pixel sizes
    pny, pnx = psf_lin.shape[:2]  # psf pixel counts
    psx, psy = s_psf  # psf side lengths
    ppx, ppy = psx/(pnx-1), psy/(pny-1)  # psf pixel sizes
      
    if psx > 2*isx or psy > 2*isy:
        raise_(ValueError(f"m-scaled image size [{isx:.5g}, {isy:.5g}] is more than two times" 
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

    # Convolution
    ###################################################################################################################
    ###################################################################################################################

    # image padding size
    iex = 1 + np.ceil(psx / 2 / ipx).astype(int)
    iey = 1 + np.ceil(psy / 2 / ipy).astype(int)

    # pad input image
    img_lin = np.pad(img_lin, ((iey, iey), (iex, iex), (0, 0)), mode="constant", constant_values=0)

    # output image
    img2 = np.zeros_like(img_lin)

    # mean color, used for dirac pulse if PSF is too small
    psfm = np.mean(psf_lin, axis=(0, 1))

    # psf side scaling factors
    scx = ppx / ipx
    scy = ppy / ipy

    if not silent:
        bar.update(3)

    if not threading:
        for i in range(3):
            _threaded(img_lin[:, :, i], psf_lin[:, :, i], i, img2, psfm[i], scx, pnx, scy, pny)
   
    else:
        # create threads
        thread_list = [Thread(target=_threaded, 
                              args=(img_lin[:, :, i], psf_lin[:, :, i], i, img2, psfm[i], scx, pnx, scy, pny))
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
    img2 = _safe_normalize(img2)

    # convert sRGB linear image to XYZ
    if not psf_color:
        img2 = color.srgb_linear_to_xyz(img2)
    
    # convert to sRGB
    img2 = color.xyz_to_srgb(img2, rendering_intent=rendering_intent) 
    
    # new image side lengths
    s2 = [isx+2*iex*ipx, isy+2*iey*ipy]

    # flip image when m is negative
    if m < 0:
        img2 = np.fliplr(np.flipud(img2))

    if not silent:
        bar.finish()

    return img2, s2

