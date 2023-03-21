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


def _grid_interp(x: np.ndarray, y:  np.ndarray, f:  np.ndarray, xi: np.ndarray, yi: np.ndarray, k: int)\
        -> np.ndarray:
    """Interpolate f on grid x, y at values xi, yi"""
    interp = scipy.interpolate.RectBivariateSpline(x, y, f.T, kx=k, ky=k)
    return interp(xi, yi).T


def _fourier_props(arr: np.ndarray, px: float, py: float)\
        -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Efficient FT of arr for real values (only calculates one half plane).
    Returns frequency image and frequency vectors
    """
    # shift fft frequencies
    A = np.fft.fftshift(arr)

    # 2D Fourier Transform with orthogonal norm
    # scipy.fft2 is a little bit faster than numpy.fft2
    workers = np.ceil(ne.detect_number_of_cores() / 3).astype(int) # divide by three since we are already threading
    A_F = scipy.fft.rfft2(A, norm='ortho', workers=workers)

    # get shifted frequencies
    fx = np.fft.rfftfreq(A.shape[1], d=px)  # axis with only positive frequencies
    fy = np.fft.fftshift(np.fft.fftfreq(A_F.shape[0], d=py))  # axis with positive and negative ones
    
    return A_F, fx, fy


def _interpolate_psf(psf: np.ndarray, k: int, iprop: list, pprop: list)\
        -> tuple[np.ndarray, float, float]:
    """
    Interpolate PSF so it has approximately the same pixel size as the image.
    (For the case that the PSF pixels are larger)
    """
    
    inx, iny, isx, isy, ipx, ipy, iex, iey = iprop
    pnx, pny, psx, psy, ppx, ppy = pprop
    
    sc = max(ppx/ipx, ppy/ipy)  # larger pixel side ratio
    sc = min(sc, 1001./max(psf.shape[0], psf.shape[1]))  # limit to a side length of 1000px

    if sc > 1:
        # scx = round(sc * psf.shape[1])
        # scy = round(sc * psf.shape[0])
        # x = np.linspace(-1, 1, psf.shape[1])
        # y = np.linspace(-1, 1, psf.shape[0])
        # xi = np.linspace(-1, 1, scx)
        # yi = np.linspace(-1, 1, scy)
        # psf_i = _grid_interp(x, y, psf, xi, yi, k)
        psf_i = scipy.ndimage.zoom(psf, sc, order=k)

        return psf_i, ppx/sc, ppy/sc

    else:
        return psf, ppx, ppy


def _threaded(img:          np.ndarray, 
              psf:          np.ndarray, 
              i:            int, 
              iprop:        tuple, 
              pprop:        tuple, 
              img2:         np.ndarray, 
              F_img:        list, 
              F_psf:        list,
              bar:          ProgressBar,
              k:            int,
              silent:       bool, 
              threading:    bool)\
        -> None:
    """

    :param img:
    :param psf:
    :param i:
    :param iprop:
    :param pprop:
    :param img2:
    :param F_img:
    :paramg F_psf:
    :param bar:
    :param k:
    :param silent:
    :param threading:
    """

    inx, iny, isx, isy, ipx, ipy, iex, iey = iprop
    pnx, pny, psx, psy, ppx, ppy = pprop

    # extend image by black borders
    img_p = np.pad(img, ((iey, iey), (iex, iex)), mode="constant", constant_values=0)

    # interpolate PSF
    psf_i, ppx, ppy = _interpolate_psf(psf, k, iprop, pprop)
    pny, pnx = psf_i.shape  # update pixel count

    if not silent and (not i or not threading):
        bar.update(3 + i*4)

    # extend psf by black borders
    psf_p = np.pad(psf_i, ((pny//2, pny//2), (pnx//2, pnx//2)), mode="constant", constant_values=0)

    # get fourier transformed images and frequency vectors
    F_psf0, fx, fy = _fourier_props(psf_p, ppx, ppy)
    F_img0, fxi, fyi = _fourier_props(img_p, ipx, ipy)

    if not silent and (not i or not threading):
        bar.update(4 + i*4)

    # interpolate PSF frequency image
    F_psf0 = np.fft.fftshift(F_psf0, axes=0)
    F_psf0i_r = _grid_interp(fx, fy, np.real(F_psf0), fxi, fyi, k)
    F_psf0i_i = _grid_interp(fx, fy, np.imag(F_psf0), fxi, fyi, k)
    F_psf0i = F_psf0i_r + F_psf0i_i*1j

    # no extrapolation, set outside values to zero
    F_psf0i[(fyi < fy[0]) | (fyi > fy[-1])] = 0
    F_psf0i[:, fxi > fx[-1]] = 0

    # # shift back
    F_psf0i = np.fft.ifftshift(F_psf0i, axes=0)

    # multiplication in fourier domain = convolution in original domain
    F_img2 = F_psf0i * F_img0

    if not silent and (not i or not threading):
        bar.update(5 + i*4)

    # transform back
    workers = np.ceil(ne.detect_number_of_cores() / 3).astype(int)
    img2_0 = scipy.fft.irfft2(F_img2, norm='ortho', s=img_p.shape, workers=workers)  # reverse transformation
    img2_0 = np.fft.ifftshift(img2_0)  # reverse quadrant shift

    if not silent and (not i or not threading):
        bar.update(6 + i*4)
   
    # assign channel to output
    img2[:, :, i] = img2_0
    F_img[:, :, i] = np.abs(F_img0)
    F_psf[:, :, i] = np.abs(F_psf0i)


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

def convolve(img:       np.ndarray | str, 
             s_img:     list[float, float], 
             psf:       np.ndarray | RImage, 
             s_psf:     list[float, float],
             m:         float = 1,
             k:         int = 3,
             silent:    bool = False, 
             threading: bool = True)\
        -> tuple[np.ndarray, list[float], dict]:
    """

    :param img:
    :param s_img:
    :param psf:
    :param s_psf:
    :param m: magnification factor, abs(m) > 1 means enlargement, abs(m) < 1 size reduction,
              m < 0 image flipping, m > 0 an upright image 
    :param k:
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
    pc.check_type("k", k, int)
    pc.check_above("k", k, 0)
    pc.check_type("silent", silent, bool)
    pc.check_type("threading", threading, bool)

    # function for raising the exception and finishing the progress bar
    def raise_(excp):
        if not silent:
            bar.finish()
        raise excp
    
    # create progress bar
    bar_steps = 7 + 2*5*(1 - threading) 
    bar = ProgressBar(fd=sys.stdout, prefix="Convolving: ", 
                      max_value=bar_steps).start() if not silent else None

    # Load image
    ###################################################################################################################
    ###################################################################################################################

    # load from files
    if isinstance(img, str):
        img_srgb = misc.load_image(img)

    # numpy array
    else:
        img_srgb = img.copy()
        if (img_max := np.max(img_srgb)):  # normalize
            img_srgb /= img_max
    
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

            # TODO why isn't this needed?
            # img_lin = color.srgb_linear_to_xyz(img_lin)
        
        if (psf_lin_max := np.max(psf_lin)):  # normalize
            psf_lin /= psf_lin_max

    # intensity array
    else:
        if psf.ndim != 2:
            raise_(ValueError(f"PSF needs to be a two dimensional array, but has shape {psf.shape}"))
        
        psf_lin = psf.copy()
        if (psf_max := np.max(psf)):  # normalize
            psf_lin /= psf_max
        
        # repeat intensities
        psf_lin = np.repeat(psf_lin[:, :, np.newaxis], 3, axis=2)
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

    # image extension size
    iex = 1 + np.ceil(psx / 2 / ipx).astype(int)
    iey = 1 + np.ceil(psy / 2 / ipy).astype(int)

    # pre-allocate output images
    img2 = np.zeros((iny+2*iey, inx+2*iex, 3))
    F_img = np.zeros((iny+2*iey, (inx+2*iex)//2 + 1, 3)) 
    F_psf = np.zeros((iny+2*iey, (inx+2*iex)//2 + 1, 3))

    # geometry parameters
    iprop = inx, iny, isx, isy, ipx, ipy, iex, iey
    pprop = pnx, pny, psx, psy, ppx, ppy

    # arguments for threading function
    args = iprop, pprop, img2, F_img, F_psf, bar, k, silent, threading

    if threading:
        thread_list = [Thread(target=_threaded, args=(img_lin[:, :, i], psf_lin[:, :, i], i, *args)) for i in range(3)]
        [thread.start() for thread in thread_list]
        [thread.join() for thread in thread_list]
    else:
        for i in range(3):
            _threaded(img_lin[:, :, i], psf_lin[:, :, i], i, *args)

    if not silent:
        bar.update(bar_steps-1)

    # Output Conversion
    ###################################################################################################################
    ###################################################################################################################

    # normalize and clip image
    if (img2_max := np.max(img2)):
        img2 /= img2_max
    img2 = np.clip(img2, 0, 1)

    # convert to sRGB
    img2 = color.xyz_to_srgb(img2) if psf_color else color.srgb_linear_to_srgb(img2)
    
    # new image side lengths
    s2 = [isx+2*iex*ipx, isy+2*iey*ipy]

    # flip image when m is negative
    if m < 0:
        img2 = np.fliplr(np.flipud(img2))

    # debug dictionary
    dbg = dict(F_img=F_img, F_psf=F_psf, psf=psf_srgb, s_psf=s_psf, img=img_srgb, s_img=s_img)

    if not silent:
        bar.finish()

    return img2, s2, dbg

