import sys
from threading import Thread  # threading

import numpy as np
import scipy.interpolate
from progressbar import progressbar, ProgressBar  

from . import color
from ..tracer import misc as misc


# TODO explain
# why rfft, why half of frequencies in x axis

def _grid_interp(x:     np.ndarray, y:     np.ndarray, 
                 f:     np.ndarray, xi:    np.ndarray, 
                 yi:    np.ndarray, 
                 k:     int)\
        -> np.ndarray:
    """
    
    """
    interp = scipy.interpolate.RectBivariateSpline(x, y, f.T, kx=k, ky=k)
    return interp(xi, yi).T


def _fourier_props(arr: np.ndarray, px: float, py: float)\
        -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    """
    # shift fft frequencies
    A = np.fft.fftshift(arr)

    # 2D Fourier Transform with orthogonal norm
    A_F = np.fft.rfft2(A, norm='ortho')
       
    # get shifted frequencies
    x = np.fft.rfftfreq(arr.shape[1], d=px)
    y = np.fft.fftshift(np.fft.fftfreq(A_F.shape[0], d=py))

    return A_F, x, y
   

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
    """

    inx, iny, isx, isy, ipx, ipy, iex, iey = iprop
    pnx, pny, psx, psy, ppx, ppy = pprop

    # extend image by black borders
    img_p = np.pad(img, (iey, iex), mode="constant", constant_values=0)

    # interpolate PSF
    sc = max(ppx/ipx, ppy/ipy)
    sc = min(sc, 3)
    if sc > 1:
        x = np.linspace(0, 1, psf.shape[1])
        y = np.linspace(0, 1, psf.shape[0])
        xi = np.linspace(0, 1, round(sc * psf.shape[1]))
        yi = np.linspace(0, 1, round(sc * psf.shape[0]))
        psf_i = _grid_interp(x, y, psf, xi, yi, k)
        ppx, ppy = psx/xi.shape[0], psy/yi.shape[0]
    else:
        psf_i = psf

    if not silent and (not i or not threading):
        bar.update(3 + i*4)

    # extend psf by black borders
    pny, pnx = psf_i.shape
    psf_p = np.pad(psf_i, (pny, pnx), mode="constant", constant_values=0)

    # get fourier transformed images and frequency vectors
    F_psf0, fx, fy = _fourier_props(psf_p, ppx, ppy)
    F_img0, fxi, fyi = _fourier_props(img_p, ipx, ipy)

    if not silent and (not i or not threading):
        bar.update(4 + i*4)

    # interpolate PSF frequency image
    F_psf0 = np.fft.ifftshift(F_psf0, axes=0)
    F_psf0i_r = _grid_interp(fx, fy, np.real(F_psf0), fxi, fyi, k)
    F_psf0i_i = _grid_interp(fx, fy, np.imag(F_psf0), fxi, fyi, k)
    F_psf0i = F_psf0i_r + F_psf0i_i*1j

    # no extrapolation, set outside values to zero
    F_psf0i[(fyi < fy[0]) | (fyi > fy[-1])] = 0
    F_psf0i[:, fxi > fx[-1]] = 0

    # shift back
    F_psf0i = np.fft.fftshift(F_psf0i, axes=0)

    # multiplication in fourier domain = convolution in original domain
    F_img2 = F_psf0i * F_img0

    if not silent and (not i or not threading):
        bar.update(5 + i*4)

    # transform back
    img2_0 = np.fft.irfft2(F_img2, norm='ortho')  # reverse transformation
    img2_0 = np.fft.ifftshift(img2_0)  # reverse quadrant shift

    if not silent and (not i or not threading):
        bar.update(6 + i*4)
    
    img2[:, :, i] = img2_0
    F_img[:, :, i] = np.abs(F_img0)
    F_psf[:, :, i] = np.abs(F_psf0i)

# TODO test psf and image format and shape

def convolve(img:       np.ndarray | str, 
             s_img:     list[float, float], 
             psf:       np.ndarray | str, 
             s_psf:     list[float, float], 
             k:         int = 3,
             silent:    bool = False, 
             threading: bool = True)\
        -> tuple[np.ndarray, list[float], dict]:
    """
    """

    # create progress bar
    bar_steps = 7 + 2*5*(1 - threading) 
    bar = ProgressBar(fd=sys.stdout, prefix="Convolving: ", 
                      max_value=bar_steps).start() if not silent else None

    # load image and convert to linear sRGB
    img = misc.load_image(img) if isinstance(img, str) else img / np.max(img)
    img_lin = color.srgb_to_srgb_linear(img)

    if not silent:
        bar.update(1)

    # load psf and convert to linear sRGB
    psf = misc.load_image(psf) if isinstance(psf, str) else psf / np.max(psf)
    psf_lin = color.srgb_to_srgb_linear(psf)
    
    if not silent:
        bar.update(2)

    # image and psf properties
    iny, inx = img.shape[:2]  # image pixel count
    isx, isy = s_img  # image side lengths
    ipx, ipy = isx/inx, isy/iny  # image pixel sizes
    pny, pnx = psf.shape[:2]  # psf pixel counts
    psx, psy = s_psf  # psf side lengths
    ppx, ppy = psx/pnx, psy/pny  # psf pixel sizes
        
    if psx > isx or psy > isy:
        raise ValueError(f"Image size [{isx:.5g}, {isy:.5g}] is smaller than PSF size [{psx:.5g}, {psy:.5g}].")

    if pnx*pny > 4e6:
        raise ValueError("PSF needs to be smaller than 4MP")

    if inx*iny > 4e6:
        raise ValueError("Image needs to be smaller than 4MP")

    if ppx > ipx or ppy > ipy:
        if not silent:
            print(f"WARNING: PSF pixel sizes [{ppx:.5g}, {ppy:.5g}] larger than image pixel sizes"
                  f" [{ipx:.5g}, {ipy:.5g}], generally you want a PSF in a higher resolution")

    if inx*iny < 1e4:
        if not silent:
            print(f"WARNING: Low resolution image.")

    if pnx*pny < 1e3:
        if not silent:
            print(f"WARNING: Low resolution PSF.")

    if pnx*pny < 100:
        raise ValueError(f"PSF too small with shape {PSF.shape}.")
    
    if inx*iny < 100:
        raise ValueError(f"Image too small with shape {img.shape}.")

    # image extension size
    iex = np.ceil(psx / 2 / ipx).astype(int)
    iey = np.ceil(psy / 2 / ipy).astype(int)

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

    # normalize and convert image
    img2 /= np.max(img2)
    img2 = np.clip(img2, 0, 1)
    img2 = color.srgb_linear_to_srgb(img2)
    
    # new image side lengths
    s2 = [s_img[0]+2*iex*ipx, s_img[1]+2*iey*ipy]
    
    # debug dictonary
    dbg = dict(F_img=F_img, F_psf=F_psf, psf=psf, s_psf=s_psf, img=img, s_img=s_img)

    if not silent:
        bar.finish()

    return img2, s2, dbg

