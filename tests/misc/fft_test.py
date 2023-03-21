#!/bin/env python3

import sys
sys.path.append('.')

import unittest
import numpy as np
import scipy.interpolate
import scipy.fft
import numexpr as ne



def _grid_interp(x: np.ndarray, y:  np.ndarray, f:  np.ndarray, xi: np.ndarray, yi: np.ndarray, k: int)\
        -> np.ndarray:
    """Interpolate f on grid x, y at values xi, yi"""
    def adapt(x):
        return x
        # return  x / (x[-1]  - x[0]) * (x[-1] - x[0] + (x[1] - x[0])/2) 
    
    interp = scipy.interpolate.RectBivariateSpline(adapt(x), adapt(y), f.T, kx=k, ky=k)
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


a = np.random.sample((101, 200))


ap = np.pad(a, ((2, 2), (3, 3)), mode="constant")


af, fx, fy = _fourier_props(ap, 1, 1)


k = 3
afr = _grid_interp(fx, fy, np.real(af), fx, fy, k)
afi = _grid_interp(fx, fy, np.imag(af), fx, fy, k)

af = afi*1j + afr

# ash = np.fft.fftshift(ap)

# af = np.fft.rfft2(ash)


ab = np.fft.irfft2(af, s=ap.shape)


absh = np.fft.ifftshift(ab)

print(np.max(absh - ap))
