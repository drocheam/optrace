#!/bin/env python3

import sys
sys.path.append('.')

import unittest
import numpy as np
import scipy.interpolate
import scipy.fft
import numexpr as ne

import scipy.signal
import scipy.ndimage

import optrace as ot
import optrace.plots as otp


psf, s_psf = ot.presets.psf.halo(1, 100, 1000, 10)


psf2 = scipy.ndimage.zoom(psf, [0.05, 0.05], order=5, mode="constant", grid_mode=True, prefilter=True)
otp.image_plot(psf2, s_psf)
otp.image_plot(psf, s_psf, block=True)
