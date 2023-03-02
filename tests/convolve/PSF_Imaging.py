#!/usr/bin/env python3

import time
import sys
from threading import Thread  # threading

import numpy as np

import optrace as ot
import optrace.plots as otp


silent = True
interp_k = 3
threading = True
debug = True
log = True
log_exp = 3

img = "/mnt/Data/Dateien/Dokumente/Job/WMA TH 2022/Optrace/optrace/ressources/images/ETDRS_Chart_inverted.png"
img = "/mnt/Data/Dateien/Dokumente/Job/WMA TH 2022/Optrace/optrace/ressources/images/interior.jpg"
# img = "/mnt/Data/Dateien/Dokumente/Job/WMA TH 2022/Optrace/optrace/ressources/images/TestScreen_square.png"


# img = np.zeros((500, 500, 3))
# x = np.linspace(-1, 1, img.shape[0])
# y = np.linspace(-1, 1, img.shape[1])
# X, Y = np.meshgrid(x, y)
# img0 = np.exp(-0.5*(X**2 + Y**2)*20)
# img[:, :, 0] = img0
# img[:, :, 1] = img0
# img[:, :, 2] = img0

ilen = [15, 10]
psflen = [1, 1]

psf = np.zeros((401, 401, 3))
x = np.linspace(-1, 1, psf.shape[0])
y = np.linspace(-1, 1, psf.shape[1])
X, Y = np.meshgrid(x, y)
psf0 = np.exp(-0.5*(X**2 + Y**2)*20)*(1+np.cos(30*X))*0.5
psf[:, :, 0] = psf0
psf[:, :, 1] = psf0
psf[:, :, 2] = psf0


import time
start = time.time()
# img = _load_img(img)
img2, s, dbg = ot.convolve(img, ilen, psf, psflen, silent=silent,
                           threading=threading, k=interp_k)
print(time.time() - start)

if debug:
    otp.convolve_debug_plots(img2, s, dbg, block=True, log=log, log_exp=log_exp)
