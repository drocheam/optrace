#!/usr/bin/env python3

import optrace as ot
import optrace.plots as otp


# load a resolution chart
img = ot.presets.image.ETDRS_chart

# define the image lengths in mm
s_img = [1.5, 1.2]

# halo preset, returns the psf and the side lengths
psf, s_psf = ot.presets.psf.halo(d1=8, d2=40, a=0.1, w=2)

# convolve
img2, s, dbg = ot.convolve(img, s_img, psf, s_psf, m=0.75)

# plot the images as well as the Fourier images
otp.convolve_debug_plots(img2, s, dbg, block=True)
