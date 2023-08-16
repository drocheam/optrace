#!/usr/bin/env python3

import optrace as ot
import optrace.plots as otp


# load a test image
img = ot.presets.image.ETDRS_chart
# img = ot.presets.image.siemens_star
# img = ot.presets.image.color_checker

# define the image lengths in mm
s_img = [1.5, 1.2]

# halo preset, returns the psf and the side lengths
# sizes d1, d2, ... are in micrometers
psf, s_psf = ot.presets.psf.halo(sig1=6, sig2=2, a=0.2, r=40)

# convolve
img2, s2 = ot.convolve(img, s_img, psf, s_psf, m=0.75)

# plot images
otp.image_plot(img, s_img, title="Initial Image")
otp.image_plot(psf, s_psf, title="PSF")
otp.image_plot(img2, s2, title="Convoluted Image", block=True)
