#!/usr/bin/env python3

import optrace as ot
import optrace.plots as otp

# Demonstrates image formation by convolution of a resolution chart and a halo PSF.

# load a test image and define the image side lengths in mm
img = ot.presets.image.ETDRS_chart([1.5, 1.2])
# img = ot.presets.image.siemens_star([1.5, 1.5])
# img = ot.presets.image.color_checker([1.5, 1.2])

# halo preset, returns the psf and the side lengths
# sizes d1, d2, ... are in micrometers
psf = ot.presets.psf.halo(sig1=4, sig2=4, a=0.05, r=30)
# psf = ot.presets.psf.gaussian(sig=4)

# convolve
# use constant padding with white color, as the background is white
# keep_size slices the output back to the original image size
img_conv = ot.convolve(img, psf, m=0.75, padding_mode="constant", padding_value=1, keep_size=True)

# plot images
otp.image_plot(img, title="Initial Image")
otp.image_plot(psf, title="PSF")
otp.image_plot(img_conv, title="Convolved Image")
otp.block()
