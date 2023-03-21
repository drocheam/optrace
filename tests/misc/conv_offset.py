#!/bin/env python3



import numpy as np




psf, s_psf = ot.presets.psf.gaussian(d=2)
img, s_img = ot.presets.psf.gaussian(d=1)

img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
img = color.srgb_linear_to_srgb(img)





