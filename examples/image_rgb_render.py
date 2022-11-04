#!/usr/bin/env python3

import numpy as np
import optrace as ot
from optrace.plots import *

Image_path = ot.presets.image.test_screen

# make raytracer
RT = ot.Raytracer(outline=[-5, 5, -5, 5, 0, 40])

# add Raysource
RSS = ot.RectangularSurface(dim=[4, 4])
RS = ot.RaySource(RSS, divergence="Lambertian", div_angle=8,
                  image=Image_path, s=[0, 0, 1], pos=[0, 0, 0])
RT.add(RS)

# add Lens 1
front = ot.SphericalSurface(r=3, R=8)
back = ot.SphericalSurface(r=3, R=-8)
nL1 = ot.RefractionIndex("Constant", n=1.5)
L1 = ot.Lens(front, back, de=0.1, pos=[0, 0, 12], n=nL1)
RT.add(L1)

# add Detector
Det = ot.Detector(ot.RectangularSurface(dim=[10, 10]), pos=[0, 0, 36])
RT.add(Det)

# render and show detector image
pos = [27., 30., 33., 36.]
_, Ims = RT.iterative_render(N_rays=15e6, N_px_D=200, pos=pos, no_sources=True)

# show rendered images
for i in np.arange(len(Ims)-1):
    r_image_plot(Ims[i], block=False, mode="sRGB (Absolute RI)")
r_image_plot(Ims[-1], block=True, mode="sRGB (Absolute RI)")
