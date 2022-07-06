#!/usr/bin/env python3
# encoding: utf-8

import sys
sys.path.append('./')

import optrace as ot
from optrace.plots import *
import numpy as np

Image_path = ot.preset_image_test_screen

# make Raytracer
RT = ot.Raytracer(outline=[-5, 5, -5, 5, 0, 40])

# add Raysource
RSS = ot.Surface("Rectangle", dim=[4, 4])
RS = ot.RaySource(RSS, direction="Diverging", div_angle=8,
                  Image=Image_path, s=[0, 0, 1], pos=[0, 0, 0])
RT.add(RS)

# add Lens 1
front = ot.Surface("Sphere", r=3, rho=1/8)
back = ot.Surface("Sphere", r=3, rho=-1/8)
nL1 = ot.RefractionIndex("Constant", n=1.5)
L1 = ot.Lens(front, back, de=0.1, pos=[0, 0, 12], n=nL1)
RT.add(L1)

# add Detector
Det = ot.Detector(ot.Surface("Rectangle", dim=[10, 10]), pos=[0, 0, 36])
RT.add(Det)

# render and show detector image
pos = [27., 30., 33., 36.]
_, Ims = RT.iterativeRender(N_rays=10e6, N_px_D=100, pos=pos)

# show rendered images
for i in np.arange(len(Ims)-1):
    RImagePlot(Ims[i], block=False, mode="sRGB (Absolute RI)")
RImagePlot(Ims[-1], block=True, mode="sRGB (Absolute RI)")

