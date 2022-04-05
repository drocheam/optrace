#!/usr/bin/env python3
# encoding: utf-8

from Backend import *
from Frontend import *
import numpy as np


# Open the image form working directory
Image = './src/tv-test-pattern.png'

# make Raytracer
RT = Raytracer(outline=[-5, 5, -5, 5, 0, 40])

# add Raysource
RSS = Surface("Rectangle", dim=[4, 4])
RS = RaySource(RSS, light_type="RGB_Image", direction_type="Diverging", sr_angle=8,
               Image=Image, s=[0, 0, 1], pos=[0, 0, 0])
RT.add(RS)

# add Lens 1
front = Surface("Sphere", r=3, rho=1/8)
back = Surface("Sphere", r=3, rho=-1/8)
nL1 = RefractionIndex("Constant", n=1.5)
L1 = Lens(front, back, de=0.1, pos=[0, 0, 12], n=nL1)
RT.add(L1)

# add Detector
Det = Detector(Surface("Rectangle", dim=[10, 10]), pos=[0, 0, 36])
RT.add(Det)

# render and show detector image
pos = [27., 30., 33., 36.]
Ims = RT.iterativeDetectorImage(N_rays=10e6, N_px=500, pos=pos)

for i in np.arange(len(Ims)-1):
    DetectorPlot(Ims[i], block=False, mode="sRGB")
DetectorPlot(Ims[-1], block=True, mode="sRGB")

