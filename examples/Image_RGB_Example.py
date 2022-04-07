#!/usr/bin/env python3
# encoding: utf-8

import sys
sys.path.append('./')

import optrace as ot
from optrace.Frontend import GUI as TraceGUI

Image = './examples/tv-test-pattern.png'

# make Raytracer
RT = ot.Raytracer(outline=[-5, 5, -5, 5, 0, 40])

# add Raysource
RSS = ot.Surface("Rectangle", dim=[4, 4])
RS = ot.RaySource(RSS, direction_type="Diverging", sr_angle=8, light_type="RGB_Image",
               Image=Image, s=[0, 0, 1], pos=[0, 0, 0])
RT.add(RS)

# add Lens 1
front = ot.Surface("Sphere", r=3, rho=1/8)
back = ot.Surface("Sphere", r=3, rho=-1/8)
nL1 = ot.RefractionIndex("Constant", n=1.5)
L1 = ot.Lens(front, back, de=0.1, pos=[0, 0, 12], n=nL1)
RT.add(L1)

# add Detector
DetS = ot.Surface("Rectangle", dim=[10, 10])
Det = ot.Detector(DetS, pos=[0, 0, 36])
RT.add(Det)

# Instantiate the class and configure its traits.
sim = TraceGUI(RT)
sim.run()

