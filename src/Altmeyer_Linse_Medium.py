#!/usr/bin/env python3
# encoding: utf-8

from Backend import *
from Frontend import *

# make Raytracer
RT = Raytracer(outline=[-5, 5, -5, 5, -40, 40], n0=RefractionIndex("Constant", n=1.3))

# add Raysource1
RSS1 = Surface("Circle", r=0.00001)
RS1 = RaySource(RSS1, direction_type="Diverging", sr_angle=0.2, light_type="D65",
                s=[0, 0, 1], pos=[0, 0, -30])
RT.add(RS1)

# Refraction Indices
nL1 = RefractionIndex("Constant", n=1.5)
nL2 = RefractionIndex("Constant", n=1.3)

# add Lens 1
front = Surface("Sphere", r=0.1, rho=1/8)
back = Surface("Sphere", r=0.1, rho=-1/8)
L1 = Lens(front, back, de=0, pos=[0, 0, 0], n=nL1, n2=nL2)
RT.add(L1)

# add Detector 1
DetS = Surface("Rectangle", dim=[10, 10])
Det = Detector(DetS, pos=[0, 0, 25])
RT.add(Det)

# Instantiate the GUI
sim = GUI(RT)
sim()


