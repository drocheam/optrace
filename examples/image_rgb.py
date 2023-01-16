#!/usr/bin/env python3

import optrace as ot
from optrace.gui import TraceGUI

Image = ot.presets.image.test_screen

# make raytracer
RT = ot.Raytracer(outline=[-5, 5, -5, 5, 0, 40])

# add Raysource
RSS = ot.RectangularSurface(dim=[4, 4])
RS = ot.RaySource(RSS, divergence="Lambertian", div_angle=8, image=Image, s=[0, 0, 1], pos=[0, 0, 0])
RT.add(RS)

# add Lens 1
front = ot.SphericalSurface(r=3, R=8)
back = ot.SphericalSurface(r=3, R=-8)
nL1 = ot.RefractionIndex("Constant", n=1.5)
L1 = ot.Lens(front, back, de=0.1, pos=[0, 0, 12], n=nL1)
RT.add(L1)

# add Detector
DetS = ot.RectangularSurface(dim=[10, 10])
Det = ot.Detector(DetS, pos=[0, 0, 36.95])
RT.add(Det)

# Instantiate the class and configure its traits.
sim = TraceGUI(RT)
sim.run()
