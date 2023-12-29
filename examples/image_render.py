#!/usr/bin/env python3

import optrace as ot
from optrace.gui import TraceGUI

# test image
image = ot.presets.image.tv_testcard1([4, 4])
# image = ot.presets.image.grid([4, 4])

# make raytracer
RT = ot.Raytracer(outline=[-5, 5, -5, 5, 0, 40])

# add Raysource
RS = ot.RaySource(image, divergence="Lambertian", div_angle=8, s=[0, 0, 1], pos=[0, 0, 0])
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
