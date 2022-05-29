#!/usr/bin/env python3
# encoding: utf-8

import sys
sys.path.append('./')

import numpy as np
import optrace as ot
from optrace.gui import TraceGUI
from optrace.plots import SpectrumPlot

# make Raytracer
RT = ot.Raytracer(outline=[-4, 5, -3, 5, -1, 27.5])

# add Raysource
RSS = ot.Surface("Circle", r=0.05)
RS = ot.RaySource(RSS, direction_type="Parallel", spectrum=ot.preset_spec_D65,
               pos=[-2.5, 0, 0], s=[0.3, 0, 0.7])
RT.add(RS)

# prism surface
P1 = ot.SurfaceFunction(r=3, func=lambda x, y: 0.5*x, derivative=lambda x, y: (np.full_like(x, 0.5), 0*y))
P2 = ot.SurfaceFunction(r=3, func=lambda x, y: -0.5*x, derivative=lambda x, y: (np.full_like(x, -0.5), 0*y))

# Prism 1
front = ot.Surface("Function", func=P1)
back = ot.Surface("Function", func=P2)
nL1 = ot.preset_n_SF10  # glass with small abbe number => much dispersion
L1 = ot.Lens(front, back, de=0.5, pos=[0, 0, 10], n=nL1)
RT.add(L1)

print("Abbe Number: ", nL1.getAbbeNumber())

# Prism 2
# back and front of Prism 1 are switched,
L2 = ot.Lens(back, front, de=0.5, pos=[0, 0, 16.5], n=nL1)
RT.add(L2)

# add Detector
Det = ot.Detector(ot.Surface("Rectangle", dim=[5, 5]), pos=[0, 0, 23.])
RT.add(Det)

# Instantiate the class and configure its traits.
sim = TraceGUI(RT, ColoringType="Wavelength")
sim.run()

