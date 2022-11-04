#!/usr/bin/env python3

import numpy as np
import optrace as ot
from optrace.gui import TraceGUI

# brewster angle at b_ang = arctan(n/n0)
# maximum transmission for p-polarized rays
# reflected rays s-polarized

n = ot.RefractionIndex("Constant", n=1.55)
b_ang = np.arctan(1.55/1)
# for n=1.55, n0=1
# transmission p-polarized: 100%
# transmission s-polarized: 83.0%
# mean transmission unpolarized: 91.5%

RT = ot.Raytracer(outline=[-3, 3, -3, 3, -8, 12], threading=True, absorb_missing=False)

# source parameters
RSS = ot.CircularSurface(r=0.05)
spectrum = ot.LightSpectrum("Monochromatic", wl=550.)
s = [0, np.sin(b_ang), np.cos(b_ang)]

# create sources
RS0 = ot.RaySource(RSS, divergence="None", spectrum=spectrum, pos=[0.5, 0, -4], 
                  polarization="x", desc="x-pol")
RS1 = ot.RaySource(RSS, divergence="None", spectrum=spectrum, pos=[0, 0, -4], 
                  polarization="y", desc="y-pol")
RS2 = ot.RaySource(RSS, divergence="None", spectrum=spectrum, pos=[-0.5, 0, -4],
                  polarization="Random", desc="no pol")
RT.add(RS0)
RT.add(RS1)
RT.add(RS2)

# add refraction index steps
surf = ot.TiltedSurface(r=2, normal=[-np.sin(b_ang), 0, np.cos(b_ang)])
for i in np.arange(12):
    L = ot.Lens(surf, surf, d=0.2, pos=[-0.4, 0, 2*i*0.25], n=n)
    RT.add(L)

# add Detector
Det = ot.Detector(ot.RectangularSurface(dim=[6, 6]), pos=[0, 0, 12])
RT.add(Det)

# Instantiate the class and configure its traits.
sim = TraceGUI(RT, coloring_type="Power", ray_opacity=0.9)
sim.run()
