#!/usr/bin/env python3

import numpy as np
import optrace as ot
from optrace.gui import TraceGUI

# make raytracer
RT = ot.Raytracer(outline=[-10, 10, -10, 10, -10, 60])

# Ray Sources
for x, sx in zip([0, -4, -8, -8, 0, -8, -8, 2.5], [0, 0, 8, 16, 4.5, 13, 6, 0.65]):
    RSS = ot.Point()
    RS = ot.RaySource(RSS, divergence="None", spectrum=ot.presets.light_spectrum.FDC,
                      pos=[x, 0, 0], s=[sx, 0, 10])
    RT.add(RS)

# Data
X, Y = np.mgrid[-3:3:200j, -3:3:200j]
Z = -(X**2 + Y**2)/5

# add Lens 1
front = ot.DataSurface2D(r=3, data=Z)
back = ot.DataSurface2D(r=3.7, data=Z)
nL1 = ot.RefractionIndex("Constant", n=1)
L1 = ot.Lens(front, back, d=1.1, pos=[0, 0, 10], n=nL1)
RT.add(L1)

# one ray for each source
cnt = len(RT.ray_sources)
# RT.trace(cnt)


gui = TraceGUI(RT, ray_count=cnt, ray_opacity=1)
gui.run()

