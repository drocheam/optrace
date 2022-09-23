#!/usr/bin/env python3

import optrace as ot
from optrace.gui import TraceGUI

# make raytracer
RT = ot.Raytracer(outline=[-10, 10, -10, 10, -25, 40])

RSS0 = ot.Surface("Circle", r=1)
RS0 = ot.RaySource(RSS0, divergence="None", spectrum=ot.presets.light_spectrum.d65,
                  pos=[0, 0, -15], s=[0, 0, 1])
RT.add(RS0)

RSS1 = ot.Surface("Ring", r=4.5, ri=1)
RS1 = ot.RaySource(RSS1, divergence="None", spectrum=ot.presets.light_spectrum.d65,
                  pos=[0, 0, -15], s=[0, 0, 1])
RT.add(RS1)

n = ot.RefractionIndex("Constant", n=1.5)

# Lens
front = ot.Surface("Sphere", r=5, R=15)
back = ot.Surface("Sphere", r=5, R=-15)
L = ot.Lens(front, back, de=0.2, pos=[0, 0, 0], n=n)
RT.add(L)

# add Detector
DETS = ot.Surface("Rectangle", dim=[20, 20])
DET = ot.Detector(DETS, pos=[0, 0, 23.])
RT.add(DET)

# run the simulator
sim = TraceGUI(RT, coloring_type="Source", ray_alpha=-0.8)
sim.run()
