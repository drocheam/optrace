#!/usr/bin/env python3

import numpy as np
import optrace as ot
from optrace.gui import TraceGUI

# make raytracer
RT = ot.Raytracer(outline=[-15, 15, -10, 10, -40, 30])

oa_angle=20
z0 = -30

x0 = np.tan(np.radians(oa_angle)) * z0

RSS = ot.Point()
RS0 = ot.RaySource(RSS, divergence="Isotropic", div_2d=True, spectrum=ot.presets.light_spectrum.d65,
                  pos=[x0, 0, z0], s=[-x0, 0, -z0], div_angle=2, div_axis_angle=90, desc="Sagittal")
RT.add(RS0)

RS1 = ot.RaySource(RSS, divergence="Isotropic", div_2d=True, spectrum=ot.presets.light_spectrum.d65,
                  pos=[x0, 0, z0], s=[-x0, 0, -z0], div_angle=2, div_axis_angle=0, desc="Meridional")
RT.add(RS1)

n = ot.RefractionIndex("Constant", n=1.5)

# Lens
front = ot.Surface("Sphere", r=3, R=10)
back = ot.Surface("Sphere", r=3, R=-10)
L = ot.Lens(front, back, d1=0, d2=1, pos=[0, 0, 0], n=n)
RT.add(L)

# add Detector
Det = ot.Detector(ot.Surface("Rectangle", dim=[30, 15]), pos=[0, 0, 23.])
RT.add(Det)

# run the simulator
sim = TraceGUI(RT, coloring_type="Source", ray_opacity=0.12)
sim.run()
