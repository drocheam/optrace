#!/usr/bin/env python3

import numpy as np
import optrace as ot
from optrace.gui import TraceGUI

# This script showcases astigmatism by simulating sagittal and meridional off-axis rays.
# You can control their angle by changing their setting in the "Custom" Tab in the GUI.
# Sagittal (blue) and meridional (red) rays are highlighted in different colors, 
# making their focal positions easier to visualize.

oa_angle = 20  # input angle of sources
z0 = -30  # position of sources

# change input angle of sagittal and meridonal sources
def change_sources(RT, oa_angle):

    RT.remove(RT.ray_sources)

    y0 = np.tan(np.radians(oa_angle)) * z0
    RSS = ot.Point()
    RS0 = ot.RaySource(RSS, divergence="Isotropic", div_2d=True, spectrum=ot.presets.light_spectrum.d65,
                      pos=[0, y0, z0], s_sph=[oa_angle, 90], div_angle=2, div_axis_angle=0, desc="Sagittal\n")
    RT.add(RS0)

    RS1 = ot.RaySource(RSS, divergence="Isotropic", div_2d=True, spectrum=ot.presets.light_spectrum.d65,
                      pos=[0, y0, z0], s_sph=[oa_angle, 90], div_angle=2, div_axis_angle=90, desc="Meridional")
    RT.add(RS1)

# make raytracer
RT = ot.Raytracer(outline=[-10, 10, -25, 25, -40, 30])

# add sources
change_sources(RT, oa_angle)

# Lens
front = ot.SphericalSurface(r=3, R=10)
back = ot.SphericalSurface(r=3, R=-10)
n = ot.RefractionIndex("Constant", n=1.5)
L = ot.Lens(front, back, d1=0, d2=1, pos=[0, 0, 0], n=n)
RT.add(L)

# add Detector
Det = ot.Detector(ot.RectangularSurface(dim=[20, 50]), pos=[0, 0, 23.])
RT.add(Det)

# run the simulator
sim = TraceGUI(RT, coloring_mode="Source", ray_opacity=0.12)
sim.add_custom_value("Input angle (-35° - +35°)", oa_angle, lambda ang: change_sources(RT, ang))
sim.run()
