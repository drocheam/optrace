#!/usr/bin/env python3

import optrace as ot
from optrace.gui import TraceGUI
import numpy as np

# A simple imaging system consisting of a single lens. 
# Spherical aberration and distortion are apparent.
# By using the aperture stop the aberrations can be limited, approximating the paraxial case for a very small diameter.
# The size of the stop and the test image are parameterizable through the "Custom" GUI tab. 

# make raytracer
RT = ot.Raytracer(outline=[-15, 15, -15, 15, 0, 40], use_hurb=True)


# Pinhole
# RS = ot.RaySource(ot.CircularSurface(r=0.05), s=[0, 0, 1], pos=[0, 0, 0])
# RT.add(RS)
# ap_surf = ot.RingSurface(r=5, ri=0.052)
# ap = ot.Aperture(ap_surf, pos=[0, 0, 5])
# RT.add(ap)



# Ideal Lens
# RS = ot.RaySource(ot.CircularSurface(r=2), s=[0, 0, 1], pos=[0, 0, 0])
# RT.add(RS)
# ap_surf = ot.RingSurface(r=3.001, ri=2.001)
# ap = ot.Aperture(ap_surf, pos=[0, 0, 5])
# RT.add(ap)
# L = ot.IdealLens(3, 50, pos=[0, 0, 5.01])
# RT.add(L)

# Lens
# RS = ot.RaySource(ot.CircularSurface(r=0.5), s=[0, 0, 1], pos=[0, 0, 0])
# RT.add(RS)
# front = ot.SphericalSurface(r=3, R=8)
# back = ot.SphericalSurface(r=3, R=-8)
# nL1 = ot.RefractionIndex("Abbe", n=1.5, V=40)
# L1 = ot.Lens(front, back, de=0.1, pos=[0, 0, 5.01], n=nL1)
# RT.add(L1)
# ap_surf = ot.RingSurface(r=3.001, ri=0.501)
# ap = ot.Aperture(ap_surf, pos=[0, 0, L1.front.pos[2]-0.001])
# RT.add(ap)


# slit
rect = ot.RectangularSurface(dim=[0.1/2, 0.1])
rect.rotate(15)
RS = ot.RaySource(rect, s=[0, 0, 1], pos=[0, 0, 0])
RT.add(RS)
ap_surf = ot.SlitSurface(dim=[2,2], dimi=[0.1001/2, 0.1001])
ap_surf.rotate(15)
ap = ot.Aperture(ap_surf, pos=[0, 0, 1])
RT.add(ap)

# add Detector
DetS = ot.RectangularSurface(dim=[1, 1])
# DetS = ot.RectangularSurface(dim=[0.1, 0.1])
Det = ot.Detector(DetS, pos=[0, 0, 36])
RT.add(Det)

# create and run the GUI. Add two custom GUI elements to control the aperture and test image
sim = TraceGUI(RT, ray_count=5000000, ray_opacity=0.05)
sim.run()
