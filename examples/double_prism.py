#!/usr/bin/env python3

import numpy as np
import optrace as ot
from optrace.gui import TraceGUI

# Assignments:
# 1. replace the ray source spectrum below with led_b3, F_eC_ and others from ot.presets 
#    and compare the detector image as well as the detector spectrum using the "Imaging" tab in the GUI
# 2. replace the glass refractive index below by different presets (BASF64, LAK8, ...)
#    and compare the abbe number as well as the effect on the dispersion
# 3. check the documentation for the difference between image modes "sRGB (Absolute RI)" and "sRGB (Perceptual RI)"
#    and compare the image modes by rendering both images

RS_spectrum = ot.presets.light_spectrum.d50
n = ot.presets.refraction_index.SF10
# n = ot.presets.refraction_index.LAK8

# print the abbe number
print(f"Abbe Number of {n.desc}: {n.abbe_number():.4g}")

# make raytracer
RT = ot.Raytracer(outline=[-5, 5, -5, 5, -1, 27.5])

# add Raysource
RSS = ot.CircularSurface(r=0.05)
RS = ot.RaySource(RSS, divergence="None", spectrum=RS_spectrum,
                  pos=[0, -2.5, 0], s=[0, 0.3, 0.7])
RT.add(RS)

# Prism 1
# the surfaces are tilted circles specified by a normal vector
front = ot.TiltedSurface(r=3, normal=[0, -0.45, np.sqrt(1-0.45**2)])
back = front.copy()
back.rotate(180) # back is a flipped copy of the first surface
L1 = ot.Lens(front, back, de=0.5, pos=[0, 0, 10], n=n)
RT.add(L1)

# Prism 2 is a shifted and rotated version of the first prism
L2 = L1.copy()
L2.rotate(180)
L2.move_to([0, 0, 16.5])
RT.add(L2)

# add Detector
Det = ot.Detector(ot.RectangularSurface(dim=[5, 5]), pos=[0, 0, 23.])
RT.add(Det)

# run the simulator
sim = TraceGUI(RT, coloring_type="Wavelength")
sim.run()
