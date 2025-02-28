#!/usr/bin/env python3

import numpy as np
import optrace as ot
from optrace.gui import TraceGUI

# A prism example where light is split into its spectral components.
# Light spectrum and materials are parameterizable through the "Custom" GUI tab.

RS_spectrum = ot.presets.light_spectrum.d65
n = ot.presets.refraction_index.LAK8

# print the abbe number
print(f"Abbe Number of {n.desc}: {n.abbe_number():.4g}")

# make raytracer
RT = ot.Raytracer(outline=[-5, 5, -5, 5, -5, 25])

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

# add Detector
Det = ot.Detector(ot.RectangularSurface(dim=[10, 10]), pos=[0, 0, 20])
RT.add(Det)

# function for changing the spectrum from the GUI
def change_spec(RT, spec):

    if spec == "D65":
        RT.ray_sources[0].spectrum = ot.presets.light_spectrum.d65
    elif spec == "LED-B5":
        RT.ray_sources[0].spectrum = ot.presets.light_spectrum.led_b1
    elif spec == "F11":
        RT.ray_sources[0].spectrum = ot.presets.light_spectrum.f11

# function for changing the material from the GUI
def change_material(RT, mat):

    if mat == "LAK8":
        n = ot.presets.refraction_index.LAK8
    elif mat == "BASF64":
        n = ot.presets.refraction_index.BASF64
    else:
        n = ot.presets.refraction_index.SF10

    RT.lenses[0].n = n

    # print the abbe number
    print(f"Abbe Number of {n.desc}: {n.abbe_number():.4g}")

# run the simulator
sim = TraceGUI(RT, ray_count=1000000, coloring_mode="Wavelength", image_mode="sRGB (Perceptual RI)")
sim.add_custom_selection("Spectrum", ["D65", "LED-B5", "F11"], "D65", lambda spec: change_spec(RT, spec))
sim.add_custom_selection("Material", ["LAK8", "BASF64", "SF10"], "LAK8", lambda mat: change_material(RT, mat))
sim.add_custom_button("Detector Image", sim.detector_image)
sim.add_custom_button("Detector Spectrum", sim.detector_spectrum)
sim.run()
