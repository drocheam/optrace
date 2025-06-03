#!/usr/bin/env python3

import optrace as ot
from optrace.gui import TraceGUI

# This example demonstrates the diffraction approximation through Heisenberg uncertainty ray bending (HURB)
# of multiple aperture shapes.
# You can read more about HURB in the documentation
# Different apertures can be selected in the "custom" tab of the GUI.

# initialize raytracer with ambient medium and active HURB
RT = ot.Raytracer(outline=[-5, 5, -5, 5, -1, 40], use_hurb=True, n0=ot.RefractionIndex("Constant", 1.33))

# function for changing the aperture type at runtime
def change_aperture(RT, name):

    # remove old geometry
    RT.clear()

    # add new setup

    if name == "Square":
        rect = ot.RectangularSurface(dim=[0.05, 0.05])
        RS = ot.RaySource(rect, s=[0, 0, 1], pos=[0, 0, -1])
        RT.add(RS)
        ap_surf = ot.SlitSurface(dim=[2,2], dimi=[0.05, 0.05])
        ap = ot.Aperture(ap_surf, pos=[0, 0, 0])
        RT.add(ap)

    elif name == "Slit":
        rect = ot.RectangularSurface(dim=[0.05, 2])
        RS = ot.RaySource(rect, s=[0, 0, 1], pos=[0, 0, -1])
        RT.add(RS)
        ap_surf = ot.SlitSurface(dim=[2.5, 2.5], dimi=[0.05, 2])
        ap = ot.Aperture(ap_surf, pos=[0, 0, 0])
        RT.add(ap)
    
    elif name == "Edge":
        rect = ot.RectangularSurface(dim=[0.4, 1])
        RS = ot.RaySource(rect, s=[0, 0, 1], pos=[0, 0.5, -1])
        RT.add(RS)
        ap_surf = ot.SlitSurface(dim=[2, 2], dimi=[1.8, 1.8])
        ap = ot.Aperture(ap_surf, pos=[0, 0.9, 0])
        RT.add(ap)

    elif name == "Pinhole":
        RS = ot.RaySource(ot.CircularSurface(r=0.05), s=[0, 0, 1], pos=[0, 0, -1])
        RT.add(RS)
        ap_surf = ot.RingSurface(r=2, ri=0.025)
        ap = ot.Aperture(ap_surf, pos=[0, 0, 0])
        RT.add(ap)

    # add Detector
    DetS = ot.RectangularSurface(dim=[1.5, 1.5])
    Det = ot.Detector(DetS, pos=[0, 0, 30])
    RT.add(Det)

# apply the default aperture
change_aperture(RT, "Square")

# create and run the GUI. Add a custom UI element to change the aperture shape 
sim = TraceGUI(RT, ray_count=5000000, ray_opacity=0.05)
sim.add_custom_button("Detector Image", sim.detector_image)
sim.add_custom_selection("Aperture Type", ["Square", "Slit", "Edge", "Pinhole"], "Square", 
                         lambda name: change_aperture(RT, name))
sim.run()

