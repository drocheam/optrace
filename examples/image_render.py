#!/usr/bin/env python3

import optrace as ot
from optrace.gui import TraceGUI
import numpy as np

# A simple imaging system consisting of a single lens. 
# Spherical aberration and distortion are apparent.
# By using the aperture stop the aberrations can be limited, approximating the paraxial case for a very small diameter.
# The size of the stop and the test image are parameterizable through the "Custom" GUI tab. 

# make raytracer
RT = ot.Raytracer(outline=[-5, 5, -5, 5, 0, 40])

source_aperture_properties = ["Testcard", 3]

# function for creating/replacing a ray source creating/replacing an aperture
def change_ray_source_and_aperture(RT, image_name, ap_r):

    if image_name == "Testcard":
        image = ot.presets.image.tv_testcard1([4, 4])
    else:
        image = ot.presets.image.grid([4, 4])

    # add Raysource
    # orient the light direction cone from each object point towards the lens with orientation="Converging",
    # therefore maximizing the amount of "useful" rays for simulation, minimizing rays that don't hit the lens
    # but we need to ensure light from each point can reach all positions on the lens
    # set the divergence angle accordingly
    div_angle = np.rad2deg(np.atan(ap_r/12)*1.2)  # rays need to hit lens outline with r=3 in 12mm distance. Add 20% margin
    RS = ot.RaySource(image, divergence="Isotropic", div_angle=div_angle, s=[0, 0, 1], pos=[0, 0, 0], 
                      orientation="Converging", conv_pos=[0, 0, 12])
    RT.remove(RT.ray_sources)
    RT.add(RS)

    # add aperture before first lens
    ap_surf = ot.RingSurface(r=5, ri=ap_r)
    ap = ot.Aperture(ap_surf, pos=[0, 0, 11])
    RT.remove(RT.apertures)
    RT.add(ap)

    source_aperture_properties[:] = [image_name, ap_r]


# add ray sources and aperture
ap0 = 3
change_ray_source_and_aperture(RT, "Testcard", ap0)

# add Lens 1
front = ot.SphericalSurface(r=3, R=8)
back = ot.SphericalSurface(r=3, R=-8)
nL1 = ot.RefractionIndex("Constant", n=1.5)
L1 = ot.Lens(front, back, de=0.1, pos=[0, 0, 12], n=nL1)
RT.add(L1)

# add Detector
DetS = ot.RectangularSurface(dim=[10, 10])
Det = ot.Detector(DetS, pos=[0, 0, 36])
RT.add(Det)

# create and run the GUI. Add two custom GUI elements to control the aperture and test image
sim = TraceGUI(RT, ray_count=5000000, ray_opacity=0.05)
sim.add_custom_value("Aperture radius (1 - 3mm)", ap0, 
                     lambda ap: change_ray_source_and_aperture(RT, source_aperture_properties[0], ap))
sim.add_custom_selection("Test Image", ["Testcard", "Grid", "Siemens Star"], "Testcard", 
                         lambda name: change_ray_source_and_aperture(RT, name, source_aperture_properties[1]))
# add buttons for easier access to image rendering
sim.add_custom_button("Source Image", sim.source_image)
sim.add_custom_button("Detector Image", sim.detector_image)
sim.run()
