#!/usr/bin/env python3

import optrace as ot
from optrace.gui import TraceGUI
import numpy as np

# This example is a demonstration of human eye vision with adaptation at a distance of 66 cm. 
# The Arizona eye model is employed to simulate a resolution chart.
# This eye model accurately matches on- and off-axis aberration levels from clinical data
# and accounts for wavelength and adaptation dependencies. 
# In the "Custom" tab of the GUI there are options available
# to change the pupil diameter and adaptation of the eye.

# options:
g = 0.6e3  # object distance in mm
G_alpha = 4  # angle of object in view in deg
P = 4  # pupil diameter in mm

# resulting properties
A = 1/g*1000  # adaption in dpt for given g
G = g*np.tan(G_alpha/180*np.pi) # half object size
OL = max(G, 8)  # half of x, y, outline size

# store eye parameter values here
eye_parameters = [A, P]

# add/replace Arizona eye model and store eye properties in "eye_parameters"
def make_geometry(RT, adaptation, pupil):

    # clear all objects
    RT.clear()

    # add RaySource
    # light is converging towards eye vertex at [0, 0, 0]
    RSS = ot.presets.image.ETDRS_chart_inverted([2*G, 2*G])
    # every object point emits a cone of possible light directions
    # to only use "useful" rays for image rending, we neglect most of the rays not hitting the pupil
    # this is done by directing each cone towards the pupil center (parameter "orientation" and "conv_pos")
    # calculate the divergence angle based on the requirement, 
    # that light from every object point can hit all positions inside the aperture
    sr_angle = np.rad2deg(np.arctan(1.4*pupil/2/g)) # direct light cone towards pupil, add 40% margin
    RS = ot.RaySource(RSS, divergence="Isotropic", div_angle=sr_angle,
                      pos=[0, 0, -g], orientation="Converging", conv_pos=[0, 0, 0], desc="USAF Chart")
    RT.add(RS)

    # load Arizona Eye model
    geom = ot.presets.geometry.arizona_eye(adaptation=adaptation, pupil=pupil)
    RT.add(geom)

    eye_parameters[:] = [adaptation, pupil]


# create raytracer
RT = ot.Raytracer(outline=[-OL, OL, -OL, OL, -g, 28], no_pol=False)

# add ray source and Arizona eye model
make_geometry(RT, A, P)

# Instantiate the class and configure its traits.
TG = TraceGUI(RT, ray_count=1000000, flip_detector_image=True, ray_opacity=0.01, vertical_labels=True,
              initial_camera=dict(center=[0, 0, 7.3], height=20))

# custom value fields in the GUI to change the eye parameters
TG.add_custom_value("Adaptation (0 - 2.5D)", A, lambda adaptation: make_geometry(RT, adaptation, eye_parameters[1]))
TG.add_custom_value("Pupil diameter (2 - 6mm)", P, lambda pupil: make_geometry(RT, eye_parameters[0], pupil))

# add buttons for easier access to image rendering
TG.add_custom_button("Source Image", TG.source_image)
TG.add_custom_button("Detector Image", TG.detector_image)

# run the simulation
TG.run()
