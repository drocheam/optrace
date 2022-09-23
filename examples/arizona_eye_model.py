#!/usr/bin/env python3

import optrace as ot
from optrace.gui import TraceGUI
import numpy as np

# options:
g = 0.6e3  # object distance
G_alpha = 4  # angle of object in view
P = 4  # pupil diameter
dispersion = True  # turns on chromatic dispersion

# resulting properties
A = 1/g*1000  # adaption in dpt for given g
G = g*np.tan(G_alpha/180*np.pi) # half object size
OL = max(G, 8)  # half of x, y, outline size
sr_angle = np.arctan(1.25*P/2/g)/np.pi*180  # ray divergence needed for diffuse light

# image vector pointing at the center of the cornea
def F(x, y):
    s = np.column_stack((-x, -y, np.ones_like(x)*g))
    ab = (s[:, 0]**2 + s[:, 1]**2 + s[:, 2]**2) ** 0.5
    return s / ab[:, np.newaxis]

# create raytracer
RT = ot.Raytracer(outline=[-OL, OL, -OL, OL, -g, 28], absorb_missing=True, no_pol=False)

# add RaySource
RSS = ot.Surface("Rectangle", dim=[2*G, 2*G])
RS = ot.RaySource(RSS, divergence="Lambertian", div_angle=sr_angle, image=ot.presets.image.ETDRS_chart_inverted, 
               pos=[0, 0, -g], orientation="Function", or_func=F, desc="USAF Chart")
RT.add(RS)

# load Arizona Eye model
geom = ot.presets.geometry.arizona_eye(adaptation=A, pupil=P)
RT.add(geom)

# Instantiate the class and configure its traits.
TG = TraceGUI(RT, ray_count=1000000, flip_det_image=True, ray_alpha=-2.25)
TG.run()
