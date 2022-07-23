#!/usr/bin/env python3

import sys
sys.path.append(".")

import optrace as ot
import optrace.gui.TraceGUI as TraceGUI
import numpy as np

# options:
g = 0.3e3  # object distance
G_alpha = 4  # angle of object in view
P = 4  # pupil diameter
dispersion = True  # turns on chromatic dispersion

# resulting properties
A = 1/g*1000  # adaption in dpt for given g
G = g*np.tan(G_alpha/180*np.pi) # half object size
OL = max(G, 8)  # half of x, y, outline size
sr_angle = np.arcsin(P/g)/np.pi*180  # ray divergence needed for diffuse light

# image vector pointing at the center of the cornea
def F(x, y):
    s = np.column_stack((-x, -y, np.ones_like(x)*g))
    ab = (s[:, 0]**2 + s[:, 1]**2 + s[:, 2]**2) ** 0.5
    return s / ab[:, np.newaxis]

# create Raytracer
RT = ot.Raytracer(outline=[-OL, OL, -OL, OL, -g, 28], AbsorbMissing=True, no_pol=False)

# add RaySource
RSS = ot.Surface("Rectangle", dim=[2*G, 2*G])
RS = ot.RaySource(RSS, direction="Diverging", div_angle=sr_angle, Image=ot.presets.Image.ETDRS_chart_inverted, 
               pos=[0, 0, -g], orientation="Function", or_func=F, desc="USAF Chart")
RT.add(RS)

# load Arizona Eye model
geom = ot.presets.Geometry.ArizonaEye(A=A, P=P, dispersion=dispersion)
RT.add(geom)

# Instantiate the class and configure its traits.
TG = TraceGUI(RT, RayCount=1000000, FlipDetImage=True, RayAlpha=-2.25)
TG.run()

