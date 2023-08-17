#!/usr/bin/env python3

import numpy as np
import optrace as ot
from optrace.gui import TraceGUI
import optrace.plots as otp


# init raytracer
RT = ot.Raytracer(outline=[-15, 15, -15, 15, 0, 80], absorb_missing=False)

# add Raysource
RSS = ot.CircularSurface(r=3)
RS = ot.RaySource(RSS, divergence="None", s=[0, 0, 1], pos=[0, 0, 0]) 
RT.add(RS)

# cosine front surface
surf_func = lambda x, y: 0.1*np.cos(2*np.pi*x/2)#-np.tanh(3-2*np.sqrt(x**2 + y**2))/3
front = ot.FunctionSurface2D(func=surf_func, r=5)

# back surface is flipped and rotated
back = front.copy()
back.flip()
back.rotate(90)

# create lens
nL1 = ot.presets.refraction_index.SF5
L1 = ot.Lens(front, back, de=2, pos=[0, 0, 12], n=nL1)
RT.add(L1)

# second lens is just a shifted copy
L2 = L1.copy()
L2.move_to([0, 0, 18])
RT.add(L2)

# add ideal lens for focussing
L3 = ot.IdealLens(r=9, D=50, pos=[0, 0, 40])
RT.add(L3)

# add Detector
DetS = ot.RectangularSurface(dim=[14, 14])
Det = ot.Detector(DetS, pos=[0, 0, 24.4])
RT.add(Det)

# add second Detector
Det2 = Det.copy()
Det2.move_to([0, 0, 60])
RT.add(Det2)

# trace
RT.trace(600000)

# plot detector images
img = RT.detector_image(315)
otp.r_image_plot(img, "sRGB (Absolute RI)", block=False)
img2 = RT.detector_image(315, detector_index=1)
otp.r_image_plot(img2, "sRGB (Absolute RI)", block=False)

# start GUI
sim = TraceGUI(RT, ray_opacity=0.03, minimalistic_view=True)
sim.run()
