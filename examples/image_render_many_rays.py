#!/usr/bin/env python3

import optrace as ot
import optrace.plots as otp
import numpy as np

# Comparable to the image_render example. 
# Same lens setup, but it is traced with many more rays by using the iterative render functionality.
# This is done for multiple image distances and without needing to start a GUI.

RSS = ot.presets.image.tv_testcard2([4, 3])

# make raytracer
RT = ot.Raytracer(outline=[-8, 8, -8, 8, 0, 40], no_pol=True)  # speed up by neglecting polarization dependence

# add Raysource
# orient the light direction cone from each object point towards the lens with orientation="Converging",
# therefore maximizing the amount of "useful" rays for simulation, minimizing rays that don't hit the lens
# but we need to ensure light from each point can reach all positions on the lens
# set the divergence angle accordingly
div_angle = np.rad2deg(np.atan(3/12)*1.2)  # rays need to hit lens outline with r=3 in 12mm distance. Add 20% margin
RS = ot.RaySource(RSS, divergence="Isotropic", div_angle=div_angle,
                  s=[0, 0, 1], pos=[0, 0, 0], orientation="Converging", conv_pos=[0, 0, 12])
RT.add(RS)

# add Lens 1
front = ot.SphericalSurface(r=3, R=8)
back = ot.SphericalSurface(r=3, R=-8)
nL1 = ot.RefractionIndex("Abbe", n=1.5, V=40)
L1 = ot.Lens(front, back, de=0.1, pos=[0, 0, 12], n=nL1)
RT.add(L1)

# add Detector
Det = ot.Detector(ot.RectangularSurface(dim=[16, 16]), pos=[0, 0, 36])
RT.add(Det)

# render and show detector image
pos = [[0, 0, 15], [0, 0, 20], [0, 0, 25.], [0, 0, 29.], [0, 0, 31.], [0, 0, 36.]]
Ims = RT.iterative_render(N=20e6, pos=pos, extent=[-5, 5, -5, 5])

# show rendered images
for dimg in Ims:
    img = dimg.get("sRGB (Absolute RI)", 315)
    otp.image_plot(img)
otp.block()
