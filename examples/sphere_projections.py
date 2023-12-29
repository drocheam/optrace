#!/usr/bin/env python3

import numpy as np
import optrace as ot
from optrace.gui import TraceGUI
import optrace.plots as otp

# create Tissot’s indicatrix and compare projection methods

R = 90

# make raytracer
RT = ot.Raytracer(outline=[-100, 100, -100, 100, -10, 100])

RSS0 = ot.Point()

for theta, num in zip([0, 25, 50, 75], [1, 6, 12, 12]):
    for n in np.arange(num)/num:
        RS0 = ot.RaySource(RSS0, divergence="Isotropic", spectrum=ot.presets.light_spectrum.d65, div_2d=False,
                           pos=[0, 0, 0], s_sph=[theta, 360*n], div_angle=5)
        RT.add(RS0)
    
# add Detector
DETS = ot.SphericalSurface(r=(1-1e-9)*R, R=-R)
DET = ot.Detector(DETS, pos=[0, 0, R])
RT.add(DET)

# trace some rays
RT.trace(200000)

# plot detector images for different projection types
for proj in ot.SphericalSurface.sphere_projection_methods:
    img = RT.detector_image(189, projection_method=proj)
    otp.r_image_plot(img, "Irradiance")

# run the simulator
sim = TraceGUI(RT, ray_opacity=0.5, image_type="Irradiance", 
               initial_camera=dict(center=[-50, -50, 0], direction=[-1, -1, -1], height=150, roll=-120))
sim.run()

