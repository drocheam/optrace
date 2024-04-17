#!/usr/bin/env python3

import numpy as np
import optrace as ot
import optrace.plots as otp
from optrace.gui import TraceGUI

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
RT.trace(600000)

# plot detector images for different projection types
for proj in ot.SphericalSurface.sphere_projection_methods:
    dimg = RT.detector_image(projection_method=proj)
    img = dimg.get("Irradiance", 189)
    otp.image_plot(img)

# run the simulator
sim = TraceGUI(RT, ray_opacity=0.5, image_mode="Irradiance", hide_labels=True, minimalistic_view=True,
               initial_camera=dict(center=[-50, -50, 0], direction=[-1, -1, -1], height=150, roll=-120))
sim.run()

