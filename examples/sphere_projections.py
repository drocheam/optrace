#!/usr/bin/env python3

import numpy as np
import optrace as ot
from optrace.gui import TraceGUI

# Tissot’s indicatrix


R = 90

# make raytracer
RT = ot.Raytracer(outline=[-100, 100, -100, 100, -10, 100])

RSS0 = ot.Point()

for theta, num in zip([0, 25, 50, 75], [1, 6, 12, 12]):
    tr = np.radians(theta)

    for n in np.arange(num):

        sx = np.sin(np.pi/2 + 2*np.pi*n/num)*np.sin(tr)
        sy = np.cos(np.pi/2 + 2*np.pi*n/num)*np.sin(tr)
        sz = np.cos(tr)

        RS0 = ot.RaySource(RSS0, divergence="Isotropic", spectrum=ot.presets.light_spectrum.d65, div_2d=False,
                pos=[0, 0, 0], s=[sx, sy, sz], div_angle=5)
        RT.add(RS0)
    
# add Detector
DETS = ot.Surface("Sphere", r=(1-1e-6)*R, R=-R)
DET = ot.Detector(DETS, pos=[0, 0, R])
RT.add(DET)

# run the simulator
sim = TraceGUI(RT, ray_opacity=0.5, image_type="Irradiance")
sim.run()

