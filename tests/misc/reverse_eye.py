#!/usr/bin/env python3

import optrace as ot
from optrace.gui import TraceGUI

# init raytracer 
RT = ot.Raytracer(outline=[-10, 10, -10, 10, -10, 60])

# load eye preset
eye = ot.presets.geometry.arizona_eye(pupil=3)

# flip, move and add it to the tracer
eye.flip()
eye.move_to([0, 0, 0])
RT.add(eye)

# create and add divergent point source
point = ot.Point()
RS = ot.RaySource(point, spectrum=ot.presets.light_spectrum.d50, divergence="Isotropic", div_angle=5,
                 pos=[0, 0, 0])
RT.add(RS)


sim = TraceGUI(RT)
sim.run()
