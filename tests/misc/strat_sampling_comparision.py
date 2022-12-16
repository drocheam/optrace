#!/usr/bin/env python3

import numpy as np
import optrace as ot
from optrace.gui import TraceGUI

# make raytracer
RT = ot.Raytracer(outline=[-10, 10, -10, 10, -5, 60])


# v--- uncomment the one needed

# add Rectangular Raysource
RSS = ot.RectangularSurface(dim=[2, 2])

# add Rectangular Raysource
RSS = ot.CircularSurface(r=2)

# add Rectangular Raysource
# RSS = ot.RingSurface(ri=1, r=2)
                    
RSS = ot.RectangularSurface(dim=[1, 1/3])

RS = ot.RaySource(RSS, divergence="None", spectrum=ot.presets.light_spectrum.FDC,
                  pos=[0, 5, 0], s=[0, 0, 1], polarization="y")
RT.add(RS)

# Instantiate the GUI and start it.
sim = TraceGUI(RT, minimalistic_view=True, ray_count=100000)
sim.run()
