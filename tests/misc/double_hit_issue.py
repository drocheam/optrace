#!/usr/bin/env python3

import numpy as np
import optrace as ot
from optrace.gui import TraceGUI


# make raytracer
RT = ot.Raytracer(outline=[-50, 50, -50, 50, -20, 200])

ot.color.WL_BOUNDS[1] = 1000

RS = ot.RaySource(ot.Point(), spectrum=ot.LightSpectrum("Monochromatic", wl=750), 
                  pos=[0, -2, -0.1], s_sph=[85, 90])
RT.add(RS)

RS = ot.RaySource(ot.Point(), spectrum=ot.LightSpectrum("Monochromatic", wl=750), 
                  pos=[0, 2, -0.1], s_sph=[-85, 90])
RT.add(RS)


# sph = ot.SphericalSurface(r=5, R=5.1)
# sph = ot.ConicSurface(r=12, R=0.1, k=-10000)
sph = ot.FunctionSurface(r=5, func=lambda x, y: -np.sqrt(1 - (x**2 + y**2)/5.1**2)*5.1)
circ = ot.CircularSurface(r=5)
L = ot.Lens(sph, circ, pos=[0, 0, 0], d1=0, d2=5, n=ot.presets.refraction_index.SF10)
RT.add(L)


# Instantiate the GUI and start it.
sim = TraceGUI(RT, minimalistic_view=True, wl_cmap="plasma")
sim.run()
