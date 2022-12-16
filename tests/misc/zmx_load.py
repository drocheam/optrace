#!/usr/bin/env python3

import numpy as np
import optrace as ot
from optrace.gui import TraceGUI

# make raytracer
RT = ot.Raytracer(outline=[-50, 50, -50, 50, -20, 200])

ot.color.WL_BOUNDS[1] = 1000

RS = ot.RaySource(ot.CircularSurface(r=3), spectrum=ot.LightSpectrum("Monochromatic", wl=750), 
                  pos=[0, 0, -10])
RT.add(RS)
RS = ot.RaySource(ot.CircularSurface(r=3), spectrum=ot.LightSpectrum("Monochromatic", wl=930), 
                  pos=[0, 0, -10])
RT.add(RS)

n_schott = ot.load.agf("./tests/schott.agf")

n_dict = dict(LAFN21=n_schott["N-LAF21"])#, SF15=n_schott["N-SF15"], 
              # F5=ot.RefractionIndex("Schott", coeff=[2.13e-06, 1.65e-08, -6.98e-11, 1.02e-06, 6.56e-10, 0.208]))

# G = ot.load.zmx("./tests/Smith1998b.zmx", n_schott | n_dict)
# G = ot.load.zmx("./tests/9201224.zmx")
# G = ot.load.zmx("./tests/4037934a.zmx", n_schott)
G = ot.load.zmx("./tests/AC254-100-B-Zemax(ZMX).zmx", n_schott)
# G = ot.load.zmx("./tests/7558005b.zmx")
# G = ot.load.zmx("./tests/1843519.zmx")
# G = ot.load.zmx("./tests/Yang2016b.zmx")

# G.remove(G.markers)
G.remove(G.detectors)
# G = G.reverse()
RT.add(G)

# RT.detectors[-1].set_surface(ot.CircularSurface(r=5))

RT.add(ot.Detector(ot.CircularSurface(r=5), pos=[0, 0, 25]))
# n = ot.RefractionIndex("Abbe", n=1.337, V=55)
# RT.add(ot.Lens(ot.CircularSurface(r=5), ot.CircularSurface(r=5), d1=0, d2=0.5, n=n, n2=n, pos=[0, 0, 98]))


# D = (n-1)*(1/R1 - 1/R2) = (n-1)*2/R


# Instantiate the GUI and start it.
sim = TraceGUI(RT, minimalistic_view=True, wl_cmap="plasma")
sim.run()
