#!/usr/bin/env python3

import numpy as np
import optrace as ot
from optrace.gui import TraceGUI

# make raytracer
RT = ot.Raytracer(outline=[-5, 5, -5, 5, -5, 60])

# add Raysource
RSS = ot.CircularSurface(r=1)
RS = ot.RaySource(RSS, divergence="None", spectrum=ot.presets.light_spectrum.FDC,
                  pos=[0, 0, 0], s=[0, 0, 1], polarization="y", desc="Centered Circle")
RT.add(RS)

RSS2 = ot.CircularSurface(r=1)
RS2 = ot.RaySource(RSS2, divergence="None", s=[0, 0, 1], spectrum=ot.LightSpectrum("Blackbody", T=5000),
                   pos=[0, 1, -3], polarization="Constant", pol_angle=25, power=2, desc="Decentered Circle")
RT.add(RS2)

# add Lens 1
front = ot.ConicSurface(r=3, R=10, k=-0.444)
back = ot.ConicSurface(r=3, R=-10, k=-7.25)
nL1 = ot.RefractionIndex("Cauchy", coeff=[1.49, 0.00354, 0, 0])
L1 = ot.Lens(front, back, de=0.1, pos=[0, 0, 10], n=nL1)
RT.add(L1)

# add Lens 2
front = ot.ConicSurface(r=3, R=5, k=-0.31)
back = ot.ConicSurface(r=3, R=-5, k=-3.04)
nL2 = ot.RefractionIndex("Constant", n=1.8)
L2 = ot.Lens(front, back, de=0.6, pos=[0, 0, 25], n=nL2)
RT.add(L2)

# add Aperture
ap = ot.RingSurface(r=1, ri=0.01)
RT.add(ot.Aperture(ap, pos=[0, 0, 20.3]))

# add Lens 3
front = ot.SphericalSurface(r=1, R=2.2)
back = ot.SphericalSurface(r=1, R=-5)
nL3 = ot.RefractionIndex("Function", func=lambda l: 1.8 - 0.007*(l - 380)/400)
nL32 = ot.RefractionIndex("Constant", n=1.1)
L3 = ot.Lens(front, back, de=0.1, pos=[0, 0, 47], n=nL3, n2=nL32)
RT.add(L3)

# add Aperture2
ap = ot.CircularSurface(r=1)

def func(l):
    return np.exp(-0.5*(l-460)**2/20**2)

fspec = ot.TransmissionSpectrum("Function", func=func)
RT.add(ot.Filter(ap, pos=[0, 0, 45.2], spectrum=fspec))

# add Detectors
Det = ot.Detector(ot.RectangularSurface(dim=[2,2]), pos=[0,0,60])
RT.add(Det)
Det2 = ot.Detector(ot.SphericalSurface(R=-1.1, r=1), pos=[0, 0, 40])
RT.add(Det2)

# Instantiate the GUI and start it.
sim = TraceGUI(RT, minimalistic_view=True, func2=lambda: RT.clear())
sim.run()
