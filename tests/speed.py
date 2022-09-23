#!/bin/env python3

import time
import sys
sys.path.append('./')

from optrace import *


def func2(N2):
    # make raytracer
    RT = Raytracer(outline=[-5, 5, -5, 5, 0, 60], absorb_missing=True,
                   silent=True, no_pol=False, threading=True)

    # add Raysource
    RSS = Surface("Circle", r=1)
    RS = RaySource(RSS, divergence="None", spectrum=LightSpectrum("Blackbody", T=5500),
                   pos=[0, 0, 0], s=[0, 0, 1], polarization='y')
    RT.add(RS)

    RS2 = RaySource(RSS, divergence="None", spectrum=presets.light_spectrum.d65,
                    pos=[0, 1, 0], s=[0, 0, 1], polarization='x', power=2)
    RT.add(RS2)

    front = Surface("Circle", r=3, R=10, k=-0.444)
    back = Surface("Circle", r=3, R=-10, k=-7.25)
    nL2 = RefractionIndex("Constant", n=1.8)
    L1 = Lens(front, back, de=0.1, pos=[0, 0, 2], n=nL2)
    RT.add(L1)

    # add Lens 1
    front = Surface("Conic", r=3, R=10, k=-0.444)
    back = Surface("Conic", r=3, R=-10, k=-7.25)
    nL1 = RefractionIndex("Cauchy", coeff=[1.49, 0.00354])
    L1 = Lens(front, back, de=0.1, pos=[0, 0, 10], n=nL1)
    RT.add(L1)

    # add Lens 2
    front = Surface("Conic", r=3, R=5, k=-0.31)
    back = Surface("Conic", r=3, R=-5, k=-3.04)
    nL2 = RefractionIndex("Constant", n=1.8)
    L2 = Lens(front, back, de=0.6, pos=[0, 0, 25], n=nL2)
    RT.add(L2)

    # add Aperture
    ap = Surface("Ring", r=1, ri=0.01)
    RT.add(Aperture(ap, pos=[0, 0, 20.3]))

    # add Lens 3
    front = Surface("Sphere", r=1, R=2.2)
    back = Surface("Sphere", r=1, R=-5)
    nL3 = RefractionIndex("Function", func=lambda l: 1.8 - 0.007*(l - 380)/400)
    nL32 = RefractionIndex("Constant", n=1.1)
    L3 = Lens(front, back, de=0.1, pos=[0, 0, 47], n=nL3, n2=nL32)
    RT.add(L3)

    # # add Aperture2
    ap = Surface("Circle", r=1, ri=0.005)

    def func(l):
        w = l.copy()
        w[l > 500] = 0
        w[l <= 500] = 1
        return w

    fspec = TransmissionSpectrum("Function", func=func)
    RT.add(Filter(ap, pos=[0, 0, 45.2], spectrum=fspec))

    # add Detector
    Det = Detector(Surface("Rectangle", dim=[3, 3]), pos=[0, 0, 60])
    RT.add(Det)

    start = time.time()    
    RT.trace(N=N2)
    # Im = RT.detector_image(500, extent="auto")
    print(time.time()-start)


N = 1000000
func2(N)
