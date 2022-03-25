#!/usr/bin/env python3
# encoding: utf-8

from Backend import *
from Frontend import *

import numpy as np


def runExample(testrun=False):

    # make Raytracer
    RT = Raytracer(outline=[-5, 5, -5, 5, -5, 60], silent=testrun)

    # add Raysource
    RSS = Surface("Circle", r=1)
    RS = RaySource(RSS, direction_type="Parallel", light_type="Lines",
                   pos=[0, 0, 0], s=[0, 0, 1], polarization_type="y")
    RT.add(RS)

    RSS2 = Surface("Circle", r=1)
    RS2 = RaySource(RSS2, direction_type="Parallel", s=[0, 0, 1], light_type="D65",
                    pos=[0, 1, -3], polarization_type="Angle", pol_ang=25, power=2)
    # print(RS2)
    ident = RT.add(RS2)
    # RS2i = RT.get(2)
    # print(RS2i)
    # print(RT.remove(ident))
    # print(RT.remove(2))

    front = Surface("Circle", r=3, rho=1/10, k=-0.444)
    back = Surface("Circle", r=3, rho=-1/10, k=-7.25)
    nL2 = RefractionIndex("Constant", n=1.8)
    L1 = Lens(front, back, de=0.1, pos=[0, 0, 2], n=nL2)
    RT.add(L1)

    # add Lens 1
    front = Surface("Asphere", r=3, rho=1/10, k=-0.444)
    back = Surface("Asphere", r=3, rho=-1/10, k=-7.25)
    nL1 = RefractionIndex("Cauchy", A=1.49, B=0.00354)
    L1 = Lens(front, back, de=0.1, pos=[0, 0, 10], n=nL1)
    RT.add(L1)

    # print(L1.estimateFocalLength())

    # add Lens 2
    front = Surface("Asphere", r=3, rho=1/5, k=-0.31)
    back = Surface("Asphere", r=3, rho=-1/5, k=-3.04)
    nL2 = RefractionIndex("Constant", n=1.8)
    L2 = Lens(front, back, de=0.6, pos=[0, 0, 25], n=nL2)
    RT.add(L2)


    # add Aperture
    ap = Surface("Ring", r=1, ri=0.01)
    RT.add(Filter(ap, pos=[0, 0, 20.3]))

    # add Lens 3
    front = Surface("Sphere", r=1, rho=1/2.2)
    back = Surface("Sphere", r=1, rho=-1/5)
    nL3 = RefractionIndex("Function", func=lambda l: 1.8 - 0.007*(l - 380)/400)
    nL32 = RefractionIndex("Constant", n=1.1)
    L3 = Lens(front, back, de=0.1, pos=[0, 0, 47], n=nL3, n2=nL32)
    RT.add(L3)

    # # add Aperture2
    ap = Surface("Circle", r=1, ri=0.005)

    def func(l):
        # w = l.copy()
        # w[l > 500] = 0
        # w[l <= 500] = 1
        w = np.exp(-0.5*(l-460)**2/20**2)
        return w

    RT.add(Filter(ap, pos=[0, 0, 45.2], filter_type="Function", func=func))


    # add Detector
    # Det = Detector(Surface(surface_type="Sphere", rho=-1/1.1, r=1), pos=[0, 0, 60])
    # Det = Detector(Surface(surface_type="Ring", pos=[0, 0, 60], rho=-1/1.1, r=1))
    Det = Detector(Surface("Rectangle", dim=[2,2]), pos=[0,0,60])
    # Det = Detector(pos=[0, 0, 60], dim=[6, 6], rho=-1/3.1)
    RT.add(Det)

    Det2 = Detector(Surface("Sphere", rho=-1/1.1, r=1), pos=[0, 0, 40])
    RT.add(Det2)

    # Instantiate the GUI and start it.
    sim = GUI(RT)
    sim(exit=testrun)


# run Example if this script is called by itself
if __name__ == '__main__':
    runExample(testrun=False)
