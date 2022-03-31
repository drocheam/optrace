#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
from Backend import *
from Frontend import *

def runExample(testrun=False):
    # make Raytracer
    RT = Raytracer(outline=[-4, 5, -3, 5, -1, 27.5], silent=testrun)

    # add Raysource
    RSS = Surface("Circle", r=0.05)
    RS = RaySource(RSS, direction_type="Parallel", light_type="D65",
                   pos=[-2.5, 0, 0], s=[0.3, 0, 0.7])
    RT.add(RS)

    # prism surface
    x = np.linspace(-3, 3, 500)
    y = np.linspace(-3, 3, 500)
    X, Y = np.meshgrid(x, y)
    Z = 0.5*X.copy()


    P1 = SurfaceFunction(r=3, func=lambda x, y: 0.5*x, derivative=lambda x, y: (np.full_like(x, 0.5), 0*y))#, normals=normals1)
    P2 = SurfaceFunction(r=3, func=lambda x, y: -0.5*x, derivative=lambda x, y: (np.full_like(x, -0.5), 0*y))#, normals=normals2)


    # Z[X**2 + Y**2 > 9] = np.nan  # cut to circular surface

    # Prism 1
    # front = Surface("Function", func=P1)
    # back = Surface("Function", func=P2)
    front = Surface("Data", r=3, Data=Z)
    back = Surface("Data", r=3, Data=-Z)
    nL1 = RefractionIndex("SF10")
    L1 = Lens(front, back, de=0.5, pos=[0, 1, 10], n=nL1)
    RT.add(L1)

    # Prism 2
    # back and front of Prism 1 are switched,
    L2 = Lens(back, front, de=0.5, pos=[0, 0, 16.5], n=nL1)
    RT.add(L2)

    # add Detector
    Det = Detector(Surface("Rectangle", dim=[5, 5]), pos=[0, 0, 27.5])
    RT.add(Det)

    # Instantiate the class and configure its traits.
    sim = GUI(RT, ColoringType="Wavelength")
    sim.interact()


# run Example if this script is called by itself
if __name__ == '__main__':
    runExample(testrun=False)
 
