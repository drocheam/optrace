#!/usr/bin/env python3
# encoding: utf-8

from Backend import *
from Frontend import *

def runExample(testrun=False):

    Image = './src/tv-test-pattern.png'

    # make Raytracer
    RT = Raytracer(outline=[-5, 5, -5, 5, 0, 40], silent=testrun)

    # add Raysource
    RSS = Surface("Rectangle", dim=[4, 4])
    RS = RaySource(RSS, direction_type="Diverging", sr_angle=8, light_type="RGB_Image",
                   Image=Image, s=[0, 0, 1], pos=[0, 0, 0])
    RT.add(RS)

    # add Lens 1
    front = Surface("Sphere", r=3, rho=1/8)
    back = Surface("Sphere", r=3, rho=-1/8)
    nL1 = RefractionIndex("Constant", n=1.5)
    L1 = Lens(front, back, de=0.1, pos=[0, 0, 12], n=nL1)
    RT.add(L1)

    # add Detector
    DetS = Surface("Rectangle", dim=[10, 10])
    Det = Detector(DetS, pos=[0, 0, 36])
    RT.add(Det)

    # Instantiate the class and configure its traits.
    sim = GUI(RT)
    sim(exit=testrun)


# run Example if this script is called by itself
if __name__ == '__main__':
    runExample(testrun=False)


