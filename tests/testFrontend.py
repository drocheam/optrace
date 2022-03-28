#!/bin/env python3

import sys
sys.path.append('./src/')

import unittest
import warnings
import time

import numpy as np
from Backend import *
from Frontend import *

from threading import Thread

class FrontendTests(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter("ignore")
    
    def tearDown(self):
        warnings.simplefilter("default")
    
    def test_GUI_inits(self):

        Image = './src/tv-test-pattern.png'

        # make Raytracer
        RT = Raytracer(outline=[-5, 5, -5, 5, 0, 40], silent=True)

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

       
        # subtest function with count and args output
        def GUI_Run(**kwargs):
            GUI_Run.i += 1
            with self.subTest(i=GUI_Run.i, args=kwargs):
                sim = GUI(RT, **kwargs)
                sim(exit=True)
        GUI_Run.i = 0


        GUI_Run()  # default init

        GUI_Run(ColoringType="Wavelength")
        GUI_Run(ColoringType="Power")
        GUI_Run(ColoringType="Source")
        GUI_Run(ColoringType="Polarization")
        GUI_Run(ColoringType="White")

        GUI_Run(PlottingType="Points")
        GUI_Run(PlottingType="Rays")
        GUI_Run(PlottingType="None")

        GUI_Run(Rays=100000)
        GUI_Run(Rays_s=-2.)
        GUI_Run(AbsorbMissing=False)
        GUI_Run(Ray_alpha=-1.)
        GUI_Run(Ray_width=5)
        GUI_Run(Pos_Det=8.9)

        GUI_Run(Rays=100000, Rays_s=-2., AbsorbMissing=False, Ray_alpha=-1., Ray_width=5,
                ColoringType="Wavelength", PlottingType="Points", Pos_Det=8.9)


    def test_MultipleSourceDetectors(self):
        pass
        # these are checked in the More_Complex_Example

    def test_Interaction(self):

        # make Raytracer
        RT = Raytracer(outline=[-5, 5, -5, 5, -5, 60], silent=True)

        # add Raysource
        RSS = Surface("Circle", r=1)
        RS = RaySource(RSS, direction_type="Parallel", light_type="Lines",
                       pos=[0, 0, 0], s=[0, 0, 1], polarization_type="y")
        RT.add(RS)

        RSS2 = Surface("Circle", r=1)
        RS2 = RaySource(RSS2, direction_type="Parallel", s=[0, 0, 1], light_type="D65",
                        pos=[0, 1, -3], polarization_type="Angle", pol_ang=25, power=2)
        RT.add(RS2)

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
            return np.exp(-0.5*(l-460)**2/20**2)

        RT.add(Filter(ap, pos=[0, 0, 45.2], filter_type="Function", func=func))


        # add Detector
        Det = Detector(Surface("Rectangle", dim=[2,2]), pos=[0,0,60])
        RT.add(Det)

        Det2 = Detector(Surface("Sphere", rho=-1/1.1, r=1), pos=[0, 0, 40])
        RT.add(Det2)

        # Instantiate the GUI and start it.
        sim = GUI(RT)

        def interact():

            sim.waitWhile("Init")
            sim.waitWhile("Tracing")
            sim.waitWhile("Drawing")

            # check if Detector is moving
            sim.Pos_Det = 5.3
            self.assertEqual(sim.Pos_Det, RT.DetectorList[0].pos[2])
        
            # Source Image Tests
            sim.showSourceImage()
            sim.SourceSelection = sim.SourceNames[1]
            sim.showSourceImage()
            
            # Detector Image Tests
            sim.showDetectorImage()
            sim.DetectorSelection = sim.DetectorNames[1]
            time.sleep(0.001)  # wait for parameters to be set
            self.assertTrue(sim.DetInd == 1)
            self.assertTrue(sim.Pos_Det == sim.Raytracer.DetectorList[1].pos[2])
            sim.showDetectorImage()

            # Image Type Tests standard
            sim.ImageType = "Irradiance"
            sim.showDetectorImage()
            sim.waitWhile("DetectorImage")
            sim.ImageType = "Illuminance"
            sim.showDetectorImage()
            sim.waitWhile("DetectorImage")
            sim.ImageType = "sRGB"
            sim.showDetectorImage()
            sim.waitWhile("DetectorImage")

            # Image Type Tests log scaling
            sim.LogImage = ['Logarithmic Scaling']
            sim.ImageType = "Irradiance"
            sim.showDetectorImage()
            sim.waitWhile("DetectorImage")
            sim.ImageType = "Illuminance"
            sim.showDetectorImage()
            sim.waitWhile("DetectorImage")
            sim.ImageType = "sRGB"
            sim.showDetectorImage()
            sim.waitWhile("DetectorImage")

            # Image Tests Flip
            sim.LogImage = []
            sim.FlipImage = ['Flip Image']
            sim.showDetectorImage()
            sim.waitWhile("DetectorImage")

            # Image Tests Higher Res
            sim.ImagePixels = 300
            sim.showDetectorImage()
            sim.waitWhile("DetectorImage")

            # Image Test Source, but actually we should test all parameter combinations,
            sim.showSourceImage()

            # Focus Test 1
            pos0 = sim.Raytracer.DetectorList[1].pos
            sim.DetectorSelection = sim.DetectorNames[1]
            sim.FocusType = "Position Variance"
            sim.moveToFocus()
            sim.waitWhile("Focussing")
            sim.Pos_Det = pos0[2]
            
            # Focus Test 2
            sim.FocusType = "Irradiance Variance"
            sim.moveToFocus()
            sim.waitWhile("Focussing")
            sim.Pos_Det = pos0[2]
            
            # Focus Tests 3
            sim.FocusType = "Irradiance Variance"
            sim.moveToFocus()
            sim.waitWhile("Focussing")
            sim.Pos_Det = pos0[2]

            # Focus Test 4, show Debug Plot
            sim.FocusDebugPlot = ['Show Cost Function']
            sim.moveToFocus()
            sim.waitWhile("Focussing")

            # Ray Coloring Tests
            sim.ColoringType = "Power"
            sim.waitWhile("Drawing")
            sim.ColoringType = "White"
            sim.waitWhile("Drawing")
            sim.ColoringType = "Wavelength"
            sim.waitWhile("Drawing")
            sim.ColoringType = "Polarization"
            sim.waitWhile("Drawing")
            sim.ColoringType = "Source"
            sim.waitWhile("Drawing")

            # PlottingType Tests
            sim.PlottingType = "Points"
            sim.waitWhile("Drawing")
            sim.PlottingType = "None"
            sim.waitWhile("Drawing")
            sim.PlottingType = "Rays"
            sim.waitWhile("Drawing")
          
            # AbsorbMissing test
            sim.AbsorbMissing = []
            sim.waitWhile("Tracing")
            sim.waitWhile("Drawing")

            # retrace Tests
            sim.Rays = 100000
            sim.waitWhile("Tracing")
            sim.waitWhile("Drawing")
            
            sim.Rays_s = -2.5
            sim.waitWhile("Drawing")

            sim.Rays_s = -2.
            sim.waitWhile("Drawing")

            sim.close()
            time.sleep(0.1)  # wait for it actually being closed

        th = Thread(target=interact)
        th.start()
        sim()

        th.join()


if __name__ == '__main__':
    unittest.main()
