#!/bin/env python3

import sys
sys.path.append('./')

import unittest
import warnings
import time

import numpy as np

import optrace as ot
import optrace.gui.GUI as TraceGUI

from threading import Thread

class FrontendTests(unittest.TestCase):

    def setUp(self) -> None:
        warnings.simplefilter("ignore")
    
    def tearDown(self) -> None:
        warnings.simplefilter("default")
    
    def test_GUI_inits(self) -> None:

        Image = './examples/tv-test-pattern.png'

        # make Raytracer
        RT = ot.Raytracer(outline=[-5, 5, -5, 5, 0, 40], silent=True)

        # add Raysource
        RSS = ot.Surface("Rectangle", dim=[4, 4])
        RS = ot.RaySource(RSS, direction_type="Diverging", sr_angle=8, light_type="RGB_Image",
                       Image=Image, s=[0, 0, 1], pos=[0, 0, 0])
        RT.add(RS)

        # add Lens 1
        front = ot.Surface("Sphere", r=3, rho=1/8)
        back = ot.Surface("Sphere", r=3, rho=-1/8)
        nL1 = ot.RefractionIndex("Constant", n=1.5)
        L1 = ot.Lens(front, back, de=0.1, pos=[0, 0, 12], n=nL1)
        RT.add(L1)

        # add Detector
        DetS = ot.Surface("Rectangle", dim=[10, 10])
        Det = ot.Detector(DetS, pos=[0, 0, 36])
        RT.add(Det)

       
        # subtest function with count and args output
        def TraceGUI_Run(**kwargs):
            TraceGUI_Run.i += 1
            with self.subTest(i=TraceGUI_Run.i, args=kwargs):
                sim = TraceGUI(RT, **kwargs)
                sim.run(exit=True, no_server=True, silent=True)
        TraceGUI_Run.i = 0

        TraceGUI_Run()  # default init

        TraceGUI_Run(ColoringType="Wavelength")
        TraceGUI_Run(ColoringType="Power")
        TraceGUI_Run(ColoringType="Source")
        TraceGUI_Run(ColoringType="Polarization")
        TraceGUI_Run(ColoringType="White")

        TraceGUI_Run(PlottingType="Points")
        TraceGUI_Run(PlottingType="Rays")

        TraceGUI_Run(RayCount=100000)
        TraceGUI_Run(RayAmountShown=-2.)
        TraceGUI_Run(AbsorbMissing=False)
        TraceGUI_Run(RayAlpha=-1.)
        TraceGUI_Run(RayWidth=5)
        TraceGUI_Run(PosDet=8.9)

        TraceGUI_Run(RayCount=100000, RaysAmountShown=-2., AbsorbMissing=False, RayAlpha=-1., RayWidth=5,
                ColoringType="Wavelength", PlottingType="Points", PosDet=8.9)


    def RT_Example(self) -> ot.Raytracer:

        # make Raytracer
        RT = ot.Raytracer(outline=[-5, 5, -5, 5, -5, 60], silent=True)

        # add Raysource
        RSS = ot.Surface("Circle", r=1)
        RS = ot.RaySource(RSS, direction_type="Parallel", light_type="Lines",
                       pos=[0, 0, 0], s=[0, 0, 1], polarization_type="y")
        RT.add(RS)

        RSS2 = ot.Surface("Circle", r=1)
        RS2 = ot.RaySource(RSS2, direction_type="Parallel", s=[0, 0, 1], light_type="D65",
                        pos=[0, 1, -3], polarization_type="Angle", pol_ang=25, power=2)
        RT.add(RS2)

        front = ot.Surface("Circle", r=3, rho=1/10, k=-0.444)
        back = ot.Surface("Circle", r=3, rho=-1/10, k=-7.25)
        nL2 = ot.RefractionIndex("Constant", n=1.8)
        L1 = ot.Lens(front, back, de=0.1, pos=[0, 0, 2], n=nL2)
        RT.add(L1)

        # add Lens 1
        front = ot.Surface("Asphere", r=3, rho=1/10, k=-0.444)
        back = ot.Surface("Asphere", r=3, rho=-1/10, k=-7.25)
        nL1 = ot.RefractionIndex("Cauchy", A=1.49, B=0.00354)
        L1 = ot.Lens(front, back, de=0.1, pos=[0, 0, 10], n=nL1)
        RT.add(L1)

        # add Lens 2
        front = ot.Surface("Asphere", r=3, rho=1/5, k=-0.31)
        back = ot.Surface("Asphere", r=3, rho=-1/5, k=-3.04)
        nL2 = ot.RefractionIndex("Constant", n=1.8)
        L2 = ot.Lens(front, back, de=0.6, pos=[0, 0, 25], n=nL2)
        RT.add(L2)

        # add Aperture
        ap = ot.Surface("Ring", r=1, ri=0.01)
        RT.add(ot.Filter(ap, pos=[0, 0, 20.3]))

        # add Lens 3
        front = ot.Surface("Sphere", r=1, rho=1/2.2)
        back = ot.Surface("Sphere", r=1, rho=-1/5)
        nL3 = ot.RefractionIndex("Function", func=lambda l: 1.8 - 0.007*(l - 380)/400)
        nL32 = ot.RefractionIndex("Constant", n=1.1)
        L3 = ot.Lens(front, back, de=0.1, pos=[0, 0, 47], n=nL3, n2=nL32)
        RT.add(L3)

        # # add Aperture2
        ap = ot.Surface("Circle", r=1, ri=0.005)

        def func(l):
            return np.exp(-0.5*(l-460)**2/20**2)

        RT.add(ot.Filter(ap, pos=[0, 0, 45.2], filter_type="Function", func=func))

        # add Detector
        Det = ot.Detector(ot.Surface("Rectangle", dim=[2,2]), pos=[0,0,60])
        RT.add(Det)

        Det2 = ot.Detector(ot.Surface("Sphere", rho=-1/1.1, r=1), pos=[0, 0, 40])
        RT.add(Det2)

        return RT

    def test_Interaction(self) -> None:

        RT = self.RT_Example()

        # Instantiate the TraceGUI and start it.
        sim = TraceGUI(RT)

        def interact(sim):

            sim.waitForIdle()

            # check if Detector is moving
            sim.PosDet = 5.3
            sim.waitForIdle()
            self.assertEqual(sim.PosDet, RT.DetectorList[0].pos[2])
        
            # Source Image Tests
            sim.showSourceImage()
            sim.waitForIdle()
            sim.SourceSelection = sim.SourceNames[1]
            sim.waitForIdle()
            sim.showSourceImage()
            sim.waitForIdle()
            
            # Detector Image Tests
            sim.showDetectorImage()
            sim.waitForIdle()
            sim.DetectorSelection = sim.DetectorNames[1]
            sim.waitForIdle()
            self.assertTrue(sim.DetInd == 1)
            self.assertTrue(sim.PosDet == sim.Raytracer.DetectorList[1].pos[2])
            sim.showDetectorImage()
            sim.waitForIdle()

            # Image Type Tests standard
            sim.ImageType = "Irradiance"
            sim.showDetectorImage()
            sim.waitForIdle()
            sim.ImageType = "Illuminance"
            sim.showDetectorImage()
            sim.waitForIdle()
            sim.ImageType = "sRGB (Absolute RI)"
            sim.showDetectorImage()
            sim.waitForIdle()

            # Image Type Tests log scaling
            sim.LogImage = ['Logarithmic Scaling']
            sim.ImageType = "Irradiance"
            sim.showDetectorImage()
            sim.waitForIdle()
            sim.ImageType = "Illuminance"
            sim.showDetectorImage()
            sim.waitForIdle()
            sim.ImageType = "sRGB (Absolute RI)"
            sim.showDetectorImage()
            sim.waitForIdle()
            sim.ImageType = "sRGB (Perceptual RI)"
            sim.showDetectorImage()
            sim.waitForIdle()
            sim.ImageType = "Hue (CIELUV)"
            sim.showDetectorImage()
            sim.waitForIdle()
            sim.ImageType = "Chroma (CIELUV)"
            sim.showDetectorImage()
            sim.waitForIdle()
            sim.ImageType = "Saturation (CIELUV)"
            sim.showDetectorImage()
            sim.waitForIdle()
            sim.ImageType = "Lightness (CIELUV)"
            sim.showDetectorImage()
            sim.waitForIdle()

            # Image Tests Flip
            sim.LogImage = []
            sim.FlipImage = ['Flip Image']
            sim.showDetectorImage()
            sim.waitForIdle()

            # Image Tests Higher Res
            sim.ImagePixels = 300
            sim.showDetectorImage()
            sim.waitForIdle()

            # Image Test Source, but actually we should test all parameter combinations,
            sim.showSourceImage()

            # Focus Test 1
            pos0 = sim.Raytracer.DetectorList[1].pos
            sim.DetectorSelection = sim.DetectorNames[1]
            sim.FocusType = "Position Variance"
            sim.moveToFocus()
            sim.waitForIdle()
            sim.PosDet = pos0[2]
            
            # Focus Test 2
            sim.FocusType = "Irradiance Variance"
            sim.moveToFocus()
            sim.waitForIdle()
            sim.PosDet = pos0[2]
            
            # Focus Tests 3
            sim.FocusType = "Irradiance Variance"
            sim.moveToFocus()
            sim.waitForIdle()
            sim.PosDet = pos0[2]

            # Focus Test 4, show Debug Plot
            sim.FocusDebugPlot = ['Show Cost Function']
            sim.moveToFocus()
            sim.waitForIdle()

            # Ray Coloring Tests
            sim.ColoringType = "Power"
            sim.waitForIdle()
            sim.ColoringType = "White"
            sim.waitForIdle()
            sim.ColoringType = "Wavelength"
            sim.waitForIdle()
            sim.ColoringType = "Polarization"
            sim.waitForIdle()
            sim.ColoringType = "Source"
            sim.waitForIdle()

            # PlottingType Tests
            sim.PlottingType = "Points"
            sim.waitForIdle()
            sim.PlottingType = "Rays"
            sim.waitForIdle()
          
            # AbsorbMissing test
            sim.AbsorbMissing = []
            sim.waitForIdle()

            # retrace Tests
            sim.RayCount = 100000
            sim.waitForIdle()
            
            sim.RaysAmountShown = -2.5
            sim.waitForIdle()

            sim.RaysAmountShown = -2.
            sim.waitForIdle()

            sim.close()

        sim.run(func=interact, no_server=True, silent=True, args=(sim,))

    def test_Missing(self) -> None:
        """test TraceGUI operation when Filter, Lenses, Detectors or Sources are missing"""

        def testFeatures(RT):
            sim = TraceGUI(RT)
            def interact(sim):
                sim.waitForIdle()
                sim.showDetectorImage()
                sim.waitForIdle()
                sim.moveToFocus()
                sim.waitForIdle()
                sim.PosDet = 10.
                sim.waitForIdle()
                sim.showSourceImage()
                sim.waitForIdle()
                sim.RayCount = 100000
                sim.waitForIdle()
                sim.AbsorbMissing = []
                sim.waitForIdle()
                sim.RaysAmountShown = -3.
                sim.waitForIdle()
                sim.ColoringType = "Power"
                sim.waitForIdle()
                sim.PlottingType = "Points"
                sim.waitForIdle()
                sim.close()
            sim.run(func=interact, no_server=True, silent=True, args=(sim,))

        RT = self.RT_Example()

        [RT.remove(id(F)) for F in RT.FilterList.copy()]
        self.assertTrue(not RT.FilterList)
        testFeatures(RT)

        [RT.remove(id(L)) for L in RT.LensList.copy()]
        RT.trace(N=RT.Rays.N)
        self.assertTrue(not RT.LensList)
        testFeatures(RT)

        [RT.remove(id(D)) for D in RT.DetectorList.copy()]
        self.assertTrue(not RT.DetectorList)
        testFeatures(RT)

        [RT.remove(id(RS)) for RS in RT.RaySourceList.copy()]
        self.assertTrue(not RT.RaySourceList)
        testFeatures(RT)

if __name__ == '__main__':
    unittest.main()
