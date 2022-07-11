#!/bin/env python3

import sys
sys.path.append('.')

import unittest
import warnings
import time

import numpy as np

import optrace as ot
from optrace.gui import TraceGUI

from threading import Thread

import matplotlib.pyplot as plt

# TODO check if scene resizing works
# TODO check if race conditions avoided while raytracing
# TODO check if replot() deletes old objects
# TODO check if change and removal of sources and detectors is handled correctly by the UI


def RT_Example() -> ot.Raytracer:

    # make Raytracer
    RT = ot.Raytracer(outline=[-5, 5, -5, 5, -5, 60], silent=True)

    # add Raysource
    RSS = ot.Surface("Circle", r=1)
    RS = ot.RaySource(RSS, direction="Parallel", spectrum=ot.preset_spec_FDC,
                   pos=[0, 0, 0], s=[0, 0, 1], polarization="y")
    RT.add(RS)

    RSS2 = ot.Surface("Circle", r=1)
    RS2 = ot.RaySource(RSS2, direction="Parallel", s=[0, 0, 1], spectrum=ot.preset_spec_D65,
                    pos=[0, 1, -3], polarization="Angle", pol_angle=25, power=2)
    RT.add(RS2)

    front = ot.Surface("Circle", r=3, rho=1/10, k=-0.444)
    back = ot.Surface("Circle", r=3, rho=-1/10, k=-7.25)
    nL2 = ot.RefractionIndex("Constant", n=1.8)
    L1 = ot.Lens(front, back, de=0.1, pos=[0, 0, 2], n=nL2)
    RT.add(L1)

    # add Lens 1
    front = ot.Surface("Asphere", r=3, rho=1/10, k=-0.444)
    back = ot.Surface("Asphere", r=3, rho=-1/10, k=-7.25)
    nL1 = ot.RefractionIndex("Cauchy", coeff=[1.49, 0.00354])
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
    RT.add(ot.Aperture(ap, pos=[0, 0, 20.3]))

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

    fspec = ot.TransmissionSpectrum("Function", func=func)
    RT.add(ot.Filter(ap, pos=[0, 0, 45.2], spectrum=fspec))

    # add Detector
    Det = ot.Detector(ot.Surface("Rectangle", dim=[2,2]), pos=[0,0,60])
    RT.add(Det)

    Det2 = ot.Detector(ot.Surface("Sphere", rho=-1/1.1, r=1), pos=[0, 0, 40])
    RT.add(Det2)

    return RT

class FrontendTests(unittest.TestCase):

    def setUp(self) -> None:
        warnings.simplefilter("ignore")
    
    def tearDown(self) -> None:
        warnings.simplefilter("default")
        time.sleep(1)
    
    def test_GUI_inits(self) -> None:

        Image = ot.preset_image_test_screen

        # make Raytracer
        RT = ot.Raytracer(outline=[-5, 5, -5, 5, 0, 40], silent=True)

        # add Raysource
        RSS = ot.Surface("Rectangle", dim=[4, 4])
        RS = ot.RaySource(RSS, direction="Diverging", div_angle=8,
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
                sim.run(_exit=True, no_server=True, silent=True)
        TraceGUI_Run.i = 0

        TraceGUI_Run()  # default init

        for ctype in TraceGUI.ColoringTypes:
            TraceGUI_Run(ColoringType=ctype)

        for ptype in TraceGUI.PlottingTypes:
            TraceGUI_Run(PlottingType=ptype)

        TraceGUI_Run(RayCount=100000)
        TraceGUI_Run(RayAmountShown=-2.)
        TraceGUI_Run(AbsorbMissing=False)
        TraceGUI_Run(RayAlpha=-1.)
        TraceGUI_Run(RayWidth=5)
        TraceGUI_Run(PosDet=8.9)

        TraceGUI_Run(RayCount=100000, RaysAmountShown=-2., AbsorbMissing=False, RayAlpha=-1., RayWidth=5,
                ColoringType="Wavelength", PlottingType="Points", PosDet=8.9)

    def test_Interaction(self) -> None:

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
            for mode in ot.RImage.display_modes:
                sim.ImageType = mode
                sim.showDetectorImage()
                sim.waitForIdle()

            # Image Tests Higher Res
            sim.ImagePixels = 300
            sim.showDetectorImage()
            sim.waitForIdle()

            # Image Test Source, but actually we should test all parameter combinations,
            sim.showSourceImage()

            # Focus Tests
            pos0 = sim.Raytracer.DetectorList[1].pos
            sim.DetectorSelection = sim.DetectorNames[1]

            for mode in ot.Raytracer.AutofocusModes:
                sim.FocusType = mode
                sim.moveToFocus()
                sim.waitForIdle()
                sim.PosDet = pos0[2]

            # Focus Test 4, show Debug Plot
            sim.FocusDebugPlot = True
            sim.moveToFocus()
            sim.waitForIdle()

            # Focus Test 5, one source only
            sim.FocusDebugPlot = False
            sim.PosDet = pos0[2]
            sim.AFOneSource = True
            sim.moveToFocus()
            sim.waitForIdle()
            
            # Ray Coloring Tests
            for type_ in sim.ColoringTypes:
                sim.ColoringType = type_
                sim.waitForIdle()

            # PlottingType Tests
            for type_ in sim.PlottingTypes:
                sim.PlottingType = type_
                sim.waitForIdle()
          
            # AbsorbMissing test
            sim.AbsorbMissing = False
            sim.waitForIdle()

            # retrace Tests
            sim.RayCount = 100000
            sim.waitForIdle()
            
            sim.RaysAmountShown = -2.5
            sim.waitForIdle()

            sim.RaysAmountShown = -2.
            sim.waitForIdle()

            sim.close()

        RT = RT_Example()
        
        sim = TraceGUI(RT)
        sim.run(_func=interact, no_server=True, silent=True, _args=(sim,))
        plt.close('all')


        def interact2(sim):
            
            sim.waitForIdle()

            # Image Type Tests log scaling
            sim.LogImage = True

            # display all image modes with log
            for mode in ot.RImage.display_modes:
                sim.ImageType = mode
                sim.showDetectorImage()
                sim.waitForIdle()

            # Image Tests Flip
            sim.LogImage = False
            sim.FlipImage = True
            sim.showDetectorImage()
            sim.waitForIdle()

            # one source only
            sim.DetImageOneSource = True
            sim.showDetectorImage()
            sim.waitForIdle()
            
            sim.close()

        sim = TraceGUI(RT)
        sim.run(_func=interact2, no_server=True, silent=True, _args=(sim,))
        plt.close('all')


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
                sim.AbsorbMissing = False
                sim.waitForIdle()
                sim.RaysAmountShown = -3.
                sim.waitForIdle()
                sim.CleanerView = True
                sim.waitForIdle()
                sim.ColoringType = "Power"
                sim.waitForIdle()
                sim.PlottingType = "Points"
                sim.waitForIdle()
                sim.close()
                time.sleep(1)

            sim.run(_func=interact, no_server=True, silent=True, _args=(sim,))

        RT = RT_Example()

        [RT.remove(F) for F in RT.FilterList.copy()]
        self.assertTrue(not RT.FilterList)
        testFeatures(RT)

        [RT.remove(L) for L in RT.LensList.copy()]
        RT.trace(N=RT.Rays.N)
        self.assertTrue(not RT.LensList)
        testFeatures(RT)

        [RT.remove(D) for D in RT.DetectorList.copy()]
        self.assertTrue(not RT.DetectorList)
        testFeatures(RT)

        [RT.remove(RS) for RS in RT.RaySourceList.copy()]
        self.assertTrue(not RT.RaySourceList)
        testFeatures(RT)

    def test_non2D(self):

        # initially there were problems with non-plotable "surfaces" like Line and Point
        # check if initial plotting, replotting and trying to color the parent sources (CorloringType=...)
        # works correctly

        # make Raytracer
        RT = ot.Raytracer(outline=[-5, 5, -5, 5, -5, 60], silent=True)

        # add Raysource
        RSS = ot.Surface("Point")
        RS = ot.RaySource(RSS, direction="Diverging", spectrum=ot.preset_spec_D65,
                          pos=[0, 0, 0], s=[0, 0, 1], div_angle=75)
        RT.add(RS)
        
        # add Raysource2
        RSS = ot.Surface("Line")
        RS2 = ot.RaySource(RSS, direction="Parallel", spectrum=ot.preset_spec_D65,
                          pos=[0, 0, 0], s=[0, 0, 1])
        RT.add(RS2)

        sim = TraceGUI(RT, ColoringType="Wavelength")

        def interact(sim):
            sim.waitForIdle()
            sim.ColoringType = "Wavelength"
            sim.waitForIdle()
            sim.PlottingType = "Points"
            sim.waitForIdle()
            sim.replot()
            sim.waitForIdle()
            sim.close()

        sim.run(_func=interact, no_server=True, silent=True, _args=(sim,))

    def test_Picker(self):
       
        # test picker interaction in the scene
        # this is done without keypress and mouse click simulation,
        # so it is not tested, if the scene reacts to those correctly
        # only the pick handling is checked here

        RT = RT_Example()
        
        def interact(sim):
            
            sim.waitForIdle()

            # change to z+ view, so there are are rays at the middle of the scene
            sim.Scene.z_plus_view()
            sim.waitForIdle()
       
            default_text = sim.RayText.text  # save default text for comparison

            # ray picked -> show default infos
            sim._RayPicker.pick(sim.SceneSize[0]/2, sim.SceneSize[1]/2, 0, sim.Scene.renderer)
            sim.waitForIdle()
            time.sleep(0.2) # delay so a human user can check the text
            text1 = sim.RayText.text
            self.assertNotEqual(text1, default_text)  # shows a ray info text
            
            # ray picked -> show verbose info
            sim.ShiftPressed = True
            sim._RayPicker.pick(sim.SceneSize[0]/2, sim.SceneSize[1]/2, 0, sim.Scene.renderer)
            sim.waitForIdle()
            time.sleep(0.2)
            text2 = sim.RayText.text
            self.assertNotEqual(text2, default_text)  # no default text
            self.assertNotEqual(text1, text2)  # not the old text
            
            # no ray picked -> default text
            sim._RayPicker.pick(0, 0, 0, sim.Scene.renderer)
            sim.waitForIdle()
            time.sleep(0.2)
            text2 = sim.RayText.text
            self.assertEqual(text2, default_text)  # shows default text
          
            # we have an extra picker sim._SpacePicker for right clicking in the scene,
            # but I don't know how to make the Picker.pick() function pick with a right click
            # so currently we overide the RayPicker with the onSpacePick method
            sim._RayPicker = sim.Scene.mlab.gcf().on_mouse_pick(sim.onSpacePick, button='Left')

            # space picked -> show coordinates
            sim.ShiftPressed = False
            sim._RayPicker.pick(sim.SceneSize[0]/3, sim.SceneSize[1]/3, 0, sim.Scene.renderer)
            sim.waitForIdle()
            time.sleep(0.2)
            text3 = sim.RayText.text
            self.assertNotEqual(text3, default_text)  # not the default text
            self.assertNotEqual(text3, text2)  # not the old text
            
            # valid space picked with shift -> move detector
            sim.ShiftPressed = True
            old_pos = RT.DetectorList[0].pos
            sim._RayPicker.pick(sim.SceneSize[0]/3, sim.SceneSize[1]/3, 0, sim.Scene.renderer)
            sim.waitForIdle()
            time.sleep(0.2)
            text4 = sim.RayText.text
            self.assertEqual(text4, default_text)  # not the default text
            self.assertNotEqual(RT.DetectorList[0].pos[2], old_pos[2])
            
            # space outside outline picked with shift -> move detector
            sim.Scene.y_plus_view()  # position 0, 0 in the window is outside the RT outline
            old_pos = RT.DetectorList[0].pos
            sim._RayPicker.pick(0, 0, 0, sim.Scene.renderer)
            sim.waitForIdle()
            time.sleep(0.2)
            text4 = sim.RayText.text
            self.assertEqual(text4, default_text)  # not the default text
            self.assertEqual(RT.DetectorList[0].pos[2], RT.outline[4])  # detector moved to the outline beginning
            
            sim.close()

        sim = TraceGUI(RT)
        sim.run(_func=interact, no_server=True, silent=True, _args=(sim,))

if __name__ == '__main__':
    unittest.main()

