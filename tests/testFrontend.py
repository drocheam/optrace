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

from pynput.keyboard import Controller, Key
from pyface.qt import QtGui  # closing UI elements

# TODO check if replot() deletes old objects
# TODO check if change and removal of sources and detectors is handled correctly by the UI


# the twisted reactor is not restartable
# that's why we we only run it in once (and not in this file)


def RT_Example() -> ot.Raytracer:

    # make Raytracer
    RT = ot.Raytracer(outline=[-5, 5, -5, 5, -5, 60], silent=True)

    # add Raysource
    RSS = ot.Surface("Circle", r=1)
    RS = ot.RaySource(RSS, direction="Parallel", spectrum=ot.presets.LightSpectrum.FDC,
                   pos=[0, 0, 0], s=[0, 0, 1], polarization="y")
    RT.add(RS)

    RSS2 = ot.Surface("Circle", r=1)
    RS2 = ot.RaySource(RSS2, direction="Parallel", s=[0, 0, 1], spectrum=ot.presets.LightSpectrum.D65,
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

        Image = ot.presets.Image.test_screen

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
                sim.debug(_exit=True, no_server=True, silent=True)
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

            sim._waitForIdle()

            # check if Detector is moving
            sim._setInMain("PosDet", 5.3)
            sim._waitForIdle()
            self.assertEqual(sim.PosDet, RT.DetectorList[0].pos[2])
        
            # Source Image Tests
            sim._doInMain(sim.showSourceImage)
            sim._waitForIdle()
            sim._setInMain("SourceSelection", sim.SourceNames[1])
            sim._waitForIdle()
            sim._doInMain(sim.showSourceImage)
            sim._waitForIdle()
            
            # Detector Image Tests
            sim._doInMain(sim.showDetectorImage)
            sim._waitForIdle()
            sim._setInMain("DetectorSelection", sim.DetectorNames[1])
            sim._waitForIdle()
            self.assertTrue(sim._DetInd == 1)
            self.assertTrue(sim.PosDet == sim.Raytracer.DetectorList[1].pos[2])
            sim._doInMain(sim.showDetectorImage)
            sim._waitForIdle()

            # Image Type Tests standard
            for mode in ot.RImage.display_modes:
                sim._setInMain("ImageType", mode)
                sim._waitForIdle()
                sim._doInMain(sim.showDetectorImage)
                sim._waitForIdle()

            # Image Tests Higher Res
            sim._setInMain("ImagePixels", 300)
            sim._waitForIdle()
            sim._doInMain(sim.showDetectorImage)
            sim._waitForIdle()

            # Image Test Source, but actually we should test all parameter combinations,
            sim._doInMain(sim.showSourceImage)
            sim._waitForIdle()

            # Focus Tests
            pos0 = sim.Raytracer.DetectorList[1].pos
            sim._setInMain("DetectorSelection", sim.DetectorNames[1])
            sim._waitForIdle()

            for mode in ot.Raytracer.AutofocusModes:
                sim._setInMain("FocusType", mode)
                sim._waitForIdle()
                sim._doInMain(sim.moveToFocus)
                sim._waitForIdle()
                sim._setInMain("PosDet", pos0[2])
                sim._waitForIdle()

            # Focus Test 4, show Debug Plot
            sim._setInMain("FocusDebugPlot", True)
            sim._waitForIdle()
            sim._doInMain(sim.moveToFocus)
            sim._waitForIdle()

            # Focus Test 5, one source only
            sim._setInMain("FocusDebugPlot", False)
            sim._waitForIdle()
            sim._setInMain("PosDet", pos0[2])
            sim._waitForIdle()
            sim._setInMain("AFOneSource", True)
            sim._doInMain(sim.moveToFocus)
            sim._waitForIdle()
            
            # Ray Coloring Tests
            for type_ in sim.ColoringTypes:
                sim._setInMain("ColoringType", type_)
                sim._waitForIdle()

            # PlottingType Tests
            for type_ in sim.PlottingTypes:
                sim._setInMain("PlottingType", type_)
                sim._waitForIdle()
          
            # AbsorbMissing test
            sim._setInMain("AbsorbMissing", False)
            sim._waitForIdle()

            # retrace Tests
            sim._setInMain("RayCount", 100000)
            sim._waitForIdle()
            
            sim._setInMain("RaysAmountShown", -2.5)
            sim._waitForIdle()

            sim._setInMain("RaysAmountShown", -2.)
            sim._waitForIdle()

            sim._doInMain(sim.close)

        RT = RT_Example()
        
        sim = TraceGUI(RT)
        sim.debug(_func=interact, no_server=True, silent=True, _args=(sim,))
        plt.close('all')


        def interact2(sim):
            
            sim._waitForIdle()

            # Image Type Tests log scaling
            sim._setInMain("LogImage", True)

            # display all image modes with log
            for mode in ot.RImage.display_modes:
                sim._setInMain("ImageType", mode)
                sim._waitForIdle()
                sim._doInMain(sim.showDetectorImage)
                sim._waitForIdle()

            # Image Tests Flip
            sim._setInMain("LogImage", False)
            sim._waitForIdle()
            sim._setInMain("FlipImage", True)
            sim._waitForIdle()
            sim._doInMain(sim.showDetectorImage)
            sim._waitForIdle()

            # one source only
            sim._setInMain("DetImageOneSource", True)
            sim._waitForIdle()
            sim._doInMain(sim.showDetectorImage)
            sim._waitForIdle()
            
            sim._doInMain(sim.close)

        sim = TraceGUI(RT)
        sim.debug(_func=interact2, no_server=True, silent=True, _args=(sim,))
        plt.close('all')


    def test_Missing(self) -> None:
        """test TraceGUI operation when Filter, Lenses, Detectors or Sources are missing"""

        def testFeatures(RT):
            sim = TraceGUI(RT)
            def interact(sim):
                sim._waitForIdle()
                sim._doInMain(sim.showDetectorImage)
                sim._waitForIdle()
                sim._doInMain(sim.moveToFocus)
                sim._waitForIdle()
                sim._setInMain("PosDet", 10.)
                sim._waitForIdle()
                sim._doInMain(sim.showSourceImage)
                sim._waitForIdle()
                sim._setInMain("RayCount", 100000)
                sim._waitForIdle()
                sim._setInMain("AbsorbMissing", False)
                sim._waitForIdle()
                sim._setInMain("RaysAmountShown", -3.)
                sim._waitForIdle()
                sim._setInMain("CleanerView", True)
                sim._waitForIdle()
                sim._setInMain("ColoringType", "Power")
                sim._waitForIdle()
                sim._setInMain("PlottingType", "Points")
                sim._waitForIdle()
                sim._doInMain(sim.close)

            sim.debug(_func=interact, no_server=True, silent=True, _args=(sim,))
            time.sleep(1)

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

    def test_ActionSpam(self):

        RT = RT_Example()
        sim = TraceGUI(RT)

        def interact(sim):

            sim._waitForIdle()

            N0 = sim.RayCount
            sim.RayCount = int(N0*1.3)
            sim.showDetectorImage()
            sim.showSourceImage()
            sim.moveToFocus()

            time.sleep(0.01)
            sim.RayCount = int(N0/1.3)
            sim.PosDet = (RT.outline[5] - RT.outline[4])/2
            sim.moveToFocus()
            sim.DetectorSelection = sim.DetectorNames[1]
            sim.showDetectorImage()
            sim.replotRays()
            sim.showSourceImage()

            time.sleep(0.1)
            sim.replotRays()
            sim.PosDet = (RT.outline[5] - RT.outline[4])/2
            sim.showSourceImage()
            sim.moveToFocus()
            sim.RayCount = int(N0*1.6)
            sim.showDetectorImage()
            sim.DetectorSelection = sim.DetectorNames[1]

            sim.RayCount = 1000000
            sim.replotRays()
            sim.DetectorSelection = sim.DetectorNames[1]
            sim.moveToFocus()
            
            sim._waitForIdle()

            self.assertEqual(sim.RayCount, 1000000)
            self.assertEqual(sim.Raytracer.Rays.N, 1000000)
            sim.close()

        sim.debug(_func=interact, no_server=True, silent=True, _args=(sim,))

    def test_KeyPresses(self):

        RT = RT_Example()
        sim = TraceGUI(RT)

        keyboard = Controller()

        def sendKey(sim, key):
            sim._doInMain(sim.Scene.scene_editor._content.setFocus)
            keyboard.press(key)
            keyboard.release(key)


        def interact(sim):

            sim._waitForIdle()

            # check CleanerView shortcut
            self.assertTrue(len(sim.CleanerView) == 0)
            sendKey(sim, "c")
            sim._waitForIdle()
            self.assertTrue(len(sim.CleanerView) != 0)
            
            # check Ray-Point toggle shortcut
            self.assertTrue(sim.PlottingType == "Rays")
            sendKey(sim, "r")
            sim._waitForIdle()
            self.assertTrue(sim.PlottingType == "Points")
            
            # y plus view
            # TODO how to check this?
            sendKey(sim, "y")
            sim._waitForIdle()

            # replot rays
            # TODO how to check this, if rays are chosen random and can therefore stay the same?
            sendKey(sim, "n")
            sim._waitForIdle()
          
            # detect shift key
            self.assertFalse(sim._ShiftPressed)
            keyboard.press(Key.shift)
            self.assertTrue(sim._ShiftPressed)
            keyboard.release(Key.shift)
            self.assertFalse(sim._ShiftPressed)

            # do this one last, since it raises another window
            # and I don't know how to focus the scene after this
            # check DetectorImage shortcut
            self.assertTrue(sim.lastDetImage is None)
            sendKey(sim, "d")
            sim._waitForIdle()
            self.assertTrue(sim.lastDetImage is not None)
            
            sim._waitForIdle()
            sim._doInMain(sim.close)

        sim.debug(_func=interact, no_server=True, silent=True, _args=(sim,))
    

    def test_Resize(self):
        
        # this test checks if
        # * some UI properties get resized correctly
        # * no exceptions while doing so
        # * resizing to initial state resizes everything back to normal

        RT = RT_Example()
        sim = TraceGUI(RT)

        def interact(sim):
            sim._waitForIdle()

            sim._setInMain("ColoringType", "Power") # shows a side bar, that also needs to be rescaled
            sim._waitForIdle()

            SceneSize0 = sim._SceneSize
            Window = sim.Scene.scene_editor._content.window()

            # properties before resizing
            ff = sim._AxisPlotObjects[0][0].axes.font_factor
            zoom = sim._OrientationAxes.widgets[0].zoom
            pos2 = sim._RaysPlot.parent.scalar_lut_manager.scalar_bar_representation.position2

            qsize = Window.size()
            ss0 = np.array([qsize.width(), qsize.height()])
            ss1 = ss0 * 1.3
            ss2 = ss1 / 1.2

            # enlarge
            sim._doInMain(Window.resize, *ss1.astype(int))
            time.sleep(0.5)  # how to check how much time it takes?

            # check if scale properties changed
            self.assertNotAlmostEqual(sim._SceneSize[0], SceneSize0[0])  # scene size variable changed
            self.assertNotAlmostEqual(sim._SceneSize[1], SceneSize0[1])  # scene size variable changed
            self.assertNotAlmostEqual(ff, sim._AxisPlotObjects[0][0].axes.font_factor) 
            self.assertNotAlmostEqual(zoom, sim._OrientationAxes.widgets[0].zoom) 
            self.assertNotAlmostEqual(pos2[0], sim._RaysPlot.parent.scalar_lut_manager.scalar_bar_representation.position2[0])
            self.assertNotAlmostEqual(pos2[1], sim._RaysPlot.parent.scalar_lut_manager.scalar_bar_representation.position2[1])

            sim._doInMain(Window.resize, *ss2.astype(int))
            time.sleep(0.5)
            sim._doInMain(Window.showFullScreen)
            time.sleep(0.5)
            sim._doInMain(Window.showMaximized)
            time.sleep(0.5)
            sim._doInMain(Window.showMinimized)
            time.sleep(0.5)
            sim._doInMain(Window.showNormal)
            time.sleep(0.5)
            sim._doInMain(Window.resize, *ss0.astype(int))
            time.sleep(0.5)
           
            # check if scale properties are back at their default state
            self.assertAlmostEqual(sim._SceneSize[0], SceneSize0[0]) 
            self.assertAlmostEqual(sim._SceneSize[1], SceneSize0[1]) 
            self.assertAlmostEqual(ff, sim._AxisPlotObjects[0][0].axes.font_factor) 
            self.assertAlmostEqual(zoom, sim._OrientationAxes.widgets[0].zoom) 
            self.assertAlmostEqual(pos2[0], sim._RaysPlot.parent.scalar_lut_manager.scalar_bar_representation.position2[0])
            self.assertAlmostEqual(pos2[1], sim._RaysPlot.parent.scalar_lut_manager.scalar_bar_representation.position2[1])

            sim._waitForIdle()
            sim._doInMain(sim.close)

        sim.debug(_func=interact, no_server=True, silent=True, _args=(sim,))
        
    def test_non2D(self):

        # initially there were problems with non-plotable "surfaces" like Line and Point
        # check if initial plotting, replotting and trying to color the parent sources (CorloringType=...)
        # works correctly

        # make Raytracer
        RT = ot.Raytracer(outline=[-5, 5, -5, 5, -5, 60], silent=True)

        # add Raysource
        RSS = ot.Surface("Point")
        RS = ot.RaySource(RSS, direction="Diverging", spectrum=ot.presets.LightSpectrum.D65,
                          pos=[0, 0, 0], s=[0, 0, 1], div_angle=75)
        RT.add(RS)
        
        # add Raysource2
        RSS = ot.Surface("Line")
        RS2 = ot.RaySource(RSS, direction="Parallel", spectrum=ot.presets.LightSpectrum.D65,
                          pos=[0, 0, 0], s=[0, 0, 1])
        RT.add(RS2)

        sim = TraceGUI(RT, ColoringType="Wavelength")

        def interact(sim):
            sim._waitForIdle()
            sim._setInMain("ColoringType", "Wavelength")
            sim._waitForIdle()
            sim._setInMain("PlottingType", "Points")
            sim._waitForIdle()
            sim._doInMain(sim.replot)
            sim._waitForIdle()
            sim._doInMain(sim.close)

        sim.debug(_func=interact, no_server=True, silent=True, _args=(sim,))

    def test_Picker(self):
       
        # test picker interaction in the scene
        # this is done without keypress and mouse click simulation,
        # so it is not tested, if the scene reacts to those correctly
        # only the pick handling is checked here

        RT = RT_Example()
        
        def interact(sim):
            
            sim._waitForIdle()

            # change to z+ view, so there are are rays at the middle of the scene
            sim._doInMain(sim.Scene.z_plus_view)
            sim._waitForIdle()
       
            default_text = sim._RayText.text  # save default text for comparison

            # ray picked -> show default infos
            sim._doInMain(sim._RayPicker.pick, sim._SceneSize[0]/2, sim._SceneSize[1]/2, 0, sim.Scene.renderer)
            sim._waitForIdle()
            time.sleep(0.2) # delay so a human user can check the text
            text1 = sim._RayText.text
            self.assertNotEqual(text1, default_text)  # shows a ray info text
            
            # ray picked -> show verbose info
            sim._ShiftPressed = True
            sim._doInMain(sim._RayPicker.pick, sim._SceneSize[0]/2, sim._SceneSize[1]/2, 0, sim.Scene.renderer)
            sim._waitForIdle()
            time.sleep(0.2)
            text2 = sim._RayText.text
            self.assertNotEqual(text2, default_text)  # no default text
            self.assertNotEqual(text1, text2)  # not the old text
            
            # no ray picked -> default text
            sim._doInMain(sim._RayPicker.pick, 0, 0, 0, sim.Scene.renderer)
            sim._waitForIdle()
            time.sleep(0.2)
            text2 = sim._RayText.text
            self.assertEqual(text2, default_text)  # shows default text
          
            # we have an extra picker sim._SpacePicker for right clicking in the scene,
            # but I don't know how to make the Picker.pick() function pick with a right click
            # so currently we overide the RayPicker with the onSpacePick method
            # TODO this should be done in main thread
            sim._RayPicker = sim.Scene.mlab.gcf().on_mouse_pick(sim._onSpacePick, button='Left')
            # space picked -> show coordinates
            sim._ShiftPressed = False
            sim._doInMain(sim._RayPicker.pick, sim._SceneSize[0]/3, sim._SceneSize[1]/3, 0, sim.Scene.renderer)
            sim._waitForIdle()
            time.sleep(0.2)
            text3 = sim._RayText.text
            self.assertNotEqual(text3, default_text)  # not the default text
            self.assertNotEqual(text3, text2)  # not the old text
            
            # valid space picked with shift -> move detector
            sim._ShiftPressed = True
            old_pos = RT.DetectorList[0].pos
            sim._doInMain(sim._RayPicker.pick, sim._SceneSize[0]/3, sim._SceneSize[1]/3, 0, sim.Scene.renderer)
            sim._waitForIdle()
            time.sleep(0.2)
            text4 = sim._RayText.text
            self.assertEqual(text4, default_text)  # not the default text
            self.assertNotEqual(RT.DetectorList[0].pos[2], old_pos[2])
            
            # space outside outline picked with shift -> move detector
            sim._doInMain(sim.Scene.y_plus_view)
            old_pos = RT.DetectorList[0].pos
            sim._doInMain(sim._RayPicker.pick, 0, 0, 0, sim.Scene.renderer)
            sim._waitForIdle()
            time.sleep(0.2)
            text4 = sim._RayText.text
            self.assertEqual(text4, default_text)  # not the default text
            self.assertEqual(RT.DetectorList[0].pos[2], RT.outline[4])  # detector moved to the outline beginning
            
            sim._doInMain(sim.close)

        sim = TraceGUI(RT)
        sim.debug(_func=interact, no_server=True, silent=True, _args=(sim,))

if __name__ == '__main__':
    unittest.main()

