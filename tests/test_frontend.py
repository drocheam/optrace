#!/bin/env python3

import sys
sys.path.append('.')

import unittest
import warnings
import time
import numpy as np

import matplotlib.pyplot as plt
from pynput.keyboard import Controller, Key

import optrace as ot
from optrace.gui import TraceGUI


# TODO check if replot() deletes old objects
# TODO check if change and removal of sources and detectors is handled correctly by the UI


def rt_example() -> ot.Raytracer:

    # make Raytracer
    RT = ot.Raytracer(outline=[-5, 5, -5, 5, -5, 60], silent=True)

    # add Raysource
    RSS = ot.Surface("Circle", r=1)
    RS = ot.RaySource(RSS, direction="Parallel", spectrum=ot.presets.light_spectrum.FDC,
                      pos=[0, 0, 0], s=[0, 0, 1], polarization="y")
    RT.add(RS)

    RSS2 = ot.Surface("Circle", r=1)
    RS2 = ot.RaySource(RSS2, direction="Parallel", s=[0, 0, 1], spectrum=ot.presets.light_spectrum.D65,
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
    Det = ot.Detector(ot.Surface("Rectangle", dim=[2, 2]), pos=[0, 0, 0])
    RT.add(Det)

    Det2 = ot.Detector(ot.Surface("Sphere", rho=-1/1.1, r=1), pos=[0, 0, 40])
    RT.add(Det2)

    return RT

class FrontendTests(unittest.TestCase):

    def setUp(self) -> None:
        warnings.simplefilter("ignore")
    
    def tearDown(self) -> None:
        warnings.simplefilter("default")
    
    def test_gui_inits(self) -> None:

        Image = ot.presets.Image.test_screen

        # make Raytracer
        RT = ot.Raytracer(outline=[-5, 5, -5, 5, 0, 40], silent=True)

        # add Raysource
        RSS = ot.Surface("Rectangle", dim=[4, 4])
        RS = ot.RaySource(RSS, direction="Diverging", div_angle=8,
                          image=Image, s=[0, 0, 1], pos=[0, 0, 0])
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
        def trace_gui_run(**kwargs):
            trace_gui_run.i += 1
            with self.subTest(i=trace_gui_run.i, args=kwargs):
                sim = TraceGUI(RT, **kwargs)
                sim.debug(_exit=True, silent=True)
        trace_gui_run.i = 0

        trace_gui_run()  # default init

        for ctype in TraceGUI.ColoringTypes:
            trace_gui_run(ColoringType=ctype)

        for ptype in TraceGUI.PlottingTypes:
            trace_gui_run(PlottingType=ptype)

        trace_gui_run(RayCount=100000)
        trace_gui_run(RayAmountShown=-2.)
        trace_gui_run(AbsorbMissing=False)
        trace_gui_run(RayAlpha=-1.)
        trace_gui_run(RayWidth=5)
        trace_gui_run(PosDet=8.9)

        trace_gui_run(RayCount=100000, RaysAmountShown=-2., AbsorbMissing=False, RayAlpha=-1., RayWidth=5,
                      ColoringType="Wavelength", PlottingType="Points", PosDet=8.9)

    def test_interaction0(self) -> None:

        def interact(sim):

            sim._wait_for_idle()

            # check if Detector is moving
            sim._set_in_main("PosDet", 5.3)
            sim._wait_for_idle()
            self.assertEqual(sim.PosDet, RT.DetectorList[0].pos[2])
        
            # Source Image Tests
            sim._do_in_main(sim.show_source_image)
            sim._wait_for_idle()
            sim._set_in_main("SourceSelection", sim.SourceNames[1])
            sim._wait_for_idle()
            sim._do_in_main(sim.show_source_image)
            sim._wait_for_idle()
            
            # Detector Image Tests
            sim._do_in_main(sim.show_detector_image)
            sim._wait_for_idle()
            sim._set_in_main("DetectorSelection", sim.DetectorNames[1])
            sim._wait_for_idle()
            self.assertTrue(sim._DetInd == 1)
            self.assertTrue(sim.PosDet == sim.Raytracer.DetectorList[1].pos[2])
            sim._do_in_main(sim.show_detector_image)
            sim._wait_for_idle()

            # Image Type Tests standard
            for mode in ot.RImage.display_modes:
                sim._set_in_main("ImageType", mode)
                sim._wait_for_idle()
                sim._do_in_main(sim.show_detector_image)
                sim._wait_for_idle()
            plt.close('all')

            # Image Tests Higher Res
            sim._set_in_main("ImagePixels", 300)
            sim._wait_for_idle()
            sim._do_in_main(sim.show_detector_image)
            sim._wait_for_idle()

            # Image Test Source, but actually we should test all parameter combinations,
            sim._do_in_main(sim.show_source_image)
            sim._wait_for_idle()

            # Focus Tests
            pos0 = sim.Raytracer.DetectorList[1].pos
            sim._set_in_main("DetectorSelection", sim.DetectorNames[1])
            sim._wait_for_idle()

            for mode in ot.Raytracer.autofocus_modes:
                sim._set_in_main("FocusType", mode)
                sim._wait_for_idle()
                sim._do_in_main(sim.move_to_focus)
                sim._wait_for_idle()
                sim._set_in_main("PosDet", pos0[2])
                sim._wait_for_idle()

            # Focus Test 4, show Debug Plot
            sim._set_in_main("FocusCostPlot", True)
            sim._wait_for_idle()
            sim._do_in_main(sim.move_to_focus)
            sim._wait_for_idle()

            # Focus Test 5, one source only
            sim._set_in_main("FocusCostPlot", False)
            sim._wait_for_idle()
            sim._set_in_main("PosDet", pos0[2])
            sim._wait_for_idle()
            sim._set_in_main("AFOneSource", True)
            sim._do_in_main(sim.move_to_focus)
            sim._wait_for_idle()
            
            # Ray Coloring Tests
            for type_ in sim.ColoringTypes:
                sim._set_in_main("ColoringType", type_)
                sim._wait_for_idle()

            # PlottingType Tests
            for type_ in sim.PlottingTypes:
                sim._set_in_main("PlottingType", type_)
                sim._wait_for_idle()
          
            # AbsorbMissing test
            sim._set_in_main("AbsorbMissing", False)
            sim._wait_for_idle()

            # retrace Tests
            sim._set_in_main("RayCount", 100000)
            sim._wait_for_idle()
            
            sim._set_in_main("RaysAmountShown", -2.5)
            sim._wait_for_idle()

            sim._set_in_main("RaysAmountShown", -2.)
            sim._wait_for_idle()

            sim._do_in_main(sim.close)

        RT = rt_example()
        sim = TraceGUI(RT)
        sim.debug(_func=interact, silent=True, _args=(sim,))
        plt.close('all')

    def test_interaction1(self) -> None:

        def interact2(sim):
            
            sim._wait_for_idle()

            # Image Type Tests log scaling
            sim._set_in_main("LogImage", True)

            # display all image modes with log
            for mode in ot.RImage.display_modes:
                sim._set_in_main("ImageType", mode)
                sim._wait_for_idle()
                sim._do_in_main(sim.show_detector_image)
                sim._wait_for_idle()
            plt.close('all')

            # Image Tests Flip
            sim._set_in_main("LogImage", False)
            sim._wait_for_idle()
            sim._set_in_main("FlipImage", True)
            sim._wait_for_idle()
            sim._do_in_main(sim.show_detector_image)
            sim._wait_for_idle()

            # one source only
            sim._set_in_main("DetImageOneSource", True)
            sim._wait_for_idle()
            sim._do_in_main(sim.show_detector_image)
            sim._wait_for_idle()
            
            sim._do_in_main(sim.close)

        RT = rt_example()
        sim = TraceGUI(RT)
        sim.debug(_func=interact2, silent=True, _args=(sim,))
        plt.close('all')

    def test_interaction3(self) -> None:

        def interact3(sim):
            
            sim._wait_for_idle()

            # test source and detector spectrum
            sim._do_in_main(sim.show_source_spectrum)
            sim._wait_for_idle()
            sim._do_in_main(sim.show_detector_spectrum)
            sim._wait_for_idle()
            sim._set_in_main("DetSpectrumOneSource", True)
            sim._do_in_main(sim.show_detector_spectrum)
            sim._wait_for_idle()

            # test image cuts
            sim._do_in_main(sim.show_source_cut)
            sim._wait_for_idle()
            sim._set_in_main("DetImageOneSource", True)
            sim._wait_for_idle()
            sim._do_in_main(sim.show_detector_cut)
            sim._wait_for_idle()
            sim._set_in_main("CutDimension", "y dimension")
            sim._do_in_main(sim.show_detector_cut)
            sim._wait_for_idle()
            sim._set_in_main("CutValue", 0)  # exists inside image
            sim._do_in_main(sim.show_detector_cut)
            sim._wait_for_idle()
            sim._set_in_main("CutValue", 100)  # does not exist inside image
            sim._do_in_main(sim.show_detector_cut)
            sim._wait_for_idle()
            plt.close('all')

            # open browsers
            sim._do_in_main(sim.open_property_browser)
            sim._wait_for_idle()
            time.sleep(2)  # wait some time so windows can be seen by a human user

            # tracing with only one thread
            sim._set_in_main("RaytracerSingleThread", True)
            sim._wait_for_idle()
            self.assertEqual(sim.Raytracer.threading, False)
            sim._set_in_main("RaytracerSingleThread", False)
            sim._wait_for_idle()
            self.assertEqual(sim.Raytracer.threading, True)
           
            sim._set_in_main("WireframeSurfaces", True)
            sim._wait_for_idle()
            sim._set_in_main("WireframeSurfaces", False)
            sim._wait_for_idle()
            
            # activate/deactivate all warning. How to check this?
            sim._set_in_main("ShowAllWarnings", True)
            sim._wait_for_idle()
            sim._set_in_main("ShowAllWarnings", False)
            sim._wait_for_idle()

            # activate/deactivate all warning. How to check this?
            sim._set_in_main("GarbageCollectorStats", True)
            sim._wait_for_idle()
            sim._set_in_main("GarbageCollectorStats", False)
            sim._wait_for_idle()

            sim._do_in_main(sim.close)

        RT = rt_example()
        sim = TraceGUI(RT)
        sim.debug(_func=interact3, silent=True, _args=(sim,))
        plt.close('all')

    def test_missing(self) -> None:
        """test TraceGUI operation when Filter, Lenses, Detectors or Sources are missing"""

        def test_features(RT):
            sim = TraceGUI(RT)

            def interact(sim):
                sim._wait_for_idle()
                sim._do_in_main(sim.show_detector_image)
                sim._wait_for_idle()
                sim._do_in_main(sim.move_to_focus)
                sim._wait_for_idle()
                sim._set_in_main("PosDet", 10.)
                sim._wait_for_idle()
                sim._do_in_main(sim.show_source_image)
                sim._wait_for_idle()
                sim._set_in_main("RayAlpha", -2.8946)
                sim._wait_for_idle()
                sim._set_in_main("RayWidth", 5)
                sim._wait_for_idle()
                sim._set_in_main("RayCount", 100000)
                sim._wait_for_idle()
                sim._set_in_main("AbsorbMissing", False)
                sim._wait_for_idle()
                sim._set_in_main("RaysAmountShown", -3.)
                sim._wait_for_idle()
                sim._set_in_main("CleanerView", True)
                sim._wait_for_idle()
                sim._set_in_main("ColoringType", "Power")
                sim._wait_for_idle()
                sim._set_in_main("PlottingType", "Points")

                sim._do_in_main(sim.change_selected_ray_source)
                sim._wait_for_idle()
                sim._do_in_main(sim.change_detector)
                sim._wait_for_idle()
                
                sim._do_in_main(sim.show_source_spectrum)
                sim._wait_for_idle()
                sim._do_in_main(sim.show_detector_spectrum)
                sim._wait_for_idle()

                sim._do_in_main(sim.replot)
                sim._wait_for_idle()
                
                sim._do_in_main(sim.close)

            sim.debug(_func=interact, silent=True, _args=(sim,))
            time.sleep(1)

        RT = rt_example()

        [RT.remove(F) for F in RT.FilterList.copy()]
        self.assertTrue(not RT.FilterList)
        test_features(RT)

        [RT.remove(L) for L in RT.LensList.copy()]
        RT.trace(N=RT.Rays.N)
        self.assertTrue(not RT.LensList)
        test_features(RT)

        [RT.remove(D) for D in RT.DetectorList.copy()]
        self.assertTrue(not RT.DetectorList)
        test_features(RT)

        [RT.remove(RS) for RS in RT.RaySourceList.copy()]
        self.assertTrue(not RT.RaySourceList)
        test_features(RT)

    def test_action_spam(self):

        RT = rt_example()
        sim = TraceGUI(RT)

        def interact(sim):

            sim._wait_for_idle()

            N0 = sim.RayCount
            sim.RayCount = int(N0*1.3)
            sim.show_detector_image()
            sim.show_source_image()
            sim.move_to_focus()

            time.sleep(0.01)
            sim.RayCount = int(N0/1.3)
            sim.PosDet = (RT.outline[5] - RT.outline[4])/2
            sim.show_source_cut()
            sim.move_to_focus()
            sim.DetectorSelection = sim.DetectorNames[1]
            sim.show_detector_image()
            sim.replot_rays()
            sim.show_source_image()

            time.sleep(0.1)
            sim._Cmd = "GUI.replot()"
            sim.replot_rays()
            sim.show_detector_spectrum()
            sim.PosDet = (RT.outline[5] - RT.outline[4])/2
            sim.show_source_image()
            sim.move_to_focus()
            sim.RayCount = int(N0*1.6)
            sim.send_cmd()
            sim.show_detector_image()
            sim.DetectorSelection = sim.DetectorNames[1]

            sim.RayCount = 1000000
            sim.replot_rays()
            sim.DetectorSelection = sim.DetectorNames[1]
            sim.move_to_focus()
            sim.show_source_spectrum()
            
            sim._wait_for_idle()

            self.assertEqual(sim.RayCount, 1000000)
            self.assertEqual(sim.Raytracer.Rays.N, 1000000)

            sim._wait_for_idle()
            sim.close()

        sim.debug(_func=interact, silent=True, _args=(sim,))

    def test_key_presses(self):
        RT = rt_example()
        sim = TraceGUI(RT)

        keyboard = Controller()

        def send_key(sim, key):
            sim._do_in_main(sim.Scene.scene_editor._content.setFocus)
            keyboard.press(key)
            keyboard.release(key)

        def interact(sim):

            sim._wait_for_idle()

            # check CleanerView shortcut
            self.assertTrue(len(sim.CleanerView) == 0)
            send_key(sim, "c")
            sim._wait_for_idle()
            self.assertTrue(len(sim.CleanerView) != 0)
            
            # check Ray-Point toggle shortcut
            self.assertTrue(sim.PlottingType == "Rays")
            send_key(sim, "r")
            sim._wait_for_idle()
            self.assertTrue(sim.PlottingType == "Points")
            
            # y plus view
            # TODO how to check this?
            send_key(sim, "y")
            sim._wait_for_idle()
            
            # maximize scene / hide other menus
            # TODO how to check this?
            send_key(sim, "h")
            sim._wait_for_idle()

            # key without shortcut
            send_key(sim, "i")
            sim._wait_for_idle()
            
            # replot rays
            # TODO how to check this, if rays are chosen random and can therefore stay the same?
            send_key(sim, "n")
            sim._wait_for_idle()
          
            # do this one last, since it raises another window
            # and I don't know how to focus the scene after this
            # check DetectorImage shortcut
            self.assertTrue(sim.lastDetImage is None)
            send_key(sim, "d")
            sim._wait_for_idle()
            self.assertTrue(sim.lastDetImage is not None)
           
            # unset scene focus so application does not react on keypresses
            sim.Scene.scene_editor._content.clearFocus()
            self.assertTrue(len(sim.CleanerView) == 1)
            send_key(sim, "c")
            sim._wait_for_idle()
            self.assertTrue(len(sim.CleanerView) == 1)

            sim._wait_for_idle()
            sim._do_in_main(sim.close)

        sim.debug(_func=interact, silent=True, _args=(sim,))
    
    def test_send_cmd(self):
        RT = rt_example()
        sim = TraceGUI(RT, silent=True)

        def send(cmd):
            sim._set_in_main("_Cmd", cmd)
            sim._do_in_main(sim.send_cmd)
            sim._wait_for_idle()

        def interact(sim):
            sim._wait_for_idle()

            state = RT.Rays.crepr()
            self.assertEqual(sim.Command_History, "")
            send("GUI.replot()")
            self.assertNotEqual(sim.Command_History, "")  # history changed
            self.assertFalse(state == RT.Rays.crepr())  # check if raytraced

            send("GUI.show_detector_image()")
            self.assertTrue(sim.lastDetImage is not None)  # check if raytraced
           
            # create optrace objects
            LLlen = len(RT.LensList)
            send("a = Surface(\"Circle\");"
                 "b = Surface(\"Sphere\", rho=-1/10);"
                 "L = Lens(a, b, n=presets.RefractionIndex.SF10, de=0.2, pos=[0, 0, 8]);"
                 "RT.add(L)")
            self.assertEqual(len(RT.LensList), LLlen+1)

            # numpy and time available
            send("a = np.array([1, 2, 3]);"
                 "time.time()")

            state = RT.Rays.crepr()
            send("RT.remove(APL[0])")
            self.assertEqual(len(RT.ApertureList), 0)  # check if ApertureList empty after removal
            self.assertEqual(len(sim._AperturePlotObjects), 0)  # check if aperture plot is removed
            self.assertFalse(state == RT.Rays.crepr())  # check if raytraced
          
            send("")  # check empty command

            send("throw RuntimeError()")  # check if exceptions are handled

            sim._do_in_main(sim.close)

        sim.debug(_func=interact, silent=True, _args=(sim,))

    def test_resize(self):
        
        # this test checks if
        # * some UI properties get resized correctly
        # * no exceptions while doing so
        # * resizing to initial state resizes everything back to normal

        RT = rt_example()
        sim = TraceGUI(RT)

        def interact(sim):
            sim._wait_for_idle()

            sim._set_in_main("ColoringType", "Power")  # shows a side bar, that also needs to be rescaled
            sim._wait_for_idle()

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
            sim._do_in_main(Window.resize, *ss1.astype(int))
            time.sleep(0.5)  # how to check how much time it takes?

            # check if scale properties changed
            self.assertNotAlmostEqual(sim._SceneSize[0], SceneSize0[0])  # scene size variable changed
            self.assertNotAlmostEqual(sim._SceneSize[1], SceneSize0[1])  # scene size variable changed
            self.assertNotAlmostEqual(ff, sim._AxisPlotObjects[0][0].axes.font_factor) 
            self.assertNotAlmostEqual(zoom, sim._OrientationAxes.widgets[0].zoom) 
            self.assertNotAlmostEqual(pos2[0],
                                      sim._RaysPlot.parent.scalar_lut_manager.scalar_bar_representation.position2[0])
            self.assertNotAlmostEqual(pos2[1],
                                      sim._RaysPlot.parent.scalar_lut_manager.scalar_bar_representation.position2[1])

            sim._do_in_main(Window.resize, *ss2.astype(int))
            time.sleep(0.5)
            sim._do_in_main(Window.showFullScreen)
            time.sleep(0.5)
            sim._do_in_main(Window.showMaximized)
            time.sleep(0.5)
            sim._do_in_main(Window.showMinimized)
            time.sleep(0.5)
            sim._do_in_main(Window.showNormal)
            time.sleep(0.5)
            sim._do_in_main(Window.resize, *ss0.astype(int))
            time.sleep(0.5)
           
            # check if scale properties are back at their default state
            self.assertAlmostEqual(sim._SceneSize[0], SceneSize0[0]) 
            self.assertAlmostEqual(sim._SceneSize[1], SceneSize0[1]) 
            self.assertAlmostEqual(ff, sim._AxisPlotObjects[0][0].axes.font_factor) 
            self.assertAlmostEqual(zoom, sim._OrientationAxes.widgets[0].zoom) 
            self.assertAlmostEqual(pos2[0],
                                   sim._RaysPlot.parent.scalar_lut_manager.scalar_bar_representation.position2[0])
            self.assertAlmostEqual(pos2[1],
                                   sim._RaysPlot.parent.scalar_lut_manager.scalar_bar_representation.position2[1])

            sim._wait_for_idle()
            sim._do_in_main(sim.close)

        sim.debug(_func=interact, silent=True, _args=(sim,))
        
    def test_non_2d(self):

        # initially there were problems with non-plotable "surfaces" like Line and Point
        # check if initial plotting, replotting and trying to color the parent sources (ColoringType=...)
        # works correctly

        # make Raytracer
        RT = ot.Raytracer(outline=[-5, 5, -5, 5, -5, 60], silent=True)

        # add Raysource
        RSS = ot.Surface("Point")
        RS = ot.RaySource(RSS, direction="Diverging", spectrum=ot.presets.light_spectrum.D65,
                          pos=[0, 0, 0], s=[0, 0, 1], div_angle=75)
        RT.add(RS)
        
        # add Raysource2
        RSS = ot.Surface("Line")
        RS2 = ot.RaySource(RSS, direction="Parallel", spectrum=ot.presets.light_spectrum.D65,
                           pos=[0, 0, 0], s=[0, 0, 1])
        RT.add(RS2)

        sim = TraceGUI(RT, ColoringType="Wavelength")

        def interact(sim):
            sim._wait_for_idle()
            sim._set_in_main("ColoringType", "Wavelength")
            sim._wait_for_idle()
            sim._set_in_main("PlottingType", "Points")
            sim._wait_for_idle()
            sim._do_in_main(sim.replot)
            sim._wait_for_idle()
            sim._do_in_main(sim.close)

        sim.debug(_func=interact, silent=True, _args=(sim,))

    def test_picker(self):
        # test picker interaction in the scene
        # this is done without keypress and mouse click simulation,
        # so it is not tested, if the scene reacts to those correctly
        # only the pick handling is checked here

        RT = rt_example()
        keyboard = Controller()
        
        def interact(sim):
            
            sim._wait_for_idle()

            # change to z+ view, so there are rays at the middle of the scene
            sim._do_in_main(sim.Scene.z_plus_view)
            sim._wait_for_idle()
       
            default_text = sim._RayText.text  # save default text for comparison

            # ray picked -> show default infos
            sim._do_in_main(sim._RayPicker.pick, sim._SceneSize[0] / 2, sim._SceneSize[1] / 2, 0, sim.Scene.renderer)
            sim._wait_for_idle()
            time.sleep(0.2)  # delay so a human user can check the text
            text1 = sim._RayText.text
            self.assertNotEqual(text1, default_text)  # shows a ray info text
            
            # ray picked -> show verbose info
            keyboard.press(Key.shift)
            sim._do_in_main(sim._RayPicker.pick, sim._SceneSize[0] / 2, sim._SceneSize[1] / 2, 0, sim.Scene.renderer)
            sim._wait_for_idle()
            time.sleep(0.2)
            text2 = sim._RayText.text
            self.assertNotEqual(text2, default_text)  # no default text
            self.assertNotEqual(text1, text2)  # not the old text
            
            # no ray picked -> default text
            sim._do_in_main(sim._RayPicker.pick, 0, 0, 0, sim.Scene.renderer)
            sim._wait_for_idle()
            time.sleep(0.2)
            text2 = sim._RayText.text
            self.assertEqual(text2, default_text)  # shows default text
          
            # we have an extra picker sim._SpacePicker for right+clicking in the scene,
            # but I don't know how to make the Picker.pick() function pick with a right click
            # so currently we overide the RayPicker with the onSpacePick method
            # do via command string, so it is guaranteed to run in main thread
            sim._set_in_main("_Cmd", "self._RayPicker = self.Scene.mlab.gcf().on_mouse_pick("
                                     "self._on_space_pick, button='Left')")
            sim._do_in_main(sim.send_cmd)
            sim._wait_for_idle()

            # space picked -> show coordinates
            keyboard.release(Key.shift)
            sim._do_in_main(sim._RayPicker.pick, sim._SceneSize[0] / 3, sim._SceneSize[1] / 3, 0, sim.Scene.renderer)
            sim._wait_for_idle()
            time.sleep(0.2)
            text3 = sim._RayText.text
            self.assertNotEqual(text3, default_text)  # not the default text
            self.assertNotEqual(text3, text2)  # not the old text
            
            # valid space picked with shift -> move detector
            keyboard.press(Key.shift)
            old_pos = RT.DetectorList[0].pos
            sim._do_in_main(sim._RayPicker.pick, sim._SceneSize[0] / 3, sim._SceneSize[1] / 3, 0, sim.Scene.renderer)
            sim._wait_for_idle()
            time.sleep(0.2)
            text4 = sim._RayText.text
            self.assertEqual(text4, default_text)  # not the default text
            self.assertNotEqual(RT.DetectorList[0].pos[2], old_pos[2])
            
            # space outside outline picked with shift -> move detector
            sim._do_in_main(sim.Scene.y_plus_view)
            sim._do_in_main(sim._RayPicker.pick, 0, 0, 0, sim.Scene.renderer)
            sim._wait_for_idle()
            time.sleep(0.2)
            text4 = sim._RayText.text
            self.assertEqual(text4, default_text)  # not the default text
            self.assertEqual(RT.DetectorList[0].pos[2], RT.outline[4])  # detector moved to the outline beginning
            keyboard.release(Key.shift)
            
            sim._do_in_main(sim.close)

        sim = TraceGUI(RT)
        sim.debug(_func=interact, silent=True, _args=(sim,))


if __name__ == '__main__':
    unittest.main()
