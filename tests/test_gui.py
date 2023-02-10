#!/bin/env python3

import sys
sys.path.append('.')
import os

import copy
import unittest
import warnings
import time
import numpy as np
from contextlib import contextmanager  # context manager for _no_trait_action()
import pytest

import matplotlib.pyplot as plt
import pyautogui
from pyface.qt import QtGui

import optrace as ot
from optrace.gui import TraceGUI


def rt_example() -> ot.Raytracer:
    
    # this geometry has multiple detector, sources, lenses
    # as well as a filter, aperture and a marker and regions with different ambient n

    # make raytracer
    RT = ot.Raytracer(outline=[-5, 5, -5, 5, -5, 60], silent=True)

    # add Raysource
    RSS = ot.CircularSurface(r=1)
    RS = ot.RaySource(RSS, divergence="None", spectrum=ot.presets.light_spectrum.FDC,
                      pos=[0, 0, 0], s=[0, 0, 1], polarization="y")
    RT.add(RS)

    RSS2 = ot.CircularSurface(r=1)
    RS2 = ot.RaySource(RSS2, divergence="None", s=[0, 0, 1], spectrum=ot.presets.light_spectrum.d65,
                       pos=[0, 1, -3], polarization="Constant", pol_angle=25, power=2)
    RT.add(RS2)

    front = ot.CircularSurface(r=3)
    back = ot.CircularSurface(r=3)
    nL2 = ot.RefractionIndex("Constant", n=1.8)
    L1 = ot.Lens(front, back, de=0.1, pos=[0, 0, 2], n=nL2)
    RT.add(L1)

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

    # add marker
    RT.add(ot.PointMarker("sdghj", [0, 1, 5]))
    RT.add(ot.LineMarker(r=2, angle=5, desc="sdghj", pos=[0, 1, 5]))
    
    # add Lens 3
    front = ot.SphericalSurface(r=1, R=2.2)
    back = ot.SphericalSurface(r=1, R=-5)
    nL3 = ot.RefractionIndex("Function", func=lambda l: 1.8 - 0.007*(l - 380)/400)
    nL32 = ot.RefractionIndex("Constant", n=1.1)
    L3 = ot.Lens(front, back, de=0.1, pos=[0, 0, 47], n=nL3, n2=nL32)
    RT.add(L3)

    # # add Aperture2
    ap = ot.CircularSurface(r=1)

    def func(l):
        return np.exp(-0.5*(l-460)**2/20**2)

    fspec = ot.TransmissionSpectrum("Function", func=func)
    RT.add(ot.Filter(ap, pos=[0, 0, 45.2], spectrum=fspec))

    # add Detector
    Det = ot.Detector(ot.RectangularSurface(dim=[2, 2]), pos=[0, 0, 0])
    RT.add(Det)

    Det2 = ot.Detector(ot.SphericalSurface(R=-1.1, r=1), pos=[0, 0, 40])
    RT.add(Det2)

    # add ideal lens
    RT.add(ot.IdealLens(r=3, D=1, pos=[0, 0, RT.outline[5]-1]))

    # add some volume
    RT.add(ot.BoxVolume(dim=[3, 2], length=1, pos=[0, 0, 9]))

    return RT

class GUITests(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        self.exc_info = False

        # set seed to PYTHONHASHSEED, so the results are reproducible
        if "PYTHONHASHSEED" in os.environ:
            np.random.seed(int(os.environ["PYTHONHASHSEED"]))

        super().__init__(*args, **kwargs)

    def setUp(self) -> None:
        warnings.simplefilter("ignore")

    def tearDown(self) -> None:
        warnings.simplefilter("default")
    
    def raise_thread_exceptions(self):
        """raise saved exception from thread"""
        if self.exc_info:  # raise unhandled thread exception
            raise self.exc_info[0](self.exc_info[1]).with_traceback(self.exc_info[2])
        self.exc_info = False

    @contextmanager
    def _try(self, sim, *args, **kwargs):
        """try TraceGUI actions. Exceptions are catched and saved in the class to be later
        raised by raise_thread_exceptions(). In all cases the gui is closed normally"""
        time.sleep(2)
        sim._wait_for_idle()
        try:
            yield
        except Exception as e:
            self.exc_info = sys.exc_info()  # assign thread exception
        finally:
            sim._wait_for_idle()
            sim._do_in_main(sim.close)
            time.sleep(2)

    @pytest.mark.slow 
    def test_gui_inits(self) -> None:

        Image = ot.presets.image.test_screen

        # make raytracer
        RT = ot.Raytracer(outline=[-5, 5, -5, 5, 0, 40], silent=True)

        # add Raysource
        RSS = ot.RectangularSurface(dim=[4, 4])
        RS = ot.RaySource(RSS, divergence="Isotropic", div_angle=8,
                          image=Image, s=[0, 0, 1], pos=[0, 0, 0])
        RT.add(RS)

        # add Lens 1
        front = ot.SphericalSurface(r=3, R=8)
        back = ot.SphericalSurface(r=3, R=-8)
        nL1 = ot.RefractionIndex("Constant", n=1.5)
        L1 = ot.Lens(front, back, de=0.1, pos=[0, 0, 12], n=nL1)
        RT.add(L1)

        # add Detector
        DetS = ot.RectangularSurface(dim=[10, 10])
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

        for ctype in TraceGUI.coloring_types:
            trace_gui_run(coloring_type=ctype)

        for ptype in TraceGUI.plotting_types:
            trace_gui_run(plotting_type=ptype)

        trace_gui_run(ray_count=100000)
        trace_gui_run(rays_visible=200)
        trace_gui_run(absorb_missing=False)
        trace_gui_run(minimalistic_view=True)
        trace_gui_run(high_contrast=True)
        trace_gui_run(raytracer_single_thread=True)
        trace_gui_run(ray_opacity=0.15)
        trace_gui_run(ray_width=5)
        trace_gui_run(vertical_labels=True)
        trace_gui_run(ui_style="Windows")
        trace_gui_run(ui_style="Fusion")

        # many parameters
        trace_gui_run(ray_count=100000, rays_visible=1500, absorb_missing=False, ray_opacity=0.1, ray_width=5,
                      coloring_type="Wavelength", plotting_type="Points", minimalistic_view=True, 
                      raytracer_single_thread=True, vertical_labels=True, ui_style="Windows")

        # this attributes can't bet set initially
        self.assertRaises(RuntimeError, TraceGUI, RT, detector_selection="DET1")
        self.assertRaises(RuntimeError, TraceGUI, RT, source_selection="RS1")
        self.assertRaises(RuntimeError, TraceGUI, RT, z_det=0)

    @pytest.mark.slow
    def test_interaction1(self) -> None:

        def interact(sim):
            with self._try(sim):

                # check moving of detectors and selection and name updates
                for i, z0 in zip(range(len(RT.detectors)), [5.3, 27.3]):
                    # change detector
                    dname0 = sim.detector_selection
                    sim._set_in_main("detector_selection", sim.detector_names[i])
                    sim._wait_for_idle()

                    dname1 = sim.detector_names[i]
                    if i:  # check if detector name changed from initial
                        self.assertNotEqual(dname0, dname1)
                    
                    assert z0 != RT.detectors[i].pos[2]
                    self.assertEqual(sim.z_det, RT.detectors[i].pos[2])
                    # position updated after changing detector
                   
                    # change detector position
                    sim._set_in_main("z_det", z0)
                    sim._wait_for_idle()
                    dname2 = sim.detector_selection
                    self.assertEqual(sim.z_det, RT.detectors[i].pos[2])  # position updated
                    self.assertNotEqual(dname1, sim.detector_names[i])  # name updated after position change
                    self.assertEqual(dname2, sim.detector_names[i])  # name updated after position change
                    
                # Source Image Tests
                sim._do_in_main(sim.show_source_image)
                sim._wait_for_idle()
                sim._set_in_main("source_selection", sim.source_names[1])
                sim._wait_for_idle()
                sim._do_in_main(sim.show_source_image)
                sim._wait_for_idle()
                
                # Detector Image Tests
                sim._do_in_main(sim.show_detector_image)
                sim._wait_for_idle()
                sim._set_in_main("detector_selection", sim.detector_names[1])
                sim._wait_for_idle()
                self.assertTrue(sim._det_ind == 1)
                self.assertTrue(sim.z_det == sim.raytracer.detectors[1].pos[2])
                sim._do_in_main(sim.show_detector_image)
                sim._wait_for_idle()

                # Image Type Tests standard
                for mode in ot.RImage.display_modes:
                    sim._set_in_main("image_type", mode)
                    sim._wait_for_idle()
                    sim._do_in_main(sim.show_detector_image)
                    sim._wait_for_idle()
                plt.close('all')

                # Image Tests Higher Res
                sim._set_in_main("image_pixels", 315)
                sim._wait_for_idle()
                sim._do_in_main(sim.show_detector_image)
                sim._wait_for_idle()

                sim._set_in_main("maximize_scene", True)
                sim._wait_for_idle()

                # Image Test Source, but actually we should test all parameter combinations,
                sim._do_in_main(sim.show_source_image)
                sim._wait_for_idle()

                # Focus Tests
                pos0 = sim.raytracer.detectors[1].pos
                sim._set_in_main("detector_selection", sim.detector_names[1])
                sim._wait_for_idle()

                for mode in ot.Raytracer.autofocus_methods:
                    sim._set_in_main("focus_type", mode)
                    sim._wait_for_idle()
                    sim._do_in_main(sim.move_to_focus)
                    sim._wait_for_idle()
                    sim._set_in_main("z_det", pos0[2])
                    sim._wait_for_idle()

                # Focus Test 4, show Debug Plot
                sim._set_in_main("focus_cost_plot", True)
                sim._wait_for_idle()
                sim._do_in_main(sim.move_to_focus)
                sim._wait_for_idle()

                # Focus Test 5, one source only
                sim._set_in_main("focus_cost_plot", False)
                sim._wait_for_idle()
                sim._set_in_main("z_det", pos0[2])
                sim._wait_for_idle()
                sim._set_in_main("af_one_source", True)
                sim._do_in_main(sim.move_to_focus)
                sim._wait_for_idle()
                
                # Ray Coloring Tests
                for type_ in sim.coloring_types:
                    sim._set_in_main("coloring_type", type_)
                    sim._wait_for_idle()

                # plotting_type Tests
                for type_ in sim.plotting_types:
                    sim._set_in_main("plotting_type", type_)
                    sim._wait_for_idle()
              
                # absorb_missing test
                sim._set_in_main("absorb_missing", False)
                sim._wait_for_idle()

                # retrace Tests
                sim._set_in_main("ray_count", 100000)
                sim._wait_for_idle()
                
                sim._set_in_main("rays_visible", 50)
                sim._wait_for_idle()

                sim._set_in_main("rays_visible", 500)

        RT = rt_example()
        sim = TraceGUI(RT)
        sim.debug(_func=interact, silent=True, _args=(sim,))
        plt.close('all')
        self.raise_thread_exceptions()

    def test_interaction2(self) -> None:

        def interact2(sim):
            with self._try(sim):
                # Image Type Tests log scaling
                sim._set_in_main("log_image", True)

                # display all image modes with log
                for mode in ot.RImage.display_modes:
                    sim._set_in_main("image_type", mode)
                    sim._wait_for_idle()
                    sim._do_in_main(sim.show_detector_image)
                    sim._wait_for_idle()
                plt.close('all')

                # images with filter
                sim._set_in_main("filter_constant", 10)
                sim._set_in_main("activate_filter", True)
                sim._do_in_main(sim.show_detector_image)
                sim._wait_for_idle()
                sim._do_in_main(sim.show_source_image)
                sim._wait_for_idle()

                # Image Tests Flip
                sim._set_in_main("log_image", False)
                sim._wait_for_idle()
                sim._set_in_main("flip_image", True)
                sim._wait_for_idle()
                sim._do_in_main(sim.show_detector_image)
                sim._wait_for_idle()

                # one source only
                sim._set_in_main("det_image_one_source", True)
                sim._wait_for_idle()
                sim._do_in_main(sim.show_detector_image)
                sim._wait_for_idle()

                # make labels vertical
                sim._set_in_main("vertical_labels", True)
                sim._wait_for_idle()
                sim._set_in_main("vertical_labels", False)
                sim._wait_for_idle()

                # high contrast mode
                sim._set_in_main("high_contrast", True)
                sim._wait_for_idle()
                sim._set_in_main("high_contrast", False)
                sim._wait_for_idle()

                # test different projection methods
                sim._set_in_main("detector_selection", sim.detector_names[1])
                sim._wait_for_idle()
                plt.close('all')
                for pm in ot.SphericalSurface.sphere_projection_methods:
                    sim._set_in_main("projection_method", pm)
                    sim._do_in_main(sim.show_detector_image)
                    sim._wait_for_idle()
                    sim._do_in_main(sim.show_detector_cut)
                    sim._wait_for_idle()
                    self.assertEqual(sim.last_det_image.projection, pm)
                
                # open browsers
                sim._do_in_main(sim.open_property_browser)
                sim._wait_for_idle()
                time.sleep(2)  # wait some time so windows can be seen by a human user

        RT = rt_example()
        sim = TraceGUI(RT)
        sim.debug(_func=interact2, silent=True, _args=(sim,))
        plt.close('all')
        self.raise_thread_exceptions()

    def test_interaction3(self) -> None:

        def interact3(sim):
            with self._try(sim):

                # test image cuts
                sim._do_in_main(sim.show_source_cut)
                sim._wait_for_idle()
                sim._set_in_main("det_image_one_source", True)
                sim._wait_for_idle()
                sim._do_in_main(sim.show_detector_cut)
                sim._wait_for_idle()
                sim._set_in_main("cut_dimension", "x")
                sim._do_in_main(sim.show_detector_cut)
                sim._wait_for_idle()
                sim._set_in_main("cut_dimension", "y")
                sim._do_in_main(sim.show_detector_cut)
                sim._wait_for_idle()
                sim._set_in_main("cut_value", 0)  # exists inside image
                sim._do_in_main(sim.show_detector_cut)
                sim._wait_for_idle()
                sim._set_in_main("cut_value", 100)  # does not exist inside image
                sim._do_in_main(sim.show_detector_cut)
                sim._wait_for_idle()
                plt.close('all')

                # tracing with only one thread
                sim._set_in_main("raytracer_single_thread", True)
                sim._wait_for_idle()
                self.assertEqual(sim.raytracer.threading, False)
                sim._set_in_main("raytracer_single_thread", False)
                sim._wait_for_idle()
                self.assertEqual(sim.raytracer.threading, True)
                sim._wait_for_idle()
               
                sim._set_in_main("wireframe_surfaces", True)
                sim._wait_for_idle()
                sim._set_in_main("wireframe_surfaces", False)
                sim._wait_for_idle()
                
                # activate/deactivate all warning. How to check this?
                sim._set_in_main("show_all_warnings", True)
                sim._wait_for_idle()
                sim._set_in_main("show_all_warnings", False)
                sim._wait_for_idle()

                # activate/deactivate all warning. How to check this?
                sim._set_in_main("garbage_collector_stats", True)
                sim._wait_for_idle()
                sim._set_in_main("garbage_collector_stats", False)
                sim._wait_for_idle()
                
        RT = rt_example()
        sim = TraceGUI(RT)
        sim.debug(_func=interact3, silent=True, _args=(sim,))
        plt.close('all')
        self.raise_thread_exceptions()

    @pytest.mark.os
    def test_interaction4(self) -> None:

        def interact4(sim):
            with self._try(sim):
                # test source and detector spectrum
                sim._do_in_main(sim.show_source_spectrum)
                sim._wait_for_idle()
                sim._do_in_main(sim.show_detector_spectrum)
                sim._wait_for_idle()
                sim._set_in_main("det_spectrum_one_source", True)
                sim._do_in_main(sim.show_detector_spectrum)
                sim._wait_for_idle()

                # open property browser
                sim._do_in_main(sim.open_property_browser)
                time.sleep(2)
                sim._wait_for_idle()
                
        RT = rt_example()
        sim = TraceGUI(RT)
        sim.debug(_func=interact4, silent=True, _args=(sim,))
        self.raise_thread_exceptions()

    @pytest.mark.slow
    def test_missing(self) -> None:
        """test TraceGUI operation when Filter, Lenses, Detectors or Sources are missing"""

        def test_features(RT):
            sim = TraceGUI(RT)

            def interact(sim):
                with self._try(sim):
                    sim._do_in_main(sim.show_detector_image)
                    sim._wait_for_idle()
                    sim._do_in_main(sim.move_to_focus)
                    sim._wait_for_idle()
                    sim._set_in_main("z_det", 10.)
                    sim._wait_for_idle()
                    sim._do_in_main(sim.show_source_image)
                    sim._wait_for_idle()
                    sim._set_in_main("ray_opacity", 0.07)
                    sim._wait_for_idle()
                    sim._set_in_main("ray_width", 5)
                    sim._wait_for_idle()
                    sim._set_in_main("ray_count", 100000)
                    sim._wait_for_idle()
                    sim._set_in_main("absorb_missing", False)
                    sim._wait_for_idle()
                    sim._set_in_main("rays_visible", 4000)
                    sim._wait_for_idle()
                    sim._set_in_main("minimalistic_view", True)
                    sim._wait_for_idle()
                    sim._set_in_main("coloring_type", "Power")
                    sim._wait_for_idle()
                    sim._set_in_main("plotting_type", "Points")

                    sim._set_in_main("high_contrast", True)
                    sim._wait_for_idle()
                    
                    sim._do_in_main(sim._change_selected_ray_source)
                    sim._wait_for_idle()
                    sim._do_in_main(sim._change_detector)
                    sim._wait_for_idle()
                    
                    sim._do_in_main(sim.show_source_spectrum)
                    sim._wait_for_idle()
                    sim._do_in_main(sim.show_detector_spectrum)
                    sim._wait_for_idle()

                    sim._do_in_main(sim.replot)

            sim.debug(_func=interact, silent=True, _args=(sim,))
            self.raise_thread_exceptions()
            time.sleep(1)

        RT = rt_example()

        [RT.remove(F) for F in RT.filters.copy()]
        self.assertTrue(not RT.filters)
        test_features(RT)

        [RT.remove(L) for L in RT.lenses.copy()]
        RT.trace(N=RT.rays.N)
        self.assertTrue(not RT.lenses)
        test_features(RT)

        [RT.remove(D) for D in RT.detectors.copy()]
        self.assertTrue(not RT.detectors)
        test_features(RT)

        [RT.remove(RS) for RS in RT.ray_sources.copy()]
        self.assertTrue(not RT.ray_sources)
        test_features(RT)

        RT.clear()
        test_features(RT)

    @pytest.mark.slow
    @pytest.mark.skip(reason="there seems to be some issue with qt4, leads to segmentation faults.")
    def test_action_spam(self):
        """spam the gui with many possible actions to check threading locks and for race conditions"""

        RT = rt_example()
        sim = TraceGUI(RT)

        def interact(sim):
            
            with self._try(sim):
                N0 = sim.ray_count
                sim._set_in_main("ray_count", int(N0*1.3))
                sim._do_in_main(sim.show_detector_image)
                sim._do_in_main(sim.show_source_image)
                sim._do_in_main(sim.move_to_focus)

                time.sleep(0.01)
                sim._set_in_main("ray_count", int(N0/1.3))
                sim._set_in_main("z_det", (RT.outline[5] - RT.outline[4])/2)
                sim._do_in_main(sim.show_source_cut)
                sim._do_in_main(sim.move_to_focus)
                sim._set_in_main("detector_selection", sim.detector_names[1])
                sim._do_in_main(sim.show_detector_image)
                sim._do_in_main(sim.replot_rays)
                sim._do_in_main(sim.show_source_image)

                time.sleep(0.1)
                sim._do_in_main(sim.replot_rays)
                sim._do_in_main(sim.send_cmd, "GUI.replot()")
                sim._do_in_main(sim.show_detector_image)
                sim._set_in_main("detector_selection", sim.detector_names[1])

                sim._set_in_main("ray_count", 1000000)
                sim._do_in_main(sim.replot_rays)
                sim._set_in_main("detector_selection", sim.detector_names[1])
                sim._do_in_main(sim.move_to_focus)
                sim._do_in_main(sim.show_source_spectrum)
                
                sim._wait_for_idle()

                self.assertEqual(sim.ray_count, 1000000)
                self.assertEqual(sim.raytracer.rays.N, 1000000)

        sim.debug(_func=interact, silent=True, _args=(sim,))
        self.raise_thread_exceptions()

    @pytest.mark.slow
    def test_replot_combinations(self):
        """check replotting by setting random combinations inside the change dictionary for replot()"""

        RT = rt_example()
        sim = TraceGUI(RT)

        def interact(sim):
            with self._try(sim):
                # check if replot actually removes old plot elements
                L0id = id(sim._plot._lens_plots[0][0])
                F0id = id(sim._plot._filter_plots[0][0])
                sim._do_in_main(sim.replot)
                sim._wait_for_idle()
                self.assertNotEqual(L0id, id(sim._plot._lens_plots[0][0]))
                self.assertNotEqual(F0id, id(sim._plot._filter_plots[0][0]))

                sim._set_in_main("ray_count", 20000)  # fewer rays so tracing is faster
                sim._wait_for_idle()

                rtp = RT.property_snapshot()
                cmp = RT.compare_property_snapshot(rtp, rtp)

                for i in np.arange(20):
                    cmp = copy.deepcopy(cmp)
                    any_ = False
                    # random change combinations
                    for key, val in cmp.items():
                        val = bool(np.random.randint(0, 2))
                        cmp[key] = val
                        any_ = any_ | val

                    cmp["Any"] = any_

                    sim._do_in_main(sim.replot, cmp)
                    sim._wait_for_idle()

        sim.debug(_func=interact, silent=True, _args=(sim,))
        self.raise_thread_exceptions()

    @pytest.mark.slow
    @pytest.mark.skip(reason="there seems to be some issue with qt4, leads to segmentation faults.")
    def test_action_spam_random(self):
        """do random actions with random delays and check if there are race conditions or other special issues"""

        RT = rt_example()

        sim = TraceGUI(RT)

        def interact(sim):
            with self._try(sim):
                funcs = [sim.show_detector_image, sim.show_source_image, sim.show_source_spectrum, sim.move_to_focus,
                         sim.replot_rays, sim.open_property_browser, sim.open_command_window,
                         sim.show_detector_spectrum, sim.show_detector_cut, sim.show_source_cut, sim.replot]

                props = [('rays_visible', 500), ('rays_visible', 2000), ('minimalistic_view', True),
                         ('minimalistic_view', False), ('plotting_type', "Rays"), ('plotting_type', "Points"),
                         ('absorb_missing', True), ('absorb_missing', False), ('ray_opacity', 0.06), ('ray_width', 12),
                         ('ray_opacity', 0.8), ('ray_width', 2), ('cut_value', 0), ('cut_dimension', 'x'), 
                         ('cut_dimension', 'y'),
                         ('af_one_source', False), ('af_one_source', True), ('det_image_one_source', False),
                         ('det_image_one_source', True), ('cut_value', 0.1), ('flip_det_image', True), 
                         ('flip_det_image', False), ('det_spectrum_one_source', True), ('det_image_one_source', False),
                         ('log_image', False), ('log_image', True), ('raytracer_single_thread', False),
                         ('raytracer_single_thread', True), ('wireframe_surfaces', True), 
                         ('wireframe_surfaces', False), ('focus_cost_plot', True), ('focus_cost_plot', False), 
                         ('maximize_scene', False), ('maximize_scene', True), ('vertical_labels', False), 
                         ('vertical_labels', True), ('activate_filter', False), ('activate_filter', True), 
                         ('high_contrast', False), ('high_contrast', True)]

                for i in np.arange(200):
                    # the expected value (mean time for a large number) is integral n*10^n from n0 to n1
                    # n0=-4, n1=0.5 the mean time would therefore be around 90ms per loop iteration for sleeping
                    n = np.random.uniform(-4, 0.5)
                    time.sleep(10**n)

                    match np.random.randint(0, 13):
                        case 0:
                            sim._do_in_main(np.random.choice(funcs))
                        case 1 | 11:
                            ch = np.random.randint(0, len(props))
                            sim._set_in_main(*props[ch])
                        case 2:
                            sim._set_in_main("coloring_type", np.random.choice(sim.coloring_types))
                        case 3:
                            # only every second time, otherwise this would happen to often
                            if np.random.randint(0, 2):
                                sim._set_in_main("ray_count", np.random.randint(10000, 500000))
                        case 4:
                            sim._set_in_main("focus_type", np.random.choice(RT.autofocus_methods))
                        case 5:
                            sim._set_in_main("image_type", np.random.choice(ot.RImage.display_modes))
                        case 6:
                            sim._set_in_main("detector_selection", np.random.choice(sim.detector_names))
                        case 7:
                            sim._set_in_main("source_selection", np.random.choice(sim.source_names))
                        case 8:
                            sim._set_in_main("z_det", np.random.uniform(RT.outline[4], RT.outline[5]))
                        case 9:
                            sim._set_in_main("image_pixels", np.random.choice(ot.RImage.SIZES))
                        case 10:
                            cmds = ['GUI.replot()', 'scene.render()', 'scene.z_minus_view()']
                            cmd = np.random.choice(cmds)
                            sim._do_in_main(sim.send_cmd, cmd)
                        case 12:
                            if isinstance(sim.raytracer.detectors[sim._det_ind].front, ot.SphericalSurface):
                                sim._set_in_main("projection_method", 
                                                 np.random.choice(ot.SphericalSurface.sphere_projection_methods))

                    # close plots from time to time
                    if i % 20 == 0:
                        plt.close('all')
        
        sim.debug(_func=interact, silent=True, _args=(sim,))
        self.raise_thread_exceptions()
   
    def test_key_presses(self):
        """test keyboard shortcuts inside the scene while simulating key presses"""

        RT = rt_example()
        sim = TraceGUI(RT)

        def send_key(sim, key):
            sim._do_in_main(sim.scene.scene_editor._content.setFocus)
            pyautogui.press(key)
            time.sleep(0.2)

        def interact(sim):
            with self._try(sim):
                # check minimalistic_view shortcut
                self.assertTrue(len(sim.minimalistic_view) == 0)
                send_key(sim, "v")
                sim._wait_for_idle()
                self.assertTrue(len(sim.minimalistic_view) != 0)
                
                # check high_contrast shortcut
                self.assertTrue(len(sim.high_contrast) == 0)
                send_key(sim, "c")
                sim._wait_for_idle()
                self.assertTrue(len(sim.high_contrast) != 0)
                
                # check Ray-Point toggle shortcut
                self.assertTrue(sim.plotting_type == "Rays")
                send_key(sim, "r")
                sim._wait_for_idle()
                self.assertTrue(sim.plotting_type == "Points")
                
                # y plus view
                sim._do_in_main(sim.scene.mlab.view, 0, 1, 0)
                sim._wait_for_idle()
                cv = sim.scene.camera.position
                send_key(sim, "y")
                sim._wait_for_idle()
                cv2 = sim.scene.camera.position
                self.assertFalse(np.allclose(cv, cv2))
                
                # maximize scene / hide other menus
                self.assertTrue(sim.scene.scene_editor._tool_bar.isVisible())
                self.assertTrue(sim._scene_not_maximized)
                send_key(sim, "h")
                sim._wait_for_idle()
                self.assertFalse(sim.scene.scene_editor._tool_bar.isVisible())
                self.assertFalse(sim._scene_not_maximized)

                # don't press m warning
                sim.silent = False
                send_key(sim, "m")
                sim._wait_for_idle()
                sim.silent = True
                send_key(sim, "m")
                sim._wait_for_idle()
                
                # don't press a warning
                sim.silent = False
                send_key(sim, "a")
                sim._wait_for_idle()
                sim.silent = True
                send_key(sim, "a")
                sim._wait_for_idle()
               
                # key without shortcut
                send_key(sim, "i")
                sim._wait_for_idle()
                
                # replot rays
                id0 = id(sim._plot._ray_property_dict["p"])
                send_key(sim, "n")
                sim._wait_for_idle()
                self.assertNotEqual(id0, id(sim._plot._ray_property_dict["p"]))  # array changed, therefore the id too
              
                # do this one last, since it raises another window
                # and I don't know how to focus the scene after this
                # check DetectorImage shortcut
                self.assertTrue(sim.last_det_image is None)
                send_key(sim, "d")
                sim._wait_for_idle()
                self.assertTrue(sim.last_det_image is not None)

        sim.debug(_func=interact, silent=True, _args=(sim,))
        self.raise_thread_exceptions()
   
    # os test because clipboard is system dependent
    @pytest.mark.os
    def test_send_cmd(self):
        """test command setting and sending as well as automatic replotting"""

        RT = rt_example()
        sim = TraceGUI(RT, silent=True)

        def send(cmd):
            sim._do_in_main(sim.send_cmd, cmd)
            sim._wait_for_idle()

        def interact(sim):
            with self._try(sim):
                state = RT.rays.crepr()
                send("GUI.replot()")
                self.assertFalse(state == RT.rays.crepr())  # check if raytraced

                send("GUI.show_detector_image()")
                self.assertTrue(sim.last_det_image is not None)  # check if raytraced
               
                # create optrace objects
                LLlen = len(RT.lenses)
                send("a = CircularSurface(r=2);"
                     "b = SphericalSurface(r=2.5, R=-10);"
                     "L = Lens(a, b, n=presets.refraction_index.SF10, de=0.2, pos=[0, 0, 8]);"
                     "RT.add(L)")
                self.assertEqual(len(RT.lenses), LLlen + 1)

                # numpy and time available
                send("a = np.array([1, 2, 3]);"
                     "time.time()")

                state = RT.rays.crepr()
                send("RT.remove(APL[0])")
                self.assertEqual(len(RT.apertures), 0)  # check if apertures empty after removal
                self.assertEqual(len(sim._plot._aperture_plots), 0)  # check if aperture plot is removed
                self.assertFalse(state == RT.rays.crepr())  # check if raytraced
              
                sim._do_in_main(sim.open_command_window)
                
                send("")  # check empty command

                send("throw RuntimeError()")  # check if exceptions are handled

                # this time with output
                sim.silent = False
                send("throw RuntimeError()")  # check if exceptions are handled

                # send empty command using command window
                sim._cdb._cmd = ""
                sim._do_in_main(sim._cdb.send_cmd)
                sim._wait_for_idle()

                # send actual command using command window
                sim._cdb._cmd = "self.replot()"
                sim._do_in_main(sim._cdb.send_cmd)
                sim._wait_for_idle()

                # resend command, history does not get updated with same command
                sim._do_in_main(sim._cdb.send_cmd)
                sim._wait_for_idle()
               
                # clear history
                sim._do_in_main(sim._cdb.clear_history)
                sim._wait_for_idle()

                # check if setting and getting clipboard works globally
                clipboard = QtGui.QApplication.clipboard()
                clipboard.clear(mode=clipboard.Clipboard)
                clipboard.setText("a", mode=clipboard.Clipboard)
                can_copy = clipboard.text(mode=clipboard.Clipboard) == "a"

                if not can_copy:
                    warnings.warn("Copying to clipboard failed. Can by a system or library issue, "
                                  "skipping additional clipboard tests.")
                
                else:
                    # check that empty strings are copied correctly in the command window
                    sim._do_in_main(sim._cdb.copy_history)
                    sim._wait_for_idle()
                    self.assertEqual(clipboard.text(), "")
                   
                    # check if full history is copied
                    sim._cdb._cmd = "self.replot()"
                    sim._do_in_main(sim._cdb.send_cmd)
                    sim._wait_for_idle()
                    sim._cdb._cmd = "a=5"
                    sim._do_in_main(sim._cdb.send_cmd)
                    sim._wait_for_idle()
                    sim._do_in_main(sim._cdb.copy_history)
                    sim._wait_for_idle()
                    self.assertEqual(clipboard.text(), "self.replot()\na=5\n")

        sim.debug(_func=interact, silent=True, _args=(sim,))
        self.raise_thread_exceptions()

    def test_resize(self):
        """
        this test checks if
        * some UI properties get resized correctly
        * no exceptions while doing so
        * resizing to initial state resizes everything back to normal
        """

        RT = rt_example()
        sim = TraceGUI(RT)

        def interact(sim):
            with self._try(sim):

                sim._set_in_main("coloring_type", "Power")  # shows a side bar, that also needs to be rescaled
                sim._wait_for_idle()

                SceneSize0 = sim._plot._scene_size.copy()
                Window = sim.scene.scene_editor._content.window()

                # properties before resizing
                ff = sim._plot._axis_plots[0][0].axes.font_factor
                zoom = sim._plot._orientation_axes.widgets[0].zoom
                pos2 = sim._plot._ray_plot.parent.scalar_lut_manager.scalar_bar_representation.position2

                qsize = Window.size()
                ss0 = np.array([qsize.width(), qsize.height()])
                ss1 = ss0 * 1.3
                ss2 = ss1 / 1.2

                # enlarge
                sim._do_in_main(Window.resize, *ss1.astype(int))
                time.sleep(0.5)  # how to check how much time it takes?

                # check if scale properties changed
                self.assertNotAlmostEqual(sim._plot._scene_size[0], SceneSize0[0])  # scene size variable changed
                self.assertNotAlmostEqual(sim._plot._scene_size[1], SceneSize0[1])  # scene size variable changed
                self.assertNotAlmostEqual(ff, sim._plot._axis_plots[0][0].axes.font_factor)
                self.assertNotAlmostEqual(zoom, sim._plot._orientation_axes.widgets[0].zoom)
                self.assertNotAlmostEqual(pos2[0],
                                          sim._plot._ray_plot.parent.scalar_lut_manager.scalar_bar_representation.position2[0])
                self.assertNotAlmostEqual(pos2[1],
                                          sim._plot._ray_plot.parent.scalar_lut_manager.scalar_bar_representation.position2[1])

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
                self.assertAlmostEqual(sim._plot._scene_size[0], SceneSize0[0])
                self.assertAlmostEqual(sim._plot._scene_size[1], SceneSize0[1])
                self.assertAlmostEqual(ff, sim._plot._axis_plots[0][0].axes.font_factor)
                self.assertAlmostEqual(zoom, sim._plot._orientation_axes.widgets[0].zoom)
                self.assertAlmostEqual(pos2[0],
                                       sim._plot._ray_plot.parent.scalar_lut_manager.scalar_bar_representation.position2[0])
                self.assertAlmostEqual(pos2[1],
                                       sim._plot._ray_plot.parent.scalar_lut_manager.scalar_bar_representation.position2[1])


                # coverage test: delete orientation:axes and resize
                sim._plot._orientation_axes = None
                sim._do_in_main(Window.resize, *ss2.astype(int))
                time.sleep(0.5)
                
        sim.debug(_func=interact, silent=True, _args=(sim,))
        self.raise_thread_exceptions()
        
    def test_non_2d(self):
        """
        initially there were problems with non-plotable "surfaces" like Line and Point
        check if initial plotting, replotting and trying to color the parent sources (coloring_type=...)
        works correctly
        """
       
       # make raytracer
        RT = ot.Raytracer(outline=[-5, 5, -5, 5, -5, 60], silent=True)

        # add Raysource
        RSS = ot.Point()
        RS = ot.RaySource(RSS, divergence="Isotropic", spectrum=ot.presets.light_spectrum.d65,
                          pos=[0, 0, 0], s=[0, 0, 1], div_angle=75)
        RT.add(RS)
        
        # add Raysource2
        RSS = ot.Line()
        RS2 = ot.RaySource(RSS, divergence="None", spectrum=ot.presets.light_spectrum.d65,
                           pos=[0, 0, 0], s=[0, 0, 1])
        RT.add(RS2)

        sim = TraceGUI(RT, ColoringType="Wavelength")

        def interact(sim):
            with self._try(sim):
                sim._set_in_main("coloring_type", "Wavelength")
                sim._wait_for_idle()
                sim._set_in_main("plotting_type", "Points")
                sim._wait_for_idle()
                sim._do_in_main(sim.replot)

        sim.debug(_func=interact, silent=True, _args=(sim,))
        self.raise_thread_exceptions()

    @pytest.mark.slow
    def test_additional_coverage_1(self):
        """additionial coverage tests"""

        RT = rt_example()
        sim = TraceGUI(RT)

        def interact(sim):
            with self._try(sim):
                # check handling of cut values outside the image
                sim._set_in_main("cut_value", 500)
                sim._do_in_main(sim.show_source_cut)
                sim._wait_for_idle()
                sim._do_in_main(sim.show_source_cut)
                sim._wait_for_idle()

                # same with output messages
                sim.silent = False
                sim._do_in_main(sim.show_source_cut)
                sim._wait_for_idle()
                sim._do_in_main(sim.show_detector_cut)
                sim._wait_for_idle()

                # refraction index box replotting with vertical labels
                sim._set_in_main("vertical_labels", True)
                sim._do_in_main(sim.replot)
                sim._wait_for_idle()

                # mock no_pol mode, check if coloring type handles this
                RT.no_pol = True
                sim._set_in_main("coloring_type", "Polarization yz")
                sim._wait_for_idle()

                # same but without messages
                sim.silent = True
                sim._set_in_main("coloring_type", "Polarization yz")
                sim._wait_for_idle()
             
                # make sure x polarization gets plotted at least once
                RT.no_pol = False
                RT.ray_sources[0].polarization = "x"
                sim._set_in_main("coloring_type", "Polarization yz")
                sim._wait_for_idle()
                sim._do_in_main(sim.retrace)
                sim._wait_for_idle()

                # skip on InitScene
                sim._status["InitScene"] = 1
                sim._do_in_main(sim._change_minimalistic_view)
                time.sleep(2)  # make sure it gets executed, can't call wait_for_idle since we set InitScene
                sim._status["InitScene"] = 0
                sim._wait_for_idle()

                # check if rays are removed correctly in replot()
                assert len(RT.ray_sources) and RT.rays.N and sim._plot._ray_plot is not None
                sim._do_in_main(sim.send_cmd, "RT.clear()")
                sim._wait_for_idle()
                self.assertTrue(sim._plot._ray_plot is None)
                self.assertFalse(sim._plot._ray_property_dict)

        sim.debug(_func=interact, silent=True, _args=(sim,))
        self.raise_thread_exceptions()
    
    def test_additional_coverage_2(self):
        """additionial coverage tests"""

        RT = rt_example()
        sim = TraceGUI(RT)

        def interact(sim):
            with self._try(sim):

                # delete one raysource plot but still try to color it
                # do with silent = True and False
                sim._plot._ray_source_plots = sim._plot._ray_source_plots[:-1]
                sim._do_in_main(sim.retrace)
                sim._wait_for_idle()
                sim.silent = False
                sim._do_in_main(sim.retrace)
                sim._wait_for_idle()
                sim.silent = True

                # only time replotting the orientation axes, since normally they don't need to change
                sim._do_in_main(sim._plot.plot_orientation_axes)
                sim._wait_for_idle()

                # assign some weird outline so plot axes finds no optimal number of labels 
                RT.outline = [-123., 10*np.pi, -12, 12.468786, -500.46504654065/np.pi, 124.456]
                sim._do_in_main(sim.replot)
                sim._wait_for_idle()

                # open property browser with cardinal points, although we now don't have rotational symmetry
                RT.lenses[0].move_to([0, 0.02, 2])
                sim._do_in_main(sim.open_property_browser)
                sim._wait_for_idle()

                # special case where we don't need to retrace or redraw the rays
                h = RT.property_snapshot()
                cmp = RT.compare_property_snapshot(h, h)
                cmp["Any"] = True
                cmp["Detectors"] = True
                sim._do_in_main(sim.replot, cmp)
                sim._wait_for_idle()
              
                sim._plot._filter_plots[0] = (sim._plot._filter_plots[0][0], None, None, None,
                                        sim._plot._filter_plots[0][4])
                sim._plot._aperture_plots[0] = (sim._plot._aperture_plots[0][0], None, None, None, None)
                sim._wait_for_idle()

                # check exception handling of raytracer actions
                # for this we assign wrong detector and source indices
                # so the actions throw
                # however this should not interrupt the program or lead to states
                # where status flags are set incorrectly
                sim._det_ind = 50
                sim._source_ind = 50
                sim._set_in_main("af_one_source", True)
                sim._set_in_main("det_image_one_source", True)
                sim._do_in_main(sim.show_detector_image)
                sim._wait_for_idle()
                sim._do_in_main(sim.show_detector_cut)
                sim._wait_for_idle()
                sim._do_in_main(sim.show_detector_spectrum)
                sim._wait_for_idle()
                sim._do_in_main(sim.show_source_image)
                sim._wait_for_idle()
                sim._do_in_main(sim.show_source_cut)
                sim._wait_for_idle()
                sim._do_in_main(sim.show_source_spectrum)
                sim._wait_for_idle()
                sim._do_in_main(sim.move_to_focus)
                sim._wait_for_idle()
                sim._det_ind = 0
                sim._source_ind = 0

                # command debugging options
                sim._set_in_main("command_dont_skip", True)
                sim._wait_for_idle()
                sim._set_in_main("command_dont_skip", False)
                sim._wait_for_idle()
                sim._set_in_main("command_dont_replot", True)
                sim._wait_for_idle()
                sim._do_in_main(sim.send_cmd, "RT.remove(APL[0])")
                sim._wait_for_idle()
                self.assertEqual(len(sim._plot._aperture_plots), 1)  # plot object still there, not replotted
                sim._set_in_main("command_dont_replot", False)
                sim._wait_for_idle()

                # check waiting timeout, while InitScene is set GUI.busy is always true
                # timeout ensures we return back from waiting
                sim._status["InitScene"] = 1
                start = time.time()
                sim._wait_for_idle(timeout=5)
                self.assertTrue(time.time() - start < 6)

                # running the action is skipped because a different action (InitScene) is running
                sim._set_in_main("silent", True)
                sim._do_in_main(sim.send_cmd, "self.replot()")
                sim._wait_for_idle(timeout=2)
                sim._set_in_main("silent", False)
                sim._do_in_main(sim.send_cmd, "self.replot()")
                sim._wait_for_idle(timeout=2)

                sim._status["InitScene"] = 0
                sim._wait_for_idle()

                # remove old ray plot when an error tracing occurred
                RT.lenses[0].move_to(RT.lenses[1].pos)
                sim._do_in_main(sim.replot)
                sim._wait_for_idle()
                self.assertTrue(sim._plot._ray_plot is None)
                
                # still works when some objects are none
                sim._plot._orientation_axes = None
                tn4 = (None, None, None, None)
                sim._plot._index_box_plots[0] = tn4
                sim._plot._ray_source_plots[0] = tn4
                sim._plot._axis_plots[0] = tn4
                sim._do_in_main(sim._change_minimalistic_view)
                sim._wait_for_idle()

        sim.debug(_func=interact, silent=True, _args=(sim,))
        self.raise_thread_exceptions()

    def test_run(self):
        RT = rt_example()
        sim = TraceGUI(RT)
        sim._exit = True  # leads to run() exiting directly after load
        sim.run()

    def test_rt_fault_no_rays(self):
        """check handling of functionality when there is an geometry fault in the raytracer 
        and therefore no rays are simulated and plotted"""

        # make raytracer
        RT = ot.Raytracer(outline=[-10, 10, -10, 10, -10, 60], silent=True)

        # add Raysource
        RSS = ot.Point()
        RS = ot.RaySource(RSS, divergence="Isotropic", spectrum=ot.presets.light_spectrum.d65,
                          pos=[0, 0, 0], s=[0, 0, 1], div_angle=75)
        RT.add(RS)
        
        geom = ot.presets.geometry.arizona_eye()
        geom.elements[2].move_to([0, 0, 0.15])

        RT.add(geom)
        
        def interact(sim):
            with self._try(sim):
                # no rays traced because of error in geometry
                self.assertEqual(sim.raytracer.rays.N, 0)  # nothing traced
                self.assertTrue(len(sim._plot._fault_markers) > 0)  # fault markers in GUI
                self.assertTrue(len(RT.markers) > 0)  # fault markers in RT

                # retrace and print error message
                sim.silent = False
                sim._do_in_main(sim.retrace)
                sim._wait_for_idle()

                # try actions with missing rays
                sim._do_in_main(sim.show_detector_image)
                sim._wait_for_idle()
                sim._do_in_main(sim.move_to_focus)
                sim._wait_for_idle()
                sim._do_in_main(sim.show_source_image)
                sim._wait_for_idle()
                sim._set_in_main("ray_opacity", 0.05)
                sim._wait_for_idle()
                sim._set_in_main("ray_width", 5)
                sim._wait_for_idle()
                sim._set_in_main("ray_count", 100000)
                sim._wait_for_idle()
                sim._set_in_main("absorb_missing", False)
                sim._wait_for_idle()
                sim._set_in_main("rays_visible", 3000)
                sim._wait_for_idle()
                sim._set_in_main("minimalistic_view", True)
                sim._wait_for_idle()
                sim._set_in_main("coloring_type", "Power")
                sim._wait_for_idle()
                sim._set_in_main("plotting_type", "Points")
                sim._wait_for_idle()
                sim._do_in_main(sim.show_source_spectrum)
                sim._wait_for_idle()
                sim._do_in_main(sim.show_detector_spectrum)
                sim._wait_for_idle()
                sim._do_in_main(sim.open_property_browser)  # should lead to collision error
                sim._wait_for_idle()
                sim._do_in_main(sim.replot)
                sim._wait_for_idle()

                # leave only one lens, now it can be traced
                # this also removes the old fault markers
                RT.remove(RT.lenses[1:])
                sim._do_in_main(sim.replot)
                sim._wait_for_idle()
                self.assertTrue(RT.rays.N > 0)  # rays traced
                self.assertEqual(len(sim._plot._fault_markers), 0)  # no fault_marker in GUI
                self.assertEqual(len(RT.markers), 0)  # no fault markers in RT

        sim = TraceGUI(RT)
        sim.debug(_func=interact, silent=True, _args=(sim,))
        self.raise_thread_exceptions()

    def test_point_marker(self):
        """test marker plotting, replotting, removal and properties in scene"""

        RT = rt_example()
        RT.remove(RT.markers)
        
        def interact(sim):
            with self._try(sim):

                # marker 1, default size
                RT.add(ot.PointMarker("Test1", [0., 0., 0]))
                sim._do_in_main(sim.replot)
                sim._wait_for_idle()
                self.assertEqual(len(sim._plot._marker_plots), 1)  # element was added
                self.assertTrue(np.allclose(sim._plot._marker_plots[0][0].actor.actor.center, 0))  # check position

                # marker 2, enlarged
                RT.add(ot.PointMarker("Test2", [0., 1., 5.2], text_factor=2, marker_factor=2))
                sim._do_in_main(sim.replot)
                sim._wait_for_idle()
                self.assertEqual(len(sim._plot._marker_plots), 2)  # element was added
                self.assertTrue(np.allclose(sim._plot._marker_plots[1][0].actor.actor.center\
                                            - np.array([0, 1., 5.2]), 0, atol=1e-6, rtol=0))  # check position

                # check size change
                a, b = tuple(sim._plot._marker_plots)
                self.assertAlmostEqual(b[0].glyph.glyph.scale_factor/a[0].glyph.glyph.scale_factor, 2)
                self.assertAlmostEqual(b[3].property.font_size/a[3].property.font_size, 2)

                # check if text was assigned correctly
                self.assertEqual(a[3].text, "Test1")
                self.assertEqual(b[3].text, "Test2")

                # check if marker crosshair is visible
                self.assertTrue(a[0].visible)

                # change one marker to label_only
                RT.markers[0].label_only = True

                # check replotting of markers
                sim._do_in_main(sim.send_cmd, "RT.remove(ML[-1])") # also checks that alias ML exists
                sim._wait_for_idle()
                self.assertEqual(len(RT.markers), 1)  # element was removed in raytracer
                self.assertEqual(len(sim._plot._marker_plots), 1)  # element was removed in scene

                # first marker now should have no crosshair shown
                a = sim._plot._marker_plots[0]
                self.assertFalse(a[0].visible)

        sim = TraceGUI(RT)
        sim.debug(_func=interact, silent=True, _args=(sim,))
        self.raise_thread_exceptions()
    
    def test_line_marker(self):
        """test marker plotting, replotting, removal and properties in scene"""

        RT = rt_example()
        RT.remove(RT.markers)
        
        def interact(sim):
            with self._try(sim):

                # marker 1, default size
                RT.add(ot.LineMarker(r=5, desc="Test1", pos=[0., 0., 0]))
                sim._do_in_main(sim.replot)
                sim._wait_for_idle()
                self.assertEqual(len(sim._plot._line_marker_plots), 1)  # element was added
                self.assertTrue(np.allclose(sim._plot._line_marker_plots[0][0].actor.actor.center, 0))  # check position

                # marker 2, enlarged
                RT.add(ot.LineMarker(r=5, angle=-20, desc="Test2", pos=[0., 1., 5.2], text_factor=2, line_factor=2))
                sim._do_in_main(sim.replot)
                sim._wait_for_idle()
                self.assertEqual(len(sim._plot._line_marker_plots), 2)  # element was added
                self.assertTrue(np.allclose(sim._plot._line_marker_plots[1][0].actor.actor.center\
                                            - np.array([0, 1., 5.2]), 0, atol=1e-6, rtol=0))  # check position

                # check size change
                a, b = tuple(sim._plot._line_marker_plots)
                self.assertAlmostEqual(b[0].actor.actor.property.line_width/a[0].actor.actor.property.line_width, 2)
                self.assertAlmostEqual(b[3].property.font_size/a[3].property.font_size, 2)

                # check if text was assigned correctly
                self.assertEqual(a[3].text, "Test1")
                self.assertEqual(b[3].text, "Test2")

                # check replotting of markers
                sim._do_in_main(sim.send_cmd, "RT.remove(ML[-1])") # also checks that alias ML exists
                sim._wait_for_idle()
                self.assertEqual(len(RT.markers), 1)  # element was removed in raytracer
                self.assertEqual(len(sim._plot._line_marker_plots), 1)  # element was removed in scene

        sim = TraceGUI(RT)
        sim.debug(_func=interact, silent=True, _args=(sim,))
        self.raise_thread_exceptions()

    def test_picker(self):
        """
        test picker interaction in the scene
        this is done without keypress and mouse click simulation,
        so it is not tested, if the scene reacts to those correctly
        only the pick handling is checked here
        """

        RT = rt_example()
        
        def interact(sim):
            with self._try(sim):
                # change to z+ view, so there are rays at the middle of the scene
                sim._do_in_main(sim.scene.z_plus_view)
                sim._wait_for_idle()
           
                default_text = sim._plot._ray_text.text  # save default text for comparison

                # ray picked -> show default infos
                sim._do_in_main(sim._plot._ray_picker.pick, sim._plot._scene_size[0] / 2, sim._plot._scene_size[1] / 2, 0, sim.scene.renderer)
                sim._wait_for_idle()
                time.sleep(0.2)  # delay so a human user can check the text
                text1 = sim._plot._ray_text.text
                self.assertNotEqual(text1, default_text)  # shows a ray info text
               
                # ray picked -> show verbose info
                pyautogui.keyDown("shiftleft")
                time.sleep(0.1)
                sim._do_in_main(sim._plot._ray_picker.pick, sim._plot._scene_size[0] / 2, sim._plot._scene_size[1] / 2, 0, sim.scene.renderer)
                sim._wait_for_idle()
                time.sleep(0.2)
                text2 = sim._plot._ray_text.text
                self.assertNotEqual(text2, default_text)  # no default text
                self.assertNotEqual(text1, text2)  # not the old text
               
                # no ray picked -> default text
                sim._do_in_main(sim._plot._ray_picker.pick, 0, 0, 0, sim.scene.renderer)
                sim._wait_for_idle()
                time.sleep(0.2)
                text2 = sim._plot._ray_text.text
                self.assertEqual(text2, default_text)  # shows default text
              
                # redraw ray info only if it is the default one
                sim._set_in_main("minimalistic_view", True)
                sim._wait_for_idle()
                sim._set_in_main("minimalistic_view", False)
                sim._wait_for_idle()

                # remove crosshair and pick a ray
                sim._plot._crosshair = None
                sim._do_in_main(sim._plot._ray_picker.pick, sim._plot._scene_size[0] / 2, sim._plot._scene_size[1] / 2, 0, sim.scene.renderer)
                sim._wait_for_idle()

                # remove crosshair and pick no ray
                sim._do_in_main(sim._plot._ray_picker.pick, 0, 0, 0, sim.scene.renderer)
                sim._wait_for_idle()
                
                # restore crosshair
                sim._do_in_main(sim._plot.init_crosshair)
                sim._wait_for_idle()
                
                # we have an extra picker sim._space_picker for right+clicking in the scene,
                # but I don't know how to make the Picker.pick() function pick with a right click
                # so currently we overide the RayPicker with the onSpacePick method
                # do via command string, so it is guaranteed to run in main thread
                sim._do_in_main(sim.send_cmd, "self._plot._ray_picker = self.scene.mlab.gcf().on_mouse_pick("
                                         "self._plot._on_space_pick, button='Left')")
                sim._wait_for_idle()

                # space picked -> show coordinates
                pyautogui.keyUp("shiftleft")
                time.sleep(0.1)
                sim._do_in_main(sim._plot._ray_picker.pick, sim._plot._scene_size[0] / 3, sim._plot._scene_size[1] / 3, 0, sim.scene.renderer)
                sim._wait_for_idle()
                time.sleep(0.2)
                text3 = sim._plot._ray_text.text
                self.assertNotEqual(text3, default_text)  # not the default text
                self.assertNotEqual(text3, text2)  # not the old text
                
                # valid space picked with shift -> move detector
                pyautogui.keyDown("shiftleft")
                time.sleep(0.1)
                old_pos = RT.detectors[0].pos
                sim._do_in_main(sim._plot._ray_picker.pick, sim._plot._scene_size[0] / 3, sim._plot._scene_size[1] / 3, 0, sim.scene.renderer)
                sim._wait_for_idle()
                time.sleep(0.2)
                text4 = sim._plot._ray_text.text
                self.assertEqual(text4, default_text)  # not the default text
                self.assertNotEqual(RT.detectors[0].pos[2], old_pos[2])
                
                # space outside outline picked with shift -> move detector
                sim._do_in_main(sim.scene.y_plus_view)
                sim._do_in_main(sim._plot._ray_picker.pick, 0, 0, 0, sim.scene.renderer)
                sim._wait_for_idle()
                time.sleep(0.2)
                text4 = sim._plot._ray_text.text
                self.assertEqual(text4, default_text)  # not the default text
                self.assertEqual(RT.detectors[0].pos[2], RT.outline[4])  # detector moved to the outline beginning
              
                # remove detectors and try to move the detector using the space pick action
                RT.detectors = []
                sim._do_in_main(sim.replot)
                sim._wait_for_idle()
                sim._do_in_main(sim._plot._ray_picker.pick, 0, 0, 60, sim.scene.renderer)
                sim._wait_for_idle()

                # remove crosshair and pick
                sim._plot._crosshair = None
                sim._do_in_main(sim._plot._ray_picker.pick, 0, 0, 60, sim.scene.renderer)
                sim._wait_for_idle()

                # release shift key
                pyautogui.keyUp("shiftleft")
                time.sleep(0.1)
                
                # remove crosshair and pick without shift key
                sim._do_in_main(sim._plot._ray_picker.pick, 0, 0, 60, sim.scene.renderer)
                sim._wait_for_idle()
                
                # restore crosshair
                sim._do_in_main(sim._plot.init_crosshair)
                sim._wait_for_idle()

                # test case where RayText is missing
                sim._plot._ray_text = None
                sim._do_in_main(sim._plot._on_ray_pick)
                sim._wait_for_idle()

        sim = TraceGUI(RT)
        sim.debug(_func=interact, silent=True, _args=(sim,))
        self.raise_thread_exceptions()

    def test_picker_coverage(self):

        RT = ot.Raytracer(outline=[-10, 10, -10, 10, 0, 10], silent=True)
        
        def interact(sim):
            def pick():    
                sim._wait_for_idle()
                sim._do_in_main(sim._plot._ray_picker.pick, sim._plot._scene_size[0] / 2, sim._plot._scene_size[1] / 2, 0, sim.scene.renderer)
                sim._wait_for_idle()
                time.sleep(0.2)  # delay so a human user can check the text

            def pick_shift_combs():
                pick()
                pyautogui.keyDown("shiftleft")
                time.sleep(0.2)
                pick()
                pyautogui.keyUp("shiftleft")

            with self._try(sim):
                # change to z+ view, so there are rays at the middle of the scene
                sim._do_in_main(sim.scene.z_plus_view)
                sim._wait_for_idle()

                # pick RaySource Point
                RT.add(ot.RaySource(ot.Point(), pos=[0, 0, 0], s_sph=[20, 0]))
                sim._do_in_main(sim.replot)
                pick_shift_combs()

                # pick RaySource Surface
                RT.remove(RT.ray_sources[0])
                RT.add(ot.RaySource(ot.CircularSurface(r=0.1), pos=[0, 0, 0], s_sph=[20, 0]))
                sim._do_in_main(sim.replot)
                pick_shift_combs()

                # pick last ray section point (at outline)
                RT.remove(RT.ray_sources[0])
                RT.add(ot.RaySource(ot.Point(), pos=[-5, 0, 0], s=[5, 0, 10]))
                sim._do_in_main(sim.replot)
                pick_shift_combs()
                
                # pick element in between
                RT.remove(RT.ray_sources[0])
                RT.add(ot.RaySource(ot.Point(), pos=[-5, 0, 0], s=[10, 0, 10]))
                RT.add(ot.Filter(ot.CircularSurface(r=0.1), pos=[0, 0, 5], 
                                 spectrum=ot.TransmissionSpectrum("Constant", val=1)))
                sim._do_in_main(sim.replot)
                pick_shift_combs()

        sim = TraceGUI(RT)
        sim.debug(_func=interact, silent=True, _args=(sim,))
        self.raise_thread_exceptions()

    def test_volumes(self):
        """plot volume plotting and handling"""

        RT = rt_example()
        RT.clear()
        
        def interact(sim):
            with self._try(sim):

                # create some volumes without specifying the color and opacity
                sphv = ot.SphereVolume(R=5, pos=[3, 5, 6])
                RT.add(sphv)
                cylv = ot.CylinderVolume(r=5, length=5, pos=[0, 5, 6])
                RT.add(cylv)
                boxv = ot.BoxVolume(dim=[5, 6], length=4, pos=[3, 0, 6])
                RT.add(boxv)

                # user volume
                front = ot.ConicSurface(r=4, k=2, R=50)
                back = ot.RectangularSurface(dim=[3, 3])
                vol = ot.Volume(front, back, pos=[0, 1, 2], d1=front.ds, d2=back.ds+1)
                RT.add(vol)

                # replot
                sim._do_in_main(sim.replot)
                sim._wait_for_idle()
                self.assertEqual(len(sim._plot._volume_plots), 4)  # elements were added
                self.assertTrue(sim._plot._volume_plots[0][3] is None)  # no text label for volumes

                # tests that opacity and color is assigned correctly
                sphv.opacity = 0.1
                sphv.color = (1.0, 0.0, 1.0)
                sim._do_in_main(sim.replot)
                sim._wait_for_idle()
                self.assertEqual(sim._plot._volume_plots[0][0].actor.property.color, sphv.color)
                self.assertEqual(sim._plot._volume_plots[0][0].actor.property.opacity, sphv.opacity)
                
                # checks that automatic replotting works
                sim._do_in_main(sim.send_cmd, "RT.volumes[0].opacity=0.9") 
                sim._wait_for_idle()
                self.assertEqual(sim._plot._volume_plots[0][0].actor.property.opacity, 0.9)

                # toggle contrast mode
                sim._set_in_main("high_contrast", True)
                sim._wait_for_idle()
                # check that custom color gets unset with high contrast
                self.assertEqual(sim._plot._volume_plots[0][0].actor.property.color, sim._plot._volume_color)

                sim._set_in_main("high_contrast", False)
                sim._wait_for_idle()
                # check that custom color gets set again without high contrast
                self.assertEqual(sim._plot._volume_plots[0][0].actor.property.color, sphv.color)

        sim = TraceGUI(RT)
        sim.debug(_func=interact, silent=True, _args=(sim,))
        self.raise_thread_exceptions()

if __name__ == '__main__':
    unittest.main()
