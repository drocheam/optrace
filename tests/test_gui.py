#!/bin/env python3

import sys
import os
import copy
import unittest
import warnings
import time

from typing import Callable, Any
from contextlib import contextmanager  # context manager for _no_trait_action()

import numpy as np
import pytest
import pytest_xvfb
from threading import Thread
import matplotlib.pyplot as plt
import pyautogui
from pyface.qt import QtGui
from pyface.api import GUI as pyface_gui  # invoke_later() method

import optrace as ot
from optrace.gui import TraceGUI
from tracing_geometry import tracing_geometry


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

    def _wait_for_idle(self, sim, base=0.2, timeout=30) -> None:
        """wait until the GUI is Idle. Only call this from another thread"""

        def raise_timeout(keys):
            raise TimeoutError(f"Timeout while waiting for other actions to finish. Blocking actions: {keys}")

        tsum = base
        time.sleep(base)  # wait for flags to be set, this could also be trait handlers, which could take longer
        while sim.busy:
            time.sleep(0.05)
            tsum += 0.05

            if tsum > timeout:
                raise TimeoutError("")
    
    def _do_in_main(self, f: Callable, *args, **kw) -> None:
        """execute a function in the GUI main thread"""
        pyface_gui.invoke_later(f, *args, **kw)
        pyface_gui.process_events()

    def _set_in_main(self, sim, trait: str, val: Any) -> None:
        """assign a property in the GUI main thread"""
        pyface_gui.set_trait_later(sim, trait, val)
        pyface_gui.process_events()

    @contextmanager
    def _try(self, sim, *args, **kwargs):
        """try TraceGUI actions. Exceptions are catched and saved in the class to be later
        raised by raise_thread_exceptions(). In all cases the gui is closed normally"""
        time.sleep(1)
        self._wait_for_idle(sim)
        try:
            yield
        except Exception as e:
            self.exc_info = sys.exc_info()  # assign thread exception
        finally:
            self._wait_for_idle(sim)
            self._do_in_main(sim.close)
            # time.sleep(2)

    @pytest.mark.gui1
    @pytest.mark.slow
    def test_gui_inits(self) -> None:

        # make raytracer
        RT = ot.Raytracer(outline=[-5, 5, -5, 5, 0, 40])

        # add Raysource
        RSS = ot.presets.image.tv_testcard1([4, 4])
        RS = ot.RaySource(RSS, divergence="Isotropic", div_angle=8,
                          s=[0, 0, 1], pos=[0, 0, 0])
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
        def trace_gui_run(clear=True,**kwargs):
            if clear:
                RT.rays.init(RT.ray_sources, RT.rays.N, RT.rays.Nt, RT.rays.no_pol)
            trace_gui_run.i += 1
            with self.subTest(i=trace_gui_run.i, args=kwargs):
                sim = TraceGUI(RT, **kwargs)
                sim._exit = True
                sim.run()

        trace_gui_run.i = 0

        trace_gui_run()  # default init

        for ctype in TraceGUI.coloring_modes:
            trace_gui_run(coloring_mode=ctype)

        for ptype in TraceGUI.plotting_modes:
            trace_gui_run(plotting_mode=ptype)

        ot.global_options.ui_dark_mode = False
        trace_gui_run(ray_count=100000)
        trace_gui_run(rays_visible=200)
        trace_gui_run(minimalistic_view=True)
        trace_gui_run(hide_labels=True)
        trace_gui_run(high_contrast=True)
        ot.global_options.ui_dark_mode = True
        trace_gui_run(ray_opacity=0.15)
        trace_gui_run(ray_width=5)
        trace_gui_run(vertical_labels=True)
        trace_gui_run(initial_camera=dict(center=[0, 1, 0], height=5))

        # many parameters
        trace_gui_run(ray_count=100000, rays_visible=1500, ray_opacity=0.1, ray_width=5,
                      coloring_mode="Wavelength", plotting_mode="Points", minimalistic_view=True, hide_labels=True,
                      vertical_labels=True)

        # type errors
        self.assertRaises(TypeError, TraceGUI, 5)  # invalid raytracer
        self.assertRaises(TypeError, TraceGUI, RT, initial_camera=5)  # invalid initial_camera

        # this attributes can't bet set initially
        self.assertRaises(RuntimeError, TraceGUI, RT, detector_selection="DET1")
        self.assertRaises(RuntimeError, TraceGUI, RT, source_selection="RS1")
        self.assertRaises(RuntimeError, TraceGUI, RT, z_det=0)
        
        # Retrace tests. The TraceGUI tries to retrace automatically

        # calling the GUI with a traced geometry does not retrace
        rcrepr = RT.rays.crepr()
        trace_gui_run(clear=False)
        self.assertEqual(rcrepr, RT.rays.crepr())

        # explicitly providing ray_count retraces
        rcrepr = RT.rays.crepr()
        trace_gui_run(clear=False, ray_count=2000)
        self.assertNotEqual(rcrepr, RT.rays.crepr())
        
        # calling the GUI with changes in the geometry retraces
        RT.remove(RT.lenses[0])
        rcrepr = RT.rays.crepr()
        trace_gui_run(clear=False)
        self.assertNotEqual(rcrepr, RT.rays.crepr())

    @pytest.mark.gui2
    @pytest.mark.slow
    def test_interaction1(self) -> None:

        def interact(sim):
            with self._try(sim):

                # check moving of detectors and selection and name updates
                for i, z0 in zip(range(len(RT.detectors)), [5.3, 27.3]):
                    # change detector
                    dname0 = sim.detector_selection
                    self._set_in_main(sim, "detector_selection", sim.detector_names[i])
                    self._wait_for_idle(sim)

                    dname1 = sim.detector_names[i]
                    if i:  # check if detector name changed from initial
                        self.assertNotEqual(dname0, dname1)
                    
                    assert z0 != RT.detectors[i].pos[2]
                    self.assertEqual(sim.z_det, RT.detectors[i].pos[2])
                    # position updated after changing detector
                   
                    # change detector position
                    self._set_in_main(sim, "z_det", z0)
                    self._wait_for_idle(sim)
                    dname2 = sim.detector_selection
                    self.assertEqual(sim.z_det, RT.detectors[i].pos[2])  # position updated
                    self.assertNotEqual(dname1, sim.detector_names[i])  # name updated after position change
                    self.assertEqual(dname2, sim.detector_names[i])  # name updated after position change
                    
                # Source Image Tests
                self._do_in_main(sim.source_image)
                self._wait_for_idle(sim)
                self._set_in_main(sim, "source_selection", sim.source_names[1])
                self._wait_for_idle(sim)
                self._do_in_main(sim.source_image)
                self._wait_for_idle(sim)
                
                # Detector Image Tests
                self._do_in_main(sim.detector_image)
                self._wait_for_idle(sim)
                self._set_in_main(sim, "detector_selection", sim.detector_names[1])
                self._wait_for_idle(sim)
                self.assertTrue(sim._det_ind == 1)
                self.assertTrue(sim.z_det == sim.raytracer.detectors[1].pos[2])
                self._do_in_main(sim.detector_image)
                self._wait_for_idle(sim)

                # detector profile and image with extent provided
                extent = [-1, 1, -1, 1]
                self._do_in_main(sim.detector_image, extent=extent)
                self._wait_for_idle(sim)
                self.assertTrue(np.all(sim.last_det_image.extent == extent))
                extent = [-1.2, 1, -1.2, 1]
                self._do_in_main(sim.detector_profile, extent=extent)
                self._wait_for_idle(sim)
                self.assertTrue(np.all(sim.last_det_image.extent == extent))

                # Image Type Tests standard
                for mode in ot.RenderImage.image_modes:
                    self._set_in_main(sim, "image_mode", mode)
                    self._wait_for_idle(sim)
                    self._do_in_main(sim.detector_image)
                    self._wait_for_idle(sim)
                self._do_in_main(plt.close, "all")

                # Image Tests Higher Res
                self._set_in_main(sim, "image_pixels", 315)
                self._wait_for_idle(sim)
                self._do_in_main(sim.detector_image)
                self._wait_for_idle(sim)

                self._set_in_main(sim, "maximize_scene", True)
                self._wait_for_idle(sim)

                # Image Test Source, but actually we should test all parameter combinations,
                self._do_in_main(sim.source_image)
                self._wait_for_idle(sim)

                # Focus Tests
                pos0 = sim.raytracer.detectors[1].pos
                self._set_in_main(sim, "detector_selection", sim.detector_names[1])
                self._wait_for_idle(sim)

                for mode in ot.Raytracer.focus_search_methods:
                    self._set_in_main(sim, "focus_search_method", mode)
                    self._wait_for_idle(sim)
                    self._do_in_main(sim.move_to_focus)
                    self._wait_for_idle(sim)
                    self._set_in_main(sim, "z_det", pos0[2])
                    self._wait_for_idle(sim)

                # Focus Test 4, show Debug Plot
                self._set_in_main(sim, "plot_cost_function", True)
                self._wait_for_idle(sim)
                self._do_in_main(sim.move_to_focus)
                self._wait_for_idle(sim)

                # Focus Test 5, one source only
                self._set_in_main(sim, "plot_cost_function", False)
                self._wait_for_idle(sim)
                self._set_in_main(sim, "z_det", pos0[2])
                self._wait_for_idle(sim)
                self._set_in_main(sim, "focus_search_single_source", True)
                self._do_in_main(sim.move_to_focus)
                self._wait_for_idle(sim)
                
                # Ray Coloring Tests
                for type_ in sim.coloring_modes:
                    self._set_in_main(sim, "coloring_mode", type_)
                    self._wait_for_idle(sim)

                # special case: user defined spectral colormap
                ot.global_options.spectral_colormap = lambda wl: plt.cm.viridis(wl/780)
                self._set_in_main(sim, "coloring_mode", "Wavelength")
                self._wait_for_idle(sim)
                ot.global_options.spectral_colormap = None

                # plotting_type Tests
                for type_ in sim.plotting_modes:
                    self._set_in_main(sim, "plotting_mode", type_)
                    self._wait_for_idle(sim)
             
                # open documentation
                self._do_in_main(sim.open_documentation)

                # retrace Tests
                self._set_in_main(sim, "ray_count", 100000)
                self._wait_for_idle(sim)
                
                self._set_in_main(sim, "rays_visible", 50)
                self._wait_for_idle(sim)

                self._set_in_main(sim, "rays_visible", 500)

        RT = tracing_geometry()
        sim = TraceGUI(RT)
        sim.debug(interact, args=(sim,))
        self.raise_thread_exceptions()

    @pytest.mark.gui2
    @pytest.mark.slow
    def test_interaction2(self) -> None:

        def interact2(sim):
            with self._try(sim):
                # Image Type Tests log scaling
                self._set_in_main(sim, "log_image", True)

                # display all image modes with log
                for mode in ot.RenderImage.image_modes:
                    self._set_in_main(sim, "image_mode", mode)
                    self._wait_for_idle(sim)
                    self._do_in_main(sim.detector_image)
                    self._wait_for_idle(sim)
                self._do_in_main(plt.close, "all")

                # images with filter
                self._set_in_main(sim, "filter_constant", 10)
                self._set_in_main(sim, "activate_filter", True)
                self._do_in_main(sim.detector_image)
                self._wait_for_idle(sim)
                self._do_in_main(sim.source_image)
                self._wait_for_idle(sim)

                # change ui dark mode while active
                self._do_in_main(ot.global_options.__setattr__, "ui_dark_mode", not ot.global_options.ui_dark_mode)
                

                # Image Tests Flip
                self._set_in_main(sim, "log_image", False)
                self._wait_for_idle(sim)
                self._set_in_main(sim, "flip_detector_image", True)
                self._wait_for_idle(sim)
                self._do_in_main(sim.detector_image)
                self._wait_for_idle(sim)

                # one source only
                self._set_in_main(sim, "detector_image_single_source", True)
                self._wait_for_idle(sim)
                self._do_in_main(sim.detector_image)
                self._wait_for_idle(sim)

                # make labels vertical
                self._set_in_main(sim, "vertical_labels", True)
                self._wait_for_idle(sim)
                self._set_in_main(sim, "vertical_labels", False)
                self._wait_for_idle(sim)
                
                # change ui dark mode while active
                self._do_in_main(ot.global_options.__setattr__, "ui_dark_mode", not ot.global_options.ui_dark_mode)

                # high contrast mode
                self._set_in_main(sim, "high_contrast", True)
                self._wait_for_idle(sim)
                self._set_in_main(sim, "high_contrast", False)
                self._wait_for_idle(sim)

                # test different projection methods
                self._set_in_main(sim, "detector_selection", sim.detector_names[1])
                self._wait_for_idle(sim)
                self._do_in_main(plt.close, "all")
                for pm in ot.SphericalSurface.sphere_projection_methods:
                    self._set_in_main(sim, "projection_method", pm)
                    self._do_in_main(sim.detector_image)
                    self._wait_for_idle(sim)
                    self._do_in_main(sim.detector_profile)
                    self._wait_for_idle(sim)
                    self.assertEqual(sim.last_det_image.projection, pm)
                
                # open browser light mode
                self._do_in_main(ot.global_options.__setattr__, "ui_dark_mode", False)
                self._do_in_main(sim.open_property_browser)
                self._wait_for_idle(sim)
                # time.sleep(2)  # wait some time so windows can be seen by a human user
               
                # open browser dark mode
                self._do_in_main(ot.global_options.__setattr__, "ui_dark_mode", True)
                self._do_in_main(sim.open_property_browser)
                self._wait_for_idle(sim)

                # try setting many rays
                rc0 = sim.ray_count
                self._set_in_main(sim, "ray_count", 50000000-1)
                self._wait_for_idle(sim)
                self.assertEqual(rc0, sim.ray_count)

        RT = tracing_geometry()
        sim = TraceGUI(RT)
        sim.debug(func=interact2, args=(sim,))
        plt.close('all')
        self.raise_thread_exceptions()

    @pytest.mark.gui2
    @pytest.mark.slow
    def test_interaction3(self) -> None:

        def interact3(sim):
            with self._try(sim):
                
                # test source and detector spectrum
                self._do_in_main(sim.source_spectrum)
                self._wait_for_idle(sim)
                self._do_in_main(sim.detector_spectrum)
                self._wait_for_idle(sim)
                self._set_in_main(sim, "detector_spectrum_single_source", True)
                self._do_in_main(sim.detector_spectrum)
                self._wait_for_idle(sim)

                # test image profiles
                self._do_in_main(sim.source_profile)
                self._wait_for_idle(sim)
                self._set_in_main(sim, "detector_image_single_source", True)
                self._wait_for_idle(sim)
                self._do_in_main(sim.detector_profile)
                self._wait_for_idle(sim)
                self._set_in_main(sim, "profile_position_dimension", "x")
                self._do_in_main(sim.detector_profile)
                self._wait_for_idle(sim)
                self._set_in_main(sim, "profile_position_dimension", "y")
                self._do_in_main(sim.detector_profile)
                self._wait_for_idle(sim)
                self._set_in_main(sim, "profile_position", 0)  # exists inside image
                self._do_in_main(sim.detector_profile)
                self._wait_for_idle(sim)
                self._set_in_main(sim, "profile_position", 100)  # does not exist inside image
                self._do_in_main(sim.detector_profile)
                self._wait_for_idle(sim)

        RT = tracing_geometry()
        sim = TraceGUI(RT)
        sim.debug(interact3, args=(sim,))
        plt.close('all')
        self.raise_thread_exceptions()

    @pytest.mark.gui1
    @pytest.mark.slow
    def test_missing(self) -> None:
        """test TraceGUI operation when Filter, Lenses, Detectors or Sources are missing"""

        def test_features(RT):
            sim = TraceGUI(RT)

            def interact(sim):
                with self._try(sim):
                    self._do_in_main(sim.detector_image)
                    self._wait_for_idle(sim)
                    self._do_in_main(sim.move_to_focus)
                    self._wait_for_idle(sim)
                    self._set_in_main(sim, "z_det", 10.)
                    self._wait_for_idle(sim)
                    self._do_in_main(sim.source_image)
                    self._wait_for_idle(sim)
                    self._set_in_main(sim, "ray_opacity", 0.07)
                    self._wait_for_idle(sim)
                    self._set_in_main(sim, "ray_width", 5)
                    self._wait_for_idle(sim)
                    self._set_in_main(sim, "ray_count", 100000)
                    self._wait_for_idle(sim)
                    self._set_in_main(sim, "rays_visible", 4000)
                    self._wait_for_idle(sim)
                    self._set_in_main(sim, "minimalistic_view", True)
                    self._wait_for_idle(sim)
                    self._set_in_main(sim, "coloring_mode", "Power")
                    self._wait_for_idle(sim)
                    self._set_in_main(sim, "hide_labels", True)
                    self._wait_for_idle(sim)
                    self._set_in_main(sim, "plotting_mode", "Points")

                    self._set_in_main(sim, "high_contrast", True)
                    self._wait_for_idle(sim)
                    
                    self._do_in_main(sim._change_selected_ray_source)
                    self._wait_for_idle(sim)
                    self._do_in_main(sim._change_detector)
                    self._wait_for_idle(sim)
                    
                    self._do_in_main(sim.source_spectrum)
                    self._wait_for_idle(sim)
                    self._do_in_main(sim.detector_spectrum)
                    self._wait_for_idle(sim)

                    self._do_in_main(sim.replot)

            sim.debug(interact, args=(sim,))
            self.raise_thread_exceptions()
            time.sleep(1)

        RT = tracing_geometry()

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

    # @pytest.mark.skip(reason="there seems to be some issue with qt4, leads to segmentation faults.")
    @pytest.mark.slow
    @pytest.mark.gui1
    def test_action_spam(self):
        """spam the gui with many possible actions to check threading locks and for race conditions"""

        RT = tracing_geometry()
        sim = TraceGUI(RT)

        def interact(sim):
            
            with self._try(sim):
                N0 = sim.ray_count
                self._set_in_main(sim, "ray_count", int(N0*1.3))
                self._set_in_main(sim, "focus_search_method", "Image Sharpness")
                self._do_in_main(sim.detector_image)
                self._do_in_main(sim.source_image)
                self._do_in_main(sim.move_to_focus)

                time.sleep(0.01)
                self._set_in_main(sim, "ray_count", int(N0/1.3))
                self._set_in_main(sim, "z_det", (RT.outline[5] - RT.outline[4])/2)
                self._do_in_main(sim.source_profile)
                self._do_in_main(sim.move_to_focus)
                self._set_in_main(sim, "detector_selection", sim.detector_names[1])
                self._do_in_main(sim.detector_image)
                self._do_in_main(sim.replot_rays)
                self._do_in_main(sim.source_image)

                time.sleep(0.1)
                self._do_in_main(sim.replot_rays)
                self._do_in_main(sim.run_command, "GUI.replot()")
                self._do_in_main(sim.detector_image)
                self._do_in_main(sim.screenshot, "tmp.png") 
                self._set_in_main(sim, "detector_selection", sim.detector_names[1])

                self._set_in_main(sim, "ray_count", 1000000)
                self._do_in_main(sim.replot_rays)
                self._do_in_main(sim.set_camera, [5, 10, 3], 30)
                self._set_in_main(sim, "detector_selection", sim.detector_names[1])
                self._do_in_main(sim.move_to_focus)
                self._do_in_main(sim.source_spectrum)
                
                self._wait_for_idle(sim, timeout=45)

                self.assertEqual(sim.ray_count, 1000000)
                self.assertEqual(sim.raytracer.rays.N, 1000000)
                os.remove("tmp.png")

        sim.debug(interact, args=(sim,))
        self.raise_thread_exceptions()

    @pytest.mark.slow
    @pytest.mark.gui3
    def test_replot_combinations(self):
        """check replotting by setting random combinations inside the change dictionary for replot()"""

        RT = tracing_geometry()
        sim = TraceGUI(RT)

        def interact(sim):
            with self._try(sim):
                # check if replot actually removes old plot elements
                L0id = id(sim._plot._lens_plots[0][0])
                F0id = id(sim._plot._filter_plots[0][0])
                self._do_in_main(sim.replot)
                self._wait_for_idle(sim)
                self.assertNotEqual(L0id, id(sim._plot._lens_plots[0][0]))
                self.assertNotEqual(F0id, id(sim._plot._filter_plots[0][0]))

                self._set_in_main(sim, "ray_count", 20000)  # fewer rays so tracing is faster
                self._wait_for_idle(sim)

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

                    self._do_in_main(sim.replot, cmp)
                    self._wait_for_idle(sim)

        sim.debug(interact, args=(sim,))
        self.raise_thread_exceptions()

    @pytest.mark.slow
    @pytest.mark.skip(reason="there seems to be some issue with qt4, leads to segmentation faults.")
    def test_action_spam_random(self):
        """do random actions with random delays and check if there are race conditions or other special issues"""

        # somehow this hangs when a command is handled and some other different action is forced
        # could be due to locks
        # I couldn't reproduce this by clicking and interacting with the UI itself

        RT = tracing_geometry()

        sim = TraceGUI(RT)

        def interact(sim):
            with self._try(sim):
                funcs = [sim.detector_image, sim.source_image, sim.source_spectrum, sim.move_to_focus,
                         sim.replot_rays, sim.open_property_browser, sim.open_command_window,
                         sim.detector_spectrum, sim.detector_profile, sim.source_profile, sim.replot]

                props = [('rays_visible', 500), ('rays_visible', 2000), ('minimalistic_view', True),
                         ('minimalistic_view', False), ('plotting_mode', "Rays"), ('plotting_mode', "Points"),
                         ('ray_opacity', 0.06), ('ray_width', 12),
                         ('ray_opacity', 0.8), ('ray_width', 2), ('profile_position', 0),
                         ('profile_position_dimension', 'x'),
                         ('profile_position_dimension', 'y'), ('hide_labels', True), ('hide_labels', False),
                         ('focus_search_single_source', False), ('focus_search_single_source', True),
                         ('detector_image_single_source', False),
                         ('detector_image_single_source', True), ('profile_position', 0.1),
                         ('flip_detector_image', True),
                         ('flip_detector_image', False), ('detector_spectrum_single_source', True),
                         ('detector_image_single_source', False),
                         ('log_image', False), ('log_image', True),  ('plot_cost_function', True),
                         ('plot_cost_function', False),
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
                            self._do_in_main(np.random.choice(funcs))
                        case 1 | 11:
                            ch = np.random.randint(0, len(props))
                            self._set_in_main(sim, *props[ch])
                        case 2:
                            self._set_in_main(sim, "coloring_mode", np.random.choice(sim.coloring_modes))
                        case 3:
                            # only every second time, otherwise this would happen to often
                            if np.random.randint(0, 2):
                                self._set_in_main(sim, "ray_count", np.random.randint(10000, 500000))
                        case 4:
                            self._set_in_main(sim, "focus_search_method", np.random.choice(RT.focus_search_methods))
                        case 5:
                            self._set_in_main(sim, "image_mode", np.random.choice(ot.RenderImage.image_modes))
                        case 6:
                            self._set_in_main(sim, "detector_selection", np.random.choice(sim.detector_names))
                        case 7:
                            self._set_in_main(sim, "source_selection", np.random.choice(sim.source_names))
                        case 8:
                            self._set_in_main(sim, "z_det", np.random.uniform(RT.outline[4], RT.outline[5]))
                        case 9:
                            self._set_in_main(sim, "image_pixels", np.random.choice(ot.RenderImage.SIZES))
                        case 10:
                            cmds = ['GUI.replot()', 'scene.render()', 'scene.z_minus_view()']
                            cmd = np.random.choice(cmds)
                            self._do_in_main(sim.run_command, cmd)
                        case 12:
                            if isinstance(sim.raytracer.detectors[sim._det_ind].front, ot.SphericalSurface):
                                self._set_in_main(sim, "projection_method", 
                                                 np.random.choice(ot.SphericalSurface.sphere_projection_methods))

                    # self._wait_for_idle(sim)

                    # close plots from time to time
                    if i % 20 == 0:
                        self._do_in_main(plt.close, "all")
        
        sim.debug(interact, args=(sim,))
        self.raise_thread_exceptions()
   
    @pytest.mark.slow
    @pytest.mark.install
    @pytest.mark.gui3
    @pytest.mark.skipif(os.getenv("XDG_SESSION_DESKTOP") == "KDE" and os.getenv("XDG_SESSION_TYPE") == "wayland" \
                        and pytest_xvfb.xvfb_instance is None,
                        reason="KDE Wayland wants remote authentication. Use X11 or run with xvfb")
    @pytest.mark.skipif(os.getenv("GITHUB_ACTIONS") == "true", 
                        reason="Issues with headless display and input in github actions")
    def test_key_presses(self):
        """test keyboard shortcuts inside the scene while simulating key presses"""

        # create a pyplot window that will be closed from the gui
        # for the list of get_fignums it makes no difference, if they are shown or not
        # but showing would lead to window focus issues in test automation
        plt.figure()
        # plt.show(block=False)

        RT = tracing_geometry()
        sim = TraceGUI(RT)

        # without these high sleeping times there are errors in the github action workflows

        def send_key(sim, key):
            self._do_in_main(sim.scene.scene_editor._content.setFocus)
            self._wait_for_idle(sim)
            pyautogui.press(key)
            time.sleep(0.5)

        def interact(sim):
            with self._try(sim):

                time.sleep(3)

                # check minimalistic_view shortcut
                self.assertTrue(len(sim.minimalistic_view) == 0)
                send_key(sim, "v")
                self._wait_for_idle(sim)
                warnings.warn(str(sim._status))
                self.assertTrue(len(sim.minimalistic_view) != 0)
                
                # check high_contrast shortcut
                self.assertTrue(len(sim.high_contrast) == 0)
                send_key(sim, "c")
                self._wait_for_idle(sim)
                self.assertTrue(len(sim.high_contrast) != 0)
                
                # check hide_labels shortcut
                self.assertTrue(len(sim.hide_labels) == 0)
                send_key(sim, "b")
                self._wait_for_idle(sim)
                self.assertTrue(len(sim.hide_labels) != 0)
                
                # set camera to initial view /y plus view by default)
                self._do_in_main(sim.scene.mlab.view, 0, 1, 0)
                self._wait_for_idle(sim)
                cv = sim.scene.camera.position
                send_key(sim, "i")
                self._wait_for_idle(sim)
                cv2 = sim.scene.camera.position
                self.assertFalse(np.allclose(cv, cv2))
                
                # maximize scene / hide other menus
                self.assertTrue(sim.scene.scene_editor._tool_bar.isVisible())
                self.assertTrue(sim._scene_not_maximized)
                send_key(sim, "h")
                self._wait_for_idle(sim)
                self.assertFalse(sim.scene.scene_editor._tool_bar.isVisible())
                self.assertFalse(sim._scene_not_maximized)

                # don't press m warning
                send_key(sim, "m")
                self._wait_for_idle(sim)
                send_key(sim, "m")
                self._wait_for_idle(sim)
                
                # don't press a warning
                send_key(sim, "a")
                self._wait_for_idle(sim)
                send_key(sim, "a")
                self._wait_for_idle(sim)
               
                # key without shortcut
                send_key(sim, "w")
                self._wait_for_idle(sim)
                
                # replot rays
                id0 = id(sim._plot._ray_property_dict["p"])
                send_key(sim, "n")
                self._wait_for_idle(sim)
                self.assertNotEqual(id0, id(sim._plot._ray_property_dict["p"]))  # array changed, therefore the id too
             
                # check if pyplot closing works.
                # as I don't know how to refocus the scene after a plot has been shown,
                # we will close a pyplot that was created before the gui
                wlen = len(plt.get_fignums())
                send_key(sim, "0")
                self._wait_for_idle(sim)
                wlen2 = len(plt.get_fignums())
                self.assertNotEqual(wlen, wlen2)
                
                # do this one last, since it raises another window
                # and I don't know how to focus the scene after this
                # check DetectorImage shortcut
                self.assertTrue(sim.last_det_image is None)
                send_key(sim, "d")
                self._wait_for_idle(sim)
                self.assertTrue(sim.last_det_image is not None)

        sim.debug(interact, args=(sim,))
        self.raise_thread_exceptions()
   
    # os test because clipboard is system dependent
    @pytest.mark.os
    @pytest.mark.install
    @pytest.mark.slow
    @pytest.mark.gui1
    def test_run_command(self):
        """test command setting and sending as well as automatic replotting (also tests TraceGUI.smart_replot()"""

        RT = tracing_geometry()
        sim = TraceGUI(RT)

        def send(cmd):
            sim.command_window.cmd = cmd
            self._do_in_main(sim.command_window.send_command)
            self._wait_for_idle(sim)

        def interact(sim):
            with self._try(sim):

                state = RT.rays.crepr()
                send("GUI.replot()")
                self.assertFalse(state == RT.rays.crepr())  # check if raytraced

                send("GUI.detector_image()")
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
                id0 = id(sim._plot._ray_plot)
                send("RT.remove(APL[0])")
                id1 = id(sim._plot._ray_plot)
                self.assertEqual(len(RT.apertures), 0)  # check if apertures empty after removal
                self.assertEqual(len(sim._plot._aperture_plots), 0)  # check if aperture plot is removed
                self.assertFalse(state == RT.rays.crepr())  # check if raytraced
                self.assertNotEqual(id0, id1)
          
                # check if removing a marker only redraws markers and no retracing etc.
                ml0 = sim._plot._marker_plots.copy()
                id0 = id(sim._plot._ray_plot)
                send("RT.remove(ML[0])")
                id1 = id(sim._plot._ray_plot)
                ml1 = sim._plot._marker_plots.copy()
                self.assertEqual(id0, id1)
                self.assertNotEqual(ml0, ml1)

                # check if automatic replot/retrace setting is applied
                self._do_in_main(sim.open_command_window)
                sim.command_window.automatic_replot = False
                id0 = id(sim._plot._ray_property_dict["p"])
                send("RT.remove(LL[0])")
                # time.sleep(5)
                id1 = id(sim._plot._ray_property_dict["p"])
                self.assertEqual(id0, id1)  # same ID -> not retraced
                
                sim.command_window.automatic_replot = True
                self._wait_for_idle(sim)

                # Misc tests

                sim.command_window._replot()  # replot button
                self._wait_for_idle(sim)

                self._do_in_main(sim.open_command_window)
                self._wait_for_idle(sim)
                self._do_in_main(sim.open_command_window)  # reopen
                self._wait_for_idle(sim)
                
                send("")  # check empty command

                send("throw RuntimeError()")  # check if exceptions are handled

                # send empty command using command window
                sim.command_window.cmd = ""
                self._do_in_main(sim.command_window.send_command)
                self._wait_for_idle(sim)

                # send actual command using command window
                sim.command_window.cmd = "self.replot()"
                self._do_in_main(sim.command_window.send_command)
                self._wait_for_idle(sim)

                # resend command, history does not get updated with same command
                self._do_in_main(sim.command_window.send_command)
                self._wait_for_idle(sim)
              
                # toggle dark mode
                self._do_in_main(ot.global_options.__setattr__, "ui_dark_mode", not ot.global_options.ui_dark_mode)
                time.sleep(0.5)

                # clear history
                self._do_in_main(sim.command_window.clear_history)
                self._wait_for_idle(sim)
                self.assertEqual(sim.command_window.history, [])

                # check if setting and getting clipboard works globally
                clipboard = QtGui.QApplication.clipboard()
                clipboard.clear(mode=QtGui.QClipboard.Clipboard)
                clipboard.setText("a", mode=QtGui.QClipboard.Clipboard)
                can_copy = clipboard.text(mode=QtGui.QClipboard.Clipboard) == "a"

                if not can_copy:
                    warnings.warn("Copying to clipboard failed. Can by a system or library issue.")
                
                # check that empty strings are copied correctly in the command window
                self._do_in_main(sim.command_window.copy_history)
                self._wait_for_idle(sim)
                if can_copy:
                    self.assertEqual(clipboard.text(), "")
                   
                # check if full history is copied
                sim.command_window.cmd = "self.replot()"
                self._do_in_main(sim.command_window.send_command)
                self._wait_for_idle(sim)
                sim.command_window.cmd = "a=5"
                self._do_in_main(sim.command_window.send_command)
                self._wait_for_idle(sim)
                self._do_in_main(sim.command_window.copy_history)
                self._wait_for_idle(sim)
                time.sleep(0.5)  # somehow needed, maybe system handles clipboard with delay
                if can_copy:
                    self.assertEqual(clipboard.text(), "self.replot()\na=5\n")

        sim.debug(interact, args=(sim,))
        self.raise_thread_exceptions()


    # @pytest.mark.os
    @pytest.mark.gui2
    @pytest.mark.skipif(os.getenv("GITHUB_ACTIONS") == "true", reason="Issues with headless display in github actions")
    def test_resize(self):
        """
        this test checks if
        * some UI properties get resized correctly
        * no exceptions while doing so
        * resizing to initial state resizes everything back to normal
        """

        RT = tracing_geometry()
        sim = TraceGUI(RT)

        def interact(sim):
            with self._try(sim):

                self._set_in_main(sim, "coloring_mode", "Power")  # shows a side bar, that also needs to be rescaled
                self._wait_for_idle(sim)

                SceneSize0 = sim._plot._scene_size.copy()
                SceneSizei0 = sim.scene.scene_editor.interactor.size.copy()
                Window = sim.scene.scene_editor._content.window()

                # properties before resizing
                ff = sim._plot._axis_plots[0][0].axes.font_factor
                zoom = sim._plot._orientation_axes.widgets[0].zoom
                pos2 = sim._plot._ray_plot.parent.scalar_lut_manager.scalar_bar_representation.position2

                qsize = Window.size()
                ss0 = np.array([qsize.width(), qsize.height()])
                ss1 = ss0 * 1.2
                ss2 = ss1 / 1.1

                # time to wait after resizing
                dt = 0.5

                # enlarge
                self._do_in_main(Window.resize, *ss1.astype(int))
                time.sleep(dt)  # how to check how much time it takes?

                # check if window was actually resized
                qsize2 = sim.scene.scene_editor._content.window().size()
                self.assertFalse(np.all(ss0 == np.array([qsize2.width(), qsize2.height()])))

                # check if scene was actually resized
                SceneSizei1 = sim.scene.scene_editor.interactor.size.copy()
                self.assertFalse(np.all(SceneSizei0 == SceneSizei1))

                # check if sizing parents exist
                self.assertFalse(sim.scene.scene_editor is None or sim.scene.scene_editor.interactor\
                        is None or sim.scene._renwin is None)

                # check if scale properties changed
                self.assertNotAlmostEqual(sim._plot._scene_size[0], SceneSize0[0])  # scene size variable changed
                self.assertNotAlmostEqual(sim._plot._scene_size[1], SceneSize0[1])  # scene size variable changed
                self.assertNotAlmostEqual(ff, sim._plot._axis_plots[0][0].axes.font_factor)
                self.assertNotAlmostEqual(zoom, sim._plot._orientation_axes.widgets[0].zoom)
                self.assertNotAlmostEqual(pos2[0],
                                          sim._plot._ray_plot.parent.scalar_lut_manager.scalar_bar_representation.position2[0])
                self.assertNotAlmostEqual(pos2[1],
                                          sim._plot._ray_plot.parent.scalar_lut_manager.scalar_bar_representation.position2[1])

                self._do_in_main(Window.resize, *ss2.astype(int))
                time.sleep(dt)
                self._do_in_main(Window.showFullScreen)
                time.sleep(dt)
                self._do_in_main(Window.showMaximized)
                time.sleep(dt)
                self._do_in_main(Window.showMinimized)
                time.sleep(dt)
                self._do_in_main(Window.showNormal)
                time.sleep(dt)
                self._do_in_main(Window.resize, *ss0.astype(int))
                time.sleep(dt)
               
                # check if scale properties are back at their default state
                self.assertAlmostEqual(sim._plot._scene_size[0], SceneSize0[0], delta=2)
                self.assertAlmostEqual(sim._plot._scene_size[1], SceneSize0[1], delta=2)
                self.assertAlmostEqual(ff, sim._plot._axis_plots[0][0].axes.font_factor, delta=0.001)
                self.assertAlmostEqual(zoom, sim._plot._orientation_axes.widgets[0].zoom, delta=0.001)
                self.assertAlmostEqual(pos2[0],
                                       sim._plot._ray_plot.parent.scalar_lut_manager.scalar_bar_representation.position2[0], delta=0.001)
                self.assertAlmostEqual(pos2[1],
                                       sim._plot._ray_plot.parent.scalar_lut_manager.scalar_bar_representation.position2[1], delta=0.001)


                # coverage test: delete orientation:axes and resize
                sim._plot._orientation_axes = None
                self._do_in_main(Window.resize, *ss2.astype(int))
                time.sleep(dt)
                
        sim.debug(interact, args=(sim,))
        self.raise_thread_exceptions()
        
    @pytest.mark.gui2
    def test_non_2d(self):
        """
        initially there were problems with non-plotable "surfaces" like Line and Point
        check if initial plotting, replotting and trying to color the parent sources (coloring_type=...)
        works correctly
        """
       
       # make raytracer
        RT = ot.Raytracer(outline=[-5, 5, -5, 5, -5, 60])

        # add Raysource
        RSS = ot.Point()
        RS = ot.RaySource(RSS, divergence="Isotropic", spectrum=ot.presets.light_spectrum.d65,
                          pos=[0, 0, 0], s=[0, 0, 1], div_angle=75)
        RT.add(RS)
        
        # add Raysource2
        RSS = ot.Line(r=3)
        RS2 = ot.RaySource(RSS, divergence="None", spectrum=ot.presets.light_spectrum.d65,
                           pos=[0, 0, 0], s=[0, 0, 1])
        RT.add(RS2)

        sim = TraceGUI(RT, coloring_mode="Wavelength")

        def interact(sim):
            with self._try(sim):
                self._set_in_main(sim, "coloring_mode", "Wavelength")
                self._wait_for_idle(sim)
                self._set_in_main(sim, "plotting_mode", "Points")
                self._wait_for_idle(sim)
                self._do_in_main(sim.replot)

        sim.debug(interact, args=(sim,))
        self.raise_thread_exceptions()

    @pytest.mark.slow
    @pytest.mark.gui3
    def test_additional_coverage_1(self):
        """additionial coverage tests"""

        RT = tracing_geometry()
        sim = TraceGUI(RT)

        def interact(sim):
            with self._try(sim):
                # check handling of profile values outside the image
                self._set_in_main(sim, "profile_position", 500)
                self._do_in_main(sim.source_profile)
                self._wait_for_idle(sim)
                self._do_in_main(sim.source_profile)
                self._wait_for_idle(sim)

                # refraction index box replotting with vertical labels
                self._set_in_main(sim, "vertical_labels", True)
                self._do_in_main(sim.replot)
                self._wait_for_idle(sim)

                # mock no_pol mode, check if coloring type handles this
                RT.no_pol = True
                self._set_in_main(sim, "coloring_mode", "Polarization yz")
                self._wait_for_idle(sim)

                # make sure x polarization gets plotted at least once
                RT.no_pol = False
                RT.ray_sources[0].polarization = "x"
                self._set_in_main(sim, "coloring_mode", "Polarization yz")
                self._wait_for_idle(sim)
                self._do_in_main(sim.retrace)
                self._wait_for_idle(sim)

                # skip on InitScene
                sim._status["InitScene"] = 1
                self._do_in_main(sim._change_minimalistic_view)
                time.sleep(2)  # make sure it gets executed, can't call wait_for_idle since we set InitScene
                sim._status["InitScene"] = 0
                self._wait_for_idle(sim)

                # check if rays are removed correctly in replot()
                assert len(RT.ray_sources) and RT.rays.N and sim._plot._ray_plot is not None
                self._do_in_main(sim.run_command, "RT.clear()")
                self._wait_for_idle(sim)
                self.assertTrue(sim._plot._ray_plot is None)
                self.assertFalse(sim._plot._ray_property_dict)

        sim.debug(interact, args=(sim,))
        self.raise_thread_exceptions()
    
    @pytest.mark.slow
    @pytest.mark.gui3
    def test_additional_coverage_2(self):
        """additionial coverage tests"""

        RT = tracing_geometry()
        sim = TraceGUI(RT)

        def interact(sim):
            with self._try(sim):

                # delete one raysource plot but still try to color it
                sim._plot._ray_source_plots = sim._plot._ray_source_plots[:-1]
                self._do_in_main(sim.retrace)
                self._wait_for_idle(sim)

                # only time replotting the orientation axes, since normally they don't need to change
                self._do_in_main(sim._plot.plot_orientation_axes)
                self._wait_for_idle(sim)

                # assign some weird outline so plot axes finds no optimal number of labels 
                RT.outline = [-123., 10*np.pi, -12, 12.468786, -500.46504654065/np.pi, 124.456]
                self._do_in_main(sim.replot)
                self._wait_for_idle(sim)

                # open property browser with cardinal points, although we now don't have rotational symmetry
                RT.lenses[0].move_to([0, 0.02, 2])
                # coverage for property browser: must unroll single ndarray element
                sim.abc = np.array([1.]) 
                self._do_in_main(sim.open_property_browser)
                self._wait_for_idle(sim)
                self._do_in_main(sim.open_property_browser) # reopen
                self._wait_for_idle(sim)

                # special case where we don't need to retrace or redraw the rays
                h = RT.property_snapshot()
                cmp = RT.compare_property_snapshot(h, h)
                cmp["Any"] = True
                cmp["Detectors"] = True
                self._do_in_main(sim.replot, cmp)
                self._wait_for_idle(sim)
              
                sim._plot._filter_plots[0] = (sim._plot._filter_plots[0][0], None, None, None,
                                        sim._plot._filter_plots[0][4])
                sim._plot._aperture_plots[0] = (sim._plot._aperture_plots[0][0], None, None, None, None)
                self._wait_for_idle(sim)

                # check exception handling of raytracer actions
                # for this we assign wrong detector and source indices
                # so the actions throw
                # however this should not interrupt the program or lead to states
                # where status flags are set incorrectly
                sim._det_ind = 50
                sim._source_ind = 50
                self._set_in_main(sim, "focus_search_single_source", True)
                self._set_in_main(sim, "detector_image_single_source", True)
                self._do_in_main(sim.detector_image)
                self._wait_for_idle(sim)
                self._do_in_main(sim.detector_profile)
                self._wait_for_idle(sim)
                self._do_in_main(sim.detector_spectrum)
                self._wait_for_idle(sim)
                self._do_in_main(sim.source_image)
                self._wait_for_idle(sim)
                self._do_in_main(sim.source_profile)
                self._wait_for_idle(sim)
                self._do_in_main(sim.source_spectrum)
                self._wait_for_idle(sim)
                self._do_in_main(sim.move_to_focus)
                self._wait_for_idle(sim)
                sim._det_ind = 0
                sim._source_ind = 0

                # remove old ray plot when an error tracing occurred
                RT.lenses[0].move_to(RT.lenses[1].pos)
                self._do_in_main(sim.replot)
                self._wait_for_idle(sim)
                self.assertTrue(sim._plot._ray_plot is None)
                
                # still works when some objects are none
                sim._plot._orientation_axes = None
                tn4 = (None, None, None, None)
                sim._plot._index_box_plots[0] = tn4
                sim._plot._ray_source_plots[0] = tn4
                sim._plot._axis_plots[0] = tn4
                self._do_in_main(sim._change_minimalistic_view)
                self._wait_for_idle(sim)

        sim.debug(interact, args=(sim,))
        self.raise_thread_exceptions()

    @pytest.mark.gui2
    def test_run(self):
        RT = tracing_geometry()
        sim = TraceGUI(RT)
        sim._exit = True  # leads to run() exiting directly after load
        sim.run()
   
    @pytest.mark.gui2
    @pytest.mark.timeout(30)  # for some reasons failed checks lead to AssertionErrors 
    # that aren't handled and the program is blocked. See NOTE 5 below
    def test_control(self):

        RT = tracing_geometry()
        
        def automated(GUI):

            for yp in np.linspace(0, 4, 3):

                # store rays and ray_plot id
                rays0 = RT.rays.p_list
                ray_plot_id0 = id(GUI._plot._ray_plot)
                
                # move source and replot
                RT.ray_sources[0].move_to([0, yp, -3])
                GUI.replot()
                
                # store rays and ray_plot id afterwards
                rays1 = RT.rays.p_list
                ray_plot_id1 = id(GUI._plot._ray_plot)

                # rays and ray_plot id must have changed. Also check if sequential mode.
                # NOTE 5 when these fail, they block the main thread it seems and the program stops execution
                self.assertNotEqual(ray_plot_id0, ray_plot_id1)
                self.assertTrue(np.any(rays0 != rays1))
                self.assertEqual(GUI._sequential, True)
                
                time.sleep(0.2)
       
            GUI.close()

        sim = TraceGUI(RT, ray_count=30000, high_contrast=False, hide_labels=True, minimalistic_view=True)
        sim.control(func=automated, args=(sim,))
        self.raise_thread_exceptions()

        # exceptions
        self.assertRaises(TypeError, sim.control, func=3)
        self.assertRaises(TypeError, sim.control, func=self._wait_for_idle, args=3)
        self.assertRaises(TypeError, sim.control, func=self._wait_for_idle, kwargs=3)

    @pytest.mark.slow
    @pytest.mark.gui2
    def test_rt_fault_no_rays(self):
        """check handling of functionality when there is an geometry fault in the raytracer 
        and therefore no rays are simulated and plotted"""

        # make raytracer
        RT = ot.Raytracer(outline=[-10, 10, -10, 10, -10, 60])

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
                self._do_in_main(sim.retrace)
                self._wait_for_idle(sim)

                # try actions with missing rays
                self._do_in_main(sim.detector_image)
                self._wait_for_idle(sim)
                self._do_in_main(sim.move_to_focus)
                self._wait_for_idle(sim)
                self._do_in_main(sim.source_image)
                self._wait_for_idle(sim)
                self._set_in_main(sim, "ray_opacity", 0.05)
                self._wait_for_idle(sim)
                self._set_in_main(sim, "ray_width", 5)
                self._wait_for_idle(sim)
                self._set_in_main(sim, "ray_count", 100000)
                self._wait_for_idle(sim)
                self._set_in_main(sim, "rays_visible", 3000)
                self._wait_for_idle(sim)
                self._set_in_main(sim, "minimalistic_view", True)
                self._wait_for_idle(sim)
                self._set_in_main(sim, "coloring_mode", "Power")
                self._wait_for_idle(sim)
                self._set_in_main(sim, "plotting_mode", "Points")
                self._wait_for_idle(sim)
                self._do_in_main(sim.source_spectrum)
                self._wait_for_idle(sim)
                self._do_in_main(sim.detector_spectrum)
                self._wait_for_idle(sim)
                self._do_in_main(sim.open_property_browser)  # should lead to collision error
                self._wait_for_idle(sim)
                self._do_in_main(sim.replot)
                self._wait_for_idle(sim)

                # leave only one lens, now it can be traced
                # this also removes the old fault markers
                RT.remove(RT.lenses[1:])
                self._do_in_main(sim.replot)
                time.sleep(0.5)
                self._wait_for_idle(sim)
                self.assertTrue(RT.rays.N > 0)  # rays traced
                self.assertEqual(len(sim._plot._fault_markers), 0)  # no fault_marker in GUI
                self.assertEqual(len(RT.markers), 0)  # no fault markers in RT

        sim = TraceGUI(RT)
        sim.debug(interact, args=(sim,))
        self.raise_thread_exceptions()

    @pytest.mark.gui1
    def test_point_marker(self):
        """test marker plotting, replotting, removal and properties in scene"""

        RT = tracing_geometry()
        RT.remove(RT.markers)
        
        def interact(sim):
            with self._try(sim):

                # marker 1, default size
                RT.add(ot.PointMarker("Test1", [0., 0., 0]))
                self._do_in_main(sim.replot)
                self._wait_for_idle(sim)
                self.assertEqual(len(sim._plot._marker_plots), 1)  # element was added
                self.assertTrue(np.allclose(sim._plot._marker_plots[0][0].actor.actor.center, 0))  # check position

                # marker 2, enlarged
                RT.add(ot.PointMarker("Test2", [0., 1., 5.2], text_factor=2, marker_factor=2))
                self._do_in_main(sim.replot)
                self._wait_for_idle(sim)
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
                self._do_in_main(sim.run_command, "RT.remove(ML[-1])") # also checks that alias ML exists
                self._wait_for_idle(sim)
                self.assertEqual(len(RT.markers), 1)  # element was removed in raytracer
                self.assertEqual(len(sim._plot._marker_plots), 1)  # element was removed in scene

                # first marker now should have no crosshair shown
                a = sim._plot._marker_plots[0]
                self.assertFalse(a[0].visible)

        sim = TraceGUI(RT)
        sim.debug(interact, args=(sim,))
        self.raise_thread_exceptions()
    
    @pytest.mark.slow
    @pytest.mark.gui1
    def test_line_marker(self):
        """test marker plotting, replotting, removal and properties in scene"""

        RT = tracing_geometry()
        RT.remove(RT.markers)
        
        def interact(sim):
            with self._try(sim):

                # marker 1, default size
                RT.add(ot.LineMarker(r=5, desc="Test1", pos=[0., 0., 0]))
                self._do_in_main(sim.replot)
                self._wait_for_idle(sim)
                self.assertEqual(len(sim._plot._line_marker_plots), 1)  # element was added
                self.assertTrue(np.allclose(sim._plot._line_marker_plots[0][0].actor.actor.center, 0))  # check position

                # marker 2, enlarged
                RT.add(ot.LineMarker(r=5, angle=-20, desc="Test2", pos=[0., 1., 5.2], text_factor=2, line_factor=2))
                self._do_in_main(sim.replot)
                self._wait_for_idle(sim)
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
                self._do_in_main(sim.run_command, "RT.remove(ML[-1])") # also checks that alias ML exists
                self._wait_for_idle(sim)
                self.assertEqual(len(RT.markers), 1)  # element was removed in raytracer
                self.assertEqual(len(sim._plot._line_marker_plots), 1)  # element was removed in scene

        sim = TraceGUI(RT)
        sim.debug(interact, args=(sim,))
        self.raise_thread_exceptions()

    @pytest.mark.slow
    @pytest.mark.gui2
    @pytest.mark.skipif(os.getenv("XDG_SESSION_DESKTOP") == "KDE" and os.getenv("XDG_SESSION_TYPE") == "wayland" \
                        and pytest_xvfb.xvfb_instance is None,
                        reason="KDE Wayland wants remote authentication. Use X11 or run with xvfb")
    @pytest.mark.skipif(os.getenv("GITHUB_ACTIONS") == "true", 
                        reason="Issues with headless display and input in github actions")
    def test_picker(self):
        """
        test picker interaction in the scene
        this is done without keypress and mouse click simulation,
        so it is not tested, if the scene reacts to those correctly
        only the pick handling is checked here
        """

        RT = tracing_geometry()
        
        def interact(sim):
            with self._try(sim):
        
                # check that pyautogui works
                pyautogui.keyDown("shiftleft")
                time.sleep(0.3)
                self.assertTrue(sim.scene.interactor.shift_key)
                pyautogui.keyUp("shiftleft")
                time.sleep(0.3)

                # change to z+ view, so there are rays at the middle of the scene
                self._do_in_main(sim.scene.z_plus_view)
                self._wait_for_idle(sim)
           
                default_text = sim._plot._ray_text.text  # save default text for comparison

                # ray picked -> show default infos
                self._do_in_main(sim._plot._ray_picker.pick, sim._plot._scene_size[0] / 2,
                                 sim._plot._scene_size[1] / 2, 0, sim.scene.renderer)
                self._wait_for_idle(sim)
                time.sleep(0.2)  # delay so a human user can check the text
                text1 = sim._plot._ray_text.text
                self.assertNotEqual(text1, default_text)  # shows a ray info text
                self.assertTrue(sim._plot._ray_highlight_plot.visible)  # show highlighted ray
               
                # ray picked -> show verbose info
                pyautogui.keyDown("shiftleft")
                time.sleep(0.3)
                self._do_in_main(sim._plot._ray_picker.pick, sim._plot._scene_size[0] / 2,
                                 sim._plot._scene_size[1] / 2, 0, sim.scene.renderer)
                self._wait_for_idle(sim)
                time.sleep(0.2)
                text2 = sim._plot._ray_text.text
                self.assertNotEqual(text2, default_text)  # no default text
                self.assertNotEqual(text1, text2)  # not the old text
                self.assertTrue(sim._plot._ray_highlight_plot.visible)  # show highlighted ray
               
                # no ray picked -> default text
                self._do_in_main(sim._plot._ray_picker.pick, 0, 0, 0, sim.scene.renderer)
                self._wait_for_idle(sim)
                time.sleep(0.2)
                text2 = sim._plot._ray_text.text
                self.assertEqual(text2, default_text)  # shows default text
                self.assertFalse(sim._plot._ray_highlight_plot.visible)  # don't show any highlighted ray
              
                # redraw ray info only if it is the default one
                self._set_in_main(sim, "minimalistic_view", True)
                self._wait_for_idle(sim)
                self._set_in_main(sim, "minimalistic_view", False)
                self._wait_for_idle(sim)

                # remove crosshair and pick a ray
                sim._plot._crosshair = None
                self._do_in_main(sim._plot._ray_picker.pick, sim._plot._scene_size[0] / 2,
                                 sim._plot._scene_size[1] / 2, 0, sim.scene.renderer)
                self._wait_for_idle(sim)

                # remove crosshair and pick no ray
                self._do_in_main(sim._plot._ray_picker.pick, 0, 0, 0, sim.scene.renderer)
                self._wait_for_idle(sim)
                
                # restore crosshair
                self._do_in_main(sim._plot.init_crosshair)
                self._wait_for_idle(sim)
                
                # we have an extra picker sim._space_picker for right+clicking in the scene,
                # but I don't know how to make the Picker.pick() function pick with a right click
                # so currently we overide the RayPicker with the onSpacePick method
                # do via command string, so it is guaranteed to run in main thread
                self._do_in_main(sim.run_command, "self._plot._ray_picker = self.scene.mlab.gcf().on_mouse_pick("
                                         "self._plot._on_space_pick, button='Left')")
                self._wait_for_idle(sim)

                # space picked -> show coordinates
                pyautogui.keyUp("shiftleft")
                time.sleep(0.3)
                self._do_in_main(sim._plot._ray_picker.pick, sim._plot._scene_size[0] / 3,
                                 sim._plot._scene_size[1] / 3, 0, sim.scene.renderer)
                self._wait_for_idle(sim)
                time.sleep(0.2)
                text3 = sim._plot._ray_text.text
                self.assertNotEqual(text3, default_text)  # not the default text
                self.assertNotEqual(text3, text2)  # not the old text
                self.assertFalse(sim._plot._ray_highlight_plot.visible)  # don't show any highlighted ray
                
                # valid space picked with shift -> move detector
                pyautogui.keyDown("shiftleft")
                time.sleep(0.3)
                old_pos = RT.detectors[0].pos
                self._do_in_main(sim._plot._ray_picker.pick, sim._plot._scene_size[0] / 3,
                                 sim._plot._scene_size[1] / 3, 0, sim.scene.renderer)
                self._wait_for_idle(sim)
                time.sleep(0.2)
                text4 = sim._plot._ray_text.text
                self.assertEqual(text4, default_text)  # not the default text
                self.assertNotEqual(RT.detectors[0].pos[2], old_pos[2])
                self.assertFalse(sim._plot._ray_highlight_plot.visible)  # don't show any highlighted ray
                
                # space outside outline picked with shift -> move detector
                self._do_in_main(sim.scene.y_plus_view)
                self._do_in_main(sim._plot._ray_picker.pick, 0, 0, 0, sim.scene.renderer)
                self._wait_for_idle(sim)
                time.sleep(0.2)
                text4 = sim._plot._ray_text.text
                self.assertEqual(text4, default_text)  # not the default text
                self.assertEqual(RT.detectors[0].pos[2], RT.outline[4])  # detector moved to the outline beginning
                self.assertFalse(sim._plot._ray_highlight_plot.visible)  # don't show any highlighted ray
              
                # remove detectors and try to move the detector using the space pick action
                RT.detectors = []
                self._do_in_main(sim.replot)
                self._wait_for_idle(sim)
                self._do_in_main(sim._plot._ray_picker.pick, 0, 0, 60, sim.scene.renderer)
                self._wait_for_idle(sim)

                # remove crosshair and pick
                sim._plot._crosshair = None
                self._do_in_main(sim._plot._ray_picker.pick, 0, 0, 60, sim.scene.renderer)
                self._wait_for_idle(sim)

                # release shift key
                pyautogui.keyUp("shiftleft")
                time.sleep(0.3)
                
                # remove crosshair and pick without shift key
                self._do_in_main(sim._plot._ray_picker.pick, 0, 0, 60, sim.scene.renderer)
                self._wait_for_idle(sim)
                
                # restore crosshair
                self._do_in_main(sim._plot.init_crosshair)
                self._wait_for_idle(sim)

                # test case where RayText is missing
                sim._plot._ray_text = None
                self._do_in_main(sim._plot._on_ray_pick)
                self._wait_for_idle(sim)

        sim = TraceGUI(RT)
        sim.debug(interact, args=(sim,))
        self.raise_thread_exceptions()

    @pytest.mark.slow
    @pytest.mark.gui3
    @pytest.mark.skipif(os.getenv("XDG_SESSION_DESKTOP") == "KDE" and os.getenv("XDG_SESSION_TYPE") == "wayland" \
                        and pytest_xvfb.xvfb_instance is None,
                        reason="KDE Wayland wants remote authentication. Use X11 or run with xvfb")
    @pytest.mark.skipif(os.getenv("GITHUB_ACTIONS") == "true", 
                        reason="Issues with headless display and input in github actions")
    def test_picker_coverage(self):

        RT = ot.Raytracer(outline=[-10, 10, -10, 10, 0, 10])
        
        def interact(sim):
            def pick():    
                self._wait_for_idle(sim)
                self._do_in_main(sim._plot._ray_picker.pick, sim._plot._scene_size[0] / 2,
                                 sim._plot._scene_size[1] / 2, 0, sim.scene.renderer)
                self._wait_for_idle(sim)
                time.sleep(0.2)  # delay so a human user can check the text

            def pick_shift_combs():
                pick()
                pyautogui.keyDown("shiftleft")
                time.sleep(0.2)
                pick()
                pyautogui.keyUp("shiftleft")
                time.sleep(0.2)

            with self._try(sim):
                # change to z+ view, so there are rays at the middle of the scene
                self._do_in_main(sim.scene.z_plus_view)
                self._wait_for_idle(sim)

                # pick RaySource Point
                RT.add(ot.RaySource(ot.Point(), pos=[0, 0, 0], s_sph=[20, 0]))
                self._do_in_main(sim.replot)
                pick_shift_combs()

                # pick RaySource Surface
                RT.remove(RT.ray_sources[0])
                RT.add(ot.RaySource(ot.CircularSurface(r=0.1), pos=[0, 0, 0], s_sph=[20, 0]))
                self._do_in_main(sim.replot)
                pick_shift_combs()

                # pick last ray section point (at outline)
                RT.remove(RT.ray_sources[0])
                RT.add(ot.RaySource(ot.Point(), pos=[-5, 0, 0], s=[5, 0, 10]))
                self._do_in_main(sim.replot)
                pick_shift_combs()
                
                # pick element in between
                RT.remove(RT.ray_sources[0])
                RT.add(ot.RaySource(ot.Point(), pos=[-5, 0, 0], s=[10, 0, 10]))
                RT.add(ot.Filter(ot.CircularSurface(r=0.1), pos=[0, 0, 5], 
                                 spectrum=ot.TransmissionSpectrum("Constant", val=1)))
                self._do_in_main(sim.replot)
                pick_shift_combs()

        sim = TraceGUI(RT)
        sim.debug(interact, args=(sim,))
        self.raise_thread_exceptions()

    @pytest.mark.gui2
    def test_picker_command(self):
        """test calling of picker commands pick_ray, pick_ray_section, reset_picking"""

        RT = tracing_geometry()
        
        def interact(sim):
            with self._try(sim):

                # TypeErrors
                self.assertRaises(TypeError, sim._plot.pick_ray_section, [])  # invalid index
                self.assertRaises(TypeError, sim._plot.pick_ray_section, 0, [])  # invalid section
                
                # ValueErrors
                self.assertRaises(ValueError, sim._plot.pick_ray_section, -2, 0)  # negative indices not supported
                self.assertRaises(ValueError, sim._plot.pick_ray_section, 20000000, 0)  # index too large
                self.assertRaises(ValueError, sim._plot.pick_ray_section, 0, -2)  # negative sections not supported
                self.assertRaises(ValueError, sim._plot.pick_ray_section, 0, 20000000)  # section too large

                # pick just the ray, check that highlight plot is shown
                self._do_in_main(sim.pick_ray, 0)
                self._wait_for_idle(sim)
                self.assertTrue(sim._plot._ray_highlight_plot.visible)
                
                # pick a ray section, check that crosshair, highlight plot and info text are shown
                self._do_in_main(sim.pick_ray_section, 0, 5)
                self._wait_for_idle(sim)
                self.assertTrue(sim._plot._ray_highlight_plot.visible)
                self.assertTrue(sim._plot._crosshair.visible)
                self.assertTrue(sim._plot._ray_text.text != "")
                
                # reset picking, check that things are hidden
                self._do_in_main(sim.reset_picking)
                self._wait_for_idle(sim)
                self.assertFalse(sim._plot._ray_highlight_plot.visible)
                self.assertFalse(sim._plot._crosshair.visible)
                self.assertFalse(sim._plot._ray_text.text != "")

                # check if detail parameter is applied by comparing texts
                text = ""
                for detail in [False, True]:
                    self._do_in_main(sim.pick_ray_section, 100, 6, detail)
                    self._wait_for_idle(sim)
                    text2 = sim._plot._ray_text.text
                    self.assertNotEqual(text, text2)
                    text2 = text

                # coverage: select section and change contrast (affects crosshair and rayhighlight plot)
                self._do_in_main(sim.pick_ray_section, 0, 5)
                self._wait_for_idle(sim)
                self._set_in_main(sim, "high_contrast", True)
                self._wait_for_idle(sim)
                self._set_in_main(sim, "high_contrast", False)
                self._wait_for_idle(sim)

                # coverage: pick but ray property dict missing
                # raises exception, but gets caught
                sim._plot._ray_property_dict = []
                self._do_in_main(sim.pick_ray_section, 0, 5)

        sim = TraceGUI(RT)
        sim.debug(interact, args=(sim,))
        self.raise_thread_exceptions()

    @pytest.mark.gui3
    def test_select_rays(self):
        """
        tests for manually selecting rays. Tests TraceGUI.select_rays and TraceGUI.ray_selection.
        Therefore also checks ScenePlotting.select_rays, ScenePlotting.ray_selection, TraceGUI.replot_rays
        """
        
        RT = tracing_geometry()
        N = 500000
        RT.trace(N)
        
        def interact(sim):
            with self._try(sim):
                
                # test: get mask for actually displayed selection
                # check if set values match rays_visible
                selection = sim.ray_selection
                self.assertEqual(np.count_nonzero(sim.ray_selection), sim.rays_visible)
                self.assertTrue(np.all(sim.ray_selection == sim._plot.ray_selection))

                # exceptions
                id0 = id(sim._plot._ray_plot)
                self.assertRaises(ValueError, sim._plot.select_rays, np.ones(100, dtype=bool))
                # ^-- mask size not same as ray number
                self.assertRaises(ValueError, sim._plot.select_rays, np.ones((N, 2), dtype=bool))  # mask not 1D
                self.assertRaises(ValueError, sim._plot.select_rays, np.ones(N, dtype=np.uint8))  # mask not bool
                self.assertRaises(ValueError, sim._plot.select_rays, np.zeros(N, dtype=bool))  # no elements set
                self.assertRaises(ValueError, sim._plot.select_rays, np.ones(N, dtype=bool), -1)  # max_show negative
                self.assertRaises(TypeError, sim.select_rays, 1)  # invalid mask
                self.assertRaises(TypeError, sim.select_rays, np.ones(N, dtype=bool), [])  # invalid max_show
                id1 = id(sim._plot._ray_plot)
                self.assertEqual(id0, id1)  # rays were not replotted

                # test cases below. Each test tests if ray_selection array and rays_visible trait are set correctly
                # additionally ray_plots are compared, as they also must have been updated

                # test case 1: mask provided, but number of displayed rays is limited
                mask = (sim.raytracer.rays.wl_list >= 400) & (sim.raytracer.rays.wl_list <= 750)
                assert np.count_nonzero(mask) > sim._plot.MAX_RAYS_SHOWN
                sim.select_rays(mask) # no max_show provided, but might be limited by this function
                self._wait_for_idle(sim)
                self.assertEqual(np.count_nonzero(sim.ray_selection), sim._plot.MAX_RAYS_SHOWN)
                self.assertEqual(sim.rays_visible, sim._plot.MAX_RAYS_SHOWN)
                id2 = id(sim._plot._ray_plot)
                self.assertNotEqual(id1, id2)

                # test case 2: mask and max_show provided
                mask = sim.raytracer.rays.p_list[:, :, 0] > 0
                sim.select_rays(mask[:, 0], 2000) 
                self._wait_for_idle(sim)
                self.assertEqual(np.count_nonzero(sim.ray_selection), 2000)
                self.assertEqual(sim.rays_visible, 2000)
                id3 = id(sim._plot._ray_plot)
                self.assertNotEqual(id2, id3)
                
                # test case 3: mask provided, number of selected rays is below limit
                mask = (sim.raytracer.rays.wl_list >= 400) & (sim.raytracer.rays.wl_list <= 405)
                assert np.count_nonzero(mask) < sim._plot.MAX_RAYS_SHOWN
                sim.select_rays(mask)
                self._wait_for_idle(sim)
                self.assertEqual(np.count_nonzero(sim.ray_selection), np.count_nonzero(mask))
                self.assertEqual(sim.rays_visible, np.count_nonzero(mask))
                id4 = id(sim._plot._ray_plot)
                self.assertNotEqual(id3, id4)
                
                # test case 4: mask provided, max_show above limit
                mask = (sim.raytracer.rays.wl_list >= 400) & (sim.raytracer.rays.wl_list <= 750)
                assert np.count_nonzero(mask) > sim._plot.MAX_RAYS_SHOWN
                sim.select_rays(mask, sim._plot.MAX_RAYS_SHOWN+5)
                self._wait_for_idle(sim)
                self.assertEqual(np.count_nonzero(sim.ray_selection), sim._plot.MAX_RAYS_SHOWN)
                self.assertEqual(sim.rays_visible, sim._plot.MAX_RAYS_SHOWN)
                id5 = id(sim._plot._ray_plot)
                self.assertNotEqual(id4, id5)
                
                # test case 5: max_show above true number of set values
                mask = (sim.raytracer.rays.wl_list >= 400) & (sim.raytracer.rays.wl_list <= 405)
                assert np.count_nonzero(mask) < sim._plot.MAX_RAYS_SHOWN
                sim.select_rays(mask, sim._plot.MAX_RAYS_SHOWN)
                self._wait_for_idle(sim)
                self.assertEqual(np.count_nonzero(sim.ray_selection), np.count_nonzero(mask))
                self.assertEqual(sim.rays_visible, np.count_nonzero(mask))
                id6 = id(sim._plot._ray_plot)
                self.assertNotEqual(id5, id6)
                
                # coverage: trace and try applying a selection
                self._set_in_main(sim, "ray_count", 1000000)
                while(not sim._status["Tracing"]):
                    time.sleep(0.01)
                sim.select_rays(np.zeros(10, dtype=bool))
                self._wait_for_idle(sim)

                # coverage: try selecting rays while tracing
                sim._status["Tracing"] = 1
                sim.select_rays(mask)
                sim._status["Tracing"] = 0
        
        sim = TraceGUI(RT)
        sim.debug(interact, args=(sim,))
        self.raise_thread_exceptions()

    @pytest.mark.gui3
    def test_volumes(self):
        """plot volume plotting and handling"""

        RT = tracing_geometry()
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
                self._do_in_main(sim.replot)
                self._wait_for_idle(sim, base=1)
                self.assertEqual(len(sim._plot._volume_plots), 4)  # elements were added
                self.assertTrue(sim._plot._volume_plots[0][3] is None)  # no text label for volumes

                # tests that opacity and color is assigned correctly
                sphv.opacity = 0.1
                sphv.color = (1.0, 0.0, 1.0)
                self._do_in_main(sim.replot)
                self._wait_for_idle(sim)
                self.assertEqual(sim._plot._volume_plots[0][0].actor.property.color, sphv.color)
                self.assertEqual(sim._plot._volume_plots[0][0].actor.property.opacity, sphv.opacity)
                
                # checks that automatic replotting works
                self._do_in_main(sim.run_command, "RT.volumes[0].opacity=0.9")
                self._wait_for_idle(sim)
                self.assertEqual(sim._plot._volume_plots[0][0].actor.property.opacity, 0.9)

                # toggle contrast mode
                self._set_in_main(sim, "high_contrast", True)
                self._wait_for_idle(sim)
                # check that custom color gets unset with high contrast
                self.assertEqual(sim._plot._volume_plots[0][0].actor.property.color, sim._plot._volume_color)

                self._set_in_main(sim, "high_contrast", False)
                self._wait_for_idle(sim)
                # check that custom color gets set again without high contrast
                self.assertEqual(sim._plot._volume_plots[0][0].actor.property.color, sphv.color)

        sim = TraceGUI(RT)
        sim.debug(interact, args=(sim,))
        self.raise_thread_exceptions()

    @pytest.mark.slow
    @pytest.mark.gui2
    def test_set_get_camera(self) -> None:
        
        RT = tracing_geometry()
        
        def interact(sim):
            with self._try(sim):
       
                cam = sim.scene.scene.camera

                # test getting and setting by
                # 1. saving the camera
                # 2. changing the view
                # 3. restoring the camera and comparing if view is the same as before
                # this tests that: a) get_camera gets the correct values
                #                   b) set_camera applies the values c) correct view is restored
                view = sim.get_camera()
                pos0, fp0, sc0 = cam.position, cam.focal_point, cam.parallel_scale
                self._do_in_main(sim.set_camera, center=[10, -5, 2], height=250, roll=20, direction=[0, 0, -1])
                self._wait_for_idle(sim)
                self._do_in_main(sim.set_camera, *view)
                self._wait_for_idle(sim)
                self.assertTrue(np.allclose(cam.position, pos0))
                self.assertTrue(np.allclose(cam.focal_point, fp0))
                self.assertEqual(cam.parallel_scale, sc0)

                # roll camera
                # center and height must stay the same, roll changes
                view = sim.get_camera()
                pos0, fp0, sc0 = cam.position, cam.focal_point, cam.parallel_scale
                roll0 = sim.scene.mlab.roll()
                self._do_in_main(sim.set_camera, roll=82)
                self._wait_for_idle(sim)
                self.assertTrue(np.allclose(cam.focal_point, fp0))
                self.assertEqual(cam.parallel_scale, sc0)
                self.assertNotEqual(roll0, sim.scene.mlab.roll())
                
                # change height
                # center and focal_point must stay the same, height changes
                view = sim.get_camera()
                pos0, fp0, sc0 = cam.position, cam.focal_point, cam.parallel_scale
                self._do_in_main(sim.set_camera, height=2)
                self._wait_for_idle(sim)
                self.assertTrue(np.allclose(cam.position, pos0))
                self.assertTrue(np.allclose(cam.focal_point, fp0))
                self.assertNotEqual(sc0, cam.parallel_scale)
                
                # change center
                # height and roll must stay the same, position and focal_point change
                view = sim.get_camera()
                pos0, fp0, sc0 = cam.position, cam.focal_point, cam.parallel_scale
                roll0 = sim.scene.mlab.roll()
                self._do_in_main(sim.set_camera, center=[0, 10, 0])
                self._wait_for_idle(sim)
                self.assertFalse(np.allclose(cam.position, pos0))
                self.assertFalse(np.allclose(cam.focal_point, fp0))
                self.assertTrue(np.allclose(cam.focal_point - fp0, cam.position - pos0))  # shifted by same amount
                self.assertEqual(cam.parallel_scale, sc0)
                self.assertEqual(roll0, sim.scene.mlab.roll())

                # change direction
                # height and center stays the same, roll and position changes
                view = sim.get_camera()
                pos0, fp0, sc0 = cam.position, cam.focal_point, cam.parallel_scale
                roll0 = sim.scene.mlab.roll()
                self._do_in_main(sim.set_camera, direction=[0, 1, 1])
                self._wait_for_idle(sim)
                self.assertFalse(np.allclose(cam.position, pos0))
                self.assertTrue(np.allclose(cam.focal_point, fp0))
                self.assertEqual(cam.parallel_scale, sc0)
                self.assertNotEqual(roll0, sim.scene.mlab.roll())

                # coverage: set everything at once
                self._do_in_main(sim.set_camera, center=[10, -5, 2], height=250, roll=20, direction=[0, 0, -1])
                self._wait_for_idle(sim)

                # exceptions
                self.assertRaises(TypeError, sim.set_camera, center=5)
                self.assertRaises(TypeError, sim.set_camera, height=[])
                self.assertRaises(TypeError, sim.set_camera, direction=5)
                self.assertRaises(TypeError, sim.set_camera, roll=[])
                self.assertRaises(ValueError, sim.set_camera, height=0)
                self.assertRaises(ValueError, sim.set_camera, height=-1)

        sim = TraceGUI(RT)
        sim.debug(interact, args=(sim,))
        self.raise_thread_exceptions()

    @pytest.mark.slow
    @pytest.mark.os
    @pytest.mark.gui2
    def test_screenshot(self) -> None:
        
        RT = tracing_geometry()
        
        def interact(sim):
            with self._try(sim):
             
                self.assertRaises(TypeError, sim.screenshot, 123)  # path not string

                # check different combinations of types and magnification
                for path in ["screenshot.png", "screenshot.jpg"]:
                    for magnification in ["auto", 1, 2]:
                        
                        assert not os.path.exists(path)
                        
                        self._do_in_main(sim.screenshot, path, magnification=magnification)
                        self._wait_for_idle(sim)

                        self.assertTrue(os.path.exists(path))
                        os.remove(path)
                       
                # check handling of size parameter and passing of additional args (like "quality" and "progressive")
                pathj = "screenshot.jpg"
                self._do_in_main(sim.screenshot, pathj, size=(400, 300), quality=90, progressive=1)
                self._wait_for_idle(sim)
                self.assertTrue(os.path.exists(pathj))
                os.remove(pathj)

        sim = TraceGUI(RT)
        sim.debug(interact, args=(sim,))
        self.raise_thread_exceptions()

    @pytest.mark.os
    @pytest.mark.gui1
    def test_custom_ui(self) -> None:
        """test if custom UI elements are correctly created, initialized and execute their action"""

        RT = tracing_geometry()
        
        val1 = [0]
        val2 = [0]
        val3 = [0]

        def set_val1(a):
            val1[0] = a

        def set_val2(a):
            val2[0] = a

        def set_val3(a):
            val3[0] = a
        
        def interact(sim):
            with self._try(sim):

                # check default values
                self.assertEqual(len(sim.custom_checkbox_1), 1)
                self.assertEqual(len(sim.custom_checkbox_2), 0)
                self.assertEqual(len(sim.custom_checkbox_3), 1)
                self.assertEqual(sim.custom_value_1, 5)
                self.assertEqual(sim.custom_value_2, 6)
                self.assertEqual(sim.custom_value_3, 7)
                self.assertEqual(sim.custom_selection_1, "a")
                self.assertEqual(sim.custom_selection_2, "e")
                self.assertEqual(sim.custom_selection_3, "i")

                # check setting of boxes by toggling them
                self._set_in_main(sim, "custom_checkbox_1", False)
                self._wait_for_idle(sim)
                self.assertEqual(val1[0], False)
                self._set_in_main(sim, "custom_checkbox_2", True)
                self._wait_for_idle(sim)
                self.assertEqual(val2[0], True)
                self._set_in_main(sim, "custom_checkbox_3", False)
                self._wait_for_idle(sim)
                self.assertEqual(val3[0], False)

                # check value setting
                self._set_in_main(sim, "custom_value_1", 2.3)
                self._wait_for_idle(sim)
                self.assertEqual(val1[0], 2.3)
                self._set_in_main(sim, "custom_value_2", -0.34)
                self._wait_for_idle(sim)
                self.assertEqual(val2[0], -0.34)
                self._set_in_main(sim, "custom_value_3", 12.23)
                self._wait_for_idle(sim)
                self.assertEqual(val3[0], 12.23)

                # check selection setting
                self._set_in_main(sim, "custom_selection_1", "b")
                self._wait_for_idle(sim)
                self.assertEqual(val1[0], "b")
                self._set_in_main(sim, "custom_selection_2", "d")
                self._wait_for_idle(sim)
                self.assertEqual(val2[0], "d")
                self._set_in_main(sim, "custom_selection_3", "g")
                self._wait_for_idle(sim)
                self.assertEqual(val3[0], "g")

                # check button actions
                self._do_in_main(sim.custom_button_action_1)
                self._wait_for_idle(sim)
                self.assertEqual(val1[0], 1)
                self._do_in_main(sim.custom_button_action_2)
                self._wait_for_idle(sim)
                self.assertEqual(val1[0], 2)
                self._do_in_main(sim.custom_button_action_3)
                self._wait_for_idle(sim)
                self.assertEqual(val1[0], 3)

                # set bound functions to None. Handling should still work
                sim._custom_selection_functions[0] = None
                sim._custom_button_functions[0] = None
                sim._custom_value_functions[0] = None
                sim._custom_checkbox_functions[0] = None
                self._set_in_main(sim, "custom_value_1", 2.35)
                self._set_in_main(sim, "custom_selection_1", "c")
                self._set_in_main(sim, "custom_checkbox_1", True)

                # time.sleep(20)

        sim = TraceGUI(RT)

        # custom boxes
        sim.add_custom_checkbox("Box 1", True, set_val1)
        sim.add_custom_checkbox("Box 2", False, set_val2)
        sim.add_custom_checkbox("Box 3", True, set_val3)
        
        # custom buttons
        sim.add_custom_button("Button 1", lambda: set_val1(1))
        sim.add_custom_button("Button 2", lambda: set_val1(2))
        sim.add_custom_button("Button 3", lambda: set_val1(3))

        # custom values
        sim.add_custom_value("Value 1", 5, set_val1)
        sim.add_custom_value("Value 2", 6, set_val2)
        sim.add_custom_value("Value 3", 7, set_val3)
        
        # custom selection
        sim.add_custom_selection("Selection 1", ["a", "b", "c"], "a", set_val1)
        sim.add_custom_selection("Selection 2", ["d", "e", "f"], "e", set_val2)
        sim.add_custom_selection("Selection 3", ["g", "h", "i"], "i", set_val3)

        # each element type is limited to three each
        # check if adding additional elements is correctly handled
        sim.add_custom_checkbox("Box 3", True, set_val3)
        sim.add_custom_button("Button 3", lambda: set_val1(3))
        sim.add_custom_value("Value 3", 7, set_val3)
        sim.add_custom_selection("Selection 3", ["g", "h", "i"], "i", set_val3)
        
        sim.debug(interact, args=(sim,))
        self.raise_thread_exceptions()

    @pytest.mark.slow
    @pytest.mark.os
    @pytest.mark.gui3
    def test_plots_passdown(self) -> None:
        
        RT = tracing_geometry()
        path = "screenshot.png"
        
        def interact(sim):
            with self._try(sim):

                # enable focus_search cost plot
                self._set_in_main(sim, "plot_cost_function", True)
                time.sleep(0.05)

                # needed for this kind of geometry, detector_profile plot fails otherwise
                self._set_in_main(sim, "profile_position_dimension", "x")
                time.sleep(0.05)

                # test passdown of path and sargs parameter
                for plot in [sim.source_image, sim.source_profile, sim.detector_image,
                             sim.detector_profile, sim.detector_spectrum, sim.source_spectrum, sim.move_to_focus]:

                        assert not os.path.exists(path)
                        
                        self._do_in_main(plot, path=path, sargs=dict(dpi=120))
                        time.sleep(0.05)
                        self._wait_for_idle(sim)

                        # check if file was saved
                        self.assertTrue(os.path.exists(path))
                        os.remove(path)

        sim = TraceGUI(RT)
        sim.debug(interact, args=(sim,))
        self.raise_thread_exceptions()


if __name__ == '__main__':
    unittest.main()
