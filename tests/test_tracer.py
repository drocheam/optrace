#!/bin/env python3

import sys
sys.path.append('.')

import pytest
import unittest
import numpy as np

import optrace as ot
from optrace.tracer import misc

from test_gui import rt_example


class TracerTests(unittest.TestCase):

    def test_raytracer_init(self):
        o0 = [-5, 5, -5, 5, 0, 10]

        # type errors
        self.assertRaises(TypeError, ot.Raytracer, outline=5)  # incorrect outline
        self.assertRaises(TypeError, ot.Raytracer, outline=o0, absorb_missing=1)  # incorrect bool parameter
        self.assertRaises(TypeError, ot.Raytracer, outline=o0, no_pol=1)  # incorrect bool parameter
        self.assertRaises(TypeError, ot.Raytracer, outline=o0, threading=1)  # incorrect bool parameter
        self.assertRaises(TypeError, ot.Raytracer, outline=o0, silent=1)  # incorrect bool parameter
        self.assertRaises(TypeError, ot.Raytracer, outline=o0, n0=1)  # incorrect Refractionindex
    
        # value errors
        self.assertRaises(ValueError, ot.Raytracer, outline=[5])  # incorrect outline
        self.assertRaises(ValueError, ot.Raytracer, outline=[5, 6, 7, 8, 9])  # incorrect outline
        self.assertRaises(ValueError, ot.Raytracer, outline=[-5, -10, -5, 5, 0, 10])  # incorrect outline
        self.assertRaises(ValueError, ot.Raytracer, outline=[0, 1, 0, 1, 0, np.inf])  # non finite outline

    def test_raytracer_geometry_adding_removal_snapshot(self):
        """
        checks add, remove, property_snapshot, compare_property_snapshot, has() and clear()
        """
        
        RT = ot.Raytracer(outline=[-5, 5, -5, 5, -10, 30])

        RS = ot.RaySource(ot.Point(), pos=[0, 0, 0], spectrum=ot.presets.light_spectrum.led_b1)
        F = ot.Filter(ot.Surface("Circle"), spectrum=ot.TransmissionSpectrum("Constant", val=1), pos=[0, 0, 5])
        DET = ot.Detector(ot.Surface("Circle"), pos=[0, 0, 10])
        AP = ot.Aperture(ot.Surface("Ring"), pos=[0, 0, 10])
        M = ot.Marker("Test", pos=[0, 0, 10])
        L = ot.Lens(ot.Surface("Circle"), ot.Surface("Circle"),
                       n=ot.RefractionIndex("Constant", n=1.2), pos=[0, 0, 10])

        # test actions for different element types
        for el, list_, ckey in zip([RS, L, AP, DET, F, M], 
                [RT.ray_source_list, RT.lens_list, RT.aperture_list, RT.detector_list, RT.filter_list, RT.marker_list], 
                ["RaySources", "Lenses", "Apertures", "Detectors", "Filters", "Markers"]):
        
            el2 = el.copy()

            # adding works
            RT.add(el)
            self.assertEqual(len(list_), 1)
            RT.add(el2)
            self.assertEqual(len(list_), 2)

            # removal works
            succ = RT.remove(el2)
            self.assertTrue(succ)
            self.assertEqual(len(list_), 1)
            self.assertTrue(list_[0] is el)
            
            # remnoving it a second time fails
            succ = RT.remove(el2)
            self.assertFalse(succ)
            
            # check if ckey in compare dictionary is set
            snap = RT.property_snapshot()
            RT.add(el2)
            snap2 = RT.property_snapshot()
            cmp = RT.compare_property_snapshot(snap, snap2)
            self.assertTrue(cmp["Any"])
            self.assertTrue(cmp[ckey])
            if ckey == "Lenses":
                self.assertTrue(cmp["Ambient"])

            # adding the same element a second time does not work
            assert len(list_) == 2
            RT.add(el2)
            self.assertEqual(len(list_), 2)
            RT.remove(el2)
            
            # has() works
            self.assertTrue(RT.has(el))
            self.assertFalse(RT.has(el2))
            
            # clear actually clears
            RT.add(el2)
            RT.clear()
            self.assertEqual(len(list_), 0)

            # adding as list
            RT.add([el, el2])
            self.assertEqual(len(list_), 2)
            self.assertTrue(list_[0] is el)
            self.assertTrue(list_[1] is el2)
            
            # removing as list
            RT.remove([el, el2])
            self.assertEqual(len(list_), 0)

        # test Rays change detection
        snap = RT.property_snapshot()
        RT.add(RS)
        RT.trace(10000)
        snap2 = RT.property_snapshot()
        cmp = RT.compare_property_snapshot(snap, snap2)
        self.assertTrue(cmp["Rays"])
        RT.trace(10000)
        snap3 = RT.property_snapshot()
        cmp = RT.compare_property_snapshot(snap2, snap3)
        self.assertTrue(cmp["Rays"])

        # test detection of ambient
        # make sure these values deviate from default ones
        for key, val in zip(["n0", "outline"], [ot.presets.refraction_index.SF10, [-5, 5, -10, 10, 5, 500]]):
            snap = RT.property_snapshot()
            RT.__setattr__(key, val)
            snap2 = RT.property_snapshot()
            cmp = RT.compare_property_snapshot(snap, snap2)
            self.assertTrue(cmp["Any"])
            self.assertTrue(cmp["Ambient"])

        # test detection of trace settings
        # make sure these values deviate from default ones
        for key, val in zip(["no_pol", "absorb_missing"], [True, False]):
            snap = RT.property_snapshot()
            RT.__setattr__(key, val)
            snap2 = RT.property_snapshot()
            cmp = RT.compare_property_snapshot(snap, snap2)
            self.assertTrue(cmp["Any"])
            self.assertTrue(cmp["TraceSettings"])
     
        # removal of whole RT list, make sure remove makes a copy of a given list
        # otherwise RT.filter_list would change for each deleted element and therefore also the list iterator,
        # leading to not all elements being actually deleted
        assert len(RT.filter_list) == 0
        for i in np.arange(5):
            RT.add(ot.Filter(ot.Surface("Circle"), spectrum=ot.TransmissionSpectrum("Constant", val=1), pos=[0, 0, 5]))
        RT.remove(RT.filter_list)
        self.assertFalse(len(RT.filter_list))

        # check type checking for adding
        self.assertRaises(TypeError, RT.add, ot.Point())  # Surface is a invalid element type

    def test_0geometry_checks(self):
        """test geometry checks before tracing"""
        
        RS = ot.RaySource(ot.Surface("Circle"), pos=[0, 0, 20], spectrum=ot.presets.light_spectrum.led_b1)
        F = ot.Filter(ot.Surface("Circle"), spectrum=ot.TransmissionSpectrum("Constant", val=1), pos=[0, 0, 5])
        AP = ot.Aperture(ot.Surface("Ring"), pos=[0, 0, 10])
        L = ot.Lens(ot.Surface("Circle"), ot.Surface("Circle"),
                       n=ot.RefractionIndex("Constant", n=1.2), pos=[0, 0, 10])

        RT = ot.Raytracer(outline=[-10, 10, -10, 10, -10, 30], silent=True)
        RS0 = RS.copy()
        RS0.move_to([0, 0, RT.outline[4]])
        RT.add(RS0)

        for i in np.arange(2):
            for el in [AP, L, F, RS]:
                RT.add(el)

                if i == 0:  # case i=0: element outside outline
                    for pos in [[RT.outline[0], 0, 0],
                                [RT.outline[1], 0, 0],
                                [0, RT.outline[2], 0],
                                [0, RT.outline[3], 0],
                                [0, 0, RT.outline[4]-1],
                                [0, 0, RT.outline[5]+1]]:  # check different positions outside outline
                        el.move_to(pos)
                        self.assertRaises(RuntimeError, RT.trace, 10000)  # element outside outline

                elif el is not RS:  # case i=1: element in front of source
                    self.assertRaises(RuntimeError, RT.trace, 10000)  # for i == 1 correct order is broken

                RT.remove(el)

            RT.add(RS)  # add second source that lies behind element, lead to sequentiality errors

        RT.remove([RS, RS0])
        self.assertRaises(RuntimeError, RT.trace, 10000)  # RaySource Missing
        
        RT.add(ot.RaySource(ot.Point(), pos=[0, 0, 0]))
        self.assertRaises(ValueError, RT.trace, 10)  # too few rays
        self.assertRaises(ValueError, RT.trace, 1e10)  # too many rays


        # additional ray source geometry checks

        RT = ot.Raytracer(outline=[-5, 5, -5, 5, -10, 30], silent=True)
        RS = ot.RaySource(ot.Point(), spectrum=ot.presets.light_spectrum.d65, pos=[0, 0, -10])
        RT.add(RS)

        RT.add(ot.Lens(ot.Surface("Sphere", r=3, R=-3.2), ot.Surface("Circle"), n=ot.presets.refraction_index.SF10, pos=[0, 0, 0], d=0.2))

        RT.trace(10000)  # works
    
        RS.move_to([0, 0, -0.5])
        RT.trace(10000)  # should also work, lens encompasses source, but correct order and not collision

        RS.move_to([0, 0, 0])
        self.assertRaises(RuntimeError, RT.trace, 10000)  # source inside lens
        
        RS.move_to([0, 0, 10])
        self.assertRaises(RuntimeError, RT.trace, 10000)  # source behind lens

    @pytest.mark.slow
    def test_focus(self):
        
        RT = ot.Raytracer(outline=[-3, 3, -3, 3, -10, 40], silent=True, n0=ot.RefractionIndex("Constant", n=1.1))
        RSS = ot.Surface("Circle", r=0.5)
        RS = ot.RaySource(RSS, pos=[0, 0, -3], spectrum=ot.presets.light_spectrum.d65)
        RT.add(RS)

        front = ot.Surface("Sphere", r=3, R=30)
        back = ot.Surface("Conic", r=3, R=-20, k=1)
        n = ot.RefractionIndex("Constant", n=1.5)
        L = ot.Lens(front, back, n, de=0.1, pos=[0, 0, 0])
        RT.add(L)

        f = L.tma(n0=RT.n0).efl
        fs = 33.08433714
        self.assertAlmostEqual(f, fs, 6)

        self.assertRaises(RuntimeError, RT.autofocus, RT.autofocus_methods[0], z_start=5)  # no simulated rays
        
        RT.trace(200000)

        for method in RT.autofocus_methods:  # all methods
            for N in [50000, 200000, 500000]:  # N cases: less rays, all rays, more rays than simulated
                res, _ = RT.autofocus(method, z_start=5.0, N=N, return_cost=False)
                self.assertAlmostEqual(res.x, fs, delta=0.15)

        # source_index=0 with only one source leads to the same result
        res, _, = RT.autofocus(RT.autofocus_methods[0], z_start=5.0, source_index=0, N=N, return_cost=False)
        self.assertAlmostEqual(res.x, fs, delta=0.15)

        RS2 = ot.RaySource(ot.Point(), spectrum=ot.presets.light_spectrum.d65, divergence="Isotropic",
                           div_angle=0.5, pos=[0, 0, -60])
        RT.add(RS2)
       
        # enlargen outline so imaging works
        # the outline was smaller before to improve focus finding
        RT.outline = [-3, 3, -3, 3, -70, 80]
        RT.trace(200000)

        # source_index=0 with two raysources should lead to the same result as if the second RS was missing
        res, _ = RT.autofocus(RT.autofocus_methods[0], z_start=5.0, source_index=0, N=N, return_cost=False)
        self.assertAlmostEqual(res.x, fs, delta=0.15)

        # 1/f = 1/b + 1/g with f=33.08, g=60 should lead to b=73.73
        res, _ = RT.autofocus(RT.autofocus_methods[0], z_start=5.0, source_index=1, N=N, return_cost=False)
        self.assertAlmostEqual(res.x, 73.73, delta=0.1)

        self.assertRaises(ValueError, RT.autofocus, RT.autofocus_methods[0], z_start=-100)  # z_start outside outline
        self.assertRaises(ValueError, RT.autofocus, "AA", z_start=10)  # invalid mode
        self.assertRaises(ValueError, RT.autofocus, RT.autofocus_methods[0], z_start=10, source_index=-1)
        # index negative
        self.assertRaises(ValueError, RT.autofocus, RT.autofocus_methods[0], z_start=10, source_index=10)
        # index too large

        self.assertRaises(ValueError, RT.autofocus, RT.autofocus_methods[0], z_start=10, N=10)  # N too low

        # add a light source with only a small power
        RS2.power = 0.001
        RT.trace(200000)
        self.assertRaises(RuntimeError, RT.autofocus, RT.autofocus_methods[0], z_start=10, source_index=1)
        # actual ray number after selection too low (since this source has only few rays)

        # coverage tests

        RT.autofocus(RT.autofocus_methods[0], z_start=RT.outline[4]) # before all lenses, source_index=None
        RT.autofocus(RT.autofocus_methods[0], z_start=RT.outline[4], source_index=0) # before all lenses, source_index
        RT.autofocus(RT.autofocus_methods[0], z_start=RT.outline[5])  # behind all lenses
        RT.autofocus(RT.autofocus_methods[0], z_start=RT.lens_list[0].extent[5] + 0.01)  # between lenses
        RT.autofocus(RT.autofocus_methods[0], z_start=RT.lens_list[-1].extent[4] - 0.01)  # between lenses with n_ambient
        RT.autofocus("Irradiance Variance", z_start=RT.outline[5])  
        # leads to a warning, that minimum may not be found
        
        RT.silent = False
        RT.threading = False
        RT.autofocus("Irradiance Variance", z_start=RT.outline[5])  

        # aperture blocks all light
        RT.add(ot.Aperture(ot.Surface("Circle", r=3), pos=[0, 0, 0]))
        RT.autofocus("Position Variance", z_start=10) # no rays here

    @pytest.mark.slow
    def test_sphere_detector_range_hits(self):
        """
        this function checks if the detector hit finding correctly handles:
        * rays starting after the detector
        * rays ending before the detector
        * rays starting inside of detector extent
        """

        RT = ot.Raytracer(outline=[-10, 10, -10, 10, -100, 100], silent=True)
        RS = ot.RaySource(ot.Surface("Circle", r=0.5), pos=[0, 0, 0], spectrum=ot.presets.light_spectrum.d65, divergence="None")
        RT.add(RS)

        aps = ot.Surface("Ring", r=2, ri=1)
        ap = ot.Aperture(aps, pos=[0, 0, 3])
        RT.add(ap)

        dets = ot.Surface("Sphere", r=5, R=-6)
        det = ot.Detector(dets, pos=[0, 0, -10])
        RT.add(det)

        # angular extent for projection_method = "Equidistant"
        ext0 = np.arcsin(RS.surface.r/np.abs(det.surface.R))
        ext = np.array(ext0).repeat(4)*[-1, 1, -1, 1]

        RT.trace(400000)

        for z in [RT.outline[4], RS.pos[2]+1, ap.pos[2]-1, ap.pos[2]+1, RS.pos[2]+1+det.surface.R, RT.outline[5]-RT.N_EPS, RT.outline[5] + 1]:
            det.move_to([0, 0, z])
            img = RT.detector_image(16, projection_method="Equidistant")
            
            if RT.outline[5] > z > RS.surface.pos[2]:
                self.assertAlmostEqual(img.get_power(), RS.power)  # correctly lit
                self.assertTrue(np.allclose(img.extent-ext, 0, atol=1e-2))  # extent correct
            else:
                self.assertAlmostEqual(img.get_power(), 0)

    @pytest.mark.slow
    def test_uniform_emittance(self):
        """test that surfaces in ray source emit uniformly"""

        RT = ot.Raytracer(outline=[-5, 5, -5, 5, -10, 10], silent=True, n0=ot.RefractionIndex("Constant", n=1))

        for surf in [ot.Surface("Circle", r=2), 
                    ot.Surface("Rectangle", dim=[1, 1]), 
                    ot.Surface("Rectangle", dim=[1, 0.75]), 
                    ot.Line(), 
                    ot.Line(angle=30), 
                    ot.Line(angle=90), 
                    ot.Surface("Ring", ri=1.5, r=2),
                    ot.Surface("Ring", ri=0.25, r=2)]:

            RS = ot.RaySource(surf, pos=[0, 0, 0], spectrum=ot.presets.light_spectrum.d65)
            RT.add(RS)
            RT.trace(200000)

            im = RT.source_image(32)
            L = im.get_by_display_mode("Irradiance")
            L /= np.max(L)

            if isinstance(surf, ot.Surface):
                if surf.surface_type != "Rectangle":
                    Lx, Ly = L.shape[:2]
                    X, Y = np.mgrid[-Lx/2:Lx/2:Lx*1j, -Ly/2:Ly/2:Ly*1j]
                    m = X**2 + Y**2 < (Lx/2-1)**2

                    if surf.surface_type == "Ring":
                        m2 = X**2 + Y**2 > (Lx/2 * surf.ri/surf.r+1)**2
                        m = m & m2

                    L = L[m]
            else:
                L = L[L > 0]

            # we should have some remaining noise, but the std should be below 7%
            self.assertTrue(np.std(L) < 0.07)
            
            RT.remove(RS)

    @pytest.mark.slow
    def test_ray_source_divergence(self):
        """
        tests the behaviour of different ray source divergence modes in 2D and 3D mode
        to check the function f we multiply the irradiance curve on the detector with 1/f
        the resulting function curve should be a constant with some noise
        we check the standard deviation on this curve
        """

        # make raytracer
        RT = ot.Raytracer(outline=[-100, 100, -100, 100, -10, 100], silent=True)

        RSS0 = ot.Surface("Rectangle", dim=[0.02, 0.02])
        RS0 = ot.RaySource(RSS0, divergence="Isotropic", spectrum=ot.presets.light_spectrum.d65, div_2d=True,
                pos=[0, 0, 0], s=[0, 0, 1], div_angle=82)
        RT.add(RS0)
            
        # add Detector
        DETS = ot.Surface("Rectangle", dim=[100, 100])
        DET = ot.Detector(DETS, pos=[0, 0, 10])
        RT.add(DET)

        # Parallel Illumination 2D
        RS0.divergence = "None"
        RT.trace(2000000)
        img = RT.detector_image(64)
        x, y = img.cut(y=0, mode="Irradiance")
        y = y[0]
        self.assertTrue(np.std(y/np.max(y)) < 0.025)

        # Equal divergence in 2D, leads to 1/cos(e)**2 on detector
        RS0.divergence = "Isotropic"
        RT.trace(2000000)
        img = RT.detector_image(64)
        x, y = img.cut(y=0, mode="Irradiance")
        y = y[0]/np.cos(np.arctan(x/10))**2
        self.assertTrue(np.std(y/np.max(y)) < 0.05)

        # Equal divergence in 2D, leads to 1/cos(e)**3 on detector
        RS0.divergence = "Lambertian"
        RT.trace(2000000)
        img = RT.detector_image(64)
        x, y = img.cut(y=0, mode="Irradiance")
        y = y[0]/np.cos(np.arctan(x/10))**3
        self.assertTrue(np.std(y/np.max(y)) < 0.05)

        # Function mode, 1/cos(e)**2 leads to uniform illumination on detector
        RS0.divergence = "Function"
        RS0.div_func = lambda e: 1/np.cos(e)**2
        RT.trace(2000000)
        img = RT.detector_image(64)
        x, y = img.cut(y=0, mode="Irradiance")
        y = y[0]
        self.assertTrue(np.std(y/np.max(y)) < 0.025)

        # Function mode, needs to be rescaled by 1/function and 1/cos(e)**2 for equal illumination
        RS0.divergence = "Function"
        RS0.div_func = lambda e: 1+np.sqrt(e)
        RT.trace(2000000)
        img = RT.detector_image(64)
        x, y = img.cut(y=0, mode="Irradiance")
        y = y[0]/( 1 + np.sqrt(np.arctan(np.abs(x)/10)))/np.cos(np.arctan(x/10))**2
        self.assertTrue(np.std(y/np.max(y)) < 0.025)

        # switch to 3D divergence
        RS0.div_2d = False

        # Parallel Illumination 3D
        RS0.divergence = "None"
        RT.trace(2000000)
        img = RT.detector_image(64)
        img = img.get_by_display_mode("Irradiance")
        self.assertTrue(np.std(img/np.max(img)) < 0.025)

        # Equal divergence in 3D, leads to 1/cos(e)**3 on detector
        RS0.divergence = "Isotropic"
        RT.trace(2000000)
        img0 = RT.detector_image(64)
        img = img0.get_by_display_mode("Irradiance")
        x0, x1, y0, y1 = img0.extent
        X, Y = np.mgrid[x0:x1:64j, y0:y1:64j]
        Z = img / np.cos(np.arctan(np.sqrt(X**2 + Y**2)/10))**3
        self.assertTrue(np.std(Z/np.max(Z)) < 0.05)

        # Lambertian divergence in 3D, leads to 1/cos(e)**4 on detector
        RS0.divergence = "Lambertian"
        RT.trace(2000000)
        img0 = RT.detector_image(64)
        img = img0.get_by_display_mode("Irradiance")
        x0, x1, y0, y1 = img0.extent
        X, Y = np.mgrid[x0:x1:64j, y0:y1:64j]
        Z = img / np.cos(np.arctan(np.sqrt(X**2 + Y**2)/10))**4
        self.assertTrue(np.std(Z/np.max(Z)) < 0.075)

        # Function mode, 1/cos(e)**3 leads to uniform illumination on detector
        RS0.divergence = "Function"
        RS0.div_func = lambda e: 1/np.cos(e)**3
        RT.trace(2000000)
        img0 = RT.detector_image(64)
        Z = img0.get_by_display_mode("Irradiance")
        self.assertTrue(np.std(Z/np.max(Z)) < 0.075)

        # Function mode, needs to be rescaled by 1/function and 1/cos(e)**3 for uniform curve on detector
        RS0.divergence = "Function"
        RS0.div_func = lambda e: 1+np.sqrt(e)
        RT.trace(2000000)
        img0 = RT.detector_image(64)
        img = img0.get_by_display_mode("Irradiance")
        x0, x1, y0, y1 = img0.extent
        X, Y = np.mgrid[x0:x1:64j, y0:y1:64j]
        r = np.sqrt(X**2 + Y**2)
        Z = img / ( 1 + np.sqrt(np.arctan(r/10)))/np.cos(np.arctan(r/10))**3
        self.assertTrue(np.std(Z/np.max(Z)) < 0.05)
   
    @pytest.mark.slow
    def test_sphere_projections(self):

        # make raytracer
        RT = ot.Raytracer(outline=[-100, 100, -100, 100, -10, 100], silent=True)

        # add Detector
        R = 90
        DETS = ot.Surface("Sphere", r=(1-1e-10)*R, R=-R)
        DET = ot.Detector(DETS, pos=[0, 0, R])
        RT.add(DET)

        # Isotropic leads to uniform illumination on sphere detector,
        # in Equal Area projection mode the iraddiance should be constant over the detector
        
        # add point source with isotropic light
        RSS0 = ot.Point()
        RS0 = ot.RaySource(RSS0, divergence="Isotropic", spectrum=ot.presets.light_spectrum.d65, div_2d=False,
                pos=[0, 0, 0], s=[0, 0, 1], div_angle=89)
        RT.add(RS0)
        RT.trace(2000000)
        img0 = RT.detector_image(64, projection_method="Equal-Area")
        Z = img0.get_by_display_mode("Irradiance")
        # mask out area outside disc, leave 3 pixels margin
        X, Y = np.mgrid[-32:32:64j, -32:32:64j]
        mask = np.sqrt(X**2 + Y**2) < 29
        Z = Z[mask]
        self.assertTrue(np.std(Z/np.max(Z)) < 0.015)  # constant irradiance (neglecting noise) as it should be
        RT.remove(RS0)

        # Equidistant projection method
        # equally radially spaced points on sphere should also be equally spaced in projection
        RSS0 = ot.Surface("Rectangle", dim=[0.0001, 0.0001])
        # add point sources with parallel rays
        for theta in [-89.99, -60, -30, 0.001, 30, 60, 89.99]:  # slightly decentered central value so we only hit one pixel
            theta_r = np.radians(theta)
            RS0 = ot.RaySource(RSS0, divergence="None", spectrum=ot.presets.light_spectrum.d65, div_2d=False,
                    pos=[0, 0, 0], s=[np.sin(theta_r), 0, np.cos(theta_r)])
            RT.add(RS0)

        # check that hits are equally spaced in projection
        # leave one pixel margin, because of pixel positions
        RT.trace(20000)
        img0 = RT.detector_image(256, projection_method="Equidistant")
        z = img0.cut(y=0, mode="Irradiance")[1][0]
        fact = z.shape[0] / 6  # divide range by number of theta spacings
        zp = (z > 0).nonzero()[0]  # find hits
        zp_diff = zp/fact - np.round(zp/fact)  # difference between pixel pos and correct pos
        for zpi in zp_diff:
            # check if difference is smaller than one pixel, in that case they are equally spaced
            self.assertAlmostEqual(zpi, 0, delta=1/fact*1.1)  # radial positions equally spaced

        # remove all sources
        for rs in RT.ray_source_list.copy():
            RT.remove(rs)

        # Stereographic projection: small circles on surface keep relative their shape, therefore also their side lengths
        # only the overall size changes
        # in the different projections circles near the sphere edge are distorted
        # here we check the side ratio of the generated detector image for each circle
        RSS0 = ot.Point()
        for phi in np.linspace(0, 2*np.pi, 9):
            for theta in np.linspace(0, 1, 5)*87:
                # circular are on sphere at defined position
                tr = np.radians(theta)
                sx = np.sin(phi)*np.sin(tr)
                sy = np.cos(phi)*np.sin(tr)
                sz = np.cos(tr)
                RS0 = ot.RaySource(RSS0, divergence="Isotropic", spectrum=ot.presets.light_spectrum.d65, div_2d=False,
                        pos=[0, 0, 0], s=[sx, sy, sz], div_angle=2)
                RT.add(RS0)

                # make detector image and get extent
                RT.trace(40000)
                img0 = RT.detector_image(256, projection_method="Stereographic")
                x0, x1, y0, y1 = img0.extent
                RT.remove(RS0)

                # check if side ratio stays 1
                self.assertAlmostEqual((x1-x0)/(y1-y0), 1, delta=0.003)  
                # some remaining difference because of rays not being exactly at the edge, 
                # but positions are random inside circular area

    @pytest.mark.slow
    def test_ray_storage(self):
        
        RT = rt_example()

        # we want the ray sources to be two and have different power
        assert len(RT.ray_source_list) == 2

        # TODO check only one source and check three sources

        powers = [(1, 1), (2, 1), (0.3456465, 4.57687168)]
        Ns = [100000, 52657, 30000, 30001]

        for powersi in powers:
            for N in Ns:

                P1, P2 = powersi

                RT.ray_source_list[0].power = P1
                RT.ray_source_list[1].power = P2

                # check if tracing leads to the correct amount of rays
                RT.trace(N)
                self.assertEqual(N, RT.rays.N)  # correct number of rays
                self.assertAlmostEqual(P1+P2, np.sum(RT.rays.w_list[:, 0]), delta=10/N)  # approx. correct power

                # returns all source sections
                tup = RT.rays.get_source_sections()
                self.assertEqual(tup[0].shape[0], N)
                
                # returns only sections of first source
                tup = RT.rays.get_source_sections(0)
                self.assertEqual(tup[0].shape[0], RT.rays.n_list[0])  # correct number
                self.assertAlmostEqual(RT.rays.n_list[0]/RT.rays.N, P1/(P1+P2), delta=10/N)  # approx. correct power

                # returns only sections of second source
                tup = RT.rays.get_source_sections(1)
                self.assertEqual(tup[0].shape[0], RT.rays.n_list[1])  # correct number
                self.assertAlmostEqual(RT.rays.n_list[1]/RT.rays.N, P2/(P1+P2), delta=10/N)  # approx. correct power

        # check get_rays_by_mask

        ch = np.random.randint(0, 2, size=N).astype(bool)
        N2 = np.count_nonzero(ch)
        ch2 = np.random.randint(0, RT.rays.nt, size=N2)

        ch2l = [None, ch2]
        retl = [None, [1, 0, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0]]
        norml = [False, True]

        for ch2li in ch2l:
            for retli in retl:
                for normli in norml:
                    tup = RT.rays.get_rays_by_mask(ch, ch2li, retli, normli)
                    self.assertEqual(tup[3].shape[0], N2)  # correct shape 1
                    self.assertEqual(tup[3].ndim, (1 if ch2li is not None else 2))  # correct shape 1
                    self.assertTrue(retli is None or retli[1] or tup[1] is None)  # property omitted for ret[1] = 0
                    self.assertTrue(retli is None or retli[5] or tup[5] is None)  # property omitted for ret[5] = 0

        # coverage tests

        # warns that there are sources without rays
        RT.ray_source_list[0].power = 0.000001
        RT.ray_source_list[1].power = 1
        RT.trace(10000)

    @pytest.mark.slow
    def test_raytracer_output_threading_nopol(self):
        RT = rt_example()
        RT.threading = False  # on by default, therefore test only off case
        RT.silent = False

        # raytracer not silent -> outputs messages and progress bar for actions
        RT.trace(10000)
        RT.autofocus(RT.autofocus_methods[0], 12)
        RT.source_image(100)
        RT.detector_image(100)
        RT.source_spectrum()
        RT.detector_spectrum()
        RT.iterative_render(100000, silent=False)

        # show all tracing messages
        for i in np.arange(len(RT.INFOS)):
            RT._msgs = np.zeros((len(RT.INFOS), 2), dtype=int)
            RT._msgs[i, 0] = 1

            # some infos throw
            try:
                RT._show_messages(1000)
            except Exception as err:
                print(err)

        # simulate with no_pol
        RT.no_pol = True
        RT.trace(10000)

    def test_offset_system_equality(self):
        """
        this function tests if the same geometry behaves the same if it is shifted by a vector
        also checks numeric precision for some larger offset values
        """

        im = None
        abcd = None

        for pos0 in [(0, 0, 0), (5.789, 0.123, -45.6), (0, 16546.789789, -4654), (1e8, -1e10, 15)]:

            x0, y0, z0 = (*pos0,)
            pos0 = np.array(pos0)
            
            RT = ot.Raytracer(outline=[-5+x0, 5+x0, -5+y0, 5+y0, -10+z0, 50+z0], silent=True)

            RSS = ot.Surface("Circle", r=0.2)
            RS = ot.RaySource(RSS, spectrum=ot.LightSpectrum("Monochromatic", wl=555), 
                              divergence="None", pos=pos0+[0, 0, -3])
            RT.add(RS)

            front = ot.Surface("Sphere", r=3, R=50)
            back = ot.Surface("Conic", r=3, R=-50, k=-1.5)
            L0 = ot.Lens(front, back, n=ot.presets.refraction_index.SF10, pos=pos0+[0, 0.01, 0])
            RT.add(L0)

            front = ot.Surface("Function", r=3,
                    func=ot.SurfaceFunction(func=lambda x, y: (x**2 + y**2)/50 + (x**2 +y**2)**2/5000, silent=True),
                    curvature_circle=25)
            back = ot.Surface("Circle", r=2)
            L1 = ot.Lens(front, back, n=ot.presets.refraction_index.SF10, pos=pos0+[0, 0.01, 10])
            RT.add(L1)

            X, Y = np.mgrid[-1:1:100j, -1:1:100j]
            data = 3 - (X**2 + Y**2)
            front = ot.Surface("Data", data=data, r=4, silent=True)
            back = ot.Surface("Circle", r=4, normal=[0, 0.01, 1])
            L2 = ot.Lens(front, back, n=ot.presets.refraction_index.K5, pos=pos0+[0, 0, 20])
            RT.add(L2)

            rect = ot.Surface("Rectangle", dim=[10, 10])
            Det = ot.Detector(rect, pos=pos0+[0., 0., 45])
            RT.add(Det)

            # render image, so we can compare side length and power
            RT.trace(100000)
            im2 = RT.detector_image(100)

            # remove not symmetric element and compare abcd matrix
            RT.remove(L2)
            abcd2 = RT.tma().abcd

            # first run: save reference values
            if im is None:
                abcd = abcd2
                im = im2.copy()
            # run > 1: compare values
            if im is not None:
                self.assertTrue(np.allclose(im.extent-im2.extent+pos0[:2].repeat(2), 0, atol=0.001))
                self.assertAlmostEqual(im.extent[1]-im.extent[0], im2.extent[1]-im2.extent[0], delta=0.001)
                self.assertAlmostEqual(im.extent[3]-im.extent[2], im2.extent[3]-im2.extent[2], delta=0.001)
                self.assertAlmostEqual(im.get_power(), im2.get_power(), places=4)
                self.assertTrue(np.allclose(abcd-abcd2, 0, atol=0.0001))

    def test_numeric_tracing(self):
        """checks RT.find_hit for a numeric surface and Surface of type "Data" """

        RT = ot.Raytracer(outline=[-3, 3, -3, 3, -10, 50], silent=True)

        x = np.linspace(-2, 2, 100)
        y = np.linspace(-2, 2, 100)
        X, Y = np.meshgrid(x, y)
        R = 12

        # sphere equation
        Z = R*(1-np.sqrt(1-(X**2 + Y**2)/R**2))

        n = ot.RefractionIndex("Constant", n=1.5)

        front = ot.Surface("Data", data=Z, r=2, silent=True)
        back = ot.Surface("Data", data=-Z, r=2, silent=True)
        L = ot.Lens(front, back, n, pos=[0, 0, 0], d=0.4)
        RT.add(L)

        RSS = ot.Surface("Circle", r=0.5)
        RS = ot.RaySource(RSS, spectrum=ot.LightSpectrum("Monochromatic", wl=555), divergence="None", pos=[0, 0, -3])
        RT.add(RS)

        RT.trace(100000)
        res, _ = RT.autofocus(RT.autofocus_methods[0], 5)

        f_should = self.lens_maker(R, -R, n(555), 1, L.d)

        self.assertAlmostEqual(res.x, f_should, delta=0.2)

    # lens maker equation
    def lens_maker(self, R1, R2, n, n0, d):
        D = (n-n0)/n0 * (1/R1 - 1/R2 + (n - n0) * d / (n*R1*R2))
        return 1 / D if D else np.inf 

    @pytest.mark.slow
    def test_same_surface_behavior(self):
        """check if different surface modes describing the same surface shape actually behave the same"""

        RT = ot.Raytracer(outline=[-3, 3, -3, 3, -10, 500], absorb_missing=False, silent=True)

        RSS = ot.Surface("Circle", r=0.2)
        RS = ot.RaySource(RSS, spectrum=ot.LightSpectrum("Monochromatic", wl=555), divergence="None", pos=[0, 0, -3])
        RT.add(RS)

        # illuminated area is especially important for numeric surface_type="Data"
        for RS_r in [0.001, 0.01, 0.1, 0.5]:

            RS.set_surface(ot.Surface("Circle", r=RS_r))

            # check different curvatures
            for R_ in [3, 10.2, 100]:
                
                r = 2

                asph = ot.Surface("Conic", R=R_, k=-1, r=r)
                sph = ot.Surface("Sphere", R=R_, r=r)

                func = lambda x, y, R: 1/2/R*(x**2 + y**2)
                func2 = lambda x, y: 0.78785 + 1/2/R_*(x**2 + y**2)  # some offset that needs to be removed
                sfunc1 = ot.SurfaceFunction(func2, silent=True)
                sfunc2 = ot.SurfaceFunction(func, func_args=dict(R=R_), silent=True)

                surff1 = ot.Surface("Function", func=sfunc1, r=r, silent=True)
                surff2 = ot.Surface("Function", func=sfunc2, r=r, z_min=0, z_max=func(0, r, R_), silent=True)

                def surf_data_gen(N):
                    x = np.linspace(-r, r, N)
                    y = np.linspace(-r, r, N)
                    X, Y = np.meshgrid(x, y)
                    return 4.657165 + 1/2/R_ * (X**2 + Y**2)  # some random offset

                # type "Data" with different resolutions and offsets
                # and odd number defines a point in the center of the lens,
                # for "even" the center is outside the grid
                surf_data0 = ot.Surface("Data", silent=True, data=surf_data_gen(900), r=r)
                surf_data1 = ot.Surface("Data", silent=True, data=surf_data_gen(200), r=r)
                surf_data2 = ot.Surface("Data", silent=True, data=surf_data_gen(50), r=r)
                surf_data3 = ot.Surface("Data", silent=True, data=surf_data_gen(901), r=r)
                surf_data4 = ot.Surface("Data", silent=True, data=surf_data_gen(201), r=r)
                surf_data5 = ot.Surface("Data", silent=True, data=surf_data_gen(51), r=r)

                surf_circ = ot.Surface("Circle", r=r)

                n = 1.5
                d = 0.1 + func(0, r, R_)
                f = self.lens_maker(R_, np.inf, n, 1, d)
                f_list = []

                # create lens, trace and find focus
                for surf in [sph, asph, surff1, surff2, surf_data0, surf_data1, surf_data2,
                            surf_data3, surf_data4, surf_data5]:
                    L = ot.Lens(surf, surf_circ, n=ot.RefractionIndex("Constant", n=n),
                                pos=[0, 0, +d/2], d=d)
                    RT.add(L)

                    RT.trace(100000)
                    res, _ = RT.autofocus(RT.autofocus_methods[0], 5)
                    f_list.append(res.x)

                    RT.remove(L)
            
                self.assertAlmostEqual(f, f_list[1], delta=0.2)  # f and asphere almost equal
                self.assertAlmostEqual(f_list[0], f_list[1], delta=0.2)  # sphere and asphere almost equal

                # other surfaces almost equal
                # since those use the exact same function, they should be nearly identical
                f_list2 = f_list[1:]
                for i, fl in enumerate(f_list2):
                    if i + 1 < len(f_list2):
                        self.assertAlmostEqual(fl, f_list2[i+1], delta=0.001)

    def test_abnormal_rays(self):
        """
        rays that hit the lens cylinder edge in any way are absorbed
        case 1: ray hits lens front, but not back
        case 2: ray misses front, but hits back
        """

        RT = ot.Raytracer(outline=[-3, 3, -3, 3, -10, 50], absorb_missing=False, silent=True)

        RSS = ot.Surface("Circle", r=2)
        RS = ot.RaySource(RSS, spectrum=ot.LightSpectrum("Monochromatic", wl=555), divergence="None", pos=[0, 0, -3])
        RT.add(RS)

        surf1 = ot.Surface("Circle", r=3)
        surf2 = ot.Surface("Circle", r=1e-6)
        L = ot.Lens(surf2, surf1, n=ot.RefractionIndex("Constant", n=1.5), pos=[0, 0, 0], d=0.1)
        RT.add(L)

        N = 10000
        
        RT.trace(N)
        self.assertAlmostEqual(1, RT._msgs[RT.INFOS.ONLY_HIT_BACK, 2] / N, places=3)

        RT.lens_list[0] = ot.Lens(surf1, surf2, n=ot.RefractionIndex("Constant", n=1.5), pos=[0, 0, 0], d=0.1)
        RT.trace(N)
        self.assertAlmostEqual(1, RT._msgs[RT.INFOS.ONLY_HIT_FRONT, 1] / N, places=3)
       
    def test_absorb_missing(self):
        """
        infinitely small lens -> almost all rays miss and are set to absorbed (due to absorb_missing = False)
        """

        RT = ot.Raytracer(outline=[-3, 3, -3, 3, -10, 50], absorb_missing=False, silent=True)

        RSS = ot.Surface("Circle", r=2)
        RS = ot.RaySource(RSS, spectrum=ot.LightSpectrum("Monochromatic", wl=555), divergence="None", pos=[0, 0, -3])
        RT.add(RS)

        surf = ot.Surface("Circle", r=1e-6)
        L = ot.Lens(surf, surf, n=ot.RefractionIndex("Constant", n=1.5), pos=[0, 0, 0], d=0.1)
        RT.add(L)

        N = 10000
        RT.absorb_missing = True
        RT.trace(N)
        self.assertAlmostEqual(1, RT._msgs[RT.INFOS.ABSORB_MISSING, 1] / N, places=3)
        
        # don't absorb with absorb_missing = False
        RT.absorb_missing = False
        RT.trace(N)
        self.assertEqual(RT._msgs[RT.INFOS.ABSORB_MISSING, 1], 0)

        # absorb_missing is forced when there are different ambient n0
        L.n2 = ot.RefractionIndex("Constant", 1.1)
        RT.trace(N)
        self.assertTrue(RT.absorb_missing)

    def test_outline_intersection(self):
        """
        strongly diverging rays, but tracing geometry is a long corridor
        -> almost all rays are absorbed by outline 
        """

        RT = ot.Raytracer(outline=[-3, 3, -3, 3, -10, 5000], absorb_missing=False, silent=True)

        RSS = ot.Point()
        RS = ot.RaySource(RSS, spectrum=ot.LightSpectrum("Monochromatic", wl=555), divergence="Isotropic", 
                          div_angle=80, pos=[0, 0, -3])
        RT.add(RS)

        N = 10000
        RT.trace(N)
        self.assertAlmostEqual(1, RT._msgs[RT.INFOS.OUTLINE_INTERSECTION, 0] / N, places=3)

    def test_tir(self):
        """
        unrealistically high refractive index, a slight angle leads to TIR at the next surface
        """

        RT = ot.Raytracer(outline=[-10, 10, -10, 10, -10, 50], absorb_missing=False, silent=True, 
                          n0=ot.RefractionIndex("Constant", 100))

        RSS = ot.Surface("Circle", r=2)
        RS = ot.RaySource(RSS, spectrum=ot.LightSpectrum("Monochromatic", wl=555), divergence="None",
                          pos=[0, 0, -3], s=[0, 0.1, 0.99])
        RT.add(RS)

        surf1 = ot.Surface("Circle", r=10)
        surf2 = ot.Surface("Circle", r=10)
        L = ot.Lens(surf2, surf1, n=ot.RefractionIndex("Constant", n=1.5), pos=[0, 0, 0], d=0.1)
        RT.add(L)

        N = 10000
        RT.trace(N)
        self.assertEqual(RT._msgs[RT.INFOS.TIR, 0], N)
   
    def test_t_th(self):
        """
        to avoid ghost rays, small transmission factors are set to 0
        """

        RT = ot.Raytracer(outline=[-10, 10, -10, 10, -10, 50], absorb_missing=False, silent=True)

        RSS = ot.Surface("Circle", r=2)
        RS = ot.RaySource(RSS, spectrum=ot.LightSpectrum("Constant"), divergence="None",
                          pos=[0, 0, -3])
        RT.add(RS)

        surf1 = ot.Surface("Circle", r=10)
        surf2 = ot.Surface("Circle", r=10)
        F = ot.Filter(surf1, pos=[0, 0, 0], spectrum=ot.TransmissionSpectrum("Gaussian", mu=750, sig=0.1))
        RT.add(F)

        RT.trace(10000)
        self.assertTrue(RT._msgs[RT.INFOS.T_BELOW_TTH, 0] > 0)

        # don't do t_th on a constant spectrum
        F.spectrum = ot.TransmissionSpectrum("Constant", val=RT.T_TH/2)
        RT.trace(10000)
        self.assertFalse(RT._msgs[RT.INFOS.T_BELOW_TTH, 0] > 0)

    def test_object_collision(self):

        # make raytracer
        RT = ot.Raytracer(outline=[-10, 10, -10, 10, -10, 60], silent=True)

        # add Raysource
        RSS = ot.Point()
        RS = ot.RaySource(RSS, divergence="Isotropic", spectrum=ot.presets.light_spectrum.d65,
                          pos=[0, 0, 0], s=[0, 0, 1], div_angle=75)
        RT.add(RS)
        
        geom = ot.presets.geometry.arizona_eye()
        geom[2].move_to([0, 0, 0.15])
        RT.add(geom)

        self.assertRaises(RuntimeError, RT.trace, 10000)  # object collision
        self.assertEqual(RT.rays.N, 0)  # not traced

    def test_seq_violation(self):

        # make raytracer
        RT = ot.Raytracer(outline=[-10, 10, -10, 10, -10, 60], silent=True)
        RT._no_geometry_checks = True  # deactivate checks so we can enforce a seq violation

        # add Raysource
        RSS = ot.Point()
        RS = ot.RaySource(RSS, divergence="Isotropic", spectrum=ot.presets.light_spectrum.d65,
                          pos=[0, 0, 0], s=[0, 0, 1], div_angle=75)
        RT.add(RS)
        
        geom = ot.presets.geometry.arizona_eye()
        geom[1].move_to([0, 0, 0.01])
        RT.add(geom)

        RT.trace(10000)
        self.assertTrue(RT._msgs[RT.INFOS.SEQ_BROKEN, 2] > 0)  # check if seq violation set

    def test_ray_reach(self):

        RT = rt_example()

        # blocking aperture
        RT.aperture_list[0] = ot.Aperture(ot.Surface("Circle", r=5), pos=RT.aperture_list[0].pos)

        # see if it is handled
        RT.trace(10000)

    def test_image_render_parameter(self):

        RT = rt_example()

        self.assertRaises(RuntimeError, RT.detector_image, 500)  # no rays traced
        self.assertRaises(RuntimeError, RT.detector_spectrum)  # no rays traced
        self.assertRaises(RuntimeError, RT.source_image, 500)  # no rays traced
        self.assertRaises(RuntimeError, RT.source_spectrum)  # no rays traced

        RT.trace(10000)

        self.assertRaises(ValueError, RT.detector_image, -5)  # negative number of pixels
        self.assertRaises(ValueError, RT.detector_image, 0)  # zero number of pixels
        self.assertRaises(ValueError, RT.source_image, -5)  # negative number of pixels
        self.assertRaises(ValueError, RT.source_image, 0)  # zero number of pixels
        self.assertRaises(ValueError, RT.detector_image, 500, detector_index=3)  # invalid index
        self.assertRaises(ValueError, RT.detector_spectrum, detector_index=3)  # invalid index
        self.assertRaises(ValueError, RT.detector_image, 500, detector_index=-3)  # invalid index
        self.assertRaises(ValueError, RT.detector_spectrum, detector_index=-3)  # invalid index
        self.assertRaises(ValueError, RT.detector_image, 500, source_index=3)  # invalid index
        self.assertRaises(ValueError, RT.detector_spectrum, source_index=3)  # invalid index
        self.assertRaises(ValueError, RT.detector_image, 500, source_index=-3)  # invalid index
        self.assertRaises(ValueError, RT.detector_spectrum, source_index=-3)  # invalid index
        self.assertRaises(ValueError, RT.source_image, 500, source_index=3)  # invalid index
        self.assertRaises(ValueError, RT.source_spectrum, source_index=3)  # invalid index
        self.assertRaises(ValueError, RT.source_image, 500, source_index=-3)  # invalid index
        self.assertRaises(ValueError, RT.source_spectrum, source_index=-3)  # invalid index

        self.assertRaises(ValueError, RT.detector_image, 500, extent="abc")  # invalid extent
        self.assertRaises(ValueError, RT.detector_image, 500, extent=[1, 2, 1, np.inf])  # invalid extent
        self.assertRaises(ValueError, RT.detector_spectrum, extent="abc")  # invalid extent
        self.assertRaises(ValueError, RT.detector_spectrum, extent=[1, 2, 1, np.inf])  # invalid extent

        RT.detector_list = []
        self.assertRaises(RuntimeError, RT.detector_image, 200)  # no detectors
        self.assertRaises(RuntimeError, RT.detector_spectrum)  # no detectors
        RT.ray_source_list = []
        self.assertRaises(RuntimeError, RT.source_image, 200)  # no sources
        self.assertRaises(RuntimeError, RT.source_spectrum)  # no sources

    @pytest.mark.slow
    def test_iterative_render(self):
        RT = rt_example()

        # testing only makes sense with multiple sources and detectors
        assert len(RT.ray_source_list) > 1
        assert len(RT.detector_list) > 1
        
        N_rays = int(RT.ITER_RAYS_STEP/10)
        
        # default call
        sim, dim = RT.iterative_render(N_rays, silent=True)
        self.assertEqual(len(sim), len(RT.ray_source_list))
        self.assertEqual(len(dim), 1)
        
        # default call with no sources
        sim, dim = RT.iterative_render(N_rays, no_sources=True, silent=True)
        self.assertEqual(len(sim), 0)

        # call with explicit position
        sim, dim = RT.iterative_render(N_rays, pos=13.3, silent=True)
        
        # call with explicit extent = [...]
        ext2 = [0, *RT.detector_list[0].extent[1:4]]
        sim, dim = RT.iterative_render(N_rays, extent=ext2, silent=True)
        self.assertTrue(np.all(dim[0].extent == ext2))
        
        # call with explicit detector_index
        sim, dim = RT.iterative_render(N_rays, detector_index=1, silent=True)
        
        # call with explicit detector pixel number
        sim, dim = RT.iterative_render(N_rays, N_px_D=400, silent=True)
        
        # call with explicit source pixel number
        sim, dim = RT.iterative_render(N_rays, N_px_S=400, silent=True)
        
        # call with multiple positions
        sim, dim = RT.iterative_render(N_rays, pos=[0, 5], silent=True)
        self.assertEqual(len(dim), 2)
        
        # call with multiple positions and detector_indices
        sim, dim = RT.iterative_render(N_rays, pos=[0, 5], 
                                       detector_index=[0, 1], silent=True)
        self.assertEqual(len(dim), 2)
        self.assertNotEqual(dim[0].coordinate_type, dim[1].coordinate_type) 
        # the detectors have different coordinate types

        # call with multiple positions and detector pixel numbers
        sim, dim = RT.iterative_render(int(RT.ITER_RAYS_STEP/4), pos=[0, 5], 
                                       N_px_D=[5, 500], silent=True)
        self.assertNotEqual(dim[0].Nx, dim[1].Nx) 
        
        # call with multiple source pixel numbers
        sim, dim = RT.iterative_render(int(RT.ITER_RAYS_STEP/4),
                                       N_px_S=[5, 500], silent=True)
        self.assertNotEqual(sim[0].Nx, sim[1].Nx) 
        
        # call with multiple positions and detector pixel numbers
        sim, dim = RT.iterative_render(int(RT.ITER_RAYS_STEP/4), pos=[0, 5],
                                       extent=[None, [-1, 1, 2, 3]], silent=True)
        self.assertFalse(np.all(dim[0].extent == dim[1].extent)) 
        
        self.assertRaises(ValueError, RT.iterative_render, 0)  # zero ray number
        self.assertRaises(ValueError, RT.iterative_render, -10)  # negative ray number
        self.assertRaises(ValueError, RT.iterative_render, 10000, -2)  # negative pixel number
        self.assertRaises(ValueError, RT.iterative_render, 10000, 0)  # zero pixel number
        self.assertRaises(ValueError, RT.iterative_render, 10000, 400, -2)  # negative pixel number
        self.assertRaises(ValueError, RT.iterative_render, 10000, 400, 0)  # zero pixel number
        
        self.assertRaises(ValueError, RT.iterative_render, 10000, N_px_D=[2, 500])  # len(N_px_D) != len(pos)
        self.assertRaises(ValueError, RT.iterative_render, 10000, N_px_S=[2, 500, 600])  
        # len(N_px_S) != len(RT.ray_source_list)
        self.assertRaises(ValueError, RT.iterative_render, 10000, extent=[None, None])  # len(extent) != len(pos)
        self.assertRaises(ValueError, RT.iterative_render, 10000, detector_index=[0, 1])  
        # len(detector_index) != len(pos)
        self.assertRaises(ValueError, RT.iterative_render, 10000, detector_index=[0, 1], pos=[0])  
        # len(detector_index) != len(pos)
        
        # coverage tests:

        # do multiple iterative steps
        sim, dim = RT.iterative_render(RT.ITER_RAYS_STEP*3, silent=True)

        # do multiple iterative steps with an un-full last iteration
        sim, dim = RT.iterative_render(RT.ITER_RAYS_STEP*3+100, silent=True)
        
        # render without detectors
        RT.detector_list = []
        sim, dim = RT.iterative_render(N_rays, silent=True)

        self.assertRaises(RuntimeError, RT.iterative_render, 10000, pos=0)  # no detectors to move

        RT.ray_source_list = []
        self.assertRaises(RuntimeError, RT.iterative_render, 10000)  # no ray_sources

    def test_tilted_plane_different_surface_types(self):
        """
        similar to brewster polarizer example
        checks bound fixing in numerical RT.find_surface_hit
        """

        n = ot.RefractionIndex("Constant", n=1.55)
        b_ang = np.arctan(1.55/1)

        RT = ot.Raytracer(outline=[-3, 3, -3, 3, -8, 12], silent=True)

        # source parameters
        RSS = ot.Surface("Circle", r=0.05)
        spectrum = ot.LightSpectrum("Monochromatic", wl=550.)
        s = [0, np.sin(b_ang), np.cos(b_ang)]

        # create sources
        RS0 = ot.RaySource(RSS, divergence="None", spectrum=spectrum, pos=[0.5, 0, -4], 
                          polarization="x", desc="x-pol")
        RS1 = ot.RaySource(RSS, divergence="None", spectrum=spectrum, pos=[0, 0, -4], 
                          polarization="y", desc="y-pol")
        RS2 = ot.RaySource(RSS, divergence="None", spectrum=spectrum, pos=[-0.5, 0, -4],
                          polarization="Random", desc="no pol")
        RT.add(RS0)
        RT.add(RS1)
        RT.add(RS2)


        surf_f = ot.SurfaceFunction(func=lambda x, y: np.tan(b_ang)*x, silent=True)
        surf1 = ot.Surface("Function", func=surf_f, r=0.7, silent=True)

        surf2 = ot.Surface("Circle", r=0.7, normal=[-np.sin(b_ang), 0, np.cos(b_ang)])

        X, Y = np.mgrid[-0.7:0.7:100j, -0.7:0.7:100j]
        Z = np.tan(b_ang)*X
        surf3 = ot.Surface("Data", r=0.7, data=Z, silent=True)

        for surf in [surf1, surf2, surf3]:
            L = ot.Lens(surf, surf, d=0.2, pos=[0, 0, 0.5], n=n)
            RT.add(L)

            RT.trace(10000)
            RT.remove(L)

            # check if all rays are straight and get parallel shifted
            p, s, _, _, _, _ = RT.rays.get_rays_by_mask(np.ones(RT.rays.N, dtype=bool), ret=[1, 1, 0, 0, 0, 0])
            self.assertAlmostEqual(np.mean(s[:, -2, 2]), 1)  # still straight
            self.assertTrue(np.allclose(p[:, -2, 0] - p[:, -3, 0], -0.0531867, atol=0.00001))  # parallel shift

    @pytest.mark.slow
    def test_non_sequental_surface_extent(self):
        """
        one surface has a larger z-extent and second surface starts at a lower z-value
        or second case, first surface ends at larger z-extent
        make sure we get no tracing or geometry check errors and cylinder ray detection works
        """

        RT = ot.Raytracer(outline=[-15, 15, -15, 15, -20, 50], silent=True)

        RSS = ot.Surface("Circle", r=4)
        RS = ot.RaySource(RSS, spectrum=ot.presets.light_spectrum.d65, pos=[0, 0, -20])
        RT.add(RS)

        # )) - lens, second surface embracing first
        R1, R2 = 5, 10
        front = ot.Surface("Sphere", r=0.999*R1, R=-R1)
        back = ot.Surface("Sphere", r=0.999*R2, R=-R2)
        L = ot.Lens(front, back, pos=[0, 0, 0], n=ot.RefractionIndex(), d=0.5)
        RT.add(L)

        FS = ot.Surface("Circle")
        F = ot.Filter(FS, spectrum=ot.TransmissionSpectrum("Constant", val=1), pos=[0, 0, 20])
        RT.add(F)

        N = 100000
        RT.trace(N)
        self.assertTrue(np.all(RT.rays.w_list[:, -2] > 0))
        
        # (( - lens, first surface embracing second
        RT.remove(L)
        front = ot.Surface("Sphere", r=0.999*R2, R=R2)
        back = ot.Surface("Sphere", r=0.999*R1, R=R1)
        L = ot.Lens(front, back, pos=[0, 0, 0], n=ot.RefractionIndex(), d=0.5)
        RT.add(L)
        RT.trace(N)
        self.assertTrue(np.all(RT.rays.w_list[:, -2] > 0))

        # )), but this time hitting space between surfaces
        RT.remove(RS)
        RS = ot.RaySource(ot.Surface("Ring", r=7, ri=6), spectrum=ot.presets.light_spectrum.d65, pos=[0, 0, -20])
        RT.add(RS)
        RT.trace(N)
        self.assertTrue(np.all(RT.rays.w_list[:, -2] == 0))
        
        # ((, but this time hitting space between surfaces
        RT.remove(L)
        front = ot.Surface("Sphere", r=0.999*R1, R=-R1)
        back = ot.Surface("Sphere", r=0.999*R2, R=-R2)
        L = ot.Lens(front, back, pos=[0, 0, 0], n=ot.RefractionIndex(), d=0.5)
        RT.add(L)
        RT.trace(N)
        self.assertTrue(np.all(RT.rays.w_list[:, -2] == 0))

    def test_brewster_and_fresnel_transmission(self):
        """
        test polarization and transmission for a setup with brewster angle
        """

        RT = ot.Raytracer(outline=[-5, 5, -5, 5, -10, 10], silent=True)

        # source parameters
        n = ot.RefractionIndex("Constant", n=1.55)
        b_ang = np.arctan(1.55/1)  # brewster angle
        RSS = ot.Surface("Circle", r=0.05)
        spectrum = ot.LightSpectrum("Monochromatic", wl=550.)
        s = [0, np.sin(b_ang), np.cos(b_ang)]

        # create sources
        RS0 = ot.RaySource(RSS, divergence="None", spectrum=spectrum, pos=[2, -2.5, -2], s=s, 
                          polarization="x", desc="x-pol", power=1)
        RS1 = ot.RaySource(RSS, divergence="None", spectrum=spectrum, pos=[0, -2.5, -2], s=s, 
                          polarization="y", desc="y-pol", power=1)
        RS2 = ot.RaySource(RSS, divergence="None", spectrum=spectrum, pos=[-2, -2.5, -2], s=s, 
                          polarization="Random", desc="no pol", power=10)
        RT.add(RS0)
        RT.add(RS1)
        RT.add(RS2)

        # add refraction index step
        rect = ot.Surface("Rectangle", dim=[10, 10])
        L1 = ot.Lens(rect, rect, de=0.5, pos=[0, 0, 0], n=n, n2=n)
        RT.add(L1)

        RT.trace(100000)

        def get_s_w_pol_source(index):

            mask = np.zeros(RT.rays.N, dtype=bool)
            Ns, Ne = RT.rays.b_list[index:index + 2]
            mask[Ns:Ne] = True

            p, s, pol, w, wl, _ = RT.rays.get_rays_by_mask(mask, slice(None))
            return s, pol, w

        sx, polx, wx = get_s_w_pol_source(0)
        sy, poly, wy = get_s_w_pol_source(1)
        sn, poln, wn = get_s_w_pol_source(2)

        # check power after transmission
        # values for n=1.55, n0=1
        # see https://de.wikipedia.org/wiki/Brewster-Winkel
        w0 = wn[0, 0]
        self.assertTrue(np.allclose(wx[:, 1]/w0, 0.8301, atol=0.001))
        self.assertTrue(np.allclose(wy[:, 1]/w0, 1.0000, atol=0.001))
        self.assertAlmostEqual(np.mean(wn[:, 1])/w0, 0.915, delta=0.001)

        # check pol projection values
        self.assertTrue(np.allclose(polx[:, 0, 1]**2 + polx[:, 0, 2]**2, 0, atol=0.00001))
        self.assertTrue(np.allclose(polx[:, 1, 1]**2 + polx[:, 1, 2]**2, 0, atol=0.00001))
        self.assertTrue(np.allclose(poly[:, 0, 1]**2 + poly[:, 0, 2]**2, 1, atol=0.00001))
        self.assertTrue(np.allclose(poly[:, 1, 1]**2 + poly[:, 1, 2]**2, 1, atol=0.00001))

        mean_yz_proj = lambda pol: np.mean(np.sqrt(pol[:, 1]**2 + pol[:, 2]**2)) 
        self.assertAlmostEqual(mean_yz_proj(poln[:, 0]), 2/np.pi, delta=0.005)
        self.assertAlmostEqual(mean_yz_proj(poln[:, 1]), 2/np.pi, delta=0.005)

        for pols, ss in zip([polx[:, 0], polx[:, 1], poly[:, 0], poly[:, 1], poln[:, 0], poln[:, 1]],
                            [sx[:, 0], sx[:, 1], sy[:, 0], sy[:, 1], sn[:, 0], sn[:, 1]]):
            # pols stays unity vector
            polss = pols[:, 0]**2 + pols[:, 1]**2 + pols[:, 2]**2
            self.assertTrue(np.allclose(polss, 1, atol=0.0001))

            # pol and s are always perpendicular
            cross = misc.cross(pols, ss)
            crosss = cross[:, 0]**2 + cross[:, 1]**2 + cross[:, 2]**2
            self.assertTrue(np.allclose(crosss, 1, atol=0.0001))


if __name__ == '__main__':
    unittest.main()
