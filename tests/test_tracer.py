#!/bin/env python3

import sys
sys.path.append('.')

import pytest
import unittest
import numpy as np

import optrace as ot
from optrace.tracer import misc
from optrace.tracer.geometry.surface import Surface

from test_gui import rt_example


# lens maker equation
def lens_maker(R1, R2, n, n0, d):
    D = (n-n0)/n0 * (1/R1 - 1/R2 + (n - n0) * d / (n*R1*R2))
    return 1 / D if D else np.inf 

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

        # coverage: init with refraction index
        ot.Raytracer(outline=[-1, 1, -1, 1, -1, 1], n0=ot.RefractionIndex("Constant", n=1.2))

    def test_raytracer_snapshot(self):
        
        RT = ot.Raytracer(outline=[-5, 5, -5, 5, -10, 30])

        RS = ot.RaySource(ot.Point(), pos=[0, 0, 0], spectrum=ot.presets.light_spectrum.led_b1)
        F = ot.Filter(ot.CircularSurface(r=3), spectrum=ot.TransmissionSpectrum("Constant", val=1), pos=[0, 0, 5])
        DET = ot.Detector(ot.CircularSurface(r=3), pos=[0, 0, 10])
        AP = ot.Aperture(ot.RingSurface(r=3, ri=0.2), pos=[0, 0, 10])
        M = ot.Marker("Test", pos=[0, 0, 10])
        L = ot.Lens(ot.CircularSurface(r=3), ot.CircularSurface(r=3),
                       n=ot.RefractionIndex("Constant", n=1.2), pos=[0, 0, 10])

        # test actions for different element types
        for el, list_, ckey in zip([RS, L, AP, DET, F, M], 
                [RT.ray_sources, RT.lenses, RT.apertures, RT.detectors, RT.filters, RT.markers],
                ["RaySources", "Lenses", "Apertures", "Detectors", "Filters", "Markers"]):
        
            RT.add(el)
            el2 = el.copy()
            
            # check if ckey in compare dictionary is set
            snap = RT.property_snapshot()
            RT.add(el2)
            snap2 = RT.property_snapshot()
            cmp = RT.compare_property_snapshot(snap, snap2)
            self.assertTrue(cmp["Any"])
            self.assertTrue(cmp[ckey])
            if ckey == "Lenses":
                self.assertTrue(cmp["Ambient"])
            RT.remove(el2)

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
    
    def test_raytracer_misc(self):
        """checks tma, clear, extent and pos"""

        RT = rt_example()

        # check extent and pos
        self.assertTrue(np.allclose(RT.extent - RT.outline, 0))  # extent is outline
        posrt = [(RT.outline[0] + RT.outline[1])/2, (RT.outline[2] + RT.outline[3])/2, RT.outline[4]]
        self.assertTrue(np.allclose(np.array(RT.pos) - posrt, 0))  # position is position of first xy plane

        # check if tma works correctly
        abcd0 = RT.tma(489).abcd
        abcd1 = ot.Group(RT.lenses).tma(489, n0=RT.n0).abcd
        self.assertTrue(np.allclose(abcd0 - abcd1, 0))

        # clear also deletes rays
        RT.trace(10000)
        self.assertTrue(RT.rays.N > 0)
        RT.clear()
        self.assertFalse(RT.rays.N > 0)

    def test_geometry_checks(self):
        """test geometry checks before tracing"""
        
        RS = ot.RaySource(ot.CircularSurface(r=3), pos=[0, 0, 20], spectrum=ot.presets.light_spectrum.led_b1)
        F = ot.Filter(ot.CircularSurface(r=3), spectrum=ot.TransmissionSpectrum("Constant", val=1), pos=[0, 0, 5])
        AP = ot.Aperture(ot.RingSurface(r=3, ri=0.5), pos=[0, 0, 10])
        L = ot.Lens(ot.CircularSurface(r=3), ot.CircularSurface(r=3),
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
                        RT.trace(10000) 
                        self.assertTrue(RT.geometry_error)  # element outside outline
                        RT.geometry_error = False

                elif el is not RS:  # case i=1: element in front of source
                    RT.trace(10000) 
                    self.assertTrue(RT.geometry_error)  # for i == 1 correct order is broken
                    RT.geometry_error = False

                RT.remove(el)

            RT.add(RS)  # add second source that lies behind element, lead to sequentiality errors

        RT.remove([RS, RS0])
        RT.trace(10000) 
        self.assertTrue(RT.geometry_error)  # RaySource Missing
        RT.geometry_error = False
        
        RT.add(ot.RaySource(ot.Point(), pos=[0, 0, 0]))
        self.assertRaises(ValueError, RT.trace, 0)  # too few rays
        self.assertRaises(ValueError, RT.trace, 1e10)  # too many rays


        # additional ray source geometry checks

        RT = ot.Raytracer(outline=[-5, 5, -5, 5, -10, 30], silent=True)
        RS = ot.RaySource(ot.Point(), pos=[0, 0, -10])
        RT.add(RS)

        RT.add(ot.Lens(ot.SphericalSurface(r=3, R=-3.2), ot.CircularSurface(r=3), n=ot.presets.refraction_index.SF10, pos=[0, 0, 0], d=0.2))

        RT.trace(10000)  # works
        self.assertFalse(RT.geometry_error)
    
        RS.move_to([0, 0, -0.5])
        RT.trace(10000)  # should also work, lens encompasses source, but correct order and not collision
        self.assertFalse(RT.geometry_error)

        RS.move_to([0, 0, 0])
        RT.trace(1000)  # source inside lens
        self.assertTrue(RT.geometry_error)
        
        RT.geometry_error = False
        RS.move_to([0, 0, 10])
        RT.trace(1000)  # source behind lens
        self.assertTrue(RT.geometry_error)

        # load eye, flip back of cornea (=meniscus lens)
        # this leads to a surface overlap of front and back surface,
        # which should be detected by the geometry check
        RT.clear()
        RT.add(ot.presets.geometry.arizona_eye())
        RT.lenses[-2].back.flip()
        RT.trace(1000)
        self.assertTrue(RT.geometry_error)

    def test_element_collisions(self):

        # check collisions

        # get geometry
        geom = ot.presets.geometry.arizona_eye()
        geome = geom.elements

        # z overlap but no x-y overlap -> no collision
        geome[1].move_to(geome[0].front.pos + [0, 20, 0.01])
        self.assertFalse(ot.Raytracer.check_collision(geome[0].front, geome[1].front)[0])
        
        # no z-overlap -> no collision
        self.assertFalse(ot.Raytracer.check_collision(geome[0].front, geome[2].front)[0])
        # z-overlap but no collision
        self.assertFalse(ot.Raytracer.check_collision(geome[0].front, geome[0].back)[0])
        
        # collision
        geome[1].move_to(geome[0].front.pos + [0, 0, 0.01])  # cornea is now inside empty area of pupil
        coll, x, y, _ = ot.Raytracer.check_collision(geome[0].front, geome[1].front)
        self.assertTrue(coll)
        self.assertTrue(np.all(geome[0].front.get_values(x, y) >= geome[1].front.get_values(x, y)))

        # another collision
        geom = ot.presets.geometry.arizona_eye()
        geome = geom.elements
        geome[2].move_to(geome[0].front.pos + [0, 0, 0.2])
        coll, x, y, _ = ot.Raytracer.check_collision(geome[0].front, geome[2].front)
        self.assertTrue(coll)
        self.assertTrue(np.all(geome[0].front.get_values(x, y) >= geome[2].front.get_values(x, y)))

        # collision point - surface

        surf = ot.SphericalSurface(r=1, R=-5)
        point = ot.Point()

        # point in front of surface, only "hit" when order is reversed
        point.move_to([0, 0, -1])
        hit, _, _, _ = ot.Raytracer.check_collision(point, surf)
        self.assertFalse(hit)
        hit, _, _, _ = ot.Raytracer.check_collision(surf, point)
        self.assertTrue(hit)

        # point in behind surface, only "hit" when order is reversed
        point.move_to([0, 0, 1])
        hit, _, _, _ = ot.Raytracer.check_collision(point, surf)
        self.assertTrue(hit)
        hit, _, _, _ = ot.Raytracer.check_collision(surf, point)
        self.assertFalse(hit)
        
        # collision line - surface

        surf = ot.SphericalSurface(r=4, R=-5)
        line = ot.Line(r=0.1)

        # line in front of surface, only "hit" when order is reversed
        line.move_to([0, 0, -10])
        hit, _, _, _ = ot.Raytracer.check_collision(line, surf)
        self.assertFalse(hit)
        hit, _, _, _ = ot.Raytracer.check_collision(surf, line)
        self.assertTrue(hit)

        # line behind surface, only "hit" when order is reversed
        line.move_to([0, 0, 1])
        hit, _, _, _ = ot.Raytracer.check_collision(line, surf)
        self.assertTrue(hit)
        hit, _, _, _ = ot.Raytracer.check_collision(surf, line)
        self.assertFalse(hit)
        
        # Line no intersects surface, so regarless of order we get a collision
        line = ot.Line(r=5)
        line.move_to([0, 0, -0.1])
        hit, _, _, _ = ot.Raytracer.check_collision(line, surf)
        self.assertTrue(hit)
        hit, _, _, _ = ot.Raytracer.check_collision(surf, line)
        self.assertTrue(hit)

        # Coverage

        # type errors
        self.assertRaises(TypeError, ot.Raytracer.check_collision, ot.Point(), ot.Point())
        self.assertRaises(TypeError, ot.Raytracer.check_collision, ot.Line(), ot.Line())

    @pytest.mark.slow
    def test_focus(self):
        
        RT = ot.Raytracer(outline=[-3, 3, -3, 3, -10, 40], silent=True, n0=ot.RefractionIndex("Constant", n=1.1))
        RSS = ot.CircularSurface(r=0.5)
        RS = ot.RaySource(RSS, pos=[0, 0, -3])
        RT.add(RS)

        front = ot.SphericalSurface(r=3, R=30)
        back = ot.ConicSurface(r=3, R=-20, k=1)
        n = ot.RefractionIndex("Constant", n=1.5)
        L = ot.Lens(front, back, n, de=0.1, pos=[0, 0, 0])
        RT.add(L)

        f = L.tma(n0=RT.n0).efl
        fs = 33.08433714
        self.assertAlmostEqual(f, fs, 6)

        self.assertRaises(RuntimeError, RT.autofocus, RT.autofocus_methods[0], z_start=5)  # no simulated rays
        
        RT.trace(20000)

        for method in RT.autofocus_methods:  # all methods
            for N in [5000, 20000, 50000]:  # N cases: less rays, all rays, more rays than simulated
                for N_th in [1, 4, 8]:  # different thread counts
                    RT._force_threads = N_th
                    res, _ = RT.autofocus(method, z_start=5.0, N=N, return_cost=False)
                    self.assertAlmostEqual(res.x, fs, delta=0.15)

        # reset thread count overwrite
        RT._force_threads = None

        # source_index=0 with only one source leads to the same result
        res, _, = RT.autofocus(RT.autofocus_methods[0], z_start=5.0, source_index=0, N=N, return_cost=False)
        self.assertAlmostEqual(res.x, fs, delta=0.15)

        RS2 = ot.RaySource(ot.Point(), divergence="Isotropic", div_angle=0.5, pos=[0, 0, -60])
        RT.add(RS2)
       
        # enlargen outline so imaging works
        # the outline was smaller before to improve focus finding
        RT.outline = [-3, 3, -3, 3, -70, 80]
        RT.trace(100000)

        # source_index=0 with two raysources should lead to the same result as if the second RS was missing
        res, _ = RT.autofocus(RT.autofocus_methods[0], z_start=5.0, source_index=0, N=N, return_cost=False)
        self.assertAlmostEqual(res.x, fs, delta=0.15)

        # 1/f = 1/b + 1/g with f=33.08, g=60 should lead to b=73.73
        res, _ = RT.autofocus(RT.autofocus_methods[0], z_start=5.0, source_index=1, N=N, return_cost=False)
        self.assertAlmostEqual(res.x, 73.73, delta=0.1)

        self.assertRaises(ValueError, RT.autofocus, RT.autofocus_methods[0], z_start=0, N=0)  # N too small
        self.assertRaises(ValueError, RT.autofocus, RT.autofocus_methods[0], z_start=-100)  # z_start outside outline
        self.assertRaises(ValueError, RT.autofocus, "AA", z_start=10)  # invalid mode
        self.assertRaises(IndexError, RT.autofocus, RT.autofocus_methods[0], z_start=10, source_index=-1)
        # index negative
        self.assertRaises(IndexError, RT.autofocus, RT.autofocus_methods[0], z_start=10, source_index=10)
        # index too large

        # coverage tests

        RT.autofocus(RT.autofocus_methods[0], z_start=RT.outline[4]) # before all lenses, source_index=None
        RT.autofocus(RT.autofocus_methods[0], z_start=RT.outline[4], source_index=0) # before all lenses, source_index
        RT.autofocus(RT.autofocus_methods[0], z_start=RT.outline[5])  # behind all lenses
        RT.autofocus(RT.autofocus_methods[0], z_start=RT.lenses[0].extent[5] + 0.01)  # between lenses
        RT.autofocus(RT.autofocus_methods[0], z_start=RT.lenses[-1].extent[4] - 0.01)  # between lenses with n_ambient
        RT.autofocus("Irradiance Variance", z_start=RT.outline[5])  
        # leads to a warning, that minimum may not be found
        
        RT.silent = False
        RT.threading = False
        RT.autofocus("Irradiance Variance", z_start=RT.outline[5])  

        # aperture blocks all light
        RT.add(ot.Aperture(ot.CircularSurface(r=3), pos=[0, 0, 0]))
        RT.autofocus("Position Variance", z_start=10) # no rays here

    @pytest.mark.slow
    def test_uniform_emittance(self):
        """test that surfaces in ray source emit uniformly"""

        RT = ot.Raytracer(outline=[-5, 5, -5, 5, -10, 10], silent=True, n0=ot.RefractionIndex("Constant", n=1))

        for surf in [ot.CircularSurface(r=2), 
                    ot.RectangularSurface(dim=[1, 1]), 
                    ot.RectangularSurface(dim=[1, 0.75]), 
                    ot.RectangularSurface(dim=[1, 1/3]),
                    ot.Line(), 
                    ot.Line(angle=30), 
                    ot.Line(angle=90), 
                    ot.RingSurface(ri=1.5, r=2),
                    ot.RingSurface(ri=0.25, r=2)]:

            RS = ot.RaySource(surf, pos=[0, 0, 0])
            RT.add(RS)
            RT.trace(200000)

            im = RT.source_image(35)
            L = im.get_by_display_mode("Irradiance")
            L /= np.max(L)

            if isinstance(surf, Surface):
                if not isinstance(surf, ot.RectangularSurface):
                    Lx, Ly = L.shape[:2]
                    X, Y = np.mgrid[-Lx/2:Lx/2:Lx*1j, -Ly/2:Ly/2:Ly*1j]
                    m = X**2 + Y**2 < (Lx/2-1)**2

                    if isinstance(surf, ot.RingSurface):
                        m2 = X**2 + Y**2 > (Lx/2 * surf.ri/surf.r+1)**2
                        m = m & m2

                    L = L[m]
            else:
                L = L[L > 0]

            # we should have some remaining noise, but the std should be below 7%
            self.assertTrue(np.std(L) < 0.07)
            
            RT.remove(RS)

    @pytest.mark.slow
    def test_0ray_source_divergence(self):
        """
        tests the behaviour of different ray source divergence modes in 2D and 3D mode
        to check the function f we multiply the irradiance curve on the detector with 1/f
        the resulting function curve should be a constant with some noise
        we check the standard deviation on this curve
        """

        # we can't use spherical detectorsf for testing, 
        # since the projection is not equidistant and equal-area at the same time

        # make raytracer
        RT = ot.Raytracer(outline=[-100, 100, -100, 100, -10, 100], silent=True)

        RSS0 = ot.RectangularSurface(dim=[0.02, 0.02])
        RS0 = ot.RaySource(RSS0, divergence="Isotropic", div_2d=True,
                pos=[0, 0, 0], s=[0, 0, 1], div_angle=82, spectrum=ot.LightSpectrum("Monochromatic", wl=550))
        RT.add(RS0)
            
        # add Detector
        DETS = ot.RectangularSurface(dim=[100, 100])
        DET = ot.Detector(DETS, pos=[0, 0, 10])
        RT.add(DET)

        # Parallel Illumination 2D
        RS0.divergence = "None"
        RT.trace(2000000)
        img = RT.detector_image(63)
        x, y = img.cut(y=0, mode="Irradiance")
        y = y[0]
        self.assertTrue(np.std(y/np.max(y)) < 0.025)

        # Equal divergence in 2D, leads to 1/cos(e)**2 on detector
        RS0.divergence = "Isotropic"
        RT.trace(2000000)
        img = RT.detector_image(63)
        x, y = img.cut(y=0, mode="Irradiance")
        x = x[:-1] + (x[1] - x[0])/2  # bin edge to bin center position
        y = y[0]/np.cos(np.arctan(x/10))**2
        self.assertTrue(np.std(y/np.max(y)) < 0.05)

        # Equal divergence in 2D, leads to 1/cos(e)**3 on detector
        RS0.divergence = "Lambertian"
        RT.trace(2000000)
        img = RT.detector_image(63)
        x, y = img.cut(y=0, mode="Irradiance")
        x = x[:-1] + (x[1] - x[0])/2  # bin edge to bin center position
        y = y[0]/np.cos(np.arctan(x/10))**3
        self.assertTrue(np.std(y/np.max(y)) < 0.05)

        # Function mode, 1/cos(e)**2 leads to uniform illumination on detector
        RS0.divergence = "Function"
        RS0.div_func = lambda e: 1/np.cos(e)**2
        RT.trace(2000000)
        img = RT.detector_image(63)
        x, y = img.cut(y=0, mode="Irradiance")
        y = y[0]
        self.assertTrue(np.std(y/np.max(y)) < 0.025)

        # Function mode, needs to be rescaled by 1/function and 1/cos(e)**2 for equal illumination
        RS0.divergence = "Function"
        RS0.div_func = lambda e: 1+np.sqrt(e)
        RT.trace(2000000)
        img = RT.detector_image(63)
        x, y = img.cut(y=0, mode="Irradiance")
        x = x[:-1] + (x[1] - x[0])/2  # bin edge to bin center position
        y = y[0]/( 1 + np.sqrt(np.arctan(np.abs(x)/10)))/np.cos(np.arctan(x/10))**2
        self.assertTrue(np.std(y/np.max(y)) < 0.025)

        # switch to 3D divergence
        RS0.div_2d = False

        # Parallel Illumination 3D
        RS0.divergence = "None"
        RT.trace(2000000)
        img = RT.detector_image(63)
        img = img.get_by_display_mode("Irradiance")
        self.assertTrue(np.std(img/np.max(img)) < 0.025)

        # Equal divergence in 3D, leads to 1/cos(e)**3 on detector
        RS0.divergence = "Isotropic"
        RT.trace(2000000)
        img0 = RT.detector_image(63)
        img = img0.get_by_display_mode("Irradiance")
        x0, x1, y0, y1 = img0.extent
        X, Y = np.mgrid[x0:x1:63j, y0:y1:63j]
        Z = img / np.cos(np.arctan(np.sqrt(X**2 + Y**2)/10))**3
        self.assertTrue(np.std(Z/np.max(Z)) < 0.05)

        # Lambertian divergence in 3D, leads to 1/cos(e)**4 on detector
        RS0.divergence = "Lambertian"
        RT.trace(4000000)
        img0 = RT.detector_image(63)
        img = img0.get_by_display_mode("Irradiance")
        x0, x1, y0, y1 = img0.extent
        X, Y = np.mgrid[x0:x1:63j, y0:y1:63j]
        Z = img / np.cos(np.arctan(np.sqrt(X**2 + Y**2)/10))**4
        self.assertTrue(np.std(Z/np.max(Z)) < 0.075)

        # Function mode, 1/cos(e)**3 leads to uniform illumination on detector
        RS0.divergence = "Function"
        RS0.div_func = lambda e: 1/np.cos(e)**3
        RT.trace(2000000)
        img0 = RT.detector_image(63)
        Z = img0.get_by_display_mode("Irradiance")
        self.assertTrue(np.std(Z/np.max(Z)) < 0.075)

        # Function mode, needs to be rescaled by 1/function and 1/cos(e)**3 for uniform curve on detector
        RS0.divergence = "Function"
        RS0.div_func = lambda e: 1+np.sqrt(e)
        RT.trace(2000000)
        img0 = RT.detector_image(63)
        img = img0.get_by_display_mode("Irradiance")
        x0, x1, y0, y1 = img0.extent
        X, Y = np.mgrid[x0:x1:63j, y0:y1:63j]
        r = np.sqrt(X**2 + Y**2)
        Z = img / ( 1 + np.sqrt(np.arctan(r/10)))/np.cos(np.arctan(r/10))**3
        self.assertTrue(np.std(Z/np.max(Z)) < 0.05)
  
    @pytest.mark.slow
    def test_sphere_projections(self):

        # make raytracer
        RT = ot.Raytracer(outline=[-100, 100, -100, 100, -10, 100], silent=True)

        # add Detector
        R = 90
        DETS = ot.SphericalSurface(r=(1-1e-10)*R, R=-R)
        DET = ot.Detector(DETS, pos=[0, 0, R])
        RT.add(DET)

        # Isotropic leads to uniform illumination on sphere detector,
        # in Equal Area projection mode the iraddiance should be constant over the detector
        
        # add point source with isotropic light
        RSS0 = ot.Point()
        RS0 = ot.RaySource(RSS0, divergence="Isotropic", div_2d=False,
                pos=[0, 0, 0], s=[0, 0, 1], div_angle=89)
        RT.add(RS0)
        RT.trace(2000000)
        img0 = RT.detector_image(63, projection_method="Equal-Area")
        Z = img0.get_by_display_mode("Irradiance")
        # mask out area outside disc, leave 3 pixels margin
        X, Y = np.mgrid[-32:32:63j, -32:32:63j]
        mask = np.sqrt(X**2 + Y**2) < 29
        Z = Z[mask]
        self.assertTrue(np.std(Z/np.max(Z)) < 0.015)  # constant irradiance (neglecting noise) as it should be
        RT.remove(RS0)

        # Equidistant projection method
        # equally radially spaced points on sphere should also be equally spaced in projection
        RSS0 = ot.RectangularSurface(dim=[0.0001, 0.0001])
        # add point sources with parallel rays
        for theta in [-89.99, -60, -30, 0.001, 30, 60, 89.99]:  # slightly decentered central value so we only hit one pixel
            RS0 = ot.RaySource(RSS0, divergence="None", div_2d=False,
                    pos=[0, 0, 0], s_sph=[theta, 0])
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
        for rs in RT.ray_sources.copy():
            RT.remove(rs)

        # Stereographic projection: small circles on surface keep relative their shape, therefore also their side lengths
        # only the overall size changes
        # in the different projections circles near the sphere edge are distorted
        # here we check the side ratio of the generated detector image for each circle
        RSS0 = ot.Point()
        for phi in np.linspace(0, 360, 9):
            for theta in np.linspace(0, 1, 5)*87:
                # circular are on sphere at defined position
                RS0 = ot.RaySource(RSS0, divergence="Isotropic", div_2d=False,
                        pos=[0, 0, 0], s_sph=[theta, phi], div_angle=2)
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

        # for parallel light a circle and a sphere with orthographic projection produce the same image,
        # since the x,y hit coordinates are unchanged when projecting
        # and because of the parallel rays both surfaces have the same hit x,y coordinates
        RT.clear()
        RS = ot.RaySource(ot.CircularSurface(r=3), pos=[0, 0, 0], divergence="None", s=[0, 0, 1])
        det0 = ot.Detector(ot.CircularSurface(r=3.01), pos=[0, 0, 10])
        det1 = ot.Detector(ot.SphericalSurface(r=3.01, R=-3.2), pos=[0, 0, 10])

        RT.add([RS, det0, det1])
        RT.trace(100000)

        img0 = RT.detector_image(100, detector_index=0)
        img1 = RT.detector_image(100, detector_index=1, projection_method="Orthographic")

        # check that images on both detectors are equal
        ma = np.max(img0._img)
        mdev = np.mean(np.abs(img0._img - img1._img)/ma)
        self.assertAlmostEqual(mdev, 0, delta=1e-7)

    @pytest.mark.slow
    def test_ray_storage(self):
        
        RT = rt_example()
        RT.add(ot.RaySource(ot.Point(), spectrum=ot.LightSpectrum("Monochromatic", wl=550), pos=[0, 0, 0]))
        
        # we want the ray sources to be two and have different power
        assert len(RT.ray_sources) == 3
        
        RS0, RS1, RS2 = tuple(RT.ray_sources)
        powers = [(1, 1, 1), (2, 1, 1), (0.3456465, 4.57687168, np.pi/2)] 
        Ns = [30000, 30001, 52657]

        # test tracing for different number of rays, sources and different power ratios and threads
        for i in np.arange(2):  # different number of sources (1 source gets removed after each iteration)
            for powersi in powers:  # different power ratios
                for N in Ns:  # different rays numbers
                    for N_th in [1, 4, 8]:  # different number of threads

                        RS0.power, RS1.power, RS2.power = powersi

                        RT._force_threads = N_th
                        RT.trace(N)

                        # check number of rays
                        self.assertEqual(N, RT.rays.N)

                        # check power
                        P_s = np.sum([RS.power for RS in RT.ray_sources])
                        self.assertAlmostEqual(P_s, np.sum(RT.rays.w_list[:, 0]), delta=10/N)

                        # check power and number for each source
                        for Ni, RSi in enumerate(RT.ray_sources):
                            tup = RT.rays.source_sections(Ni)
                            self.assertEqual(tup[0].shape[0], RT.rays.N_list[Ni])
                            self.assertAlmostEqual(RT.rays.N_list[Ni]/RT.rays.N, RSi.power/P_s, delta=10/N)

                        # check source section call for all rays, call for each source has been tested above
                        tup = RT.rays.source_sections()
                        self.assertEqual(tup[0].shape[0], N)

                        # check rays_by_mask without parameters
                        tp1 = RT.rays.rays_by_mask()
                        tp2 = RT.rays.rays_by_mask(np.ones(N, dtype=bool))
                        for t1, t2 in zip(tp1, tp2):
                            np.testing.assert_array_equal(t1, t2)

                        # check rays_by_mask
                        ch = np.random.randint(0, 2, size=N).astype(bool)
                        N2 = np.count_nonzero(ch)
                        ch2 = np.random.randint(0, RT.rays.Nt, size=N2)
                        for ch2li in [None, ch2]:
                            for retli in [None, [1, 0, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0, 0]]:
                                for normli in [False, True]:
                                    tup = RT.rays.rays_by_mask(ch, ch2li, retli, normli)
                                    self.assertEqual(tup[3].shape[0], N2)  # correct shape 1
                                    self.assertEqual(tup[3].ndim, (1 if ch2li is not None else 2))  # correct shape 1
                                    self.assertTrue(retli is None or retli[1] or tup[1] is None)  # property omitted for ret[1] = 0
                                    self.assertTrue(retli is None or retli[5] or tup[5] is None)  # property omitted for ret[5] = 0
                                    self.assertTrue(retli is None or retli[6] or tup[6] is None)  # property omitted for ret[5] = 0
                                    # check if s is normalized
                                    if normli and (retli is None or retli[1]):
                                        # mask out values that have zero vectors (due to absorption)
                                        mask = np.all(np.isfinite(tup[1]), axis=2 if ch2li is None else 1)
                                        self.assertTrue(np.allclose(tup[1][mask, 0]**2 + tup[1][mask, 1]**2\
                                                                    + tup[1][mask, 2]**2, 1, rtol=0, atol=1e-4))

            RT.remove(RT.ray_sources[-1])

        # ray_lengths is tested in tracing tests

    def test_optical_lengths_ray_lengths(self):
       
        # optical setup of a source, a lens, filter and aperture
        # all rays have direction strictly parallel to z-axis
        # all surfaces have no extent in z-direction

        RT = ot.Raytracer(outline=[-10, 10, -10, 10, -2, 10])

        RS = ot.RaySource(ot.Point(), pos=[0, 0, 0])
        RT.add(RS)

        n = ot.presets.refraction_index.SF10
        L = ot.Lens(ot.CircularSurface(r=3), ot.CircularSurface(r=3), n=n, pos=[0, 0, 1], d1=0, d2=1.2)
        RT.add(L)
        
        F = ot.Filter(ot.CircularSurface(r=3), spectrum=ot.TransmissionSpectrum("Constant", val=1), pos=[0, 0, 5])
        RT.add(F)
        
        AP = ot.Aperture(ot.CircularSurface(r=3), pos=[0, 0, 9])
        RT.add(AP)

        RT.trace(1000)

        # optical path lengths
        olengths = RT.rays.optical_lengths()
        olengths2 = olengths.copy()
        olengths2[:, 0] = 1
        olengths2[:, 1] = n(RT.rays.wl_list)*1.2
        olengths2[:, 2] = 2.8
        olengths2[:, 3] = 4
        self.assertTrue(np.allclose(olengths - olengths2, 0, atol=1e-9, rtol=0))
        
        # ray lengths
        lengths = RT.rays.ray_lengths()
        lengths2 = lengths.copy()
        lengths2[:, 0] = 1
        lengths2[:, 1] = 1.2
        lengths2[:, 2] = 2.8
        lengths2[:, 3] = 4
        self.assertTrue(np.allclose(lengths - lengths2, 0, atol=1e-9, rtol=0))

    def test_ray_storage_misc(self):
        # coverage tests

        # warns that there are sources without rays
        RT = rt_example()
        RT.ray_sources[0].power = 0.000001
        RT.ray_sources[1].power = 1
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

    def test_numeric_tracing(self):
        """checks RT.find_hit for a numeric surface and DataSurface """

        RT = ot.Raytracer(outline=[-3, 3, -3, 3, -10, 50], silent=True)

        x = np.linspace(-2, 2, 100)
        y = np.linspace(-2, 2, 100)
        X, Y = np.meshgrid(x, y)
        R = 12

        # sphere equation
        Z = R*(1-np.sqrt(1-(X**2 + Y**2)/R**2))

        n = ot.RefractionIndex("Constant", n=1.5)

        front = ot.DataSurface2D(data=Z, r=2, silent=True)
        back = ot.DataSurface2D(data=-Z, r=2, silent=True)
        L = ot.Lens(front, back, n, pos=[0, 0, 0], d=0.4)
        RT.add(L)

        RSS = ot.CircularSurface(r=0.5)
        RS = ot.RaySource(RSS, spectrum=ot.LightSpectrum("Monochromatic", wl=555), divergence="None", pos=[0, 0, -3])
        RT.add(RS)

        RT.trace(100000)
        res, _ = RT.autofocus(RT.autofocus_methods[0], 5)

        f_should = lens_maker(R, -R, n(555), 1, L.d)

        self.assertAlmostEqual(res.x, f_should, delta=0.2)

    def test_absorb_missing(self):
        """
        infinitely small lens -> almost all rays miss and are set to absorbed (due to absorb_missing = True)
        """

        RT = ot.Raytracer(outline=[-3, 3, -3, 3, -10, 50], absorb_missing=False, silent=True)

        RSS = ot.CircularSurface(r=2)
        RS = ot.RaySource(RSS, spectrum=ot.LightSpectrum("Monochromatic", wl=555), divergence="None", pos=[0, 0, -3])
        RT.add(RS)

        surf = ot.CircularSurface(r=1e-6)
        L = ot.Lens(surf, surf, n=ot.RefractionIndex("Constant", n=1.5), pos=[0, 0, 0], d=0.1)
        RT.add(L)

        N = 10000
        RT.absorb_missing = True
        RT.trace(N)
        self.assertAlmostEqual(1, RT._msgs[RT.INFOS.ABSORB_MISSING, 1] / N, places=3)
        self.assertTrue(np.all(RT.rays.p_list[:, -1, 2] < RT.outline[5] - 1))  # absorbed before hitting outline
        
        # don't absorb with absorb_missing = False
        RT.absorb_missing = False
        RT.trace(N)
        self.assertEqual(RT._msgs[RT.INFOS.ABSORB_MISSING, 1], 0)
        self.assertTrue(np.allclose(RT.rays.p_list[:, -1, 2] - RT.outline[5], 0))  # hitting outline
    
    def test_absorb_missing_different_media(self):
        """
        infinitely small lens -> almost all rays miss and are set to absorbed because a media transition
        """

        RT = ot.Raytracer(outline=[-3, 3, -3, 3, -10, 50], absorb_missing=False, silent=True)

        RSS = ot.CircularSurface(r=2)
        RS = ot.RaySource(RSS, spectrum=ot.LightSpectrum("Monochromatic", wl=555), divergence="None", pos=[0, 0, -3])
        RT.add(RS)

        surf = ot.CircularSurface(r=1e-6)
        L = ot.Lens(surf, surf, n=ot.RefractionIndex("Constant", n=1.5), pos=[0, 0, 0], d=0.1, n2=ot.presets.refraction_index.SF10)
        RT.add(L)

        # check if they are absorbed before reaching the outline
        # and check message
        N = 10000
        RT.trace(N)
        self.assertAlmostEqual(RT._msgs[RT.INFOS.ABSORB_MEDIA_TRANS, 1] / N, 1, places=3)
        self.assertTrue(np.all(RT.rays.p_list[:, -1, 2] < RT.outline[5] - 1))  # absorbed before hitting outline

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

        RSS = ot.CircularSurface(r=2)
        RS = ot.RaySource(RSS, spectrum=ot.LightSpectrum("Monochromatic", wl=555), divergence="None",
                          pos=[0, 0, -3], s=[0, 0.1, 0.99])
        RT.add(RS)

        surf1 = ot.CircularSurface(r=10)
        surf2 = ot.CircularSurface(r=10)
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

        RSS = ot.CircularSurface(r=2)
        RS = ot.RaySource(RSS, spectrum=ot.LightSpectrum("Constant"), divergence="None",
                          pos=[0, 0, -3])
        RT.add(RS)

        surf1 = ot.CircularSurface(r=10)
        surf2 = ot.CircularSurface(r=10)
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
        RS = ot.RaySource(RSS, divergence="Isotropic",
                          pos=[0, 0, 0], s=[0, 0, 1], div_angle=75)
        RT.add(RS)
        
        geom = ot.presets.geometry.arizona_eye()
        geom.elements[2].move_to([0, 0, 0.15])
        RT.add(geom)

        RT.trace(1000)
        self.assertTrue(RT.geometry_error)  # object collision
        self.assertEqual(RT.rays.N, 0)  # not traced

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
        self.assertRaises(IndexError, RT.detector_image, 500, detector_index=3)  # invalid index
        self.assertRaises(IndexError, RT.detector_spectrum, detector_index=3)  # invalid index
        self.assertRaises(IndexError, RT.detector_image, 500, detector_index=-3)  # invalid index
        self.assertRaises(IndexError, RT.detector_spectrum, detector_index=-3)  # invalid index
        self.assertRaises(IndexError, RT.detector_image, 500, source_index=3)  # invalid index
        self.assertRaises(IndexError, RT.detector_spectrum, source_index=3)  # invalid index
        self.assertRaises(IndexError, RT.detector_image, 500, source_index=-3)  # invalid index
        self.assertRaises(IndexError, RT.detector_spectrum, source_index=-3)  # invalid index
        self.assertRaises(IndexError, RT.source_image, 500, source_index=3)  # invalid index
        self.assertRaises(IndexError, RT.source_spectrum, source_index=3)  # invalid index
        self.assertRaises(IndexError, RT.source_image, 500, source_index=-3)  # invalid index
        self.assertRaises(IndexError, RT.source_spectrum, source_index=-3)  # invalid index

        self.assertRaises(ValueError, RT.detector_image, 500, extent="abc")  # invalid extent
        self.assertRaises(ValueError, RT.detector_image, 500, extent=[1, 2, 1, np.inf])  # invalid extent
        self.assertRaises(ValueError, RT.detector_spectrum, extent="abc")  # invalid extent
        self.assertRaises(ValueError, RT.detector_spectrum, extent=[1, 2, 1, np.inf])  # invalid extent

        RT.detectors = []
        self.assertRaises(RuntimeError, RT.detector_image, 200)  # no detectors
        self.assertRaises(RuntimeError, RT.detector_spectrum)  # no detectors
        RT.ray_sources = []
        self.assertRaises(RuntimeError, RT.source_image, 200)  # no sources
        self.assertRaises(RuntimeError, RT.source_spectrum)  # no sources

    @pytest.mark.slow
    def test_iterative_render(self):
        RT = rt_example()
        RT.no_pol = True

        # testing only makes sense with multiple sources and detectors
        assert len(RT.ray_sources) > 1
        assert len(RT.detectors) > 1
        
        N_rays = int(RT.ITER_RAYS_STEP/10)
        
        # default call
        sim, dim = RT.iterative_render(N_rays, silent=True)
        self.assertEqual(len(sim), len(RT.ray_sources))
        self.assertEqual(len(dim), 1)
        self.assertTrue(dim[0].limit is None)
        
        # default call with no sources
        sim, dim = RT.iterative_render(N_rays, no_sources=True, silent=True)
        self.assertEqual(len(sim), 0)

        # call with explicit position
        sim, dim = RT.iterative_render(N_rays, pos=13.3, silent=True)
        
        # call with explicit extent = [...]
        ext2 = [0, *RT.detectors[0].extent[1:4]]
        sim, dim = RT.iterative_render(N_rays, extent=ext2, silent=True)
        self.assertTrue(np.all(dim[0].extent == ext2))
        
        # call with explicit detector_index
        sim, dim = RT.iterative_render(N_rays, detector_index=1, silent=True)
        
        # call with explicit detector pixel number
        sim, dim = RT.iterative_render(N_rays, N_px_D=315, silent=True)
        self.assertEqual(dim[0].N, 315)
        
        # call with explicit source pixel number
        sim, dim = RT.iterative_render(N_rays, N_px_S=315, silent=True)
        self.assertEqual(sim[0].N, 315)
        
        # call with explicit projection_method
        sim, dim = RT.iterative_render(N_rays, N_px_S=400, silent=True, detector_index=1, 
                                       projection_method="Stereographic")
        self.assertEqual(dim[0].projection, "Stereographic")
        
        # call with explicit limit
        sim, dim = RT.iterative_render(N_rays, N_px_S=400, silent=True, detector_index=1, 
                                       limit=5)
        self.assertEqual(dim[0].limit, 5)
        
        # call with multiple positions
        sim, dim = RT.iterative_render(N_rays, pos=[0, 5], silent=True)
        self.assertEqual(len(dim), 2)
        
        # call with multiple positions and detector_indices
        sim, dim = RT.iterative_render(N_rays, pos=[0, 5], 
                                       detector_index=[0, 1], silent=True)
        self.assertEqual(len(dim), 2)
        self.assertNotEqual(dim[0].projection, dim[1].projection) 
        # the detectors have different coordinate types

        # call with multiple positions and detector pixel numbers
        sim, dim = RT.iterative_render(int(RT.ITER_RAYS_STEP/4), pos=[0, 5], 
                                       N_px_D=[5, 500], silent=True)
        self.assertNotEqual(dim[0].Nx, dim[1].Nx) 
        
        # call with multiple positions and projection_methods
        sim, dim = RT.iterative_render(int(RT.ITER_RAYS_STEP/4), pos=[0, 5], 
                                       N_px_D=500, silent=True, detector_index=1,
                                       projection_method=["Equidistant", "Equal-Area"])
        self.assertNotEqual(dim[0].projection, dim[1].projection) 
        
        # call with multiple positions and limits
        sim, dim = RT.iterative_render(int(RT.ITER_RAYS_STEP/4), pos=[0, 5], 
                                       N_px_D=500, silent=True, detector_index=1,
                                       limit=[10, 12])
        self.assertNotEqual(dim[0].limit, dim[1].limit) 
        
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
        # len(N_px_S) != len(RT.ray_sources)
        self.assertRaises(ValueError, RT.iterative_render, 10000, extent=[None, None])  # len(extent) != len(pos)
        self.assertRaises(ValueError, RT.iterative_render, 10000, projection_method=["Equidistant", "Equal-Area"])
        # ^--  len(projection_method) != len(pos)
        self.assertRaises(ValueError, RT.iterative_render, 10000, detector_index=[0, 1])  
        # len(detector_index) != len(pos)
        self.assertRaises(ValueError, RT.iterative_render, 10000, detector_index=[0, 1], pos=[0])  
        # len(detector_index) != len(pos)
        self.assertRaises(ValueError, RT.iterative_render, 10000, limit=[4, 1], pos=[0])  
        # len(limit) != len(pos)
        
        # coverage tests:

        # do multiple iterative steps
        sim, dim = RT.iterative_render(RT.ITER_RAYS_STEP*2, silent=True)

        # do multiple iterative steps with an un-full last iteration
        sim, dim = RT.iterative_render(RT.ITER_RAYS_STEP*2+100, silent=True)
        
        # render without detectors
        RT.detectors = []
        sim, dim = RT.iterative_render(N_rays, silent=True)

        self.assertRaises(RuntimeError, RT.iterative_render, 10000, pos=0)  # no detectors to move

        RT.ray_sources = []
        self.assertRaises(RuntimeError, RT.iterative_render, 10000)  # no ray_sources

    def test_brewster_and_fresnel_transmission(self):
        """
        test polarization and transmission for a setup with brewster angle
        """

        RT = ot.Raytracer(outline=[-5, 5, -5, 5, -10, 10], silent=True)

        # source parameters
        n = ot.RefractionIndex("Constant", n=1.55)
        b_ang = np.arctan(1.55/1)  # brewster angle
        RSS = ot.CircularSurface(r=0.05)
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
        rect = ot.RectangularSurface(dim=[10, 10])
        L1 = ot.Lens(rect, rect, de=0.5, pos=[0, 0, 0], n=n, n2=n)
        RT.add(L1)

        RT.trace(100000)

        def get_s_w_pol_source(index):

            mask = np.zeros(RT.rays.N, dtype=bool)
            Ns, Ne = RT.rays.B_list[index:index + 2]
            mask[Ns:Ne] = True

            p, s, pol, w, wl, _, _ = RT.rays.rays_by_mask(mask, slice(None))
            return s, pol, w

        sx, polx, wx = get_s_w_pol_source(0)
        sy, poly, wy = get_s_w_pol_source(1)
        sn, poln, wn = get_s_w_pol_source(2)

        # check power after transmission
        # values for n=1.55, n0=1
        # see https://de.wikipedia.org/wiki/Brewster-Winkel
        w0 = wn[0, 0]
        self.assertTrue(np.allclose(wx[:, 1]/w0, 0.8301, atol=0.001, rtol=0))
        self.assertTrue(np.allclose(wy[:, 1]/w0, 1.0000, atol=0.001, rtol=0))
        self.assertAlmostEqual(np.mean(wn[:, 1])/w0, 0.915, delta=0.001)

        # check pol projection values
        self.assertTrue(np.allclose(polx[:, 0, 1]**2 + polx[:, 0, 2]**2, 0, atol=0.00001, rtol=0))
        self.assertTrue(np.allclose(polx[:, 1, 1]**2 + polx[:, 1, 2]**2, 0, atol=0.00001, rtol=0))
        self.assertTrue(np.allclose(poly[:, 0, 1]**2 + poly[:, 0, 2]**2, 1, atol=0.00001, rtol=0))
        self.assertTrue(np.allclose(poly[:, 1, 1]**2 + poly[:, 1, 2]**2, 1, atol=0.00001, rtol=0))

        mean_yz_proj = lambda pol: np.mean(np.sqrt(pol[:, 1]**2 + pol[:, 2]**2)) 
        self.assertAlmostEqual(mean_yz_proj(poln[:, 0]), 2/np.pi, delta=0.005)
        self.assertAlmostEqual(mean_yz_proj(poln[:, 1]), 2/np.pi, delta=0.005)

        for pols, ss in zip([polx[:, 0], polx[:, 1], poly[:, 0], poly[:, 1], poln[:, 0], poln[:, 1]],
                            [sx[:, 0], sx[:, 1], sy[:, 0], sy[:, 1], sn[:, 0], sn[:, 1]]):
            # pols stays unity vector
            polss = pols[:, 0]**2 + pols[:, 1]**2 + pols[:, 2]**2
            self.assertTrue(np.allclose(polss, 1, atol=0.0001, rtol=0))

            # pol and s are always perpendicular
            cross = misc.cross(pols, ss)
            crosss = cross[:, 0]**2 + cross[:, 1]**2 + cross[:, 2]**2
            self.assertTrue(np.allclose(crosss, 1, atol=0.0001, rtol=0))

    def test_source_detector_image_spectrum(self):
        # different projection methods are tested in surface tests
        # hit_detector, hit_source are tested in iterative_render
        # different plot modes and parameter pass-through (projection_method, desc, ...) is tested in test_plots
        # hit_detector special cases are tested in test_tracer_special
        pass

    def test_ideal_lens_imaging(self):

        # an ideal lens creates an ideal image
        # this image is exactly the same as the input image, as all rays from each pixel are mapped into the same output pixel
        # check if the images are the same

        Image = ot.presets.image.test_screen

        # make raytracer
        RT = ot.Raytracer(outline=[-5, 5, -5, 5, 0, 40])

        # add Raysource
        RSS = ot.RectangularSurface(dim=[4, 4])
        RS = ot.RaySource(RSS, divergence="Lambertian", div_angle=8, image=Image, s=[0, 0, 1], pos=[0, 0, 0])
        RT.add(RS)

        # add Lens 1
        L1 = ot.IdealLens(r=5, D=120, pos=[0, 0, 12])
        RT.add(L1)

        # position of image
        zi = RT.tma().image_position(0)

        # add Detector
        DetS = ot.RectangularSurface(dim=[10, 10])
        Det = ot.Detector(DetS, pos=[0, 0, zi])
        RT.add(Det)

        RT.trace(1000000)

        # calculate image size from image magnification (A element in ABCD matrix)
        beta = -RT.tma().matrix_at(0, zi)[0, 0]
        dimg_ext = np.array(RSS.extent[:4]) * beta

        simg = RT.source_image(500)
        dimg = RT.detector_image(500, extent=dimg_ext)

        # get pixel power image, detector image must be flipped
        diff = simg.img[:, :, 3] - np.flipud(np.fliplr(dimg.img[:, :, 3]))
        diffm = np.max(simg.img[:, :, 3])

        # calculate maximum deviation
        dev = np.max(np.abs(diff/diffm))

        # should be minimal, only numerical errors
        self.assertAlmostEqual(dev, 0, delta=1e-8)

    def test_ideal_lens_polarization(self):
        """test polarization for ideal lenses. Case 1: perpendicular light, case 2: divergent/convergent light"""

        RT = ot.Raytracer(outline=[-10, 10, -10, 10, 0, 40])

        # add Raysource
        RSS = ot.RectangularSurface(dim=[4, 4])
        RS = ot.RaySource(RSS, divergence="None", s=[0, 0, 1], pos=[0, 0, 0])
        RT.add(RS)

        # add Lenses
        RT.add(ot.IdealLens(r=5, D=120, pos=[0, 0, 12]))
        RT.add(ot.IdealLens(r=6, D=-50, pos=[0, 0, 24]))

        RT.trace(200000)
        _, s, pol, w, _, _, _ = RT.rays.rays_by_mask(ret=[0, 1, 1, 0, 0, 0, 0])
        scal = np.sum(s*pol, axis=2)[:, :-1]
        self.assertAlmostEqual(np.ptp(scal), 0, delta=1e-7)  # s and pol are perpendicular
        self.assertTrue(np.allclose(np.sum(pol*pol, axis=2)[:, :-1], 1))  # length is 1

    def test_polarization(self):
        """test polarization in a more complex setup"""

        RT = rt_example()
        RT.trace(200000)

        _, s, pol, _, _, _, _ = RT.rays.rays_by_mask(ret=[0, 1, 1, 0, 0, 0, 0])
        scal = np.sum(s*pol, axis=2)
        m = np.isnan(scal)
        scal = scal[~m]
        self.assertAlmostEqual(np.ptp(scal), 0, delta=1e-7)  # s and pol are perpendicular
        polpol = np.sum(pol*pol, axis=2)
        m = ~np.isnan(polpol)
        self.assertTrue(np.allclose(polpol[m], 1))  # length is 1

if __name__ == '__main__':
    unittest.main()
