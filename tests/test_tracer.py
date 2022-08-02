#!/bin/env python3

import sys
sys.path.append('.')

import unittest
import numpy as np

import optrace as ot

from test_frontend import rt_example


class TracerTests(unittest.TestCase):

    def test_raytracer_init(self):
        o0 = [-5, 5, -5, 5, 0, 10]

        self.assertRaises(TypeError, ot.Raytracer, outline=5)  # incorrect outline
        self.assertRaises(ValueError, ot.Raytracer, outline=[5])  # incorrect outline
        self.assertRaises(ValueError, ot.Raytracer, outline=[5, 6, 7, 8, 9])  # incorrect outline
        self.assertRaises(ValueError, ot.Raytracer, outline=[-5, -10, -5, 5, 0, 10])  # incorrect outline
        self.assertRaises(TypeError, ot.Raytracer, outline=o0, AbsorbMissing=1)  # incorrect bool parameter
        self.assertRaises(TypeError, ot.Raytracer, outline=o0, no_pol=1)  # incorrect bool parameter
        self.assertRaises(TypeError, ot.Raytracer, outline=o0, threading=1)  # incorrect bool parameter
        self.assertRaises(TypeError, ot.Raytracer, outline=o0, silent=1)  # incorrect bool parameter
        self.assertRaises(TypeError, ot.Raytracer, outline=o0, n0=1)  # incorrect Refractionindex
    
    def test_raytracer_geometry(self):

        # checks add, remove, property_snapshot, compare_property_snapshot, _make_element_list and trace checks

        RT = ot.Raytracer(outline=[-5, 5, -5, 5, -10, 30])

        RT.add(ot.RaySource(ot.Surface("Point"), pos=[0, 0, 0]))
        self.assertEqual(len(RT.RaySourceList), 1)
        RT.add(ot.RaySource(ot.Surface("Point"), pos=[0, 0, 0]))
        self.assertEqual(len(RT.RaySourceList), 2)
        RT.remove(RT.RaySourceList[1])
        self.assertEqual(len(RT.RaySourceList), 1)

        RT.add(ot.Filter(ot.Surface("Circle"), spectrum=ot.TransmissionSpectrum("Constant", val=1), pos=[0, 0, 5]))
        self.assertEqual(len(RT.FilterList), 1)
        RT.remove(RT.FilterList[0])
        self.assertEqual(len(RT.FilterList), 0)

        RT.add(ot.Detector(ot.Surface("Circle"), pos=[0, 0, 10]))
        self.assertEqual(len(RT.DetectorList), 1)
        RT.remove(RT.DetectorList[0])
        self.assertEqual(len(RT.DetectorList), 0)

        RT.add(ot.Aperture(ot.Surface("Ring"), pos=[0, 0, 10]))
        self.assertEqual(len(RT.ApertureList), 1)
        RT.remove(RT.ApertureList[0])
        self.assertEqual(len(RT.ApertureList), 0)

        RT.add(ot.Lens(ot.Surface("Circle"), ot.Surface("Circle"),
                       n=ot.RefractionIndex("Constant", n=1.2), pos=[0, 0, 10]))
        self.assertEqual(len(RT.LensList), 1)
        RT.remove(RT.LensList[0])
        self.assertEqual(len(RT.LensList), 0)

        RT.add(ot.RaySource(ot.Surface("Circle"), pos=[0, 10, -10]))
        self.assertRaises(RuntimeError, RT.trace, 1000)  # RaySource outside outline
        RT.remove(RT.RaySourceList[-1])
        
        RT.add(ot.Detector(ot.Surface("Circle"), pos=[0, 10, 10]))
        self.assertRaises(RuntimeError, RT.trace, 1000)  # Detector outside outline
        RT.remove(RT.DetectorList[-1])

        RT.add(ot.Aperture(ot.Surface("Ring"), pos=[0, -6, 10]))
        self.assertRaises(RuntimeError, RT.trace, 1000)  # Aperture outside outline
        RT.remove(RT.ApertureList[-1])

        RT.remove(RT.RaySourceList[0])
        self.assertRaises(RuntimeError, RT.trace, 1000)  # RaySource Missing
        RT.add(ot.RaySource(ot.Surface("Point"), pos=[0, 0, 0]))
        
        RT.add(ot.Aperture(ot.Surface("Ring"), pos=[0, 0, -5]))
        self.assertRaises(RuntimeError, RT.trace, 1000)  # Element behind RaySources

        def countTrue(dict_):
            i = 0
            for key, val in dict_.items():
                if val:
                    i += 1
            return i
        
        # check snapshotting
        snap = RT.property_snapshot()
        RT.add(ot.Aperture(ot.Surface("Ring"), pos=[0, 0, 15]))
        snap2 = RT.property_snapshot()
        self.assertNotEqual(snap, snap2)

        # check snapshots comparison 1
        cmp = RT.compare_property_snapshot(snap, snap2)
        self.assertTrue(cmp["Any"])
        self.assertTrue(cmp["Apertures"])
        self.assertEqual(countTrue(cmp), 2)

        # check snapshots comparison 2
        RT.add(ot.Lens(ot.Surface("Circle"), ot.Surface("Circle"),
                       n=ot.RefractionIndex("Constant", n=1.2), pos=[0, 0, 10]))
        snap3 = RT.property_snapshot()
        cmp = RT.compare_property_snapshot(snap2, snap3)
        self.assertTrue(cmp["Any"])
        self.assertTrue(cmp["Lenses"])
        self.assertTrue(cmp["Ambient"])
        self.assertEqual(countTrue(cmp), 3)

        RT.remove(RT.RaySourceList[-1])
        assert not len(RT.RaySourceList)
        self.assertRaises(RuntimeError, RT.trace, 1000)  # no raysource

        # adding list to Raytracer works
        RT.add([ot.Detector(ot.Surface("Circle"), pos=[0, 0, 20]), ot.Aperture(ot.Surface("Ring"), pos=[0, 0, 30])])
        # check type checking for adding
        self.assertRaises(TypeError, RT.add, ot.Surface("Point"))  # Surface is a invalid element type

    def test_geometries(self):

        # test example geometries from presets
        for geom_func in ot.presets.Geometry.geometries:
            RT = ot.Raytracer(outline=[-10, 10, -10, 10, -10, 80], silent=True)
            RSS = ot.Surface("Circle", r=0.5)
            RS = ot.RaySource(RSS, pos=[0, 0, -3], spectrum=ot.presets.light_spectrum.D65)
            RT.add(RS)

            geom = geom_func()
            RT.add(geom)

            RT.trace(100000)

        # additionally load Arizona Model, this time with dispersion
        geom = ot.presets.Geometry.ArizonaEye(dispersion=True)

    def test_focus(self):
        
        RT = ot.Raytracer(outline=[-3, 3, -3, 3, -60, 80], silent=True, n0=ot.RefractionIndex("Constant", n=1.1))
        RSS = ot.Surface("Circle", r=0.5)
        RS = ot.RaySource(RSS, pos=[0, 0, -3], spectrum=ot.presets.light_spectrum.D65)
        RT.add(RS)

        front = ot.Surface("Sphere", r=3, rho=1/30)
        back = ot.Surface("Asphere", r=3, rho=-1/20, k=1)
        n = ot.RefractionIndex("Constant", n=1.5)
        L = ot.Lens(front, back, n, de=0.1, pos=[0, 0, 0])
        RT.add(L)

        f = L.estimate_focal_length(n0=RT.n0)
        fs = 33.08433714
        self.assertAlmostEqual(f, fs, 6)

        self.assertRaises(ValueError, RT.autofocus, RT.autofocus_modes[0], z_start=5)  # no simulated rays
        
        RT.trace(200000)

        for method in RT.autofocus_modes:  # all methods
            for N in [50000, 200000, 500000]:  # N cases: less rays, all rays, more rays than simulated
                res, _, _, _ = RT.autofocus(method, z_start=5.0, N=N, return_cost=False)
                self.assertAlmostEqual(res.x, fs, delta=0.15)

        # source_index=0 with only one source leads to the same result
        res, _, _, _ = RT.autofocus(RT.autofocus_modes[0], z_start=5.0, source_index=0, N=N, return_cost=False)
        self.assertAlmostEqual(res.x, fs, delta=0.15)

        RS2 = ot.RaySource(ot.Surface("Point"), spectrum=ot.presets.light_spectrum.D65, direction="Diverging",
                           div_angle=0.5, pos=[0, 0, -60])
        RT.add(RS2)
        
        RT.trace(200000)

        # source_index=0 with two raysources should lead to the same result as if the second RS was missing
        res, _, _, _ = RT.autofocus(RT.autofocus_modes[0], z_start=5.0, source_index=0, N=N, return_cost=False)
        self.assertAlmostEqual(res.x, fs, delta=0.15)

        # 1/f = 1/b + 1/g with f=33.08, g=60 should lead to b=73.73
        res, _, _, _ = RT.autofocus(RT.autofocus_modes[0], z_start=5.0, source_index=1, N=N, return_cost=False)
        self.assertAlmostEqual(res.x, 73.73, delta=0.1)

        self.assertRaises(ValueError, RT.autofocus, RT.autofocus_modes[0], z_start=-100)  # z_start outside outline
        self.assertRaises(ValueError, RT.autofocus, "AA", z_start=-100)  # invalid mode
        self.assertRaises(ValueError, RT.autofocus, RT.autofocus_modes[0], z_start=-100, source_index=-1)
        # index negative
        self.assertRaises(ValueError, RT.autofocus, RT.autofocus_modes[0], z_start=-100, source_index=10)
        # index too large

        self.assertRaises(ValueError, RT.autofocus, RT.autofocus_modes[0], z_start=10, N=10)  # N too low

        # TODO test r- and vals-returns of autofocus()

    def test_raytracer_output_and_threading(self):
        RT = rt_example()
        RT.threading = False  # on by default, therefore test only off case
        RT.silent = False
        # raytracer not silent -> outputs messages and progress bar

        RT.trace(10000)
        RT.autofocus(RT.autofocus_modes[0], 12)
        RT.source_image(100)
        RT.detector_image(100)
        RT.source_spectrum()
        RT.detector_spectrum()
        RT.iterative_render(100000, silent=False)

        RT._msgs = np.ones((len(RT._infos), 2), dtype=int)
        RT._show_messages(1000)

    def test_numeric_tracing(self):

        # checks RT.find_hit for a numeric surface and Surface of type "Data"

        RT = ot.Raytracer(outline=[-3, 3, -3, 3, -10, 50], silent=True)

        x = np.linspace(-2, 2, 100)
        y = np.linspace(-2, 2, 100)
        X, Y = np.meshgrid(x, y)
        R = 12

        # sphere equation
        Z = R*(1-np.sqrt(1-(X**2 + Y**2)/R**2))

        n = ot.RefractionIndex("Constant", n=1.5)

        front = ot.Surface("Data", data=Z, r=2)
        back = ot.Surface("Data", data=-Z, r=2)
        L = ot.Lens(front, back, n, pos=[0, 0, 0], d=0.4)
        RT.add(L)

        RSS = ot.Surface("Circle", r=0.5)
        RS = ot.RaySource(RSS, spectrum=ot.LightSpectrum("Monochromatic", wl=555), direction="Parallel", pos=[0, 0, -3])
        RT.add(RS)

        RT.trace(100000)
        res, _, _, _ = RT.autofocus(RT.autofocus_modes[0], 5)

        # lens maker equation
        def getf(R1, R2, n, n0, d):
            D = (n-n0)/n0 * (1/R1 - 1/R2 + (n - n0) * d / (n*R1*R2))
            return 1 / D if D else np.inf 

        f_should = getf(R, -R, n(555), 1, L.d)

        self.assertAlmostEqual(res.x, f_should, delta=0.2)

    def test_abnormal_rays(self):

        RT = ot.Raytracer(outline=[-3, 3, -3, 3, -10, 50], absorb_missing=False, silent=True)

        RSS = ot.Surface("Circle", r=2)
        RS = ot.RaySource(RSS, spectrum=ot.LightSpectrum("Monochromatic", wl=555), direction="Parallel", pos=[0, 0, -3])
        RT.add(RS)

        surf1 = ot.Surface("Circle", r=3)
        surf2 = ot.Surface("Circle", r=1e-6)
        L = ot.Lens(surf2, surf1, n=ot.RefractionIndex("Constant", n=1.5), pos=[0, 0, 0], d=0.1)
        RT.add(L)

        N = 100000
        
        RT.trace(N)
        self.assertAlmostEqual(1, RT._msgs[RT._infos.only_hit_back, 2]/N, places=4)

        RT.LensList[0] = ot.Lens(surf1, surf2, n=ot.RefractionIndex("Constant", n=1.5), pos=[0, 0, 0], d=0.1)
        RT.trace(N)
        self.assertAlmostEqual(1, RT._msgs[RT._infos.only_hit_front, 1]/N, places=4)
        
        RT.absorb_missing = True
        RT.LensList[0] = ot.Lens(surf2, surf2, n=ot.RefractionIndex("Constant", n=1.5), pos=[0, 0, 0], d=0.1)
        RT.trace(N)
        self.assertAlmostEqual(1, RT._msgs[RT._infos.absorb_missing, 1]/N, places=4)

        # TODO test TIR
        # TODO test TTH
        # TODO test outline_intersection


if __name__ == '__main__':
    unittest.main()
