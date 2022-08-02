#!/bin/env python3

import sys
sys.path.append('.')

import os
import time
import doctest
import unittest
import numpy as np
import warnings

from pathlib import Path
import optrace.tracer.misc as misc
import optrace.tracer.color as color

import optrace as ot
from optrace.tracer.geometry.s_object import SObject as SObject
from optrace.tracer.base_class import BaseClass as BaseClass

import contextlib  # redirect stdout

# test backend excluding raytracer

SPEEDUP = False


# TODO create same surface with Sphere, Data and Function and check if they are equal
class BackendModuleTests(unittest.TestCase):

    # TODO Surface Function missing
    # TODO checks Masks and PlottingMeshes
    def test_surface(self):

        # check if init is working without exceptions
        S = [ot.Surface("Circle", r=4),
             ot.Surface("Sphere", r=5, rho=1/10),
             ot.Surface("Point"),
             ot.Surface("Line", ang=45),
             ot.Surface("Asphere", r=3.2, rho=-1/7, k=2),
             ot.Surface("Rectangle", dim=[5, 6]),
             ot.Surface("Data", r=2, data=np.array([[1, 2, 3], [-1, 0, -1], [5, 3, 0]])),
             ot.Surface("Ring", r=3, ri=0.2)]

        # check if exceptions in init work
        self.assertRaises(TypeError, ot.Surface)  # no type
        self.assertRaises(ValueError, ot.Surface, "ABC")  # invalid type
        self.assertRaises(ValueError, ot.Surface, "Asphere", r=3.2, rho=-1/7, k=5)  # r too large for surface
        self.assertRaises(ValueError, ot.Surface, "Sphere", r=-1)  # size negative
        self.assertRaises(ValueError, ot.Surface, "Sphere", rho=0)  # rho zero
        self.assertRaises(ValueError, ot.Surface, "Ring", r=1, ri=-0.2)  # size negative
        self.assertRaises(ValueError, ot.Surface, "Ring", r=1, ri=1.4)  # ri larger than r
        self.assertRaises(ValueError, ot.Surface, "Rectangle", dim=[-5, 5])  # size negative
        self.assertRaises(TypeError, ot.Surface, "Function", func=lambda x, y: x + y)  # func is no ot.SurfaceFunction

        x = np.linspace(-0.7, 5.3, 1000)
        y = np.linspace(-8, -2, 1000)
        p = np.random.uniform(-2, -1, size=(100, 3))
        s = np.random.uniform(-1, 1, size=(100, 3))
        s /= np.linalg.norm(s, axis=1)[:, np.newaxis]  # normalize
        s[:, 2] = np.abs(s[:, 2])  # only positive s_z

        # check if other functions throw
        for i, Si in enumerate(S):
            with self.subTest(i=i, Surface=Si.surface_type):
                Si.get_extent()
                Si.copy()

                # moving working correctly
                pos_new = np.array([2.3, -5, 10])
                Si.move_to(pos_new)
                self.assertTrue(np.all(pos_new == Si.pos))

                if Si.surface_type not in ["Point", "Line"]:
                    z = Si.get_values(x, y)
                    # print(z)
                    Si.get_mask(x, y)
                    Si.get_edge(nc=100)
                    Si.get_plotting_mesh(N=100)
                    Si.get_normals(x, y)

                    # check if moving actually moved the origin of the function
                    self.assertEqual(Si.get_values(np.array([pos_new[0]]), np.array([pos_new[1]]))[0], pos_new[2])

                    # check values
                    if Si.is_planar():
                        self.assertTrue(np.all(z == pos_new[2]))
                    elif Si is S[1]:  # Sphere
                        self.assertAlmostEqual(z[0], 10.94461486)
                        self.assertAlmostEqual(z[500], 10.0000009)
                    elif Si is S[4]:  # Asphere
                        self.assertAlmostEqual(z[0], 10.)  # == zmax since it's outside
                        self.assertAlmostEqual(z[300], 9.78499742)

                if Si.has_hit_finding():
                    Si.find_hit(p, s)

                if Si.is_planar():
                    Si.get_random_positions(N=1000)

        # test object lock
        self.assertRaises(AttributeError, S[0].__setattr__, "r46546", 4)  # _new_lock active
        self.assertRaises(RuntimeError, S[0].__setattr__, "r", 4)  # object locked
        self.assertRaises(RuntimeError, S[0].__setattr__, "dim", np.array([4, 5]))  # object locked

        # test array lock
        self.assertRaises(RuntimeError, S[6].__setattr__, "pos", np.array([4, 5]))  # object locked
        
        def setArrayElement():
            S[6].pos[0] = 1.
        self.assertRaises(ValueError, setArrayElement)  # array elements read-only

    def test_surface_function(self):

        # turn off warnings temporarily, since not specifying zmax, zmin in __init__ leads to one
        warnings.simplefilter("ignore")

        func = lambda x, y: 1 + x - y
        der = lambda x, y: tuple(np.ones_like(x), -np.ones_like(x))
        hits = lambda p, s: p*s  # not correct

        S0 = ot.SurfaceFunction(func, r=5, deriv_func=der, hit_func=hits)

        self.assertTrue(S0.has_hits())
        self.assertTrue(S0.has_derivative())

        # max and min values are +-5*sqrt(2), since the offset of 1 is removed internally
        self.assertAlmostEqual(S0.zmax, 5*np.sqrt(2), delta=1e-6)
        self.assertAlmostEqual(S0.zmin, -5*np.sqrt(2), delta=1e-6)

        # restore warnings
        warnings.simplefilter("default")

    def test_filter(self):

        self.assertRaises(TypeError, ot.Filter, ot.Surface("Circle"), spectrum=ot.presets.Spectrum.X, 
                          pos=[0, 0, 0])  # invalid TransmissionSpectrun
        self.assertRaises(RuntimeError, ot.Filter, ot.Surface("Point"), spectrum=ot.presets.light_spectrum.D65, 
                          pos=[0, 0, 0])  # non 2D surface
        self.assertRaises(RuntimeError, ot.Filter, ot.Surface("Line"), spectrum=ot.presets.light_spectrum.D65, 
                          pos=[0, 0, 0])  # non 2D surface

        # if getColor and calling implemented
        F = ot.Filter(ot.Surface("Circle"), spectrum=ot.TransmissionSpectrum("Constant", val=0.5), pos=[0, 0, 0])
        F.get_color()
        F(np.array([400, 500]))

        def invalid_set():
            F.aaa = 1
        self.assertRaises(AttributeError, invalid_set)  # _new_lock active

    def test_detector(self):

        self.assertRaises(RuntimeError, ot.Detector, ot.Surface("Point"), pos=[0, 0, 0])  # non 2D surface
        self.assertRaises(RuntimeError, ot.Detector, ot.Surface("Line"), pos=[0, 0, 0])  # non 2D surface

        Det = ot.Detector(ot.Surface("Sphere"), pos=[0, 0, 0])
        Det.get_angle_extent() # TODO check actual values

        # TODO check Det.toAngleCoordinates()

        self.assertRaises(AttributeError, Det.__setattr__, "aaa", 1)  # _new_lock active

    def test_ray_storage(self):

        from optrace.tracer.ray_storage import RayStorage as RayStorage

        RS = RayStorage()
        self.assertRaises(RuntimeError, RS.make_thread_rays, 0, 0)  # no rays
        self.assertRaises(RuntimeError, RS.get_source_sections, 0)  # no rays
        self.assertRaises(RuntimeError, RS.get_rays_by_mask, np.array([]))  # no rays

        # TODO actual tests

    def test_s_object(self):

        front = ot.Surface("Sphere")
        back = ot.Surface("Circle")
        pos = [0, -1, 5]

        self.assertRaises(ValueError, SObject, front, pos, back)  # no d1, d2
        self.assertRaises(TypeError, SObject, front, pos, back, d1=[], d2=2)  # invalid d1 type
        self.assertRaises(TypeError, SObject, front, pos, back, d2=[], d1=1)  # invalid d2 type
        self.assertRaises(ValueError, SObject, front, pos, back, d1=1)  # d2 missing
        self.assertRaises(ValueError, SObject, front, pos, back, d2=1)  # d1 missing
        self.assertRaises(ValueError, SObject, front, pos, back, d1=-1, d2=1)  # d1 negative
        self.assertRaises(TypeError, SObject, front, pos, 5, 1, 1)  # invalid BackSurface type
        self.assertRaises(TypeError, SObject, 5, pos, back, 1, 1)  # invalid FrontSurface type
        self.assertRaises(TypeError, SObject, front, 5, back, 1, 1)  # invalid pos type
        self.assertRaises(ValueError, SObject, front, [5, 5], back, 1, 1)  # wrong pos shape

        L = ot.Lens(ot.Surface("Circle"), ot.Surface("Sphere"), n=ot.RefractionIndex("Constant", n=1.1), 
                    pos=[0, -1, 5], d=2)

        self.assertTrue(L.has_back_surface())
        self.assertTrue(np.all(L.pos == np.array([0, -1, 5])))

        self.assertRaises(RuntimeError, L.__setattr__, "Surface", ot.Surface("Circle")) 
        # surface needs to be set in a different way
        self.assertRaises(RuntimeError, L.__setattr__, "pos", [0, 1, 0]) 
        # pos needs to be set in a different way

        L.move_to([5, 6, 8])
        self.assertTrue(np.all(L.pos == np.array([5, 6, 8])))

        self.assertRaises(RuntimeError, L.set_surface, ot.Surface("Circle"))
        # only SObjects with one surface can be set

        AP = ot.Aperture(ot.Surface("Ring"), pos=[0, 1, 2])
        AP.set_surface(ot.Surface("Asphere"))
        self.assertEqual(AP.surface.surface_type, "Asphere")

        # getDesc for SObject with one and two Surfaces
        SObject(ot.Surface("Circle"), [0, 0, 0], ot.Surface("Circle"), d1=0.1, d2=0.1).get_desc()
        SObject(ot.Surface("Circle"), [0, 0, 0], d1=0.1, d2=0.1).get_desc()

    def test_base_class(self):

        self.assertRaises(TypeError, BaseClass, desc=1)  # desc not a string 
        self.assertRaises(TypeError, BaseClass, long_desc=1)  # long_desc not a string 
        self.assertRaises(TypeError, BaseClass, threading=1)  # threading not bool
        self.assertRaises(TypeError, BaseClass, silent=1)  # silent not bool
    
        BC = BaseClass(desc="a")
        BC.a = 1
        BC.crepr()
        BC.copy()
        str(BC)

        BC._new_lock = True
        BC.lock()

        self.assertRaises(RuntimeError, BC.__setattr__, "a", "")  # object locked
        self.assertRaises(AttributeError, BC.__setattr__, "dc", "")  # invalid property

        # check desc handling
        self.assertEqual(BC.get_long_desc(), BC.desc)
        self.assertEqual(BC.get_desc(), BC.desc)
        BC = BaseClass()
        self.assertEqual(BC.get_long_desc(fallback="abc"), "abc")

    def test_property_checker(self):

        pc = misc.PropertyChecker

        self.assertRaises(TypeError, pc.checkType, "", 5, bool)  # 5 not bool
        pc.checkType("", 5, int)  # 5 not bool
        pc.checkNoneOrCallable("", None)  # valid
        pc.checkNoneOrCallable("", lambda x: x)  # valid

        def test():
            return 1
        pc.checkNoneOrCallable("", test)  # valid
        
        self.assertRaises(ValueError, pc.checkBelow, "", 5, 4)  # 5 > 4
        self.assertRaises(ValueError, pc.checkNotAbove, "", 5, 4)  # 5 > 4
        self.assertRaises(ValueError, pc.checkNotBelow, "", 4, 5)  # 5 > 4
        self.assertRaises(ValueError, pc.checkAbove, "", 4, 5)  # 4 < 5
        self.assertRaises(ValueError, pc.checkIfIn, "", "test", ["test1", "test2"])  # not an element of

    def test_lens(self):

        front = ot.Surface("Sphere", r=3, rho=1/10)  # height 10 - sqrt(91)
        back = ot.Surface("Asphere", r=2, k=-5, rho=-1/20)  # height sqrt(26) - 5
        n = ot.presets.RefractionIndex.SF10
        n2 = ot.RefractionIndex("Constant", n=1.05)
        
        L = [ot.Lens(front, back, n, pos=[0, 0, 0], de=0.),
             ot.Lens(front, back, n, pos=[0, 0, 0], de=0.1),
             ot.Lens(front, back, n, pos=[0, 0, 0], d=0.8),
             ot.Lens(back, front, n, pos=[0, 0, 0], de=0.1),
             ot.Lens(back, front, n, pos=[0, 0, 0], d=0.1),
             ot.Lens(front, back, n, pos=[0, 0, 0], d1=0.5, d2=0.3, n2=n2),
             ot.Lens(front, front, n, pos=[0, 0, 0], d=0.1),
             ot.Lens(front, front, n, pos=[0, 0, 0], de=0.1),
             ot.Lens(front, front, n, pos=[0, 0, 0], de=-0.1),
             ot.Lens(front, front, n, pos=[0, 0, 0], d1=0.1, d2=0.1),
             ot.Lens(back, back, n, pos=[0, 0, 0], d=0.1),
             ot.Lens(back, back, n, pos=[0, 0, 0], d1=0.1, d2=0.1)]

        d = [10-np.sqrt(91) + np.sqrt(26)-5, 10-np.sqrt(91) + np.sqrt(26)-5 + 0.1,
             0.8, 0.1, 0.1, 0.8, 0.1, 10-np.sqrt(91) + 0.1, 0.1, 0.2, 0.1, 0.2]

        for di, Li in zip(d, L):
            self.assertAlmostEqual(di, Li.d1+Li.d2, 8)

        self.assertRaises(TypeError, ot.Lens, front, back, None, pos=[0, 0, 0], d=1)  # invalid RefractionIndex
        self.assertRaises(TypeError, ot.Lens, front, back, n, n2=front, pos=[0, 0, 0], d=1)  # invalid RefractionIndex
        self.assertRaises(ValueError, ot.Lens, front, back, n, pos=[0, 0, 0], d1=1)  # d2 missing
        self.assertRaises(ValueError, ot.Lens, front, back, n, pos=[0, 0, 0], d2=1)  # d1 missing
        self.assertRaises(RuntimeError, ot.Lens, front, ot.Surface("Point"), n, pos=[0, 0, 0])  # non 2D surface
        self.assertRaises(RuntimeError, ot.Lens, front, ot.Surface("Line"), n, pos=[0, 0, 0])  # non 2D surface
        self.assertRaises(AttributeError, L[0].__setattr__, "aaa", 2)  # _new_lock active

        # check estimate focal length
        R1 = 100
        R2 = 80
        d = 0.5
        wl = 550
        wl2 = 650
        n = ot.RefractionIndex("Abbe", n=1.5, V=60)
        n2 = ot.RefractionIndex("Abbe", n=1.1, V=60)
        spa = ot.Surface("Sphere", r=3, rho=1/R1)
        spb = ot.Surface("Sphere", r=3, rho=-1/R2)
        aspa = ot.Surface("Asphere", r=3, rho=1/R1)
        aspb = ot.Surface("Asphere", r=3, rho=-1/R2)
        circ = ot.Surface("Circle", r=3)

        # lens maker equation
        def getf(R1, R2, n, n0, d):
            D = (n-n0)/n0 * (1/R1 - 1/R2 + (n - n0) * d / (n*R1*R2))
            return 1 / D if D else np.inf 
      
        # check focal length for different surface constellations
        largs = dict(d=d, n=n, pos=[0, 0, 0])
        self.assertAlmostEqual(ot.Lens(spa, spb, **largs).estimate_focal_length(wl=wl), getf(R1, -R2, n(wl), 1, d))
        self.assertAlmostEqual(ot.Lens(spb, spa, **largs).estimate_focal_length(wl=wl), getf(-R2, R1, n(wl), 1, d))
        self.assertAlmostEqual(ot.Lens(spa, circ, **largs).estimate_focal_length(wl=wl), getf(R1, np.inf, n(wl), 1, d))
        self.assertAlmostEqual(ot.Lens(circ, spb, **largs).estimate_focal_length(wl=wl), getf(np.inf, -R2, n(wl), 1, d))
        self.assertAlmostEqual(ot.Lens(circ, spa, **largs).estimate_focal_length(wl=wl), getf(np.inf, R1, n(wl), 1, d))
        self.assertAlmostEqual(ot.Lens(spb, circ, **largs).estimate_focal_length(wl=wl), getf(-R2, np.inf, n(wl), 1, d))
        self.assertAlmostEqual(ot.Lens(circ, circ, **largs).estimate_focal_length(wl=wl), 
                               getf(np.inf, np.inf, n(wl), 1, d))
        self.assertAlmostEqual(ot.Lens(spa, spa, **largs).estimate_focal_length(wl=wl), getf(R1, R1, n(wl), 1, d))
        self.assertAlmostEqual(ot.Lens(spb, spb, **largs).estimate_focal_length(wl=wl), getf(-R2, -R2, n(wl), 1, d))
        self.assertAlmostEqual(ot.Lens(spa, spb, **largs).estimate_focal_length(n0=n2, wl=wl), 
                               getf(R1, -R2, n(wl), n2(wl), d))  # ambient n
        self.assertAlmostEqual(ot.Lens(spa, spb, **largs).estimate_focal_length(n0=n2, wl=wl), 
                               getf(R1, -R2, n(wl), n2(wl), d))  # ambient n
        self.assertAlmostEqual(ot.Lens(aspa, aspb, **largs).estimate_focal_length(n0=n2, wl=wl2), 
                               getf(R1, -R2, n(wl2), n2(wl2), d))  # different wl

        L = ot.Lens(ot.Surface("Circle"), ot.Surface("Data", data=np.ones((3, 3))), n=n, pos=[0, 0, 0], d=1)
        self.assertRaises(RuntimeError, L.estimate_focal_length)  # numeric type not valid for calculations

    def test_ray_source(self):

        if SPEEDUP:
            return

        def or_func(x, y):
            s = np.column_stack((-x, -y, np.ones_like(x)*5))
            ab = (s[:, 0]**2 + s[:, 1]**2 + s[:, 2]**2) ** 0.5
            s /= ab[:, np.newaxis]
            return s

        # use power other than default of 1
        # use position other than default of [0, 0, 0]
        # use s other than [0, 0, 1]
        rargs = dict(spectrum=ot.presets.light_spectrum.D50, or_func=or_func, pos=[0.5, -2, 3], 
                     power=2.5, s=[0, 0.5, 1], pol_angle=0.5)
       
        # possible surface types
        Surfaces = [ot.Surface("Point"), ot.Surface("Line"), ot.Surface("Circle"), 
                    ot.Surface("Rectangle"), ot.Surface("Ring")]

        # check most RaySource combinations
        # also checks validity of createRays()
        # the loop also has the nice side effect of checking of all image presets

        for Surf in Surfaces:
            for dir_type in ot.RaySource.directions:
                for or_type in ot.RaySource.orientations:
                    for pol_type in ot.RaySource.polarizations:
                        for Im in [None, *ot.presets.Image.all_presets]:

                            # only check RectangleSurface with Image being active/set
                            if Im is not None and Surf.surface_type != "Rectangle":
                                continue

                            RS = ot.RaySource(Surf, direction=dir_type, orientation=or_type, 
                                              polarization=pol_type, image=Im, **rargs)
                            RS.get_color()
                            p, s, pols, weights, wavelengths = RS.create_rays(10000)

                            self.assertGreater(np.min(s[:, 2]), 0)  # ray direction in positive direction
                            self.assertGreater(np.min(weights), 0)  # no zero weight rays
                            self.assertEqual(np.sum(weights), rargs["power"])  # rays amount to power
                            self.assertGreaterEqual(np.min(wavelengths), color.WL_MIN)  # inside visible range
                            self.assertLessEqual(np.max(wavelengths), color.WL_MAX)  # inside visible range

                            # check positions
                            self.assertTrue(np.all(p[:, 2] == rargs["pos"][2]))  # rays start at correct z-position
                            self.assertGreaterEqual(np.min(p[:, 0]), RS.surface.get_extent()[0])
                            self.assertLessEqual(np.max(p[:, 0]), RS.surface.get_extent()[1])
                            self.assertGreaterEqual(np.min(p[:, 1]), RS.surface.get_extent()[2])
                            self.assertLessEqual(np.max(p[:, 1]), RS.surface.get_extent()[3])

                            # s needs to be a unity vector
                            ss = s[:, 0]**2 + s[:, 1]**2 + s[:, 2]**2
                            self.assertAlmostEqual(np.max(ss), 1, 6)
                            self.assertAlmostEqual(np.min(ss), 1, 6)

                            # pol needs to be a unity vector
                            polss = pols[:, 0]**2 + pols[:, 1]**2 + pols[:, 2]**2
                            self.assertAlmostEqual(np.max(polss), 1, 6)
                            self.assertAlmostEqual(np.min(polss), 1, 6)

        rsargs = [ot.Surface("Rectangle"), [0, 0, 0]]
     
        self.assertRaises(TypeError, ot.RaySource, *rsargs, spectrum=1)  # invalid spectrum_type
        self.assertRaises(TypeError, ot.RaySource, *rsargs, or_func=1)  # invalid or_func
        self.assertRaises(ValueError, ot.RaySource, *rsargs, power=0)  # power needs to be above 0
        self.assertRaises(TypeError, ot.RaySource, *rsargs, power=None)  # invalid power type
        self.assertRaises(ValueError, ot.RaySource, *rsargs, div_angle=0)  # pdiv_angle needs to be above 0
        self.assertRaises(TypeError, ot.RaySource, *rsargs, div_angle=None)  # invalid div_angle type
        self.assertRaises(TypeError, ot.RaySource, *rsargs, pol_angle=None)  # invalid pol_angle type
        
        self.assertRaises(TypeError, ot.RaySource, *rsargs, orientation=1)  # invalid orientation
        self.assertRaises(TypeError, ot.RaySource, *rsargs, direction=1)  # invalid direction
        self.assertRaises(TypeError, ot.RaySource, *rsargs, polarization=1)  # invalid polarization
        self.assertRaises(ValueError, ot.RaySource, *rsargs, orientation="A")  # invalid orientation
        self.assertRaises(ValueError, ot.RaySource, *rsargs, direction="A")  # invalid direction
        self.assertRaises(ValueError, ot.RaySource, *rsargs,  polarization="A")  # invalid polarization

        # image error handling
        self.assertRaises(TypeError, ot.RaySource, *rsargs, image=1)  # invalid image type
        self.assertRaises(TypeError, ot.RaySource, *rsargs, image=[1, 2])  # invalid image type
        self.assertRaises(TypeError, ot.RaySource, *rsargs, image=[[1, 2], [3, 4]])  # invalid image type
        self.assertRaises(TypeError, ot.RaySource, *rsargs, image=np.ones((2, 2, 3, 2)))  # invalid image type
        self.assertRaises(TypeError, ot.RaySource, *rsargs, image=np.ones((2, 2, 2)))  # invalid image type
        self.assertRaises(ValueError, ot.RaySource, *rsargs, image=np.zeros((2, 2, 3)))  # image completely black
        self.assertRaises(ValueError, ot.RaySource, *rsargs, image=-np.ones((2, 2, 3)))  # invalid image values
        self.assertRaises(ValueError, ot.RaySource, *rsargs, image=2*np.ones((2, 2, 3)))  # invalid image values
        self.assertRaises(RuntimeError, ot.RaySource, *rsargs, 
                          image=np.ones((int(1.1*ot.RaySource._max_image_px), 1, 3)))  # image too large

        self.assertRaises(TypeError, ot.RaySource, *rsargs, s=1)  # invalid s type
        self.assertRaises(TypeError, ot.RaySource, *rsargs, s=[1, 3])  # invalid s type
        
        self.assertRaises(ValueError, ot.RaySource, ot.Surface("Sphere"), pos=[0, 0, 0], s=[1, 3]) 
        # currently only planar surfaces are supported

        RS = ot.RaySource(*rsargs)
        self.assertRaises(AttributeError, RS.__setattr__, "aaa", 2)  # _new_lock active
        self.assertRaises(RuntimeError, RS.create_rays, 1000)  # spectrum missing
        self.assertRaises(RuntimeError, RS.get_color)  # spectrum missing

    def test_r_image(self):

        self.assertRaises(TypeError, ot.RImage, 5)  # invalid extent
        self.assertRaises(ValueError, ot.RImage, [5, 6, 8])  # invalid extent
        self.assertRaises(ValueError, ot.RImage, [5, 6, 8, 7])  # invalid extent
        self.assertRaises(TypeError, ot.RImage, [5, 6, 8, 9], 5)  # invalid coordinate type
        self.assertRaises(ValueError, ot.RImage, [5, 6, 8, 9], "hjkhjk")  # invalid coordinate type
        self.assertRaises(TypeError, ot.RImage, [5, 6, 8, 9], offset=None)  # invalid offset type
        self.assertRaises(ValueError, ot.RImage, [5, 6, 8, 9], offset=1.2)  # invalid offset value
        self.assertRaises(ValueError, ot.RImage, [5, 6, 8, 9], offset=-0.1)  # invalid offset value

        img = ot.RImage([-1, 1, -2, 2], silent=True)
        self.assertFalse(img.has_image())
        
        N = 4
        p = np.array([[-1, -2], [-1, 2], [1, -2], [1, 2]])
        w = np.array([1, 1, 2, 2])
        wl = np.array([480, 550, 670, 750])
        img.render(N, p, w, wl)
        self.assertRaises(ValueError, img.render, N=ot.RImage.MAX_IMAGE_SIDE*1.2)  # pixel number too large
        self.assertRaises(ValueError, img.render, N=-2)  # pixel number negative
        self.assertRaises(ValueError, img.render, N=0)  # pixel number zero

        img.Nx
        img.Ny
        img.sx
        img.sy
        img.Apx
        img.get_power()
        img.get_luminous_power()
        img.get_xyz()
        img.getRGB()
        img.get_luv()
        self.assertTrue(img.has_image())

        for dm in ot.RImage.display_modes:
            img.get_by_display_mode(dm)

        self.assertRaises(ValueError, img.get_by_display_mode, "hjkhjkhjk")  # invalid mode

        self.assertEqual(img.get_power(), np.sum(w))

        # check rescaling
        P0 = img.get_power()
        img.rescale(250)  # valid value
        img.rescale(ot.RImage.MAX_IMAGE_SIDE)  # maximum value
        img.rescale(131)  # a prime number
        img.threading = False  # turn off threading
        img.rescale(1)  # smallest possible number
        self.assertAlmostEqual(P0, img.get_power())  # overall power stays the same even after rescaling
        img.rescale(ot.RImage.MAX_IMAGE_SIDE * 10)  # too large number
        self.assertRaises(ValueError, img.rescale, 0.5)  # N < 1
        self.assertRaises(ValueError, img.rescale, -1)  # N negative

        # saving and loading for valid path
        path = "test_img.npz"
        img.save(path)
        img = ot.RImage.load(path)
        img.silent = True

        # saving and loading for valid path, overwrite file
        path0 = img.save(path, overwrite=True)
        img = ot.RImage.load(path0)
        img.silent = True
        self.assertEqual(path, path0)
        
        # saving and loading for valid path, don't overwrite file
        path1 = img.save(path, overwrite=False)
        img = ot.RImage.load(path1)
        img.silent = True
        self.assertNotEqual(path, path1)
        
        # saving and loading for invalid path
        path2 = img.save(str(Path("hjkhjkljk") / "test_img.npz"))
        img = ot.RImage.load(path2)

        # delete files
        os.remove(path0)
        os.remove(path1)
        os.remove(path2)

        # png saving
        img.rescale(4)
        img.silent = True
        for imm in ot.RImage.display_modes:
            path3 = img.export_png(path, imm, log=True, overwrite=True, flip=False)
            path3 = img.export_png(path, imm, log=False, overwrite=True, flip=True)
        os.remove(path3)

        # check empty image
        RIm = ot.RImage(extent=[-1, 1, -1, 1])
        for mode in ot.RImage.display_modes:
            self.assertRaises(RuntimeError, RIm.get_by_display_mode, mode)

        # check image offset
        RIm.render()
        RIm.offset = 0.1
        for mode in ot.RImage.display_modes:
            RIm.get_by_display_mode(mode, log=False)
            RIm.get_by_display_mode(mode, log=True)

        # image with too small extent
        zero_ext = [0, ot.RImage.EPS/2, 0, ot.RImage.EPS/2]
        RIm = ot.RImage(extent=zero_ext)
        Im_zero_ext = RIm.extent.copy()
        RIm.render(keep_extent=False)
        self.assertFalse(np.all(RIm.extent == Im_zero_ext))
        RIm = ot.RImage(extent=zero_ext)
        RIm.render(keep_extent=True)
        self.assertTrue(np.all(RIm.extent == Im_zero_ext))
        
        # image with high aspect ratio in y 
        hasp_ext = [0, 1, 0, 1.2*ot.RImage.MAX_IMAGE_RATIO]
        RIm = ot.RImage(extent=hasp_ext)
        Im_hasp_ext = RIm.extent.copy()
        RIm.render(keep_extent=False)
        self.assertFalse(np.all(RIm.extent == Im_hasp_ext))
        RIm = ot.RImage(extent=hasp_ext)
        RIm.render(keep_extent=True)
        self.assertTrue(np.all(RIm.extent == Im_hasp_ext))
        
        # image with high aspect ratio in x
        hasp_ext = [0, 1.2*ot.RImage.MAX_IMAGE_RATIO, 0, 1]
        RIm = ot.RImage(extent=hasp_ext)
        Im_hasp_ext = RIm.extent.copy()
        RIm.render(keep_extent=False)
        self.assertFalse(np.all(RIm.extent == Im_hasp_ext))
        RIm = ot.RImage(extent=hasp_ext)
        RIm.render(keep_extent=True)
        self.assertTrue(np.all(RIm.extent == Im_hasp_ext))

    def test_refraction_index(self):

        func = lambda wl: 2.0 - wl/500/5 

        rargs = dict(V=50, coeff=[1.2, 1e-8], func=func, wls=[380, 780], vals=[1.5, 1.])

        for type_ in ot.RefractionIndex.n_types:
            n = ot.RefractionIndex(type_, **rargs)
            n(550)
            n.get_abbe_number()
            n.is_dispersive()

        # combinations to check for correct values
        R = [ot.RefractionIndex("Constant", n=1.2),
             ot.RefractionIndex("Cauchy", coeff=[1.2, 0.004]),
             ot.RefractionIndex("Function", func=func)]

        # check presets
        for material in ot.presets.RefractionIndex.all_presets:
            n0 = material(550)
            A0 = material.get_abbe_number()

            self.assertTrue(1 <= n0 <= 2.5)  # sane refraction index
            self.assertTrue(A0 == np.inf or A0 < 150)  # sane Abbe Number
                
        # check exceptions
        self.assertRaises(ValueError, ot.RefractionIndex, "ABC")  # invalid type
        n2 = ot.RefractionIndex("Function")
        self.assertRaises(RuntimeError, n2, 550)  # func missing
        self.assertRaises(ValueError, ot.RefractionIndex, "Constant", n=0.99)  # n < 1
        self.assertRaises(TypeError, ot.RefractionIndex, "Abbe", V=[1])  # invalid V type
        self.assertRaises(ValueError, ot.RefractionIndex, "Abbe", V=0)  # invalid V
        self.assertRaises(ValueError, ot.RefractionIndex, "Abbe", V=50, lines=[380, 480])  
        # lines need to have 3 elements
        self.assertRaises(ValueError, ot.RefractionIndex, "Abbe", V=50, lines=[380, 780, 480])  
        # lines need to be ascending
        self.assertRaises(ValueError, ot.RefractionIndex, "Function", func=lambda wl: 0.5 - wl/wl)  # func < 1
        self.assertRaises(ValueError, ot.RefractionIndex, "Data", wls=[380, 780], vals=[1.5, 0.9])  # vals < 1
        self.assertRaises(TypeError, ot.RefractionIndex, "Cauchy", coeff=1)  # invalid coeff type
        self.assertRaises(ValueError, ot.RefractionIndex, "Cauchy", coeff=[1, 0, 0, 0, 0])  # too many coeff
        self.assertRaises(RuntimeError, ot.RefractionIndex("Cauchy", coeff=[2, -1]), np.array([380., 780.]))  
        # n < 1 on runtime
        self.assertRaises(RuntimeError, ot.RefractionIndex("Cauchy"), 550)  # coeffs not specified

        # check values
        wl = np.array([450., 580.])
        Rval = np.array([[1.2, 1.2],
                         [1.219753086, 1.211890606],
                         [1.82, 1.768]])

        for i, Ri in enumerate(R):
            for j, wlj in enumerate(wl):
                self.assertAlmostEqual(Ri(wlj), Rval[i, j], places=5)

        # check if equal operator is working
        self.assertEqual(ot.presets.RefractionIndex.SF10, ot.presets.RefractionIndex.SF10)
        self.assertEqual(R[1], R[1])
        self.assertEqual(ot.RefractionIndex("Function", func=func), ot.RefractionIndex("Function", func=func))

        self.assertRaises(AttributeError, R[0].__setattr__, "aaa", 1)  # _new_lock active

    def test_color(self):
        
        if SPEEDUP:
            return

        doctest.testmod(ot.tracer.color)

        # raytracer geometry of only a Source
        RT = ot.Raytracer(outline=[-3, 3, -3, 3, 0, 6], silent=True)
        RSS = ot.Surface("Rectangle", dim=[6, 6])
        RS = ot.RaySource(RSS, pos=[0, 0, 0], spectrum=ot.presets.light_spectrum.D65)
        RT.add(RS)

        # check white color of spectrum in sRGB
        for RSci, WPi in zip(RS.get_color(), np.array([1, 1, 1])):
            self.assertAlmostEqual(RSci, WPi, delta=0.001)

        # check white color of spectrum in XYZ
        spec_XYZ = RS.spectrum.get_xyz()[0, 0]
        for RSci, WPi in zip(spec_XYZ/spec_XYZ[1], color.WP_D65_XYZ):
            self.assertAlmostEqual(RSci, WPi, delta=0.001) 

        # check color of all rendered rays on RaySource in sRGB
        RT.trace(N=500000)
        spec_RS = RT.source_spectrum(N=2000)
        for RSci, WPi in zip(spec_RS.get_color(), np.array([1, 1, 1])):
            self.assertAlmostEqual(RSci, WPi, delta=0.015)

        # check color of all rendered rays on RaySource in XYZ
        RS_XYZ = spec_RS.get_xyz()[0, 0]
        for RSci, WPi in zip(RS_XYZ/RS_XYZ[1], color.WP_D65_XYZ):
            self.assertAlmostEqual(RSci, WPi, delta=0.015) 

        # assign colored image to RaySource
        RS.image = Image = np.array([[[0, 1, 0], [1, 1, 1]], [[0.2, 0.5, 0.1], [0, 0.8, 1.0]]], dtype=np.float32)
        SIm, _ = RT.iterative_render(100000000, N_px_S=2, silent=True)

        # get Source Image
        RS_RGB = np.flipud(SIm[0].getRGB())  # flip so element [0, 0] is in lower left
        Im_px = Image.reshape((4, 3))
        RS_px = RS_RGB.reshape((4, 3))

        # check if is near original image. 
        # Unfortunately many rays are needed for the color to be near the actual value
        # TODO compare XYZ Values
        for i in range(4):
            for RSp, Cp in zip(RS_px[i], Im_px[i]):
                self.assertAlmostEqual(RSp, Cp, delta=0.015)

        self.assertRaises(ValueError, color.tristimulus, np.array([]), "ABC")  # invalid tristimulus
        self.assertRaises(ValueError, color.illuminant, np.array([]), "ABC")  # invalid illuminant

        # coverage tests and zero images
        Img0 = np.zeros((10, 10, 3), dtype=np.float64)
        sRGBL0 = color.XYZ_to_sRGBLinear(Img0)  # zero image in conversion
        sRGB0 = color.sRGBLinear_to_sRGB(sRGBL0, clip=False)  # default setting is clip=True
        Luv0 = color.XYZ_to_Luv(Img0)  # zero image in Luv conversion

    def test_srgb_primaries(self):

        # also tests XYZ -> xyY, XYZ -> Luv and Luv -> u'v'L conversion

        def primary_to_xy(func, wl):
            XYZ = np.column_stack((color.tristimulus(wl, "X"),
                                   color.tristimulus(wl, "Y"),
                                   color.tristimulus(wl, "Z"))) * func(wl)[:, np.newaxis]
            XYZ = np.sum(XYZ, axis=0)
            return color.XYZ_to_xyY(np.array([[XYZ]]))[0, 0]
        
        def primary_to_u_v_(func, wl):
            XYZ = np.column_stack((color.tristimulus(wl, "X"),
                                   color.tristimulus(wl, "Y"),
                                   color.tristimulus(wl, "Z"))) * func(wl)[:, np.newaxis]
            XYZ = np.sum(XYZ, axis=0)
            return color.Luv_to_u_v_L(color.XYZ_to_Luv(np.array([[XYZ]])))[0, 0]

        wl = color.wavelengths(10000)
        r_xy = primary_to_xy(color.sRGB_r_primary, wl)
        g_xy = primary_to_xy(color.sRGB_g_primary, wl)
        b_xy = primary_to_xy(color.sRGB_b_primary, wl)
        w_xy = primary_to_xy(ot.presets.light_spectrum.sRGB_w, wl)

        prec_arg = dict(delta=2e-5)  # we expect some uncertainty due to different constants and conversion norms
        
        self.assertAlmostEqual(r_xy[0], color.SRGB_R_XY[0], **prec_arg)
        self.assertAlmostEqual(r_xy[1], color.SRGB_R_XY[1], **prec_arg)
        self.assertAlmostEqual(g_xy[0], color.SRGB_G_XY[0], **prec_arg)
        self.assertAlmostEqual(g_xy[1], color.SRGB_G_XY[1], **prec_arg)
        self.assertAlmostEqual(b_xy[0], color.SRGB_B_XY[0], **prec_arg)
        self.assertAlmostEqual(b_xy[1], color.SRGB_B_XY[1], **prec_arg)
        self.assertAlmostEqual(w_xy[0], color.SRGB_W_XY[0], **prec_arg)
        self.assertAlmostEqual(w_xy[1], color.SRGB_W_XY[1], **prec_arg)
        
        r_uv = primary_to_u_v_(color.sRGB_r_primary, wl)
        g_uv = primary_to_u_v_(color.sRGB_g_primary, wl)
        b_uv = primary_to_u_v_(color.sRGB_b_primary, wl)
        w_uv = primary_to_u_v_(ot.presets.light_spectrum.sRGB_w, wl)

        self.assertAlmostEqual(r_uv[0], color.SRGB_R_UV[0], **prec_arg)
        self.assertAlmostEqual(r_uv[1], color.SRGB_R_UV[1], **prec_arg)
        self.assertAlmostEqual(g_uv[0], color.SRGB_G_UV[0], **prec_arg)
        self.assertAlmostEqual(g_uv[1], color.SRGB_G_UV[1], **prec_arg)
        self.assertAlmostEqual(b_uv[0], color.SRGB_B_UV[0], **prec_arg)
        self.assertAlmostEqual(b_uv[1], color.SRGB_B_UV[1], **prec_arg)
        self.assertAlmostEqual(w_uv[0], color.SRGB_W_UV[0], **prec_arg)
        self.assertAlmostEqual(w_uv[1], color.SRGB_W_UV[1], **prec_arg)

    def test_spectrum(self):

        wl = color.wavelengths(100)
        sargs = dict(func=lambda x: x**2, lines=ot.presets.Lines.FDC, line_vals=[0.5, 1., 5], 
                     wls=np.array([380, 400, 780]), vals=np.array([1, 0.5, 0]))

        for type_ in ot.Spectrum.spectrum_types:
            spec = ot.Spectrum(type_, **sargs)  # init
            if spec.is_continuous():
                spec(wl)  # call

        # check presets
        for spec in ot.presets.Spectrum.tristimulus:
            if spec.is_continuous():
                spec(wl)  # call

        spec = ot.Spectrum("Function")
        self.assertRaises(RuntimeError, spec, np.array([380., 780.]))  # no function
        spec = ot.Spectrum("Data")
        self.assertRaises(RuntimeError, spec, np.array([380., 780.]))  # wls or vals not specified
        
        self.assertRaises(ValueError, ot.Spectrum, "Monochromatic", wl=100)  # wavelength outside visible range
        self.assertRaises(ValueError, ot.Spectrum, "Rectangle", wl0=100, wl1=500)  # wavelength outside visible range
        self.assertRaises(ValueError, ot.Spectrum, "Rectangle", wl0=400, wl1=900)  # wavelength outside visible range
        self.assertRaises(ValueError, ot.Spectrum, "Blackbody", T=0)  # temperature <= 0
        self.assertRaises(ValueError, ot.Spectrum, "Gaussian", mu=100)  # wavelength outside visible range
        self.assertRaises(ValueError, ot.Spectrum, "Gaussian", sig=0)  # sigma <= 0
        self.assertRaises(ValueError, ot.Spectrum, "Gaussian", fact=0)  # fact <= 0
        self.assertRaises(ValueError, ot.Spectrum, "Constant", val=-1)  # val < 0
        self.assertRaises(ValueError, ot.Spectrum, "Lines", lines=[400, 500], line_vals=[1, -1])  
        # line weights below zero
        self.assertRaises(ValueError, ot.Spectrum, "Lines", lines=[100, 500], line_vals=[1, 2])  
        # wavelength outside visible range
        self.assertRaises(ValueError, ot.Spectrum, "AAAA")  # invalid type
        self.assertRaises(ValueError, ot.Spectrum, "Data", wls=[400, 500], vals=[5, -1])  # vals below 0

        self.assertRaises(TypeError, ot.Spectrum, "Function", func=1)  # invalid function type
        self.assertRaises(ValueError, ot.Spectrum, "Lines", lines=[], line_vals=[1, 2])  # lines empty
        self.assertRaises(ValueError, ot.Spectrum, "Lines", lines=[4, 5], line_vals=[])  # line_vals empty
        self.assertRaises(TypeError, ot.Spectrum, "Lines", lines=5, line_vals=[1, 2])  # invalid lines type
        self.assertRaises(TypeError, ot.Spectrum, "Lines", lines=[400, 500], line_vals=5)  # invalid line_vals type
        self.assertRaises(TypeError, ot.Spectrum, 5)  # invalid type
        self.assertRaises(TypeError, ot.Spectrum, "Constant", quantity=1)  # quantity not str
        self.assertRaises(TypeError, ot.Spectrum, "Constant", unit=1)  # unit not str
        self.assertRaises(TypeError, ot.Spectrum, "Data", wls=1, vals=[1, 2])  # wls not array-like
        self.assertRaises(TypeError, ot.Spectrum, "Data", wls=[400, 500], vals=3)  # vals not array-like

        self.assertRaises(RuntimeError, ot.Spectrum, "Function", func=lambda wl: 1-wl/550)  # func below 0
        self.assertRaises(AttributeError, spec.__setattr__, "aaa", 1)  # _new_lock active
        self.assertRaises(RuntimeError, ot.Spectrum("Monochromatic", wl=500), 500)  # call of discontinuous type
       
        # equal operator
        self.assertTrue(ot.Spectrum("Constant") == ot.Spectrum("Constant"))
        self.assertFalse(ot.Spectrum("Constant", val=2) == ot.Spectrum("Constant", val=1))
        self.assertFalse(ot.Spectrum("Constant", val=2) == 1)  # comparison between different types
        self.assertTrue(ot.Spectrum("Data", wls=[400, 500], vals=[1, 2]) == ot.Spectrum("Data", 
                                                                                        wls=[400, 500], vals=[1, 2]))
        self.assertFalse(ot.Spectrum("Data", wls=[400, 500], vals=[1, 2]) == ot.Spectrum("Data", 
                                                                                         wls=[400, 500], vals=[1, 1]))
        self.assertFalse(ot.Spectrum("Data", wls=[400, 500], vals=[1, 2]) == ot.Spectrum("Data", 
                                                                                         wls=[400, 501], vals=[1, 2]))

    def test_transmission_spectrum(self):

        wl = color.wavelengths(100)
        sargs = dict(func=lambda x: 0.5, wls=np.array([380, 400, 780]), vals=np.array([1, 0.5, 0]))

        for type_ in ot.TransmissionSpectrum.spectrum_types:
            spec = ot.TransmissionSpectrum(type_, **sargs)  # init
            spec(wl)
            spec.get_xyz()
            spec.get_color()
           
        self.assertRaises(ValueError, ot.TransmissionSpectrum, "Lines")  # invalid discrete type
        self.assertRaises(ValueError, ot.TransmissionSpectrum, "Monochromatic")  # invalid discrete type
        self.assertRaises(ValueError, ot.TransmissionSpectrum, "Gaussian", fact=2)  # fact above 1 
        self.assertRaises(ValueError, ot.TransmissionSpectrum, "Constant", val=2)  # val above 1 
        self.assertRaises(ValueError, ot.TransmissionSpectrum, "Data", wls=[380, 500], vals=[0, 2])  # vals above 1
        self.assertRaises(RuntimeError, ot.TransmissionSpectrum, "Function", func=lambda wl: wl/550)  # func above 1 

    def test_light_spectrum(self):

        wl = color.wavelengths(100)
        sargs = dict(func=lambda x: x**2, lines=ot.presets.Lines.FDC, line_vals=[0.5, 1., 5], 
                     wls=np.array([380, 400, 780]), vals=np.array([1, 0.5, 0]))

        for type_ in ot.LightSpectrum.spectrum_types:
            spec = ot.LightSpectrum(type_, **sargs)  # init
            if spec.is_continuous():
                spec(wl)
            spec.get_xyz()
            spec.get_color()
            spec.random_wavelengths(1000)

        # check presets
        for spec in ot.presets.light_spectrum.all_presets:
            if spec.is_continuous():
                spec(wl)
            spec.get_xyz()
            spec.get_color()
            spec.random_wavelengths(1000)

        # LightSpectrum.makeSpectrum() is tested in test_Color()

        self.assertRaises(ValueError, ot.LightSpectrum, "Constant", val=0)  # val <= 0

        # LightSpectrum has specialized functions compared to Spectrum, 
        # check if they handle missing properties correctly
        self.assertRaises(RuntimeError, ot.LightSpectrum("Data").get_xyz)  # wls or vals not specified
        self.assertRaises(RuntimeError, ot.LightSpectrum("Data").random_wavelengths, 100)  # wls or vals not specified
        self.assertRaises(RuntimeError, ot.LightSpectrum("Lines", line_vals=[1, 1, 1]).get_xyz)  
        # lines or line_vals not specified
        self.assertRaises(RuntimeError, ot.LightSpectrum("Lines", lines=[400, 500, 600]).get_xyz)  
        # lines or line_vals not specified
        self.assertRaises(RuntimeError, ot.LightSpectrum("Lines", line_vals=[1, 1, 1]).random_wavelengths, 100)  
        # lines or line_vals not specified
        self.assertRaises(RuntimeError, ot.LightSpectrum("Lines", lines=[400, 500, 600]).random_wavelengths, 100)  
        # lines or line_vals not specified
        
    def test_misc(self):
        # doctests
        doctest.testmod(ot.tracer.misc)

        # interp1d tests
        a = np.linspace(0, 1, 10)
        b = np.linspace(0, 1, 10)
        z = np.ones((10, 10), dtype=np.float64)
        self.assertRaises(ValueError, misc.interp2d, a, b, z, np.array([0.5]), np.array([0.5]), "ABC")  # invalid method
        self.assertRaises(ValueError, misc.interp2d, a, b, z, np.array([-1.5]), np.array([0.5]))  # point outside 
        self.assertRaises(ValueError, misc.interp2d, a, b, z, np.array([0.5]), np.array([1.5]))   # point outside 
        self.assertRaises(ValueError, misc.interp2d, a, b, z, np.array([1.5]), np.array([0.5]))   # point outside 
        self.assertRaises(ValueError, misc.interp2d, a, b, z, np.array([0.5]), np.array([-1.5]))  # point outside 
        self.assertRaises(TypeError, misc.interp2d, z, b, z, np.array([0.5]), np.array([-1.5]))  # x invalid dimensions
        self.assertRaises(TypeError, misc.interp2d, a, z, z, np.array([0.5]), np.array([-1.5]))  # y invalid dimensions
        self.assertRaises(TypeError, misc.interp2d, a, b, z, z, np.array([-1.5]))  # xp invalid dimensions
        self.assertRaises(TypeError, misc.interp2d, a, b, z, np.array([0.5]), z)  # yp invalid dimensions
        self.assertRaises(TypeError, misc.interp2d, a, b, b, np.array([0.5]), np.array([0.5]))  # z invalid dimensions
        self.assertRaises(TypeError, misc.interp2d, a, b, z, np.array([0.5]), z)  # z invalid dimensions
        self.assertRaises(TypeError, misc.interp2d, a, b[:-1], z, np.array([0.5]), z)  # shape z = x * y does not match
        self.assertRaises(TypeError, misc.interp2d, a, b[:-1], z, np.array([0.5]), np.array([0.5, 1]))  
        # shape xp = yp does not match
        
        # values at the edge
        edge = np.array([1.])
        assert(edge[0] == a[-1])  # check if we did not change the ranges by mistake
        assert(edge[0] == b[-1])
        misc.interp2d(a, b, z, edge, edge, method="linear")
        misc.interp2d(a, b, z, edge, edge, method="nearest")

        # test timer
        @misc.timer
        def func(time_):
            return time.sleep(time_)

        with contextlib.redirect_stdout(None):
            func(1)  # s output
            func(0.01)  # ms output

        # test random_from_distribution
        self.assertRaises(RuntimeError, misc.random_from_distribution, np.array([0, 1]), np.array([0, 0]), 10)  
        # cdf zero
        self.assertRaises(RuntimeError, misc.random_from_distribution, np.array([0, 1]), np.array([0, -1]), 10)  
        # pdf < 0


if __name__ == '__main__':
    unittest.main()
