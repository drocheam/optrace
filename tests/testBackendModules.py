#!/bin/env python3

import sys
sys.path.append('.')

import os
import doctest
import unittest
import numpy as np
import warnings

from pathlib import Path

import optrace as ot
from optrace.tracer.geometry.SObject import SObject as SObject
from optrace.tracer.BaseClass import BaseClass as BaseClass


class BackendModuleTests(unittest.TestCase):


    # Szrface Function missing
    # TODO checks Masks and PlottingMeshes
    def test_Surface(self):

        # check if init is working without exceptionss
        S =  [ot.Surface("Circle", r=4),
              ot.Surface("Sphere", r=5, rho=1/10),
              ot.Surface("Point"),
              ot.Surface("Line", ang=45),
              ot.Surface("Asphere", r=3.2, rho=-1/7, k=2),
              ot.Surface("Rectangle", dim=[5, 6]),
              ot.Surface("Data", r=2, Data=np.array([[1, 2, 3], [-1, 0, -1], [5, 3, 0]])),
              ot.Surface("Ring", r=3, ri=0.2)]

        # check if exceptions in init work
        self.assertRaises(TypeError, ot.Surface)  # no type
        self.assertRaises(ValueError, ot.Surface, "ABC")  # invalid type
        self.assertRaises(ValueError, ot.Surface, "Asphere", r=3.2, rho=-1/7, k=5) # r too large for surface
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
                Si.getExtent()
                Si.copy()

                # moving working correctly
                pos_new = np.array([2.3, -5, 10])
                Si.moveTo(pos_new)
                self.assertTrue(np.all(pos_new == Si.pos))

                if Si.surface_type not in ["Point", "Line"]:
                    z = Si.getValues(x, y)
                    # print(z)
                    Si.getMask(x, y)
                    Si.getEdge(nc=100)
                    Si.getPlottingMesh(N=100)
                    Si.getNormals(x, y)

                    # check if moving actually moved the origin of the function
                    self.assertEqual(Si.getValues(np.array([pos_new[0]]), np.array([pos_new[1]]))[0], pos_new[2])

                    # check values
                    if Si.isPlanar():
                        self.assertTrue(np.all(z == pos_new[2]))
                    elif Si is S[1]:  # Sphere
                        self.assertAlmostEqual(z[0], 10.94461486)
                        self.assertAlmostEqual(z[500], 10.0000009)
                    elif Si is S[4]: # Asphere
                        self.assertAlmostEqual(z[0], 10.)  # == zmax since it's outside
                        self.assertAlmostEqual(z[300], 9.78499742)
                    elif Si is S[6]:                    
                        self.assertAlmostEqual(z[0], 15.)  # == zmax since it's outside
                        self.assertAlmostEqual(z[200], 10.8993994)

                if Si.hasHitFinding():
                    Si.findHit(p, s)

                if Si.isPlanar():
                    Si.getRandomPositions(N=1000)

        # TODO check lock


    def test_SurfaceFunction(self):

        # turn off warnings temporarily, since not specifying zmax, zmin in __init__ leads to one
        warnings.simplefilter("ignore")

        func = lambda x, y: 1 + x - y
        der = lambda x, y: tuple(np.ones_like(x), -np.ones_like(x))
        hits = lambda p, s: p*s  # not correct

        S0 = ot.SurfaceFunction(func, r=5, derivative=der, hits=hits)

        self.assertTrue(S0.hasHits())
        self.assertTrue(S0.hasDerivative())

        # max and min values are +-5*sqrt(2), since the offset of 1 is removed internally
        self.assertAlmostEqual(S0.zmax, 5*np.sqrt(2), delta=1e-6)
        self.assertAlmostEqual(S0.zmin, -5*np.sqrt(2), delta=1e-6)

        # restore warnings
        warnings.simplefilter("default")

    def test_Filter(self):

        self.assertRaises(TypeError, ot.Filter, ot.Surface("Circle"), spectrum=ot.preset_spec_X, pos=[0, 0, 0]) # invalid TransmissionSpectrun
        self.assertRaises(RuntimeError, ot.Filter, ot.Surface("Point"), spectrum=ot.preset_spec_D65, pos=[0, 0, 0]) # non 2D surface
        self.assertRaises(RuntimeError, ot.Filter, ot.Surface("Line"), spectrum=ot.preset_spec_D65, pos=[0, 0, 0]) # non 2D surface

        # if getColor and calling implemented
        F = ot.Filter(ot.Surface("Circle"), spectrum=ot.TransmissionSpectrum("Constant", val=0.5), pos=[0, 0, 0])
        F.getColor()
        F(np.array([400, 500]))

        def invalid_set():
            F.aaa = 1
        self.assertRaises(AttributeError, invalid_set)  # _new_lock active

    def test_Detector(self):

        self.assertRaises(RuntimeError, ot.Detector, ot.Surface("Point"), pos=[0, 0, 0]) # non 2D surface
        self.assertRaises(RuntimeError, ot.Detector, ot.Surface("Line"), pos=[0, 0, 0]) # non 2D surface

        Det = ot.Detector(ot.Surface("Sphere"), pos=[0, 0, 0])
        Det.getAngleExtent() # TODO check actual values

        # TODO check Det.toAngleCoordinates()

        self.assertRaises(AttributeError, Det.__setattr__, "aaa", 1)  # _new_lock active

    def test_SObject(self):

        front = ot.Surface("Sphere")
        back = ot.Surface("Circle")
        pos = [0, -1, 5]

        self.assertRaises(ValueError, SObject, front, pos, back) # no d1, d2
        self.assertRaises(TypeError, SObject, front, pos, back, d1=[], d2=2) # invalid d1 type
        self.assertRaises(TypeError, SObject, front, pos, back, d2=[], d1=1) # invalid d2 type
        self.assertRaises(ValueError, SObject, front, pos, back, d1=1) # d2 missing
        self.assertRaises(ValueError, SObject, front, pos, back, d2=1) # d1 missing
        self.assertRaises(ValueError, SObject, front, pos, back, d1=-1, d2=1) # d1 negative
        self.assertRaises(TypeError, SObject, front, pos, 5, 1, 1) # invalid BackSurface type
        self.assertRaises(TypeError, SObject, 5, pos, back, 1, 1) # invalid FrontSurface type
        self.assertRaises(TypeError, SObject, front, 5, back, 1, 1) # invalid pos type
        self.assertRaises(ValueError, SObject, front, [5, 5], back, 1, 1) # wrong pos shape

        L = ot.Lens(ot.Surface("Circle"), ot.Surface("Sphere"), n=ot.RefractionIndex("Constant", n=1.1), pos=[0, -1, 5], d=2)

        self.assertTrue(L.hasBackSurface())
        self.assertTrue(np.all(L.pos == np.array([0, -1, 5])))

        self.assertRaises(RuntimeError, L.__setattr__, "Surface", ot.Surface("Circle")) # surface needs to be set in a different way
        self.assertRaises(RuntimeError, L.__setattr__, "pos", [0, 1, 0]) # pos needs to be set in a different way

        L.moveTo([5, 6, 8])
        self.assertTrue(np.all(L.pos == np.array([5, 6, 8])))

        self.assertRaises(RuntimeError, L.setSurface, ot.Surface("Circle")) # only SObjects with one surface can be set

        AP = ot.Aperture(ot.Surface("Ring"), pos=[0, 1, 2])
        AP.setSurface(ot.Surface("Asphere"))
        self.assertEqual(AP.Surface.surface_type, "Asphere")

    def test_BaseClass(self):

        self.assertRaises(TypeError, BaseClass, desc=1) # desc not a string 
        self.assertRaises(TypeError, BaseClass, long_desc=1) # long_desc not a string 
        self.assertRaises(TypeError, BaseClass, threading=1) # threading not bool
        self.assertRaises(TypeError, BaseClass, silent=1) # silent not bool
    
        BC = BaseClass(desc="a")
        BC.crepr()
        BC.copy()
        str(BC)

        BC._new_lock = True
        BC.lock()

        self.assertRaises(RuntimeError, BC.__setattr__, "desc", "") # object locked
        self.assertRaises(AttributeError, BC.__setattr__, "dc", "") # invalid property

        # check desc handling
        self.assertEqual(BC.getLongDesc(), BC.desc)
        self.assertEqual(BC.getDesc(), BC.desc)
        BC = BaseClass()
        self.assertEqual(BC.getLongDesc(fallback="abc"), "abc")

        self.assertRaises(TypeError, BaseClass._checkType, "", 5, bool) # 5 not bool
        BaseClass._checkType("", 5, int) # 5 not bool
        BaseClass._checkNoneOrCallable("", None) # valid
        BaseClass._checkNoneOrCallable("", lambda x: x) # valid

        def test():
            return 1
        BaseClass._checkNoneOrCallable("", test) # valid
        
        self.assertRaises(ValueError, BaseClass._checkBelow, "", 5, 4) # 5 > 4
        self.assertRaises(ValueError, BaseClass._checkNotAbove, "", 5, 4) # 5 > 4
        self.assertRaises(ValueError, BaseClass._checkNotBelow, "", 4, 5) # 5 > 4
        self.assertRaises(ValueError, BaseClass._checkAbove, "", 4, 5) # 4 < 5
        self.assertRaises(ValueError, BaseClass._checkIfIn, "", "test", ["test1", "test2"]) # not an element of

    def test_Lens(self):
        # TODO check estimateFocalLength()

        front = ot.Surface("Sphere", r=3, rho=1/10)  # height 10 - sqrt(91)
        back = ot.Surface("Asphere", r=2, k=-5, rho=-1/20) # height sqrt(26) - 5
        n = ot.preset_n_SF10
        n2 = ot.RefractionIndex("Constant", n=1.05)
        
        L = [ot.Lens(front, back, n, pos=[0, 0, 0], de=0.),
             ot.Lens(front, back, n, pos=[0, 0, 0], de=0.1),
             ot.Lens(front, back, n, pos=[0, 0, 0], d=0.8),
             ot.Lens(back, front, n, pos=[0, 0, 0], de=0.1),
             ot.Lens(back, front, n, pos=[0, 0, 0], d=0.1),
             ot.Lens(front, back, n, pos=[0, 0, 0], d1=0.5, d2=0.3, n2=n2),
             ot.Lens(front, front, n, pos=[0, 0, 0], d=0.1),
             ot.Lens(front, front, n, pos=[0, 0, 0], de=0.1),
             ot.Lens(front, front, n, pos=[0, 0, 0], d1=0.1, d2=0.1),
             ot.Lens(back, back, n, pos=[0, 0, 0], d=0.1),
             ot.Lens(back, back, n, pos=[0, 0, 0], d1=0.1, d2=0.1)]

        d = [10-np.sqrt(91) + np.sqrt(26)-5, 10-np.sqrt(91) + np.sqrt(26)-5 + 0.1,
             0.8, 0.1, 0.1, 0.8, 0.1, 10-np.sqrt(91) + 0.1, 0.2, 0.1, 0.2]

        for di, Li in zip(d, L):
            self.assertAlmostEqual(di, Li.d1+Li.d2, 8)

        self.assertRaises(TypeError, ot.Lens, front, back, None, pos=[0, 0 ,0])
        self.assertRaises(TypeError, ot.Lens, front, back, n, n2=front, pos=[0, 0, 0])
        self.assertRaises(RuntimeError, ot.Lens, front, ot.Surface("Point"), n, pos=[0, 0, 0]) # non 2D surface
        self.assertRaises(RuntimeError, ot.Lens, front, ot.Surface("Line"), n, pos=[0, 0, 0]) # non 2D surface
        self.assertRaises(AttributeError, L[0].__setattr__, "aaa", 2)  # _new_lock active

    def test_RaySource(self):

        def or_func(x, y):
            s = np.column_stack((-x, -y, np.ones_like(x)*5))
            ab = (s[:, 0]**2 + s[:, 1]**2 + s[:, 2]**2) ** 0.5
            s /= ab[:, np.newaxis]
            return s

        # use power other than default of 1
        # use position other than default of [0, 0, 0]
        # use s other than [0, 0, 1]
        rargs = dict(spectrum=ot.preset_spec_D50, or_func=or_func, pos=[0.5, -2, 3], power=2.5, s=[0, 0.5, 1], pol_angle=0.5)
       
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
                        for Im in [None, *ot.presets_image]:

                            # only check RectangleSurface with Image being active/set
                            if Im is not None and Surf.surface_type != "Rectangle":
                                continue

                            RS = ot.RaySource(Surf, direction=dir_type, orientation=or_type, 
                                              polarization=pol_type, Image=Im, **rargs)
                            RS.getColor()
                            p, s, pols, weights, wavelengths = RS.createRays(10000)

                            self.assertGreater(np.min(s[:, 2]), 0) # ray direction in positive direction
                            self.assertGreater(np.min(weights), 0) # no zero weight rays
                            self.assertEqual(np.sum(weights), rargs["power"]) # rays amount to power
                            self.assertGreaterEqual(np.min(wavelengths), ot.Color.WL_MIN) # inside visible range
                            self.assertLessEqual(np.max(wavelengths), ot.Color.WL_MAX) # inside visible range

                            # check positions
                            self.assertTrue(np.all(p[:, 2] == rargs["pos"][2])) # rays start at correct z-position
                            self.assertGreaterEqual(np.min(p[:, 0]), RS.Surface.getExtent()[0])
                            self.assertLessEqual(np.max(p[:, 0]), RS.Surface.getExtent()[1])
                            self.assertGreaterEqual(np.min(p[:, 1]), RS.Surface.getExtent()[2])
                            self.assertLessEqual(np.max(p[:, 1]),  RS.Surface.getExtent()[3])

                            # s needs to be a unity vector
                            ss = s[:, 0]**2 + s[:, 1]**2 + s[:, 2]**2
                            self.assertAlmostEqual(np.max(ss), 1, 6)
                            self.assertAlmostEqual(np.min(ss), 1, 6)

                            # pol needs to be a unity vector
                            polss = pols[:, 0]**2 + pols[:, 1]**2 + pols[:, 2]**2
                            self.assertAlmostEqual(np.max(polss), 1, 6)
                            self.assertAlmostEqual(np.min(polss), 1, 6)

        rsargs=[ot.Surface("Rectangle"), [0, 0, 0]]
     
        self.assertRaises(TypeError, ot.RaySource, *rsargs, spectrum=1) # invalid spectrum_type
        self.assertRaises(TypeError, ot.RaySource, *rsargs, or_func=1) # invalid or_func
        self.assertRaises(ValueError, ot.RaySource, *rsargs, power=0) # power needs to be above 0
        self.assertRaises(TypeError, ot.RaySource, *rsargs, power=None) # invalid power type
        self.assertRaises(ValueError, ot.RaySource, *rsargs, div_angle=0) # pdiv_angle needs to be above 0
        self.assertRaises(TypeError, ot.RaySource, *rsargs, div_angle=None) # invalid div_angle type
        self.assertRaises(TypeError, ot.RaySource, *rsargs, pol_angle=None) # invalid pol_angle type
        
        self.assertRaises(TypeError, ot.RaySource, *rsargs, orientation=1) # invalid orientationtype
        self.assertRaises(TypeError, ot.RaySource, *rsargs, direction=1) # invalid directiontype
        self.assertRaises(TypeError, ot.RaySource, *rsargs, polarization=1) # invalid polarizationtype
        self.assertRaises(ValueError, ot.RaySource, *rsargs, orientation="A") # invalid orientation
        self.assertRaises(ValueError, ot.RaySource, *rsargs, direction="A") # invalid direction
        self.assertRaises(ValueError, ot.RaySource,*rsargs,  polarization="A") # invalid polarization

        # image error handling
        self.assertRaises(TypeError, ot.RaySource, *rsargs, Image=1) # invalid image type
        self.assertRaises(TypeError, ot.RaySource, *rsargs, Image=[1, 2]) # invalid image type
        self.assertRaises(TypeError, ot.RaySource, *rsargs, Image=[[1, 2], [3, 4]]) # invalid image type
        self.assertRaises(TypeError, ot.RaySource, *rsargs, Image=np.ones((2, 2, 3, 2))) # invalid image type
        self.assertRaises(TypeError, ot.RaySource, *rsargs, Image=np.ones((2, 2, 2))) # invalid image type
        self.assertRaises(ValueError, ot.RaySource, *rsargs, Image=np.zeros((2, 2, 3))) # image completely black
        self.assertRaises(ValueError, ot.RaySource, *rsargs, Image=-np.ones((2, 2, 3))) # invalid image values
        self.assertRaises(ValueError, ot.RaySource, *rsargs, Image=2*np.ones((2, 2, 3))) # invalid image values
        self.assertRaises(RuntimeError, ot.RaySource, *rsargs, Image=np.ones((int(1.1*ot.RaySource._max_image_px), 1, 3))) # image too large

        self.assertRaises(TypeError, ot.RaySource, *rsargs, s=1) # invalid s type
        self.assertRaises(TypeError, ot.RaySource, *rsargs, s=[1, 3]) # invalid s type
        
        self.assertRaises(ValueError, ot.RaySource, ot.Surface("Sphere"), pos=[0, 0, 0], s=[1, 3]) # currently only planar surfaces are supported

        RS = ot.RaySource(*rsargs)
        self.assertRaises(AttributeError, RS.__setattr__, "aaa", 2)  # _new_lock active
        self.assertRaises(RuntimeError, RS.createRays, 1000) # spectrum missing
        self.assertRaises(RuntimeError, RS.getColor) # spectrum missing

    def test_RImage(self):

        self.assertRaises(TypeError, ot.RImage, 5) # invalid extent
        self.assertRaises(ValueError, ot.RImage, [5, 6, 8]) # invalid extent
        self.assertRaises(ValueError, ot.RImage, [5, 6, 8, 7]) # invalid extent
        self.assertRaises(TypeError, ot.RImage, [5, 6, 8, 9], 5) # invalid coordinate type
        self.assertRaises(ValueError, ot.RImage, [5, 6, 8, 9], "hjkhjk") # invalid coordinate type
        self.assertRaises(TypeError, ot.RImage, [5, 6, 8, 9], offset=None) # invalid offset type
        self.assertRaises(ValueError, ot.RImage, [5, 6, 8, 9], offset=1.2) # invalid offset value
        self.assertRaises(ValueError, ot.RImage, [5, 6, 8, 9], offset=-0.1) # invalid offset value

        img = ot.RImage([-1, 1, -2, 2], silent=True)
        self.assertFalse(img.hasImage())
        
        N = 4
        p = np.array([[-1, -2],[-1, 2],[1, -2],[1, 2]])
        w = np.array([1, 1, 2, 2])
        wl = np.array([480, 550, 670, 750])
        img.render(N, p, w, wl)

        img.Nx
        img.Ny
        img.sx
        img.sy
        img.Apx
        img.getPower()
        img.getLuminousPower()
        img.getXYZ()
        img.getRGB()
        img.getLuv()
        self.assertTrue(img.hasImage())

        for dm in ot.RImage.display_modes:
            img.getByDisplayMode(dm)

        self.assertRaises(ValueError, img.getByDisplayMode, "hjkhjkhjk") # invalid mode

        self.assertEqual(img.getPower(), np.sum(w))

        # check rescaling
        img.rescale(250) # valid value
        img.rescale(ot.RImage.MAX_IMAGE_SIDE) # maximum value
        img.rescale(131) # a prime number
        img.rescale(1) # smallest possible number
        img.rescale(ot.RImage.MAX_IMAGE_SIDE * 10) # too large number
        self.assertRaises(ValueError, img.rescale, 0.5) # N < 1
        self.assertRaises(ValueError, img.rescale, -1) # N negative

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

    def test_Raytracer_Init(self):
        o0 = [-5, 5, -5, 5, 0, 10]

        self.assertRaises(TypeError, ot.Raytracer, outline=5) # incorrect outline
        self.assertRaises(ValueError, ot.Raytracer, outline=[5]) # incorrect outline
        self.assertRaises(ValueError, ot.Raytracer, outline=[5, 6, 7, 8, 9]) # incorrect outline
        self.assertRaises(ValueError, ot.Raytracer, outline=[-5, -10, -5, 5, 0, 10]) # incorrect outline
        self.assertRaises(TypeError, ot.Raytracer, outline=o0, AbsorbMissing=1) # incorrect bool parameter
        self.assertRaises(TypeError, ot.Raytracer, outline=o0, no_pol=1) # incorrect bool parameter
        self.assertRaises(TypeError, ot.Raytracer, outline=o0, threading=1) # incorrect bool parameter
        self.assertRaises(TypeError, ot.Raytracer, outline=o0, silent=1) # incorrect bool parameter
        self.assertRaises(TypeError, ot.Raytracer, outline=o0, n0=1) # incorrect Refractionindex
    
    def test_Raytracer_Geometry(self):

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

        RT.add(ot.Lens(ot.Surface("Circle"), ot.Surface("Circle"), n=ot.RefractionIndex("Constant", n=1.2), pos=[0, 0, 10]))
        self.assertEqual(len(RT.LensList), 1)
        RT.remove(RT.LensList[0])
        self.assertEqual(len(RT.LensList), 0)

        RT.add(ot.Detector(ot.Surface("Circle"), pos=[0, 10, 10]))
        self.assertRaises(RuntimeError, RT.trace, 1000) # Detector outside outline
        RT.remove(RT.DetectorList[-1])

        RT.add(ot.Aperture(ot.Surface("Ring"), pos=[0, -6, 10]))
        self.assertRaises(RuntimeError, RT.trace, 1000) # Aperture outside outline
        RT.remove(RT.ApertureList[-1])

        RT.remove(RT.RaySourceList[0])
        self.assertRaises(RuntimeError, RT.trace, 1000) # RaySource Missing
        RT.add(ot.RaySource(ot.Surface("Point"), pos=[0, 0, 0]))
        
        RT.add(ot.Aperture(ot.Surface("Ring"), pos=[0, 0, -5]))
        self.assertRaises(RuntimeError, RT.trace, 1000) # Element behind RaySources

        def countTrue(dict_):
            i = 0
            for key, val in dict_.items():
                if val:
                    i += 1
            return i
        
        # check snapshotting
        snap = RT.PropertySnapshot()
        RT.add(ot.Aperture(ot.Surface("Ring"), pos=[0, 0, 15]))
        snap2 = RT.PropertySnapshot()
        self.assertNotEqual(snap, snap2)

        # check snapshots comparison 1
        cmp = RT.comparePropertySnapshot(snap, snap2)
        self.assertTrue(cmp["Any"])
        self.assertTrue(cmp["Apertures"])
        self.assertEqual(countTrue(cmp), 2)

        # check snapshots comparison 2
        RT.add(ot.Lens(ot.Surface("Circle"), ot.Surface("Circle"), n=ot.RefractionIndex("Constant", n=1.2), pos=[0, 0, 10]))
        snap3 = RT.PropertySnapshot()
        cmp = RT.comparePropertySnapshot(snap2, snap3)
        self.assertTrue(cmp["Any"])
        self.assertTrue(cmp["Lenses"])
        self.assertTrue(cmp["Ambient"])
        self.assertEqual(countTrue(cmp), 3)

    def test_Raytracer(self):
        pass

    def test_RefractionIndex(self):

        func = lambda wl: 2.0 - wl/500/5 

        rargs = dict(V=50, coeff=[1.2, 1e-8], func=func, wls=[380, 780], vals=[1.5, 1.])

        for type_ in ot.RefractionIndex.n_types:
            n = ot.RefractionIndex(type_, **rargs)
            n(550)
            n.getAbbeNumber()
            n.isDispersive()

        # combinations to check for correct values
        R = [ot.RefractionIndex("Constant", n=1.2),
             ot.RefractionIndex("Cauchy", coeff=[1.2, 0.004]),
             ot.RefractionIndex("Function", func=func)]

        # check presets
        for material in ot.presets_n:
            n0 = material(550)
            A0 = material.getAbbeNumber()

            self.assertTrue(n0 >= 1 and n0 <= 2.5) # sane refraction index
            self.assertTrue(A0 == np.inf or A0 < 150) # sane Abbe Number
                
        # check exceptions
        self.assertRaises(ValueError, ot.RefractionIndex, "ABC")  # invalid type
        n2 = ot.RefractionIndex("Function")
        self.assertRaises(RuntimeError, n2, 550)  # func missing
        self.assertRaises(ValueError, ot.RefractionIndex, "Constant", n=0.99)  # n < 1
        self.assertRaises(TypeError, ot.RefractionIndex, "Abbe", V=[1]) # invalid V type
        self.assertRaises(ValueError, ot.RefractionIndex, "Abbe", V=0) # invalid V
        self.assertRaises(ValueError, ot.RefractionIndex, "Abbe", V=50, lines=[380, 480]) # lines need to have 3 elements
        self.assertRaises(ValueError, ot.RefractionIndex, "Abbe", V=50, lines=[380, 780, 480]) # lines need to be ascending
        self.assertRaises(ValueError, ot.RefractionIndex, "Function", func=lambda wl: 1.5 - wl/550) # func < 1
        self.assertRaises(ValueError, ot.RefractionIndex, "Data", wls=[380, 780], vals=[1.5, 0.9]) # vals < 1
        self.assertRaises(TypeError, ot.RefractionIndex, "Cauchy", coeff=1) # invalid coeff type
        self.assertRaises(ValueError, ot.RefractionIndex, "Cauchy", coeff=[1, 0, 0, 0, 0]) # too many coeff
        self.assertRaises(RuntimeError, ot.RefractionIndex("Cauchy", coeff=[2, -1]), np.array([380., 780.])) # n < 1 on runtime

        # check values
        wl = np.array([450., 580.])
        Rval = np.array([[1.2, 1.2],
                         [1.219753086, 1.211890606],
                         [1.82, 1.768]])

        for i, Ri in enumerate(R):
            for j, wlj in enumerate(wl):
                self.assertAlmostEqual(Ri(wlj), Rval[i, j], places=5)

        # check if equal operator is working
        self.assertEqual(ot.preset_n_SF10, ot.preset_n_SF10)
        self.assertEqual(R[1], R[1])
        self.assertEqual(ot.RefractionIndex("Function", func=func), ot.RefractionIndex("Function", func=func))

        self.assertRaises(AttributeError, R[0].__setattr__, "aaa", 1)  # _new_lock active

    def test_Color(self):
        
        doctest.testmod(ot.tracer.Color)

        # raytracer geometry of only a Source
        RT = ot.Raytracer(outline=[-3, 3, -3, 3, 0, 6], silent=True)
        RSS = ot.Surface("Rectangle", dim=[6, 6])
        RS = ot.RaySource(RSS, pos=[0, 0, 0], spectrum=ot.preset_spec_D65)
        RT.add(RS)

        # check white color of spectrum in sRGB
        for RSci, WPi in zip(RS.getColor(), np.array([1, 1, 1])):
            self.assertAlmostEqual(RSci, WPi, delta=0.001)

        # check white color of spectrum in XYZ
        spec_XYZ = RS.spectrum.getXYZ()[0, 0]
        for RSci, WPi in zip(spec_XYZ/spec_XYZ[1], ot.Color._WP_D65_XYZ):
            self.assertAlmostEqual(RSci, WPi, delta=0.001) 

        # check color of all rendered rays on RaySource in sRGB
        RT.trace(N=500000)
        spec_RS = RT.SourceSpectrum(N=2000)
        for RSci, WPi in zip(spec_RS.getColor(), np.array([1, 1, 1])):
            self.assertAlmostEqual(RSci, WPi, delta=0.015)

        # check color of all rendered rays on RaySource in XYZ
        RS_XYZ = spec_RS.getXYZ()[0, 0]
        for RSci, WPi in zip(RS_XYZ/RS_XYZ[1], ot.Color._WP_D65_XYZ):
            self.assertAlmostEqual(RSci, WPi, delta=0.015) 

        # assign colored image to RaySource
        RS.Image = Image = np.array([[[0, 1, 0], [1, 1, 1]],[[0.2, 0.5, 0.1], [0, 0.8, 1.0]]], dtype=np.float32)
        RT.trace(N=5000000)

        # get Source Image
        RS_RGB = RT.SourceImage(N=2).getRGB()
        Im_px = Image.reshape((4, 3))
        RS_px = RS_RGB.reshape((4, 3))

        # check if is near original image. 
        # Unfortunately many rays are needed for the color to be near the actual value
        for i in range(4):
            for RSp, Cp in zip(RS_px[i], Im_px[i]):
                self.assertAlmostEqual(RSp, Cp, delta=0.06)

    def test_Spectrum(self):

        wl = ot.Color.wavelengths(100)
        sargs = dict(func=lambda x: x**2, lines=ot.preset_lines_FDC, line_vals=[0.5, 1., 5], 
                     wls=np.array([380, 400, 780]), vals=np.array([1, 0.5, 0]))

        for type_ in ot.Spectrum.spectrum_types:
            spec = ot.Spectrum(type_, **sargs) # init
            if spec.isContinuous():
                spec(wl) # call

        # check presets
        for spec in ot.presets_spec_tristimulus:
            if spec.isContinuous():
                spec(wl) # call

        spec  = ot.Spectrum("Function")
        self.assertRaises(RuntimeError, spec, np.array([380., 780.])) # no function
        spec  = ot.Spectrum("Data")
        self.assertRaises(RuntimeError, spec, np.array([380., 780.])) # wls or vals not specified
        
        self.assertRaises(ValueError, ot.Spectrum, "Monochromatic", wl=100) # wavelength outside visible range
        self.assertRaises(ValueError, ot.Spectrum, "Rectangle", wl0=100, wl1=500) # wavelength outside visible range
        self.assertRaises(ValueError, ot.Spectrum, "Rectangle", wl0=400, wl1=900) # wavelength outside visible range
        self.assertRaises(ValueError, ot.Spectrum, "Blackbody", T=0) # temperature <= 0
        self.assertRaises(ValueError, ot.Spectrum, "Gaussian", mu=100) # wavelength outside visible range
        self.assertRaises(ValueError, ot.Spectrum, "Gaussian", sig=0) # sigma <= 0
        self.assertRaises(ValueError, ot.Spectrum, "Gaussian", fact=0) # fact <= 0
        self.assertRaises(ValueError, ot.Spectrum, "Constant", val=-1) # val < 0
        self.assertRaises(ValueError, ot.Spectrum, "Lines", lines=[400, 500], line_vals=[1, -1]) # line weights below zero
        self.assertRaises(ValueError, ot.Spectrum, "Lines", lines=[100, 500], line_vals=[1, 2]) # wavelength outside visible range
        self.assertRaises(ValueError, ot.Spectrum, "AAAA") # invalid type
        self.assertRaises(ValueError, ot.Spectrum, "Data", wls=[400, 500], vals=[5, -1]) # vals below 0

        self.assertRaises(TypeError, ot.Spectrum, "Function", func=1) # invalid function type
        self.assertRaises(TypeError, ot.Spectrum, "Lines", lines=5, line_vals=[1, 2]) # invalid lines type
        self.assertRaises(TypeError, ot.Spectrum, "Lines", lines=[400, 500], line_vals=5) # invalid line_vals type
        self.assertRaises(TypeError, ot.Spectrum, 5) # invalid type type
        self.assertRaises(TypeError, ot.Spectrum, "Constant", quantity=1) # quantity not str
        self.assertRaises(TypeError, ot.Spectrum, "Constant", unit=1) # unit not str
        self.assertRaises(TypeError, ot.Spectrum, "Data", wls=1, vals=[1, 2]) # wls not array-like
        self.assertRaises(TypeError, ot.Spectrum, "Data", wls=[400, 500], vals=3) # vals not array-like

        self.assertRaises(RuntimeError, ot.Spectrum, "Function", func= lambda wl: 1-wl/550) # func below 0 

        self.assertRaises(AttributeError, spec.__setattr__, "aaa", 1)  # _new_lock active

    def test_TransmissionSpectrum(self):

        wl = ot.Color.wavelengths(100)
        sargs = dict(func=lambda x: 0.5, wls=np.array([380, 400, 780]), vals=np.array([1, 0.5, 0]))

        for type_ in ot.TransmissionSpectrum.spectrum_types:
            spec = ot.TransmissionSpectrum(type_, **sargs) # init
            spec(wl)
            spec.getXYZ()
            spec.getColor()
           
        self.assertRaises(ValueError, ot.TransmissionSpectrum, "Lines") # invalid discrete type
        self.assertRaises(ValueError, ot.TransmissionSpectrum, "Monochromatic") # invalid discrete type
        self.assertRaises(ValueError, ot.TransmissionSpectrum, "Gaussian", fact=2) # fact above 1 
        self.assertRaises(ValueError, ot.TransmissionSpectrum, "Constant", val=2) # val above 1 
        self.assertRaises(ValueError, ot.TransmissionSpectrum, "Data", wls=[380, 500], vals=[0, 2]) # vals above 1
        self.assertRaises(RuntimeError, ot.TransmissionSpectrum, "Function", func= lambda wl: wl/550) # func above 1 

    def test_LightSpectrum(self):

        wl = ot.Color.wavelengths(100)
        sargs = dict(func=lambda x: x**2, lines=ot.preset_lines_FDC, line_vals=[0.5, 1., 5], 
                     wls=np.array([380, 400, 780]), vals=np.array([1, 0.5, 0]))

        for type_ in ot.LightSpectrum.spectrum_types:
            spec = ot.LightSpectrum(type_, **sargs) # init
            if spec.isContinuous():
                spec(wl)
            spec.getXYZ()
            spec.getColor()
            spec.randomWavelengths(1000)

        # check presets
        for spec in ot.presets_spec_light:
            if spec.isContinuous():
                spec(wl)
            spec.getXYZ()
            spec.getColor()
            spec.randomWavelengths(1000)

        # LightSpectrum.makeSpectrum() is tested in test_Color()

        self.assertRaises(ValueError, ot.LightSpectrum, "Constant", val=0) # val <= 0

        # LightSpectrum has specialized functions compared to Spectrum, check if they handle missing properties correctly
        self.assertRaises(RuntimeError, ot.LightSpectrum("Data").getXYZ) # wls or vals not specified
        self.assertRaises(RuntimeError, ot.LightSpectrum("Data").randomWavelengths, 100) # wls or vals not specified
        self.assertRaises(RuntimeError, ot.LightSpectrum("Lines").getXYZ) # lines or line_vals not specified
        self.assertRaises(RuntimeError, ot.LightSpectrum("Lines").randomWavelengths, 100) # lines or line_vals not specified
        
    def test_Misc(self):
        doctest.testmod(ot.tracer.Misc)
        # additional tests not required here, since this functions are not exposed to the user

    def test_Focus(self):
        
        RT = ot.Raytracer(outline=[-3, 3, -3, 3, -60, 80], silent=True, n0=ot.RefractionIndex("Constant", n=1.1))
        RSS = ot.Surface("Circle", r=0.5)
        RS = ot.RaySource(RSS, pos=[0, 0, -3], spectrum=ot.preset_spec_D65)
        RT.add(RS)

        front = ot.Surface("Sphere", r=3, rho=1/30)
        back = ot.Surface("Asphere", r=3, rho=-1/20, k=1)
        n = ot.RefractionIndex("Constant", n=1.5)
        L = ot.Lens(front, back, n, de=0.1, pos=[0, 0, 0])
        RT.add(L)

        f = L.estimateFocalLength(n0=RT.n0)
        fs = 33.08433714
        self.assertAlmostEqual(f, fs, 6)

        self.assertRaises(ValueError, RT.autofocus, RT.AutofocusModes[0], z_start=5) # no simulated rays
        
        RT.trace(200000)

        for method in RT.AutofocusModes: # all methods
            for N in [50000, 200000, 500000]: # N cases: less rays, all rays, more rays than simulated
                z, _, _, _ = RT.autofocus(method, z_start=5.0, N=N, ret_cost=False)
                self.assertAlmostEqual(z, fs, delta=0.15)

        # snum=0 with only one source leads to the same result
        z, _, _, _ = RT.autofocus(RT.AutofocusModes[0], z_start=5.0, snum=0, N=N, ret_cost=False)
        self.assertAlmostEqual(z, fs, delta=0.15)

        RS2 = ot.RaySource(ot.Surface("Point"), spectrum=ot.preset_spec_D65, direction="Diverging", 
                           div_angle=0.5, pos=[0, 0, -60])
        RT.add(RS2)
        
        RT.trace(200000)

        # snum=0 with two raysources should lead to the same result as if the second RS was missing
        z, _, _, _ = RT.autofocus(RT.AutofocusModes[0], z_start=5.0, snum=0, N=N, ret_cost=False)
        self.assertAlmostEqual(z, fs, delta=0.15)

        # 1/f = 1/b + 1/g with f=33.08, g=60 should lead to b=73.73
        z, _, _, _ = RT.autofocus(RT.AutofocusModes[0], z_start=5.0, snum=1, N=N, ret_cost=False)
        self.assertAlmostEqual(z, 73.73, delta=0.1)

        self.assertRaises(ValueError, RT.autofocus, RT.AutofocusModes[0], z_start=-100) # z_start outside outline
        self.assertRaises(ValueError, RT.autofocus, "AA", z_start=-100) # invalid mode
        self.assertRaises(ValueError, RT.autofocus, RT.AutofocusModes[0], z_start=-100, snum=-1) # snum negative
        self.assertRaises(ValueError, RT.autofocus, RT.AutofocusModes[0], z_start=-100, snum=10) # snum too large

        self.assertRaises(ValueError, RT.autofocus, RT.AutofocusModes[0], z_start=10, N=10) # N too low

        # TODO test r- and vals-returns of autofocus()

if __name__ == '__main__':
    unittest.main()

