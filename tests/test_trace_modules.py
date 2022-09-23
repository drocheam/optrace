#!/bin/env python3

import sys
sys.path.append('.')

import os
import time
import doctest
import unittest
import numpy as np
import warnings
import pytest

from pathlib import Path
import optrace.tracer.misc as misc
import optrace.tracer.color as color

import optrace as ot
from optrace.tracer.geometry.element import Element as SObject
from optrace.tracer.base_class import BaseClass as BaseClass
from optrace.tracer.transfer_matrix_analysis import TMA as TMA
from optrace.tracer.ray_storage import RayStorage as RayStorage

import contextlib  # redirect stdout


# TODO how to test if uniform, uniform2 and random_from_distribution shuffled values?
# TODO how to test their dither?

class TraceTests(unittest.TestCase):

    def test_ray_storage(self):

        RS = RayStorage()
        self.assertRaises(AssertionError, RS.make_thread_rays, 0, 0)  # no rays
        self.assertRaises(AssertionError, RS.get_source_sections, 0)  # no rays
        self.assertRaises(AssertionError, RS.get_rays_by_mask, np.array([]))  # no rays

        # actual tests are done with tracing in tracer test file

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

        self.assertRaises(TypeError, pc.check_type, "", 5, bool)  # 5 not bool
        pc.check_type("", 5, int)  # 5 not bool
        pc.check_none_or_callable("", None)  # valid
        pc.check_none_or_callable("", lambda x: x)  # valid

        def test():
            return 1
        pc.check_none_or_callable("", test)  # valid
        
        self.assertRaises(ValueError, pc.check_below, "", 5, 4)  # 5 > 4
        self.assertRaises(ValueError, pc.check_not_above, "", 5, 4)  # 5 > 4
        self.assertRaises(ValueError, pc.check_not_below, "", 4, 5)  # 5 > 4
        self.assertRaises(ValueError, pc.check_above, "", 4, 5)  # 4 < 5
        self.assertRaises(ValueError, pc.check_if_element, "", "test", ["test1", "test2"])  # not an element of

    @pytest.mark.slow
    def test_r_image(self):

        self.assertRaises(TypeError, ot.RImage, 5)  # invalid extent
        self.assertRaises(ValueError, ot.RImage, [5, 6, 8])  # invalid extent
        self.assertRaises(ValueError, ot.RImage, [5, 6, 8, 7])  # invalid extent
        self.assertRaises(ValueError, ot.RImage, [5, 6, 8, np.inf])  # invalid extent
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

        # check render and rescale with different threading options
        img.threading = True
        img.render(N, p, w, wl)
        img.rescale(8)
        img.threading = False
        img.render(N, p, w, wl)
        img.rescale(256)
        img.threading = True

        img.Nx
        img.Ny
        img.sx
        img.sy
        img.Apx
        img.get_power()
        img.get_luminous_power()
        img.get_xyz()
        img.get_rgb()
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
        self.assertEqual(ot.presets.refraction_index.SF10, ot.presets.refraction_index.SF10)
        self.assertEqual(R[1], R[1])
        self.assertEqual(ot.RefractionIndex("Function", func=func), ot.RefractionIndex("Function", func=func))

        self.assertRaises(AttributeError, R[0].__setattr__, "aaa", 1)  # _new_lock active

    def test_color_doctest(self):
        doctest.testmod(ot.tracer.color)

    @pytest.mark.slow
    def test_color_accuracy(self):
        # raytracer geometry of only a Source
        RT = ot.Raytracer(outline=[-3, 3, -3, 3, 0, 6], silent=True)
        RSS = ot.Surface("Rectangle", dim=[6, 6])
        RS = ot.RaySource(RSS, pos=[0, 0, 0], spectrum=ot.presets.light_spectrum.d65)
        RT.add(RS)

        # check white color of spectrum in sRGB
        for RSci, WPi in zip(RS.get_color(), np.array([1, 1, 1])):
            self.assertAlmostEqual(RSci, WPi, delta=0.0005)

        # check white color of spectrum in XYZ
        spec_XYZ = RS.spectrum.get_xyz()[0, 0]
        for RSci, WPi in zip(spec_XYZ/spec_XYZ[1], color.WP_D65_XYZ):
            self.assertAlmostEqual(RSci, WPi, delta=0.0005) 

        # check color of all rendered rays on RaySource in sRGB
        RT.trace(N=500000)
        spec_RS = RT.source_spectrum(N=2000)
        for RSci, WPi in zip(spec_RS.get_color(), np.array([1, 1, 1])):
            self.assertAlmostEqual(RSci, WPi, delta=0.001)

        # check color of all rendered rays on RaySource in XYZ
        RS_XYZ = spec_RS.get_xyz()[0, 0]
        for RSci, WPi in zip(RS_XYZ/RS_XYZ[1], color.WP_D65_XYZ):
            self.assertAlmostEqual(RSci, WPi, delta=0.001) 

        # assign colored image to RaySource
        RS.image = Image = np.array([[[0, 1, 0], [1, 1, 1], [1, 0, 0], [0, 0, 1]],
                                    [[1, 1, 0], [1, 0, 1], [0, 1, 1], [0.1, 0.1, 0.1]],
                                    [[0, 0, 0], [0.2, 0.5, 1], [0.01, 1, 1], [0.01, 0.01, 0.01]],
                                    [[0.2, 0.5, 0.1], [0, 0.8, 1.0], [0.5, 0.3, 0.5], [0.1, 0., 0.7]]], dtype=np.float32)
        SIm, _ = RT.iterative_render(150000000, N_px_S=4, silent=True)

        # get Source Image
        RS_XYZ = np.flipud(SIm[0].get_xyz())  # flip so element [0, 0] is in lower left
        Im_px = color.srgb_to_xyz(Image).reshape((16, 3))
        RS_px = RS_XYZ.reshape((16, 3))
        Im_px /= np.max(Im_px)
        RS_px /= np.max(RS_px)

        # check if is near original image. 
        # Unfortunately many rays are needed for the color to be near the actual value
        for i in range(len(Im_px)):
            for RSp, Cp in zip(RS_px[i], Im_px[i]):
                self.assertAlmostEqual(RSp, Cp, delta=0.0012)

        # coverage tests and zero images
        Img0 = np.zeros((10, 10, 3), dtype=np.float64)
        sRGBL0 = color.xyz_to_srgb_linear(Img0)  # zero image in conversion
        sRGB0 = color.srgb_linear_to_srgb(sRGBL0, clip=False)  # default setting is clip=True
        Luv0 = color.xyz_to_luv(Img0)  # zero image in Luv conversion

    def test_misc_doctest(self):
        doctest.testmod(ot.tracer.misc)

    def test_misc_timer(self):
        # test timer
        @misc.timer
        def func(time_):
            return time.sleep(time_)

        with contextlib.redirect_stdout(None):
            func(1)  # s output
            func(0.01)  # ms output

    def test_misc_random_from_distribution(self):

        # discrete mode
        
        x = np.array([-1, 4, 5, 20])
        f = np.array([0.5, 7, 3, 0.1])
        F = misc.random_from_distribution(x, f, 100001, kind="discrete")

        # checks if values in F are all each an element of list x (discrete values)
        self.assertTrue(np.all(np.any(np.abs(F[:, np.newaxis] - x) < 1e-10, axis=1)))

        # mathematical correct values
        exp_val = np.sum(x*f) / np.sum(f)  # expectation value
        std_val = np.sqrt(np.sum((x - exp_val)**2 * f/np.sum(f))) # standard deviation

        # check if we are in proximity
        self.assertAlmostEqual(np.mean(F), exp_val, delta=0.0005)
        self.assertAlmostEqual(np.std(F), std_val, delta=0.005)

        # call with N=0
        rand = misc.random_from_distribution(x, f, 0, kind="discrete")
        self.assertEqual(rand.shape, (0, ))
        
        # continuous mode

        func = lambda x: np.exp(-(x-1)**2)
        x = np.linspace(-4, 5, 1000)
        f = func(x)
        F = misc.random_from_distribution(x, f, 10000)

        # range correct
        self.assertTrue(np.all(F <= x[-1]))
        self.assertTrue(np.all(F >= x[0]))

        # mean and variation correct
        self.assertAlmostEqual(np.mean(F), 1, delta=0.0005)
        self.assertAlmostEqual(np.std(F), 1/np.sqrt(2), delta=0.005)

        # error raising
        self.assertRaises(RuntimeError, misc.random_from_distribution, np.array([0, 1]), np.array([0, 0]), 10)  
        # cdf zero
        self.assertRaises(RuntimeError, misc.random_from_distribution, np.array([0, 1]), np.array([0, -1]), 10)  
        # pdf < 0

        # call with N=0
        rand = misc.random_from_distribution(x, f, 0)
        self.assertEqual(rand.shape, (0, ))

    def test_misc_uniform(self):

        # test uniform
        for a, b, N in zip([0, -5, 8, 7, 32.3], [1, 15, 9, 7.0001, 1000], [1, 2, 100, 501, 1000]):
            rand = misc.uniform(a, b, N)
            self.assertEqual(rand.shape[0], N)

            # check value range
            self.assertTrue(np.min(rand) >= a)
            self.assertTrue(np.max(rand) <= b)

            # check value density
            hist = np.histogram(rand, bins=N, range=[a, b])[0]
            self.assertTrue(np.allclose(hist - np.ones(N), 0))
      
        rand = misc.uniform(0, 1, 0)  # call with N = 0
        self.assertEqual(rand.shape, (0, ))

        # test uniform2
        for a, b, c, d, N in zip([0, -5, 8, 7, 32.3], [1, 15, 9, 7.0001, 1000], 
                                 [-65, 7, 12, 32.3, -10], [0.7, 8, 90, 700.0001, 100], [1, 2, 100, 501, 1000]):
            rand1, rand2 = misc.uniform2(a, b, c, d, N)

            # check shape
            self.assertEqual(rand1.shape[0], N)
            self.assertEqual(rand2.shape[0], N)
    
            # check value range
            self.assertTrue(np.min(rand1) >= a)
            self.assertTrue(np.max(rand1) <= b)
            self.assertTrue(np.min(rand2) >= c)
            self.assertTrue(np.max(rand2) <= d)

            # check value density
            N2 = int(np.sqrt(N))
            hist1 = np.histogram(rand1, bins=N2, range=[a, b])[0]
            self.assertTrue(np.all(hist1 > 0))
            hist2 = np.histogram(rand2, bins=N2, range=[c, d])[0]
            self.assertTrue(np.all(hist2 > 0))
            hist = np.histogram2d(rand1, rand2, bins=[N2, N2], range=[[a, b], [c, d]])[0]
            self.assertTrue(np.all(hist > 0))

        rand1, rand2 = misc.uniform2(0, 1, 0, 1, 0)  # call with N = 0
        self.assertEqual(rand1.shape, (0, ))
        self.assertEqual(rand2.shape, (0, ))

    def test_misc_calc(self):
       
        # actual values are tested with the docstrings in self.test_misc_doctest()

        # test rdot
        self.assertRaises(RuntimeError, misc.rdot, np.array([[5.]]), np.array([[10.]]))  # invalid number of dimension
        self.assertRaises(RuntimeError, misc.rdot, np.ones((4, 4)), np.ones((4, 4)))  # invalid number of dimension
        empty0 = np.zeros((0, 0), dtype=float)
        empty2 = np.zeros((0, 2), dtype=float)
        empty3 = np.zeros((0, 3), dtype=float)
        self.assertTrue(np.all(misc.rdot(empty2, empty2) == empty0))  # empty 2D 
        self.assertTrue(np.all(misc.rdot(empty3, empty3) == empty0))  # empty 3D

        # test part_mask
        true5 = np.ones(5, dtype=bool)
        false5 = np.zeros(5, dtype=bool)
        empty_bool = np.zeros((0, ), dtype=bool)
        self.assertTrue(np.all(misc.part_mask(true5, true5) == true5))  # everything true
        self.assertTrue(np.all(misc.part_mask(true5, false5) == false5))  # everything false
        self.assertTrue(np.all(misc.part_mask(false5, empty_bool) == false5)) # everything false
        self.assertTrue(np.all(misc.part_mask(empty_bool, empty_bool) == empty_bool))  # empty arrays

        # test uniform_resample
        for N in [2, 10, 100, 121, 500, 1001]:
            for N2 in [2, 3, 100, 500, 999]:
                x_ = np.random.sample(N)
                x = np.cumsum(x_)
                f = np.random.sample(N)

                # check shapes and bounds
                xs, fs = misc.uniform_resample(x, f, N2)
                self.assertEqual(xs.shape[0], N2)
                self.assertEqual(fs.shape[0], N2)
                self.assertAlmostEqual(xs[0], x[0])
                self.assertAlmostEqual(xs[-1], x[-1])
                self.assertAlmostEqual(fs[0], f[0])
                self.assertAlmostEqual(fs[-1], f[-1])

                # check spacing value and equalness
                self.assertTrue(np.isclose(np.std(np.diff(xs)), 0))
                self.assertTrue(np.isclose(xs[1] - xs[0], (x[-1] - x[0])/(N2-1)))

        # values stay the same if sampling points are the same
        x = np.arange(100)
        f = np.random.sample(100)
        self.assertTrue(np.allclose(misc.uniform_resample(x, f, 100)[1] - f, 0))
        
        # check linear scaling between values
        x = np.arange(100)
        f = np.arange(100)/100
        xs, fs = misc.uniform_resample(x, f, 1000)
        self.assertTrue(np.allclose(fs - xs/100, 0))

        # test normalize
        self.assertTrue(np.all(misc.normalize(empty3) == empty3))  # works with empty
        # test numerical special cases
        # since the number gets squared for normalization, the minimum or maximum value is sqrt(min) or sqrt(max)
        fmin = np.sqrt(np.finfo(np.float64).tiny)  # minimum float number
        fmax = np.sqrt(np.finfo(np.float64).max)  # maximum float number
        A = np.array([[np.nan, np.nan, np.nan], [np.inf, 0, 0], [0, 0, 0], [0, 0, fmin], [0, fmax, 0]])
        B = np.array([[np.nan, np.nan, np.nan], [np.nan, 0, 0], [np.nan, np.nan, np.nan], [0, 0, 1], [0, 1, 0]])
        B2 = misc.normalize(A)
        self.assertTrue(np.all(np.isnan(B2[0])))
        self.assertTrue(np.all(np.isnan(B2[2])))
        self.assertTrue(np.all(np.isnan(B2[1, 0])))
        self.assertTrue(np.all(B2[1, 1:] == 0))
        self.assertTrue(np.all(B2[3:] == B[3:]))

        # test cross
        self.assertTrue(np.all(misc.cross(empty3, empty3) == empty3))  # works with empty

    def test_property_checker(self):
        # check property checker
        self.assertRaises(TypeError, misc.PropertyChecker.check_type, "", 1, str)
        self.assertRaises(TypeError, misc.PropertyChecker.check_type, "", ot.Point(), int)
        self.assertRaises(TypeError, misc.PropertyChecker.check_callable, "", 5)
        self.assertRaises(TypeError, misc.PropertyChecker.check_none_or_callable, "", 5)
        self.assertRaises(ValueError, misc.PropertyChecker.check_above, "", 5, 6)
        self.assertRaises(ValueError, misc.PropertyChecker.check_above, "", 5, 5)
        self.assertRaises(ValueError, misc.PropertyChecker.check_not_below, "", 4, 5)
        self.assertRaises(ValueError, misc.PropertyChecker.check_below, "", 6, 5)
        self.assertRaises(ValueError, misc.PropertyChecker.check_below, "", 5, 5)
        self.assertRaises(ValueError, misc.PropertyChecker.check_not_above, "", 6, 5)
        self.assertRaises(ValueError, misc.PropertyChecker.check_if_element, "", "ABC", ["AB", "BC"])
        misc.PropertyChecker.check_type("", "abc", str)
        misc.PropertyChecker.check_type("", ot.Surface("Circle"), ot.Surface)
        misc.PropertyChecker.check_callable("", lambda x: x)
        misc.PropertyChecker.check_callable("", ot.Surface.get_values)
        misc.PropertyChecker.check_none_or_callable("", lambda x: x)
        misc.PropertyChecker.check_none_or_callable("", ot.Surface.get_values)
        misc.PropertyChecker.check_above("", 6, 5)
        misc.PropertyChecker.check_not_below("", 6, 5)
        misc.PropertyChecker.check_not_below("", 5, 5)
        misc.PropertyChecker.check_below("", 4, 5)
        misc.PropertyChecker.check_not_above("", 4, 5)
        misc.PropertyChecker.check_not_above("", 5, 5)
        misc.PropertyChecker.check_if_element("", "AB", ["AB", "BC"])

    def test_tma_single_lens_efl(self):

        R1 = 100
        R2 = 80
        d = 0.5
        wl = 550
        wl2 = 650
        n = ot.RefractionIndex("Abbe", n=1.5, V=60)
        nc = ot.RefractionIndex("Constant", n=1)
        n1 = ot.RefractionIndex("Abbe", n=1.3, V=60)
        n2 = ot.RefractionIndex("Abbe", n=1.1, V=60)
        spa = ot.Surface("Sphere", r=3, R=R1)
        spb = ot.Surface("Sphere", r=3, R=-R2)
        aspa = ot.Surface("Conic", r=3, R=R1)
        aspb = ot.Surface("Conic", r=3, R=-R2)
        circ = ot.Surface("Circle", r=3)

        # lens maker equation
        def getf(R1, R2, n, n1, n2, d):
            D = (n-n1)/n2/R1 - (n-n2)/R2/n2 + (n-n1)/(n*R1) * (n-n2)/(n2*R2) * d 
            return 1 / D if D else np.inf 
      
        # check focal length for different surface constellations
        largs = dict(d=d, n=n, pos=[0, 0, 0])
        self.assertAlmostEqual(ot.Lens(spa, spb, **largs).tma(wl=wl).efl, getf(R1, -R2, n(wl), 1, 1, d))
        self.assertAlmostEqual(ot.Lens(spb, spa, **largs).tma(wl=wl).efl, getf(-R2, R1, n(wl), 1, 1, d))
        self.assertAlmostEqual(ot.Lens(spa, circ, **largs).tma(wl=wl).efl, getf(R1, np.inf, n(wl), 1, 1, d))
        self.assertAlmostEqual(ot.Lens(circ, spb, **largs).tma(wl=wl).efl, getf(np.inf, -R2, n(wl), 1, 1, d))
        self.assertAlmostEqual(ot.Lens(circ, spa, **largs).tma(wl=wl).efl, getf(np.inf, R1, n(wl), 1, 1, d))
        self.assertAlmostEqual(ot.Lens(spb, circ, **largs).tma(wl=wl).efl, getf(-R2, np.inf, n(wl), 1, 1, d))
        self.assertTrue(np.isnan(ot.Lens(circ, circ, **largs).tma(wl=wl).efl))
        self.assertAlmostEqual(ot.Lens(spa, spa, **largs).tma(wl=wl).efl, getf(R1, R1, n(wl), 1, 1, d))
        self.assertAlmostEqual(ot.Lens(spb, spb, **largs).tma(wl=wl).efl, getf(-R2, -R2, n(wl), 1, 1, d))

        # ambient n on both sides
        self.assertAlmostEqual(ot.Lens(spa, spb, n2=n2, **largs).tma(wl=wl, n0=n2).efl, 
                               getf(R1, -R2, n(wl), n2(wl), n2(wl), d))  # ambient n
        self.assertAlmostEqual(ot.Lens(spa, spb, n2=n2, **largs).tma(wl=wl, n0=n2).efl, 
                               getf(R1, -R2, n(wl), n2(wl), n2(wl), d))  # ambient n
        self.assertAlmostEqual(ot.Lens(aspa, aspb, n2=n2, **largs).tma(wl=wl2, n0=n2).efl, 
                               getf(R1, -R2, n(wl2), n2(wl2), n2(wl2), d))  # different wl

        # ambient n on one side
        self.assertAlmostEqual(ot.Lens(spa, spb, n2=n2, **largs).tma(wl=wl, ).efl, 
                               getf(R1, -R2, n(wl), 1, n2(wl), d))  # ambient n
        self.assertAlmostEqual(ot.Lens(spa, spb, n2=nc, **largs).tma(wl=wl, n0=n2).efl, 
                               getf(R1, -R2, n(wl), n2(wl), 1, d))  # ambient n
        self.assertAlmostEqual(ot.Lens(aspa, aspb, n2=n2, **largs).tma(wl=wl2, n0=n1).efl, 
                               getf(R1, -R2, n(wl2), n1(wl2), n2(wl2), d))  # different wl

    def test_tma_single_lens_cardinal(self):
        R1 = 76
        R2 = 35
        wl = 450
        n = ot.RefractionIndex("Constant", n=1.5)
        nc = ot.RefractionIndex("Constant", n=1)
        n1 = ot.RefractionIndex("Constant", n=1.3)
        n2 = ot.RefractionIndex("Constant", n=1.1)
        spa = ot.Surface("Sphere", r=3, R=R1)
        spb = ot.Surface("Sphere", r=3, R=-R2)
        spc = ot.Surface("Sphere", r=3, R=-R1)
        spd = ot.Surface("Sphere", r=3, R=R2)
        circ = ot.Surface("Circle", r=3)

        def cmp_props(L, list2, n0=ot.RefractionIndex("Constant", n=1)):
            tma = L.tma(n0=n0)
            # edmund optics defines the nodal point shift NPS as sum of focal lengths
            list1 = [tma.efl_n, tma.bfl, tma.ffl,
                     tma.principal_point[0] - tma.vertex_point[0],
                     tma.principal_point[1] - tma.vertex_point[1],
                     tma.focal_length[0] + tma.focal_length[1]]

            for el1, el2 in zip(list1, list2):
                self.assertAlmostEqual(el1, el2, delta=0.0002)

        # values from
        # https://www.edmundoptics.com/knowledge-center/tech-tools/focal-length/

        # edmund optics list: efl, bfl, ffl, P, P'', NPS
        # P and P'' are lengths between principal points and vertext points
       
        # standard biconvex lens
        L = ot.Lens(spa, spb, pos=[0, 0, 0], d=0.71, n=n1)
        cmp_props(L, [79.9980, 79.8255, -79.6235, 0.3745, -0.1725, 0])

        # standard iconcave lens
        L = ot.Lens(spb, spa, pos=[0, 0, 0], d=0.2, n=n)
        cmp_props(L, [-47.8992, -47.9904, 47.9412, 0.0420, -0.0912, 0])

        # negative meniscus, R < 0
        L = ot.Lens(spb, spc, pos=[0, 0, 0], d=0.2, n=n)
        cmp_props(L, [-129.9674, -130.2150, 129.8534, -0.1140, -0.2476, 0])

        # positive meniscus, R < 0
        L = ot.Lens(spc, spb, pos=[0, 0, 0], d=0.2, n=n)
        cmp_props(L, [129.5455, 129.6591, -129.2987, 0.2468, 0.1136, 0])

        # negative meniscus, R > 0
        L = ot.Lens(spa, spd, pos=[0, 0, 0], d=0.2, n=n)
        cmp_props(L, [-129.9674, -129.8534, 130.2150, 0.2476, 0.1140, 0])

        # positive meniscus, R > 0
        L = ot.Lens(spd, spa, pos=[0, 0, 0], d=0.2, n=n)
        cmp_props(L, [129.5455, 129.2987, -129.6591, -0.1136, -0.2468, 0])

        # plano-convex, R > 0
        L = ot.Lens(spd, circ, pos=[0, 0, 0], d=0.71, n=n)
        cmp_props(L, [70, 69.5267, -70, 0, -0.4733, 0])
        
        # plano-convex, R < 0
        L = ot.Lens(circ, spc, pos=[0, 0, 0], d=0.71, n=n)
        cmp_props(L, [152, 152, -151.5267, 0.4733, 0, 0])

        # plano-concave, R < 0
        L = ot.Lens(spc, circ, pos=[0, 0, 0], d=0.71, n=n)
        cmp_props(L, [-152, -152.4733, 152, 0, -0.4733, 0])
        
        # plano-concave, R > 0
        L = ot.Lens(circ, spd, pos=[0, 0, 0], d=0.71, n=n)
        cmp_props(L, [-70, -70, 70.4733, 0.4733, 0, 0])

        # negative meniscus, R > 0, different media
        L = ot.Lens(spa, spd, pos=[0, 0, 0], d=0.2, n=n, n2=n2)
        cmp_props(L, [-113.7271, -125.0559, 148.0705, 0.2253, 0.0439, 22.7454], n1)

        # both surfaces disc
        tma = ot.Lens(circ, circ, pos=[0, 0, 0], d=0.2, n=n, n2=n2).tma()
        list1 = [tma.efl_n, tma.bfl, tma.ffl,
                 tma.principal_point[0] - tma.vertex_point[0],
                 tma.principal_point[1] - tma.vertex_point[1],
                 tma.focal_length[0] + tma.focal_length[1]]
        self.assertTrue(np.all(np.isnan(list1)))

    def test_raytracer_tma_cardinal_values(self):

        # check if RT throws with no rotational symmetry
        RT = ot.Raytracer([-5, 6, -5, 6, -12, 500])
        RT.add(ot.presets.geometry.arizona_eye())
        RT.lens_list[0].move_to([0, 5, -12])
        self.assertRaises(RuntimeError, RT.tma)  # no rotational symmetry
       
        # compare paraxial properties of LeGrand paraxial eye
        # values are from Field Guide to Visual and Ophthalmic Optics, Jim Schwiegerling, page 15
        # and Handbook of visual optics, Volume 1, Pablo, Artal, table 16.1
        # for some reason the the principal points differ between these two,
        # we find the one from Artal to be correct

        RT = ot.Raytracer([-6, 6, -6, 6, -12, 50])
        RT.add(ot.presets.geometry.legrand_eye())
        
        tma = RT.tma()
        self.assertAlmostEqual(tma.focal_point[0], -15.089, delta=0.001)
        self.assertAlmostEqual(tma.focal_point[1], 24.197, delta=0.001)
        self.assertAlmostEqual(tma.nodal_point[0], 7.200, delta=0.001)
        self.assertAlmostEqual(tma.nodal_point[1], 7.513, delta=0.001)
        self.assertAlmostEqual(tma.principal_point[0], 1.595, delta=0.001)
        self.assertAlmostEqual(tma.principal_point[1], 1.908, delta=0.001)
        self.assertAlmostEqual(tma.power_n[1], 59.940, delta=0.001)
        self.assertAlmostEqual(tma.d, 7.600, delta=0.001)
        self.assertAlmostEqual(RT.lens_list[0].tma().power_n[1], 42.356, delta=0.001)
        self.assertAlmostEqual(RT.lens_list[1].tma(n0=RT.lens_list[0].n2).power_n[1], 21.779, delta=0.001)

        # additional checks for derived properties
        self.assertAlmostEqual(tma.vertex_point[0], RT.lens_list[0].front.pos[2])
        self.assertAlmostEqual(tma.vertex_point[1], RT.lens_list[-1].back.pos[2])
        self.assertAlmostEqual(tma.n1, RT.n0(550))  # wavelength independent, so value unimportant
        self.assertAlmostEqual(tma.n2, RT.lens_list[-1].n2(550))
        self.assertAlmostEqual(tma.power[0], -59.940/tma.n1, delta=0.001)
        self.assertAlmostEqual(tma.power[1], 59.940/tma.n2, delta=0.001)
        self.assertAlmostEqual(tma.efl_n, 1000/59.940, delta=0.001)
        self.assertAlmostEqual(tma.efl, 1000/59.940*tma.n2, delta=0.001)
        self.assertAlmostEqual(tma.focal_length_n[0], -1000 / 59.940, delta=0.001)
        self.assertAlmostEqual(tma.focal_length_n[1], 1000 / 59.940, delta=0.001)
        self.assertAlmostEqual(tma.focal_length[0], -1000 / 59.940 * tma.n1, delta=0.001)
        self.assertAlmostEqual(tma.focal_length[1], 1000 / 59.940 * tma.n2, delta=0.001)
        self.assertAlmostEqual(tma.ffl, tma.focal_point[0] - tma.vertex_point[0])
        self.assertAlmostEqual(tma.bfl, tma.focal_point[1] - tma.vertex_point[1])

    def test_tma_image_object_distances(self):

        L = ot.Lens(ot.Surface("Sphere", R=100), ot.Surface("Sphere", R=-200), de=0, pos=[0, 0, 2],
                    n=ot.RefractionIndex("Constant", n=1.5))
        Ltma = L.tma()
        Lf = 2/(1/100 + 1/200)
        self.assertAlmostEqual(Ltma.efl, Lf, places=1)
        
        def check_image_pos(g_z): 
            g = L.pos[2] - g_z
            f = L.tma().efl
            image_pos_approx = L.pos[2] + f*g/(g-f)
            image_pos = L.tma().image_position(g_z)

            self.assertAlmostEqual(image_pos, image_pos_approx, delta=0.5)

        for g_z in [-2*Lf, -Lf/2, 0, Lf/2, 2*Lf]:
            check_image_pos(g_z)
        
        def check_object_pos(b_z): 
            b = b_z - L.pos[2]
            f = L.tma().efl
            object_pos_approx = L.pos[2] - f*b/(b-f)
            object_pos = L.tma().object_position(b_z)

            self.assertAlmostEqual(object_pos, object_pos_approx, delta=0.5)

        for b_z in [-2*Lf, -Lf/2, 0, Lf/2, 2*Lf]:
            check_object_pos(b_z)

        RT = ot.Raytracer([-6, 6, -6, 6, -12, 50])
        RT.add(ot.presets.geometry.legrand_eye())
        tma = RT.tma()

        # image/object distances special cases
        # object at -inf
        self.assertAlmostEqual(tma.image_position(-np.inf), tma.focal_point[1])
        self.assertAlmostEqual(tma.image_position(-100000), tma.focal_point[1], delta=0.005)
        # image at inf
        self.assertAlmostEqual(tma.object_position(np.inf), tma.focal_point[0])
        self.assertAlmostEqual(tma.object_position(100000), tma.focal_point[0], delta=0.005)
        # object at focal point 0
        self.assertAlmostEqual(1 / tma.image_position(tma.focal_point[0] - 1e-12), 0) # 1/b approaches zero
        self.assertTrue(1 / tma.image_position(tma.focal_point[0] - 1e-12) > 0)  # positive infinity
        # image at focal point 1
        self.assertAlmostEqual(1 / tma.object_position(tma.focal_point[1] + 1e-12), 0)
        self.assertTrue(1 / tma.object_position(tma.focal_point[1] + 1e-12) < 0)  # negative infinity

    def test_tma_error_cases(self):

        L = ot.Lens(ot.Surface("Circle"), ot.Surface("Circle"), pos=[0, 0, 0], n=ot.presets.refraction_index.SF10)
        
        self.assertRaises(TypeError, TMA, [L], wl=None)  # invalid wl type
        self.assertRaises(TypeError, TMA, L, wl=None)  # invalid lens_list type
        self.assertRaises(TypeError, TMA, [L], n0=1.2)  # invalid n0 type
        self.assertRaises(ValueError, TMA, [L], wl=10)  # wl to small
        self.assertRaises(ValueError, TMA, [L], wl=1000)  # wl to large

        self.assertRaises(ValueError, TMA, [])  # empty lens list

        # check if only symmetric lenses
        self.assertRaises(RuntimeError, ot.Lens(ot.Surface("Circle"), ot.Surface("Circle", normal=[0, 1, 1]),
                          n=ot.presets.refraction_index.SF10, pos=[0, 0, 0]).tma)

        # check surface collision
        LL = [ot.presets.geometry.legrand_eye()[0], ot.presets.geometry.legrand_eye()[2]]
        LL[1].move_to([0, 0, 0])
        self.assertRaises(RuntimeError, TMA, LL)  # surface collision
        
        # check sharing of same axis
        LL = [ot.presets.geometry.legrand_eye()[0], ot.presets.geometry.legrand_eye()[2]]
        LL[1].move_to([0, 1, 10])
        self.assertRaises(RuntimeError, TMA, LL)

        # positions inside z-range of lens setup
        LL = [ot.presets.geometry.legrand_eye()[0], ot.presets.geometry.legrand_eye()[2]]
        self.assertRaises(ValueError, TMA(LL).image_position, 0.25)  # object inside lens setup
        self.assertRaises(ValueError, TMA(LL).object_position, 0.25)  # image inside lens setup

        # check lock
        tma = TMA(LL)
        self.assertRaises(AttributeError, tma.__setattr__, "n78", ot.presets.refraction_index.LAK8)  # invalid property
        self.assertRaises(RuntimeError, tma.__setattr__, "wl", 560)  # object locked

    def test_tma_misc(self):

        # test tma.trace
        L = ot.Lens(ot.Surface("Sphere", R=100), ot.Surface("Sphere", R=-200), de=0, pos=[0, 0, 2],
                    n=ot.RefractionIndex("Constant", n=1.5))
        Ltma = L.tma()
        Ltma.trace([1.2, -0.7])  # list input
        res = Ltma.trace(np.array([1.2, -0.7]))  # 1D array input
        res2 = Ltma.trace(np.array([[1.2, -0.7], [5, 3]]))  # 2D array input
        self.assertTrue(np.allclose(res2[0], res))  # rows in input lead to rows in output

        # test tma.matrix_at
        ######
        # image object positions for imaging
        z_g = Ltma.focal_point[0] - 5
        z_b = Ltma.image_position(z_g)
        # calculate matrix for this image-object distance combination
        abcd = Ltma.matrix_at(z_g, z_b)
        # for imaging element B of the ABCD matrix must be zero, check if this is the case
        self.assertAlmostEqual(abcd[0, 1], 0)

if __name__ == '__main__':
    unittest.main()
