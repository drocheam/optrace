#!/bin/env python3

import sys
sys.path.append('.')

import contextlib  # redirect stdout
import time
import doctest
import unittest
import pytest

from scipy.stats import qmc
import numpy as np
import matplotlib.pyplot as plt

import optrace.tracer.misc as misc
import optrace as ot
import optrace.plots as otp
from optrace.tracer.base_class import BaseClass as BaseClass
from optrace.tracer.ray_storage import RayStorage as RayStorage


class TracerMiscTests(unittest.TestCase):

    def test_global_options(self):
        
        # test settings of options
        ot.global_options.show_progressbar = False
        ot.global_options.show_progressbar = True
        ot.global_options.multithreading = False
        ot.global_options.multithreading = True
        ot.global_options.show_warnings = False
        ot.global_options.show_warnings = True

        # spectral color map
        ot.global_options.wavelength_range = [300, 800]
        otp.spectrum_plot(ot.presets.light_spectrum.led_b1)
        ot.global_options.wavelength_range[0] = 380
        otp.spectrum_plot(ot.presets.light_spectrum.led_b1)
        ot.global_options.wavelength_range[1] = 780
        ot.global_options.spectral_colormap = lambda wl: plt.cm.viridis(wl/780)
        otp.spectrum_plot(ot.presets.light_spectrum.led_b1)
        ot.global_options.spectral_colormap = None
        otp.spectrum_plot(ot.presets.light_spectrum.led_b1)
        plt.close("all")

        # dark modes. Check by toggling. Functionality is tested elsewhere
        ot.global_options.plot_dark_mode = not ot.global_options.plot_dark_mode
        ot.global_options.plot_dark_mode = not ot.global_options.plot_dark_mode
        ot.global_options.ui_dark_mode = not ot.global_options.ui_dark_mode
        ot.global_options.ui_dark_mode = not ot.global_options.ui_dark_mode

        # Type Errors
        self.assertRaises(TypeError, ot.global_options.__setattr__, "multithreading", [])
        self.assertRaises(TypeError, ot.global_options.__setattr__, "show_progressbar", [])
        self.assertRaises(TypeError, ot.global_options.__setattr__, "show_warnings", [])
        self.assertRaises(TypeError, ot.global_options.__setattr__, "wavelength_range", 2)
        self.assertRaises(TypeError, ot.global_options.__setattr__, "plot_dark_mode", 2)
        self.assertRaises(TypeError, ot.global_options.__setattr__, "ui_dark_mode", 2)

        # Value Errors
        self.assertRaises(ValueError, ot.global_options.__setattr__, "wavelength_range", [380, 870, 1000])
        self.assertRaises(ValueError, ot.global_options.__setattr__, "wavelength_range", [381, 780])
        self.assertRaises(ValueError, ot.global_options.__setattr__, "wavelength_range", [380, 779])

    def test_global_options_context_managers(self):

        # context manager for no warnings. Case 1: True before
        ot.global_options.show_warnings = True
        with ot.global_options.no_warnings():
            self.assertFalse(ot.global_options.show_warnings)
        self.assertTrue(ot.global_options.show_warnings)
        
        # context manager for no warnings. Case 2: False before
        ot.global_options.show_warnings = False
        with ot.global_options.no_warnings():
            self.assertFalse(ot.global_options.show_warnings)
        self.assertFalse(ot.global_options.show_warnings)
        
        # context manager for no progressbar. Case 1: True before
        ot.global_options.show_progressbar = True
        with ot.global_options.no_progressbar():
            self.assertFalse(ot.global_options.show_progressbar)
        self.assertTrue(ot.global_options.show_progressbar)
        
        # context manager for no progressbar. Case 2: False before
        ot.global_options.show_progressbar = False
        with ot.global_options.no_progressbar():
            self.assertFalse(ot.global_options.show_progressbar)
        self.assertFalse(ot.global_options.show_progressbar)

    @pytest.mark.os
    def test_ray_storage(self):

        RS = RayStorage()
        self.assertRaises(AssertionError, RS.thread_rays, 0, 0)  # no rays
        self.assertRaises(AssertionError, RS.source_sections, 0)  # no rays
        self.assertRaises(AssertionError, RS.rays_by_mask, np.array([]))  # no rays

        # test storage size
        
        RS_ = ot.RaySource(ot.Point())

        for nt in [2, 3, 8, 17]:
            for N in [1, 101, 2000, 10101, 321455]:
                for pol in [False, True]:
                    RS.init([RS_], N, nt, pol)

                    calculated = RayStorage.storage_size(RS.N, RS.Nt, RS.no_pol)
                    measured = RS.p_list.nbytes + RS.s0_list.nbytes + RS.pol_list.nbytes +\
                        RS.w_list.nbytes + RS.n_list.nbytes + RS.wl_list.nbytes
                    
                    self.assertEqual(measured, calculated)
        
        # actual tests are done with tracing in tracer test file

    def test_base_class(self):

        self.assertRaises(TypeError, BaseClass, desc=1)  # desc not a string 
        self.assertRaises(TypeError, BaseClass, long_desc=1)  # long_desc not a string 
    
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

        # coverage: crepr of different member types
        L = ot.presets.geometry.arizona_eye().lenses[0]
        L.crepr()

    def test_property_checker(self):

        pc = misc.PropertyChecker

        # type checking
        self.assertRaises(TypeError, pc.check_type, "", 5, bool)  # 5 not bool
        self.assertRaises(TypeError, pc.check_type, "", 5, str)  # 5 not string
        self.assertRaises(TypeError, pc.check_type, "", 5, str | None)  # 5 not None
        self.assertRaises(TypeError, pc.check_type, "", None, ot.Point)  # None not Point
        pc.check_type("", 5, int)  # valid
        pc.check_type("", "", str)  # valid
        pc.check_type("", None, str | None)  # valid
        pc.check_type("", "", str | None)  # valid
        pc.check_type("", ot.Point(), ot.Point)  # valid

        # callables
        pc.check_none_or_callable("", None)  # valid
        pc.check_none_or_callable("", lambda x: x)  # valid
        pc.check_none_or_callable("", print)  # valid
        pc.check_callable("", lambda x: x)  # valid
        pc.check_callable("", print)  # valid
        self.assertRaises(TypeError, pc.check_callable, "", None)
        self.assertRaises(TypeError, pc.check_callable, "", 5)
        self.assertRaises(TypeError, pc.check_none_or_callable, "", 5)
        
        # values
        pc.check_above("", 5, 4)
        pc.check_not_above("", 4, 4)
        pc.check_below("", 4, 5)
        pc.check_not_below("", 4, 4)
        self.assertRaises(ValueError, pc.check_below, "", 5, 4)  # 5 > 4
        self.assertRaises(ValueError, pc.check_not_above, "", 5, 4)  # 5 > 4
        self.assertRaises(ValueError, pc.check_not_below, "", 4, 5)  # 5 > 4
        self.assertRaises(ValueError, pc.check_above, "", 4, 5)  # 4 < 5

        # elements
        pc.check_if_element("", "test", ["test", "test2"])  # valid
        self.assertRaises(ValueError, pc.check_if_element, "", "test", ["test1", "test2"])  # not an element of

    def test_misc_doctest(self):
        # normalize whitespace in doc strings and raise exceptions on errors, so pytest etc. notice errors
        optionflags = doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS
        opts = dict(optionflags=optionflags, raise_on_error=True)

        doctest.testmod(ot.tracer.misc,optionflags=optionflags)

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

    @pytest.mark.slow
    def test_misc_uniform(self):

        # test stratified_interval_sampling
        for a, b, N in zip([0, -5, 8, 7, 32.3], [1, 15, 9, 7.0001, 1000], [1, 2, 100, 501, 1000]):
            rand = misc.stratified_interval_sampling(a, b, N)
            self.assertEqual(rand.shape[0], N)

            # check value range
            self.assertTrue(np.min(rand) >= a)
            self.assertTrue(np.max(rand) <= b)

            # check value density
            hist = np.histogram(rand, bins=N, range=[a, b])[0]
            self.assertTrue(np.allclose(hist - np.ones(N), 0))
     
        rand = misc.stratified_interval_sampling(0, 1, 0)  # call with N = 0
        self.assertEqual(rand.shape, (0, ))

        # test stratified_rectangle_sampling
        for a, b, c, d, N in zip([0, -5, 8, 7, 32.3], [1, 15, 9, 7.0001, 1000], 
                                 [-65, 7, 12, 32.3, -10], [0.7, 8, 90, 700.0001, 100], [1, 2, 100, 501, 1000]):
            rand1, rand2 = misc.stratified_rectangle_sampling(a, b, c, d, N)

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

        rand1, rand2 = misc.stratified_rectangle_sampling(0, 1, 0, 1, 0)  # call with N = 0
        self.assertEqual(rand1.shape, (0, ))
        self.assertEqual(rand2.shape, (0, ))

        # the following part checks that dither and shuffling are presented
        # since everything is random, there are constellations where all values are shuffled or have no dither
        # we test many times, so the probability become very small
        # but when we encounter any random list with correct shuffling and dither, we break the loop before
        # so we don't need to test that many cases
        for shuffle in [False, True]:  # check shuffled and unshuffled case
            b1, b2 = False, False
            for i in np.arange(200):
                rand = misc.stratified_interval_sampling(0, 1, 1000000, shuffle=shuffle)
                b1 = b1 or np.std(np.diff(np.sort(rand))) > 1e-7  # check step variance (= dither)
                b2 = b2 or np.any(np.diff(rand) < 0)  # check that not all are ascending (= sorted)
                if b1 and b2:
                    break

            self.assertTrue(b1)  # check if dithered
            self.assertTrue(b2 if shuffle else (not b2))  # check if shuffled

        # same for stratified_interval_sampling 2
        b1x, b1y, b2x, b2y = False, False, False, False
        N = 1000000  # position number
        N2 = int(np.sqrt(N))  # nearest downwards integer square root of N
        dN = N - N2**2  # difference between square number and desired number
        # if have e.g. A = [[0, 1, 2], [0, 1, 2], [0, 1, 2]]
        # which is of squared shape (N2, N2)
        # in the flattend array [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2] 
        # we have therefore N2-1 negative differences between values
        # since N from above is not an exact square number, we have additional dN random values, 
        # that also could lead to a negative difference
        # so if values are randomly sorted, we typically (not always, since its random) 
        # should have more than N2+dN
        # negative differences between the flattened arrays
        # this is checked in the next part
        for i in np.arange(200):
            randy, randx = misc.stratified_rectangle_sampling(0, 1, 0, 1, 1000000)
            b1x = b1x or np.count_nonzero(np.abs(np.diff(np.sort(randx))) > 1e-9) > 2*(N2 + dN)  
            # ^-- only few zero steps between sorted values
            b2x = b2x or (np.count_nonzero(np.diff(randx) < 0) > 2*(N2+dN)) # check that not all are ascending
            b1y = b1y or np.count_nonzero(np.abs(np.diff(np.sort(randy))) > 1e-9) > 2*(N2 + dN)  
            # ^-- only few zero steps between sorted values
            b2y = b2y or (np.count_nonzero(np.diff(randy) < 0) > 2*(N2+dN)) # check that not all are ascending
            if b1x and b2x and b1y and b2y:
                break

        self.assertTrue(b1x)  # check if dithered x
        self.assertTrue(b2x)  # check if shuffled x
        self.assertTrue(b1y)  # check if dithered y
        self.assertTrue(b2y)  # check if shuffled y

        # test discrepancy

        N = 50000

        # sampling from latin hypercube
        lhc = qmc.LatinHypercube(d=2)
        sample_lhc = lhc.random(n=N)
        d_lhc = qmc.discrepancy(sample_lhc)

        # samples with optrace stratified sampling
        strat = misc.stratified_rectangle_sampling(0, 1, 0, 1, N)
        strat = np.vstack(strat).T
        d_strat = qmc.discrepancy(strat)

        # stratified sampling has smaller discrepancy
        self.assertTrue(d_strat < d_lhc)

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


if __name__ == '__main__':
    unittest.main()
