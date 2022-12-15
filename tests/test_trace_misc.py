#!/bin/env python3

import sys
sys.path.append('.')

import contextlib  # redirect stdout
from pathlib import Path
import os
import time
import doctest
import unittest

import numpy as np
import pytest

import optrace.tracer.misc as misc
import optrace.tracer.color as color

import optrace as ot
from optrace.tracer.base_class import BaseClass as BaseClass
from optrace.tracer.ray_storage import RayStorage as RayStorage



class TracerMiscTests(unittest.TestCase):

    def test_ray_storage(self):

        RS = RayStorage()
        self.assertRaises(AssertionError, RS.make_thread_rays, 0, 0)  # no rays
        self.assertRaises(AssertionError, RS.source_sections, 0)  # no rays
        self.assertRaises(AssertionError, RS.rays_by_mask, np.array([]))  # no rays

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

        # printing
        BC.silent = False
        BC.print("abc")
        BC.silent = True
        BC.print("abc")

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

    @pytest.mark.slow
    def test_r_image_misc(self):

        # init exceptions
        self.assertRaises(TypeError, ot.RImage, 5)  # invalid extent
        self.assertRaises(ValueError, ot.RImage, [5, 6, 8])  # invalid extent
        self.assertRaises(ValueError, ot.RImage, [5, 6, 8, 7])  # invalid extent
        self.assertRaises(ValueError, ot.RImage, [5, 6, 8, np.inf])  # invalid extent
        self.assertRaises(TypeError, ot.RImage, [5, 6, 8, 9], 5)  # invalid projection type
        self.assertRaises(TypeError, ot.RImage, [5, 6, 8, 9], limit=[5])  # invalid limit type
        self.assertRaises(ValueError, ot.RImage, [5, 6, 8, 9], limit=-1)  # invalid limit value

        img = ot.RImage([-1, 1, -2, 2], silent=True)
        self.assertFalse(img.has_image())  # no image yet
        
        # create some image
        img, P, L = self.gen_r_image()
        
        # now has an image
        self.assertTrue(img.has_image())
        
        # check display_modes
        for dm in ot.RImage.display_modes:
            img_ = img.get_by_display_mode(dm)
            
            # all values should be positive
            self.assertTrue(np.min(img_) > -1e-9)
            
            # in srgb modes the values should be <= 1
            if dm.find("sRGB") != -1:
                self.assertTrue(np.max(img_) < 1+1e-9)

        self.assertRaises(ValueError, img.get_by_display_mode, "hjkhjkhjk")  # invalid mode

        # get_rgb with log and rendering intent options
        for ri in color.SRGB_RENDERING_INTENTS:
            for log in [True, False]:
                im = img.get_rgb(log=log, rendering_intent=ri)
                self.assertTrue(np.min(im) > 0-1e-9)
                self.assertTrue(np.max(im) < 1+1e-9)

        # check if empty image throws
        RIm = ot.RImage(extent=[-1, 1, -1, 1])
        for mode in ot.RImage.display_modes:
            self.assertRaises(RuntimeError, RIm.get_by_display_mode, mode)
    
    @pytest.mark.slow
    def test_r_image_cut(self):
       
        for ratio in [1/6, 0.9, 3, 5]:  # different side ratios
            for Npx in ot.RImage.SIZES:  # different pixel numbers

                img, P, L = self.gen_r_image(ratio=ratio, N_px=Npx)
            
                for p in [-0.1, 0, 0.163485, 1, 1.01]: # different relative position inside image
                    for cut_ in ["x", "y"]:
                        ext = img.extent[:2] if cut_ == "x" else img.extent[2:]
                        ext2 = img.extent[2:] if cut_ == "x" else img.extent[:2]
                        kwargs = {cut_: ext[0] + p*(ext[1] - ext[0])}

                        for dm in ["sRGB (Absolute RI)", "Irradiance"]:  # modes with three and one channel

                            if p < 0 or p > 1:
                                self.assertRaises(ValueError, img.cut, dm, **kwargs)
                            else:
                                rgb_mode = dm.find("sRGB (") != -1
                                bine, vals = img.cut(dm, **kwargs)

                                # check properties
                                self.assertEqual(len(vals), 3 if rgb_mode else 1)  # channel count
                                self.assertEqual(len(bine), len(vals[0])+1)  # lens of bin edges and values
                                self.assertAlmostEqual(bine[0], ext2[0])  # first bin starts at extent
                                self.assertAlmostEqual(bine[-1], ext2[-1])  # last bin ends at extent

                                # stack rgb channels in depth dimension
                                if rgb_mode:
                                    vals = np.dstack((vals[0], vals[1], vals[2]))

                                # make cut through image, multiply with 1-1e-12 so p = 1 gets mapped into last bin
                                if cut_ == "x":
                                    img_ = img.get_by_display_mode(dm)[:, int(p*(1-1e-12)*img.Nx)]
                                else:
                                    img_ = img.get_by_display_mode(dm)[int(p*(1-1e-12)*img.Ny), :]

                                # compare with cut from before
                                self.assertTrue(np.allclose(img_ - vals, 0))

    @pytest.mark.slow
    def test_r_image_render_and_rescaling(self):

        for limit in [None, 20]:  # different resolution limits

            for i, ratio in enumerate([1/6, 0.38, 1, 5]):  # different side ratios

                img, P, L = self.gen_r_image(ratio=ratio, limit=limit, threading=bool(i % 2))  # toggle threading

                # check power sum
                P0 = img.get_power()
                L0 = img.get_luminous_power()
                self.assertAlmostEqual(P0, P)
                self.assertAlmostEqual(L0, L)

                ratio_act = img.Nx / img.Ny
                sx0, sy0 = img.sx, img.sy

                for i, Npx in enumerate([*ot.RImage.SIZES, *np.random.randint(1, 2*ot.RImage.SIZES[-1], 3)]):  # different side lengths
                    img.rescale(Npx)
                    self.assertEqual(img.N, min(img.Nx, img.Ny))

                    # check that nearest matching side length was chosen
                    siz = ot.RImage.SIZES
                    near_siz = siz[np.argmin(np.abs(Npx - np.array(siz)))]
                    cmp_siz1 = img.Nx if ratio_act < 1 else img.Ny
                    self.assertEqual(near_siz, cmp_siz1)

                    self.assertAlmostEqual(P0, img.get_power())  # overall power stays the same even after rescaling
                    self.assertAlmostEqual(ratio_act, img.Nx/img.Ny)  # ratio stayed the same
                    self.assertEqual(sx0, img.sx)  # side length stayed the same
                    self.assertEqual(sy0, img.sy)  # side length stayed the same
                    self.assertAlmostEqual(img.sx*img.sy/img.Nx/img.Ny, img.Apx)  # check pixel area

        # test exceptions render
        self.assertRaises(ValueError, img.render, N=ot.RImage.MAX_IMAGE_SIDE*1.2)  # pixel number too large
        self.assertRaises(ValueError, img.render, N=-2)  # pixel number negative
        self.assertRaises(ValueError, img.render, N=0)  # pixel number zero

        # test exceptions rescale
        self.assertRaises(ValueError, img.rescale, 0.5)  # N < 1
        self.assertRaises(ValueError, img.rescale, -1)  # N negative

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

    def gen_r_image(self, N=10000, N_px=ot.RImage.SIZES[6], ratio=1, limit=None, threading=True):
        # create some image
        img = ot.RImage([-1, 1, -1*ratio, 1*ratio], silent=True, limit=limit, threading=threading)
        p = np.zeros((N, 2))
        p[:, 0] = np.random.uniform(img.extent[0], img.extent[1], N)
        p[:, 1] = np.random.uniform(img.extent[2], img.extent[3], N)
        w = np.random.uniform(1e-9, 1, N)
        wl = np.random.uniform(*color.WL_BOUNDS, N)
        img.render(N_px, p, w, wl)
        return img, np.sum(w), np.sum(w * color.y_observer(wl) * 683)  # 683 lm/W conversion factor

    @pytest.mark.slow
    @pytest.mark.os
    def test_r_image_saving_loading(self):
       
        # create some image
        img = self.gen_r_image()[0]

        # saving and loading for valid path
        path = "test_img.npz"
        for silent in [False, True]:
            img.silent = silent
            path_ = img.save(path)
            img = ot.RImage.load(path)
        self.assertTrue(img.limit is None)  # None limit is None after loading
        os.remove(path_)

        # saving and loading for valid path, overwrite file
        img.limit = 5
        path0 = img.save(path, overwrite=True)
        img = ot.RImage.load(path0)
        img.silent = True
        self.assertEqual(path, path0)
        self.assertEqual(img.limit, 5)  # limit is loaded correctly
        
        # saving and loading for valid path, don't overwrite file
        path1 = img.save(path, overwrite=False)
        img = ot.RImage.load(path1)
        img.silent = True
        self.assertNotEqual(path, path1)
        os.remove(path0)
        os.remove(path1)
        
        # saving and loading for invalid path
        for silent in [False, True]:
            img.silent = silent
            path2 = img.save(str(Path("hjkhjkljk") / "test_img.npz"))
            img = ot.RImage.load(path2)
            os.remove(path2)

        # png saving
        for ratio in [3, 1, 1/3]:
            img = self.gen_r_image(ratio=ratio)[0]
            img.rescale(9)
            img.silent = True
            for imm in ot.RImage.display_modes:
                path3 = img.export_png(path, imm, log=True, overwrite=True, flip=False)
                path3 = img.export_png(path, imm, log=False, overwrite=True, flip=True)
            os.remove(path3)

    def test_r_image_filter(self):
    
        # render a point
        # gaussian resolution limit filter makes a gaussian function from this
        # check if standard deviation is correct
        # this also means, that the gaussian curve is completly inside, meaning the image extent was enlarged

        # rays at center of image
        p = np.zeros((1000, 3))
        w = np.ones(1000)
        wl = np.full(1000, 500)

        for k in [1, 0.3, 5]:  # different side ratios
        
            r0 = 1e-4  # smaller than resolution limit
            img = ot.RImage([-r0, r0, -k/2*r0, k/2*r0])

            for limit in [0.2, 20]:  # different limits

                img.limit = limit
                img.render(900, p, w, wl)
                irr = img.img[:, :, 3]  # irradiance image

                # radial distance in image
                ny, nx = irr.shape[:2]
                x = np.linspace(img.extent[0], img.extent[1], nx)
                y = np.linspace(img.extent[2], img.extent[3], ny)
                X, Y = np.meshgrid(x, y)
                R = np.sqrt(X**2 + Y**2)

                # get standard deviation and limit
                # we need to divide by two since we only integrate over half the region
                # because of R**2 > 0
                sigma = np.sqrt(np.sum(R**2 * irr)/np.sum(irr)/2)
                limit2 = sigma / 0.175*1000

                # compare to limit
                self.assertAlmostEqual(limit, limit2, delta=0.012)

    def test_refraction_index(self):

        func = lambda wl: 2.0 - wl/500/5 
        func2 = lambda wl, a: a - wl/500/5 
        wlf = color.wavelengths(1000)
        funcf = func(wlf)
        
        n_list = [
        # https://raw.githubusercontent.com/nzhagen/zemaxglass/master/AGF_files/archer.agf
        (ot.RefractionIndex("Schott", coeff=[2.417473, -0.008685888, 0.01396835, 0.0006180845, -5.274288e-05, 3.679696e-06], desc="S-BAL3M"), 1.568151, 52.737315),
        # https://raw.githubusercontent.com/nzhagen/zemaxglass/master/AGF_files/birefringent.agf
        (ot.RefractionIndex("Sellmeier1", coeff=[1.29899, 0.0089232927, 43.17364, 1188.531, 0.0, 0.0], desc="ADP"), 1.523454, 52.25678),
        (ot.RefractionIndex("Sellmeier4", coeff=[3.1399, 1.3786, 0.02941, 3.861, 225.9009], desc="ALN"), 2.154291, 50.733102),
        (ot.RefractionIndex("Handbook of Optics 1", coeff=[2.7405, 0.0184, 0.0179, 0.0155], desc="BBO"), 1.670737, 52.593907),
        (ot.RefractionIndex("Sellmeier3", coeff=[0.8559, 0.00345744, 0.8391, 0.019881, 0.0009, 0.038809, 0.6845, 49.070025], desc="CALCITE"), 1.658643, 48.541403),
        (ot.RefractionIndex("Sellmeier5", coeff=[0.8559, 0.00345744, 0.8391, 0.019881, 0.0009, 0.038809, 0.6845, 49.070025, 0, 0], desc="CALCITE"), 1.658643, 48.541403),
        (ot.RefractionIndex("Handbook of Optics 2", coeff=[2.81418, 0.87968, 0.09253764, 0.00711], desc="ZNO"), 2.003385, 12.424016),
        # https://raw.githubusercontent.com/nzhagen/zemaxglass/master/AGF_files/hikari.agf
        (ot.RefractionIndex("Extended3", coeff=[3.22566311, -0.0126719158, -0.000122584245, 0.0306900263, 0.000649958511, 1.0629994e-05, 1.20774149e-06, 0.0, 0.0], desc="Q-LASFH19S"), 1.82098, 42.656702),
        (ot.RefractionIndex("Extended2", coeff=[2.54662904, -0.0122972332, 0.0187464623, 0.000460296583, 7.85351066e-07, 1.72720972e-06, -0.000133476806, 0.0], desc="E-KZFH1"), 1.61266, 44.461379),
        # https://raw.githubusercontent.com/nzhagen/zemaxglass/master/AGF_files/infrared.agf
        (ot.RefractionIndex("Herzberger", coeff=[2.2596285, 0.0311097853, 0.0010251756, -0.000594355286, 4.49796618e-07, -4.12852834e-09], desc="CLRTR_OLD"), 2.367678, 15.282309),
        # https://raw.githubusercontent.com/nzhagen/zemaxglass/master/AGF_files/lightpath.agf
        (ot.RefractionIndex("Conrady", coeff=[1.47444837, 0.0103147698, 0.00026742387], desc="NICHIA_MELT1"), 1.493724, 64.358478),
        # https://refractiveindex.info/?shelf=glass&book=OHARA-PHM&page=PHM51
        (ot.RefractionIndex("Extended", coeff=[2.5759016, -0.010553544, 0.013895937, 0.00026498331, -1.9680543e-06, 1.0989977e-07, 0, 0], desc="PHM"), 1.617,  62.8008),
        # elements from presets
        (ot.presets.refraction_index.COC, 1.5324098, 56.0522),

        # different modes
        (ot.RefractionIndex("Abbe", n=1.578, V=70), 1.578, 70),
        (ot.RefractionIndex("Constant", n=1.2546), 1.2546, np.inf),
        (ot.RefractionIndex("Function", func=func), 1.764976, 11.2404),
        (ot.RefractionIndex("Function", func=func2, func_args=dict(a=2)), 1.764976, 11.2404),
        (ot.RefractionIndex("Data", wls=wlf, vals=funcf), 1.764976, 11.2404),
        ]

        for nl_ in n_list:
            n, nc, V = nl_

            n(color.wavelengths(1000))  # call with array

            self.assertEqual(n.is_dispersive(), n.spectrum_type != "Constant")
            self.assertAlmostEqual(n(np.array(ot.presets.spectral_lines.d)), nc, delta=5e-5)
            self.assertAlmostEqual(n.get_abbe_number(), V, delta=0.3)

        # check if equal operator is working
        self.assertEqual(ot.presets.refraction_index.SF10, ot.presets.refraction_index.SF10)
        self.assertEqual(n_list[1][0], n_list[1][0])
        self.assertNotEqual(n_list[1][0], n_list[2][0])
        assert n_list[-1][0].spectrum_type == "Data"
        self.assertEqual(n_list[-1][0], n_list[-1][0])  # comparision of Data type
        self.assertEqual(ot.RefractionIndex("Function", func=func), ot.RefractionIndex("Function", func=func))

    def test_refraction_index_abbe_mode(self):

        for lines in [*ot.presets.spectral_lines.all_line_combinations, None]:
            for nc in [1.01, 1.15, 1.34, 1.56, 1.913, 2.678]:
                for V in [15, 37.56, 78, 156]:
                  n = ot.RefractionIndex("Abbe", n=nc, V=V, lines=lines)
                  self.assertAlmostEqual(nc, n(n.lines[1]), delta=1e-4)
                  self.assertAlmostEqual(V, n.get_abbe_number(lines), delta=1e-2)  # enforce lines...
                  self.assertAlmostEqual(V, n.get_abbe_number(), delta=1e-2)  # ...but should correct one anyway

    def test_refraction_index_exceptions(self):

        # value exceptions
        self.assertRaises(ValueError, ot.RefractionIndex, "ABC")  # invalid type
        self.assertRaises(ValueError, ot.RefractionIndex, "Constant", n=0.99)  # n < 1
        self.assertRaises(ValueError, ot.RefractionIndex, "Constant", n=np.inf)  # n not finite
        self.assertRaises(ValueError, ot.RefractionIndex, "Abbe", V=0)  # invalid V
        self.assertRaises(ValueError, ot.RefractionIndex, "Abbe", V=-1)  # invalid V
        self.assertRaises(ValueError, ot.RefractionIndex, "Abbe", V=np.inf)  # invalid V
        self.assertRaises(ValueError, ot.RefractionIndex, "Abbe", V=np.nan)  # invalid V
        self.assertRaises(ValueError, ot.RefractionIndex, "Abbe", V=50, lines=[380, 480])  
        # lines need to have 3 elements
        self.assertRaises(ValueError, ot.RefractionIndex, "Abbe", V=50, lines=[380, 780, 480])  
        # lines need to be ascending
        self.assertRaises(ValueError, ot.RefractionIndex, "Function", func=lambda wl: 0.5 - wl/wl)  # func < 1
        self.assertRaises(ValueError, ot.RefractionIndex, "Data", wls=[380, 780], vals=[1.5, 0.9])  # vals < 1
        self.assertRaises(ValueError, ot.RefractionIndex, "Cauchy", coeff=[1, 0, 0, 0, 0])  # too many coeff

        # type errors
        self.assertRaises(TypeError, ot.RefractionIndex, "Cauchy", coeff=1)  # invalid coeff type
        self.assertRaises(TypeError, ot.RefractionIndex, "Abbe", V=[1])  # invalid V type
        self.assertRaises(TypeError, ot.RefractionIndex, "Constant", n=[5])  # n not a float
        n2 = ot.RefractionIndex("Function")
        self.assertRaises(TypeError, n2, 550)  # func missing
        self.assertRaises(TypeError, ot.RefractionIndex("Cauchy"), 550)  # coeffs not specified
        
        # misc
        self.assertRaises(AttributeError, n2.__setattr__, "aaa", 1)  # _new_lock active
        self.assertRaises(RuntimeError, ot.RefractionIndex("Cauchy", coeff=[2, -1, 0, 0]), np.array([380., 780.]))  
        # n < 1 on runtime
        
        # check exceptions when wavelengths are outside data range
        wl0, wl1 = color.WL_BOUNDS
        color.WL_BOUNDS[1] = wl1 + 100
        self.assertRaises(RuntimeError, ot.presets.refraction_index.PET, color.wavelengths(1000))
        color.WL_BOUNDS[0] = wl0-100
        color.WL_BOUNDS[1] = wl1
        self.assertRaises(RuntimeError, ot.presets.refraction_index.PET, color.wavelengths(1000))

    def test_refraction_index_equality(self):
        # equal operator
        self.assertTrue(ot.RefractionIndex("Constant") == ot.RefractionIndex("Constant"))
        self.assertFalse(ot.RefractionIndex("Constant", n=2) == ot.RefractionIndex("Constant", n=1))
        self.assertFalse(ot.RefractionIndex("Constant", n=2) == 1)  # comparison between different types
        self.assertTrue(ot.RefractionIndex("Data", wls=[400, 500], vals=[1, 2]) == ot.RefractionIndex("Data", 
                                                                                        wls=[400, 500], vals=[1, 2]))
        self.assertFalse(ot.RefractionIndex("Data", wls=[400, 500], vals=[1, 2]) == ot.RefractionIndex("Data", 
                                                                                         wls=[400, 500], vals=[1, 1]))
        self.assertFalse(ot.RefractionIndex("Data", wls=[400, 500], vals=[1, 2]) == ot.RefractionIndex("Data", 
                                                                                         wls=[400, 501], vals=[1, 2]))

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

        # the following part checks that dither and shuffling are presented
        # since everything is random, there are constellations where all values are shuffled or have not dither
        # we test many times, so the probability become very small
        # but when we encounter any random list with correct shuffling and dither, we break the loop before
        # so we don't need to test that many cases
        b1, b2 = False, False
        for i in np.arange(200):
            rand = misc.uniform(0, 1, 1000000)
            b1 = b1 or np.std(np.diff(np.sort(rand))) > 1e-7  # check step variance (= dither)
            b2 = b2 or np.any(np.diff(rand) < 0)  # check that not all are ascending (= sorted)
            if b1 and b2:
                break

        self.assertTrue(b1)  # check if dithered
        self.assertTrue(b2)  # check if shuffled

        # same for uniform 2
        b1x, b1y, b2x, b2y = False, False, False, False
        N = 1000000  # position number
        N2 = int(np.sqrt(N))  # nearest downwards integer square root of N
        dN = N - N2**2  # difference between square number and desired number
        # if have e.g. A = [[0, 1, 2], [0, 1, 2], [0, 1, 2]]
        # which is of squared shape (N2, N2)
        # in the flattend array [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2] we have therefore N2-1 negative differences between values
        # since N from above is not an exact square number, we have additional dN random values, that also could lead to a
        # negative difference
        # so if values are randomly sorted, we typically (not always, since its random) should have more than N2+dN
        # negative differences between the flattened arrays
        # this is checked in the next part
        for i in np.arange(200):
            randy, randx = misc.uniform2(0, 1, 0, 1, 1000000)
            b1x = b1x or np.count_nonzero(np.abs(np.diff(np.sort(randx))) > 1e-9) > 2*(N2 + dN)  
            # ^-- only few zero steps between sorted values
            b2x = b2x or (np.count_nonzero(np.diff(randx) < 0) > 2*(N2+dN)) # check that not all are ascending (= sorted)
            b1y = b1y or np.count_nonzero(np.abs(np.diff(np.sort(randy))) > 1e-9) > 2*(N2 + dN)  
            # ^-- only few zero steps between sorted values
            b2y = b2y or (np.count_nonzero(np.diff(randy) < 0) > 2*(N2+dN)) # check that not all are ascending (= sorted)
            if b1x and b2x and b1y and b2y:
                break

        self.assertTrue(b1x)  # check if dithered x
        self.assertTrue(b2x)  # check if shuffled x
        self.assertTrue(b1y)  # check if dithered y
        self.assertTrue(b2y)  # check if shuffled y

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
