#!/bin/env python3

import sys
sys.path.append('.')

import doctest
import unittest
import numpy as np
import pytest
import colorio

import optrace as ot
import optrace.tracer.color as color
import optrace.tracer.misc as misc



class ColorTests(unittest.TestCase):

    def test_color_doctest(self):
        doctest.testmod(ot.tracer.color.illuminants)
        doctest.testmod(ot.tracer.color.observers)
        doctest.testmod(ot.tracer.color.srgb)
        doctest.testmod(ot.tracer.color.xyz)
        doctest.testmod(ot.tracer.color.tools)
        doctest.testmod(ot.tracer.color.luv)

    @pytest.mark.os
    def test_white_d65(self):
        # raytracer geometry of only a Source
        RT = ot.Raytracer(outline=[-3, 3, -3, 3, 0, 6], silent=True)
        RSS = ot.RectangularSurface(dim=[6, 6])
        RS = ot.RaySource(RSS, pos=[0, 0, 0], spectrum=ot.presets.light_spectrum.d65)
        RT.add(RS)

        # check white color of spectrum in sRGB
        for RSci, WPi in zip(RS.get_color(), np.array([1, 1, 1])):
            self.assertAlmostEqual(RSci, WPi, delta=0.0005)

        # check white color of spectrum in XYZ
        spec_XYZ = RS.spectrum.get_xyz()
        for RSci, WPi in zip(spec_XYZ/spec_XYZ[1], color.WP_D65_XYZ):
            self.assertAlmostEqual(RSci, WPi, delta=0.0005) 

        # check color of all rendered rays on RaySource in sRGB
        RT.trace(N=500000)
        spec_RS = RT.source_spectrum()
        for RSci, WPi in zip(spec_RS.get_color(), np.array([1, 1, 1])):
            self.assertAlmostEqual(RSci, WPi, delta=0.001)

        # check color of all rendered rays on RaySource in XYZ
        RS_XYZ = spec_RS.get_xyz()
        for RSci, WPi in zip(RS_XYZ/RS_XYZ[1], color.WP_D65_XYZ):
            self.assertAlmostEqual(RSci, WPi, delta=0.001) 

    def xyz_grid(self, N=1000000):
        x, y = misc.uniform2(1e-6, 0.8, 1e-6, 0.9, N)

        # xyY -> XYZ rescales by Y/y, so multiply by y so range is [0...1]
        Y = misc.uniform(1e-6, 1.2, x.shape[0])*y
   
        # use only values with valid z = 1 - x - y >= 0
        valid = (1 - x - y) >= 0
        x, y, Y = x[valid], y[valid], Y[valid]
        xyY = np.dstack((x, y, Y))
        return color.xyY_to_xyz(xyY)
    
    def test_srgb_linear_conversion(self):
        """test xyz -> srgb linear conversion by comparing to colorios conversion"""

        # some XYZ values
        XYZ = np.random.uniform(0, 1.5, (1000000, 3))
       
        # test xyz -> srgb linear
        c = colorio.cs.ColorCoordinates(XYZ.T, "xyz1")
        c.convert("srgblinear", mode="clip")
        srgb2 = color.xyz_to_srgb_linear(np.array([XYZ]), normalize=False, rendering_intent="Ignore")[0]
        srgb2 = np.clip(srgb2, 0, 1)
        diff = np.abs(srgb2 - c.data.T)
        self.assertTrue(np.max(diff) < 0.0005)
        self.assertTrue(np.mean(diff) < 1e-4)
        
        # special case 1: completely dark
        d = color.xyz_to_srgb_linear(np.zeros((1, 1, 3)))
        self.assertTrue(np.allclose(d, 0))
        
        # special case 2: white
        # make sure we don't normalize since it needs to match absolutely
        d = color.xyz_to_srgb_linear(np.array([[[*color.WP_D65_XYZ]]]), normalize=False)
        self.assertTrue(np.allclose(d, 1))

    def test_outside_srgb(self):

        # negative srgb values are outside the gamut
        srgbl = np.random.uniform(-1, 1, (500, 500, 3))

        # convert to xyz
        xyz = color.srgb_linear_to_xyz(srgbl)

        # check which lie outside
        og = color.outside_srgb_gamut(xyz)

        # check if those are true that had negative srgb values to begin with
        self.assertTrue(np.all(og == np.any(srgbl < 0, axis=2)))

    def test_srgb_conversion(self):
        """test xyz -> srgb conversion by comparing to colorios conversion"""
    
        # some XYZ values
        XYZ = np.random.uniform(0, 1.5, (1000000, 3))
        
        # test xyz -> srgb
        c = colorio.cs.ColorCoordinates(XYZ.T, "xyz1")
        c.convert("srgb1", mode="clip")
        srgb2 = color.xyz_to_srgb(np.array([XYZ]), normalize=False, clip=True, rendering_intent="Ignore")[0]
        diff = np.abs(srgb2 - c.data.T)
        self.assertTrue(np.max(diff) < 0.008)
        self.assertTrue(np.mean(diff) < 1e-4)
    
        # special case 1: completely dark
        d = color.xyz_to_srgb(np.zeros((1, 1, 3)))
        self.assertTrue(np.allclose(d, 0))
        
        # special case 2: white
        # make sure we don't normalize since it needs to match absolutely
        d = color.xyz_to_srgb(np.array([[[*color.WP_D65_XYZ]]]), normalize=False)
        self.assertTrue(np.allclose(d, 1))

    def test_xyY_conversion(self):
        """test xyz -> xyY conversion by comparing to colorios conversion"""

        # some XYZ values
        XYZ = np.random.uniform(0, 1.5, (1000000, 3))
       
        # test xyz -> xyY
        c = colorio.cs.ColorCoordinates(XYZ.T, "xyz1")
        c.convert("xyy1")
        xyy = color.xyz_to_xyY(np.array([XYZ]))[0]
        diff = np.abs(xyy - c.data.T)
        self.assertTrue(np.max(diff) < 1e-10)
        self.assertTrue(np.mean(diff) < 1e-10)

        # special case 1: completely dark
        d = color.xyz_to_xyY(np.zeros((1, 1, 3)))
        self.assertTrue(np.allclose(d - [*color.WP_D65_XY, 0], 0, atol=0.0001))
        
        # special case 2: white
        d = color.xyz_to_xyY(np.array([[[*color.WP_D65_XYZ]]]))
        self.assertTrue(np.allclose(d - [*color.WP_D65_XY, 1], 0, atol=0.0001))

    def test_xyz_from_spectrum(self):

        wl = color.wavelengths(5000)
        spec = color.d65_illuminant(wl)

        # calculate whitepoint for a spectrum
        XYZ0 = np.array(color.WP_D65_XYZ)
        XYZ1 = color.xyz_from_spectrum(wl, spec, method="sum")
        XYZ2 = color.xyz_from_spectrum(wl, spec, method="trapz")
        XYZ1 /= XYZ1[1]
        XYZ2 /= XYZ2[1]

        # higher uncertainty since the whitepoint was calculated on more precise data than the spacing in our illuminant
        self.assertTrue(np.allclose(XYZ1 - XYZ0, 0, atol=5e-3))
        self.assertTrue(np.allclose(XYZ2 - XYZ0, 0, atol=5e-3))
        self.assertTrue(np.any(XYZ1 != XYZ2))  # different results depending on method

    def test_luv_conversion(self):
        """test xyz -> CIELUV conversion by comparing to colorios conversion"""
        
        # some XYZ values
        XYZ = np.random.uniform(0, 1.5, (1000000, 3))

        # test xyz -> luv
        c = colorio.cs.ColorCoordinates(XYZ.T, "xyz1")
        c.convert("cieluv")
        luv = color.xyz_to_luv(np.array([XYZ]), normalize=False)[0]
        diff = np.abs(luv - c.data.T)/100
        self.assertTrue(np.max(diff) < 0.00003)
        self.assertTrue(np.mean(diff) < 1e-7)
        
        # special case 1: completely dark
        d = color.xyz_to_luv(np.zeros((1, 1, 3)), normalize=False)
        self.assertTrue(np.allclose(d, 0, atol=1e-8))
        
        # special case 2: white
        d = color.xyz_to_luv(np.array([[[*color.WP_D65_XYZ]]]), normalize=False)
        self.assertTrue(np.allclose(d - [100, 0, 0], 0, atol=1e-5))
       
        # special case 3: white in Lu'v'
        d_ = color.luv_to_u_v_l(d)
        should = [*color.WP_D65_LUV[1:], color.WP_D65_LUV[0]]
        self.assertTrue(np.allclose(d_ - should, 0, atol=1e-5))

        # check normalize parameter, maximum L should be 100, rest is unchanged
        luvn = color.xyz_to_luv(np.array([XYZ]), normalize=True)[0]
        self.assertAlmostEqual(np.max(luvn[:, 0]), 100)

        # chroma gets scaled with lightness, therefore normalize color by lightness
        # indendent of lightness normalization the ratio should stay the same
        m = luvn[:, 0] > 0  # only for values with non-zero lighness
        diff = luvn[m, 1:]/luvn[m, 0, np.newaxis] - luv[m, 1:]/luv[m, 0, np.newaxis]
        self.assertTrue(np.allclose(diff, 0))

    def test_luv_chroma_hue(self):
        """test xyz -> CIELCH conversion by comparing to colorios conversion"""
        
        # some XYZ values
        XYZ = np.random.uniform(0, 1.5, (1000000, 3))

        # test xyz -> lch
        c = colorio.cs.ColorCoordinates(XYZ.T, "xyz1")
        c.convert("ciehcl")
        luv = color.xyz_to_luv(np.array([XYZ]), normalize=False)
        lch = np.dstack((luv[:, :, 0],
                         color.luv_chroma(luv),
                         color.luv_hue(luv)))

        # check l and c
        diff = np.abs(lch[:, :, :2] - c.data.T[:, :2])/100  # values are in range 0 - xxx
        where = np.argmax(diff.flatten())
        self.assertAlmostEqual(np.max(diff), 0, delta=1e-4)
        self.assertAlmostEqual(np.mean(diff), 0, delta=1e-6)
        
        # hue needs to be handled differently, see
        # https://stackoverflow.com/a/7869457
        get_diff_ang = lambda a, b: (a-b + 180) % 360 - 180
        diff_ang = np.max(np.abs(get_diff_ang(lch[:, :, 2], c.data.T[:, 2])))
        self.assertAlmostEqual(diff_ang/360, 0, delta=0.003)  # 0.03% error, equals around 0.1 deg

        # special case 1: completely dark
        d = color.xyz_to_luv(np.zeros((1, 1, 3)), normalize=False)
        self.assertAlmostEqual(color.luv_chroma(d), 0, delta=1e-5)
        # hue is hard to check since it hugely depends on numerical errors

        # # special case 2: white
        d = color.xyz_to_luv(np.array([[[*color.WP_D65_XYZ]]]), normalize=False)
        self.assertAlmostEqual(color.luv_chroma(d), 0, delta=1e-5)

    def test_luv_saturation(self):

        # some XYZ values
        XYZ = np.random.uniform(1e-9, 1.5, (1000000, 3))  # exclude zero

        # CIELUV Saturation from xyz
        luv = color.xyz_to_luv(np.array([XYZ]), normalize=False)
        lch = np.dstack((luv[:, :, 0],
                         color.luv_chroma(luv),
                         color.luv_hue(luv)))

        sat0 = color.luv_saturation(luv)
        sat1 = lch[:, :, 1]/lch[:, :, 0]
        diff = np.abs(sat1 - sat0)/100
        self.assertTrue(np.max(diff) < 1e-9)
        self.assertTrue(np.mean(diff) < 1e-9)
        
        # special case 1: completely dark
        d = color.xyz_to_luv(np.zeros((1, 1, 3)), normalize=False)
        sat = color.luv_saturation(d)
        self.assertAlmostEqual(sat, 0, delta=1e-6)

        # special case 2: white
        d = color.xyz_to_luv(np.array([[[*color.WP_D65_XYZ]]]), normalize=False)
        sat = color.luv_saturation(d)
        self.assertAlmostEqual(sat, 0, delta=1e-6)

    def test_srgb_roundtrip(self):
        """ test if converting xyz -> srgb -> xyz preserves values"""

        # generate XYZ grid
        XYZ = self.xyz_grid()

        # convert to srgb and back without losing information (normalizing, clipping, mapping)
        srgb = color.xyz_to_srgb(XYZ, normalize=False, clip=False, rendering_intent="Ignore")
        XYZ2 = color.srgb_to_xyz(srgb)

        # check if we are where we started
        self.assertTrue(np.allclose(XYZ-XYZ2, 0, atol=5e-7, rtol=0))
    
    def test_xyy_roundtrip(self):
        """ test if converting xyz -> xyY -> xyz preserves values"""

        # generate XYZ grid
        XYZ = self.xyz_grid()  # excludes Y = 0 since the conversion is not unique there

        # convert to xyY and back without losing information
        xyY = color.xyz_to_xyY(XYZ)
        XYZ2 = color.xyY_to_xyz(xyY)

        # check if we are where we started
        self.assertTrue(np.allclose(XYZ-XYZ2, 0, atol=1e-7, rtol=0))

    def test_luv_roundtrip(self):
        """ test if converting xyz -> luv -> xyz preserves values"""

        # generate XYZ grid
        XYZ = self.xyz_grid()
        
        # convert to srgb and back without losing information (normalizing, clipping, mapping)
        luv = color.xyz_to_luv(XYZ, normalize=False)
        XYZ2 = color.luv_to_xyz(luv)

        # L is discontinuous at L = k*e = 7.999625.. (see constants in LUV conversion)
        # numerical errors lead to a dither at this step
        # check area around step with a larger threshold
        diff = np.abs(XYZ-XYZ2)
        ls = np.abs(8 - luv[:, :, 0]) > 1e-2  # "save L", values away from discont. step
        self.assertTrue(np.allclose(diff[ls], 0, atol=1e-8, rtol=0))
        self.assertTrue(np.allclose(diff[~ls], 0, atol=1e-4, rtol=0))

    def test_srgb_hue_brightness_preservation(self):

        # generate XYZ grid
        XYZ0 = self.xyz_grid()
        LUV0 = color.xyz_to_luv(XYZ0, normalize=False)
        H0 = color.luv_hue(LUV0)
        L0 = LUV0[:, :, 0]

        for RI in color.SRGB_RENDERING_INTENTS:
            srgb = color.xyz_to_srgb(XYZ0, normalize=False, rendering_intent=RI, clip=False)
            xyz2 = color.srgb_to_xyz(srgb)
            luv = color.xyz_to_luv(xyz2, normalize=False)
            H = color.luv_hue(luv)
            L = luv[:, :, 0]

            # check hue
            # correct angle difference:
            # https://stackoverflow.com/a/7869457
            # otherwise 0.001 and 359.999 would be treated as far away
            get_diff_ang = lambda a, b: (a-b + 180) % 360 - 180
            diff_ang = np.max(np.abs(get_diff_ang(H, H0)))
            self.assertAlmostEqual(diff_ang/360, 0, delta=0.003)  # 0.03% error, equals around 0.1 deg

            # check lightness
            self.assertTrue(np.allclose((L-L0)/100, 0, atol=1e-4, rtol=0))

            # check if srgb coordinates are all inside [0, 1]
            # however this is only true with normalize=True in srgb conversion
            if RI != "Ignore":
                srgb2 = color.xyz_to_srgb(XYZ0, normalize=True, rendering_intent=RI, clip=False)
                self.assertTrue(np.min(srgb2) > -1e-6)
                self.assertTrue(np.max(srgb2) < 1 + 1e-6)

    @pytest.mark.slow
    def test_image_color_rendering(self):
        # return
        RT = ot.Raytracer(outline=[-3, 3, -3, 3, 0, 6], silent=True, no_pol=True)
        RSS = ot.RectangularSurface(dim=[6, 6])
        RS = ot.RaySource(RSS, pos=[0, 0, 0], spectrum=ot.presets.light_spectrum.d65)
        RT.add(RS)
        
        # assign colored image to RaySource
        RS.image = Image = np.array([[[0, 1, 0], [1, 1, 1], [1, 0, 0], [0, 0, 1], [0, 0, 0]],
                                    [[1, 1, 0], [1, 0, 1], [0, 1, 1], [0.1, 0.1, 0.1], [0, 0, 0]],
                                    [[0, 0, 0], [0.2, 0.5, 1], [0.01, 1, 1], [0.01, 0.01, 0.01], [0, 0, 0]],
                                    [[0.2, 0.5, 0.1], [0, 0.8, 1.0], [0.5, 0.3, 0.5], [0.1, 0., 0.7], [0, 0, 0]],
                                    [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]], dtype=np.float32)
        SIm, _ = RT.iterative_render(100000000, N_px_S=5, silent=True)

        # get Source Image
        RS_XYZ = np.flipud(SIm[0].get_xyz())  # flip so element [0, 0] is in lower left
        Im_px = color.srgb_to_xyz(Image).reshape((25, 3))
        RS_px = RS_XYZ.reshape((25, 3))
        Im_px /= np.max(Im_px)
        RS_px /= np.max(RS_px)

        # check if is near original image. 
        # Unfortunately many rays are needed for the color to be near the actual value
        for i in range(len(Im_px)):
            for RSp, Cp in zip(RS_px[i], Im_px[i]):
                self.assertAlmostEqual(RSp, Cp, delta=0.002)

    def test_non_default_wavelength_range(self):
        # change wavelength range
        wl0, wl1 = color.WL_BOUNDS
        color.WL_BOUNDS[:] = [200, 700]  # we need to 
        wl = color.wavelengths(2)
        self.assertEqual(wl[0], 200.)
        self.assertEqual(wl[1], 700.)

        # reset range
        color.WL_BOUNDS[:] = [wl0, wl1]
        wl = color.wavelengths(2)
        self.assertEqual(wl[0], wl0)
        self.assertEqual(wl[1], wl1)

    def test_additional_coverage(self):

        # check if random_wavelengths_from_srgb correctly handles wavelength range changes

        # some rgb image from presets
        rgb = ot.presets.image.racoon
        rgb = rgb.reshape((rgb.shape[0]*rgb.shape[1], 3))

        # initial bounds
        wl0b, wl1b = color.WL_BOUNDS
        
        # upper bound too low
        color.WL_BOUNDS[1] = wl1b - 10
        self.assertRaises(RuntimeError, color.random_wavelengths_from_srgb, rgb)

        # lower bound too high
        color.WL_BOUNDS[1] = wl1b
        color.WL_BOUNDS[0] = wl0b + 10
        self.assertRaises(RuntimeError, color.random_wavelengths_from_srgb, rgb)
   
        # larger range, but this is valid
        color.WL_BOUNDS[:] = [wl0b - 10, wl1b + 10]
        color.random_wavelengths_from_srgb(rgb)
        color.WL_BOUNDS[:] = [wl0b, wl1b]

        # zero images
        Img0 = np.zeros((10, 10, 3), dtype=np.float64)
        sRGBL0 = color.xyz_to_srgb_linear(Img0)  # zero image in conversion
        sRGB0 = color.srgb_linear_to_srgb(sRGBL0)  # default setting is clip=True
        Luv0 = color.xyz_to_luv(Img0)  # zero image in Luv conversion

if __name__ == '__main__':
    unittest.main()

