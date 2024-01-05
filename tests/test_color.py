#!/bin/env python3

import sys
sys.path.append('.')

import doctest
import unittest
import numpy as np
import pytest
import colour  # TODO maybe use cv2 instead
import cv2

import optrace as ot
import optrace.tracer.color as color
import optrace.tracer.misc as misc

from optrace.tracer.color.tools import _WL_MIN0, _WL_MAX0



class ColorTests(unittest.TestCase):

    def test_color_doctest(self):
        # normalize whitespace in doc strings and raise exceptions on errors, so pytest etc. notice errors
        optionflags = doctest.NORMALIZE_WHITESPACE
        opts = dict(optionflags=optionflags, raise_on_error=True)
        # opts = dict(optionflags=optionflags)

        doctest.testmod(ot.tracer.color.illuminants, **opts)
        doctest.testmod(ot.tracer.color.observers, **opts)
        doctest.testmod(ot.tracer.color.srgb, **opts)
        doctest.testmod(ot.tracer.color.xyz, **opts)
        doctest.testmod(ot.tracer.color.tools, **opts)
        doctest.testmod(ot.tracer.color.luv, **opts)

    @pytest.mark.os
    def test_white_d65(self):
        # raytracer geometry of only a Source
        RT = ot.Raytracer(outline=[-3, 3, -3, 3, 0, 6])
        RSS = ot.RectangularSurface(dim=[6, 6])
        RS = ot.RaySource(RSS, pos=[0, 0, 0], spectrum=ot.presets.light_spectrum.d65)
        RT.add(RS)

        # check white color of spectrum in sRGB
        for RSci, WPi in zip(RS.color(), np.array([1, 1, 1])):
            self.assertAlmostEqual(RSci, WPi, delta=0.0005)

        # check white color of spectrum in XYZ
        spec_XYZ = RS.spectrum.xyz()
        for RSci, WPi in zip(spec_XYZ/spec_XYZ[1], color.WP_D65_XYZ):
            self.assertAlmostEqual(RSci, WPi, delta=0.0005) 

        # check color of all rendered rays on RaySource in sRGB
        RT.trace(N=500000)
        spec_RS = RT.source_spectrum()
        for RSci, WPi in zip(spec_RS.color(), np.array([1, 1, 1])):
            self.assertAlmostEqual(RSci, WPi, delta=0.001)

        # check color of all rendered rays on RaySource in XYZ
        RS_XYZ = spec_RS.xyz()
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
        """test xyz -> srgb linear conversion by comparing to colour science conversion"""

        # some XYZ values
        XYZ = np.random.uniform(0, 1.5, (1000000, 3))
       
        # test xyz -> srgb linear
        c = colour.XYZ_to_sRGB(XYZ, apply_cctf_encoding=False)
        c = np.clip(c, 0, 1)

        srgb2 = color.xyz_to_srgb_linear(np.array([XYZ]), normalize=False, rendering_intent="Ignore")[0]
        srgb2 = np.clip(srgb2, 0, 1)
        diff = np.abs(srgb2 - c)
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
        """test xyz -> srgb conversion by comparing to colour-science conversion"""
    
        # some XYZ values
        XYZ = np.random.uniform(0, 1.5, (1000000, 3))
        
        # test xyz -> srgb with invalid values clipped
        c = colour.XYZ_to_sRGB(XYZ)
        c = np.clip(c, 0, 1)

        srgb2 = color.xyz_to_srgb(np.array([XYZ]), normalize=False, clip=True, rendering_intent="Ignore")[0]

        diff = np.abs(srgb2 - c)
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
        """test xyz -> xyY conversion by comparing to colour-science conversion"""

        # some XYZ values
        XYZ = np.random.uniform(0, 1.5, (1000000, 3))
       
        # test xyz -> xy
        c = colour.XYZ_to_xy(XYZ)

        xyy = color.xyz_to_xyY(np.array([XYZ]))[0, :, :2]
        diff = np.abs(xyy - c)
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
        """test xyz -> CIELUV conversion by comparing to colour-science conversion"""
        
        # some XYZ values
        XYZ = np.random.uniform(0, 1.5, (1000000, 3))

        c = colour.XYZ_to_Luv(XYZ)
        luv = color.xyz_to_luv(np.array([XYZ]), normalize=False)[0]
        diff = np.abs(luv - c)/100
        self.assertTrue(np.max(diff) < 0.0003)
        self.assertTrue(np.mean(diff) < 2e-4)
        
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
        """test xyz -> CIELCH conversion by comparing to colour-science conversion"""
        
        # some XYZ values
        XYZ = np.random.uniform(0, 1.5, (1000000, 3))

        # test xyz -> lch
        c = colour.XYZ_to_Luv(XYZ)
        c = colour.Luv_to_LCHuv(c)

        luv = color.xyz_to_luv(np.array([XYZ]), normalize=False)
        lch = np.dstack((luv[:, :, 0],
                         color.luv_chroma(luv),
                         color.luv_hue(luv)))[0]

        # check l and c
        diff = np.abs(lch[:, :2] - c[:, :2])/100  # values are in range 0 - xxx
        where = np.argmax(diff.flatten())
        self.assertAlmostEqual(np.max(diff), 0, delta=5e-4)
        self.assertAlmostEqual(np.mean(diff), 0, delta=1e-4)
       
        # hue needs to be handled differently, see
        # https://stackoverflow.com/a/7869457
        get_diff_ang = lambda a, b: (a-b + 180) % 360 - 180
        diff_ang = np.mean(np.abs(get_diff_ang(lch[:, 2], c[:, 2])))

        # colour science seems to do conversion a little differently, therefore larger mean error
        self.assertAlmostEqual(diff_ang/360, 0, delta=0.0001)  # 0.01% mean error

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
                self.assertTrue(np.max(srgb2) < 1 + 1e-6)
                
                # impossible colors are not corrected with RI == Perceptual
                # so clip=True should be provided in xyz_to_srgb
                if RI == "Absolute":
                    self.assertTrue(np.min(srgb2) > -1e-6)

    @pytest.mark.slow
    def test_image_color_rendering(self):
       
        # assign colored image to RaySource
        Image = np.array([[[0, 1, 0], [1, 1, 1], [1, 0, 0], [0, 0, 1], [0, 0, 0]],
                         [[1, 1, 0], [1, 0, 1], [0, 1, 1], [0.1, 0.1, 0.1], [0, 0, 0]],
                         [[0, 0, 0], [0.2, 0.5, 1], [0.01, 1, 1], [0.01, 0.01, 0.01], [0, 0, 0]],
                         [[0.2, 0.5, 0.1], [0, 0.8, 1.0], [0.5, 0.3, 0.5], [0.1, 0., 0.7], [0, 0, 0]],
                         [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]], dtype=np.float32)
        RSS = ot.RGBImage(Image, [6, 6])
        
        RT = ot.Raytracer(outline=[-3, 3, -3, 3, 0, 6], no_pol=True)
        RS = ot.RaySource(RSS, pos=[0, 0, 0])
        RT.add(RS)
        
        RT.add(ot.Detector(ot.RectangularSurface([6, 6]), pos=[0, 0, 1e-9]))

        # get Source Image
        SIm = RT.iterative_render(100000000, extent=[-3, 3, -3, 3])[0]
        RS_XYZ = cv2.resize(SIm._data[:, :, :3], [5, 5], interpolation=cv2.INTER_AREA)

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
        wl0, wl1 = ot.global_options.wavelength_range
        ot.global_options.wavelength_range[:] = [200, 700]  # we need to 
        wl = color.wavelengths(2)
        self.assertEqual(wl[0], 200.)
        self.assertEqual(wl[1], 700.)

        # reset range
        ot.global_options.wavelength_range[:] = [wl0, wl1]
        wl = color.wavelengths(2)
        self.assertEqual(wl[0], wl0)
        self.assertEqual(wl[1], wl1)

    def test_additional_coverage(self):

        # check if random_wavelengths_from_srgb correctly handles wavelength range changes

        # some rgb image from presets
        rgb = ot.presets.image.landscape([1, 1]).data
        rgb = rgb.reshape((rgb.shape[0]*rgb.shape[1], 3))

        # initial bounds
        wl0b, wl1b = ot.global_options.wavelength_range
        
        # upper bound too low
        ot.global_options.wavelength_range[1] = wl1b - 10
        self.assertRaises(RuntimeError, color.random_wavelengths_from_srgb, rgb)

        # lower bound too high
        ot.global_options.wavelength_range[1] = wl1b
        ot.global_options.wavelength_range[0] = wl0b + 10
        self.assertRaises(RuntimeError, color.random_wavelengths_from_srgb, rgb)
   
        # larger range, but this is valid
        ot.global_options.wavelength_range[:] = [wl0b - 10, wl1b + 10]
        color.random_wavelengths_from_srgb(rgb)
        ot.global_options.wavelength_range[:] = [wl0b, wl1b]

        # zero images
        Img0 = np.zeros((10, 10, 3), dtype=np.float64)
        sRGBL0 = color.xyz_to_srgb_linear(Img0)  # zero image in conversion
        sRGB0 = color.srgb_linear_to_srgb(sRGBL0)  # default setting is clip=True
        Luv0 = color.xyz_to_luv(Img0)  # zero image in Luv conversion

    def test_rgb_primaries_outside_range(self):
        """test that srgb primaries are strictly zero outside range"""
        
        for prim in [color.srgb_r_primary, color.srgb_g_primary, color.srgb_b_primary]:
            arr = np.array([_WL_MIN0-1, _WL_MAX0+1])
            val = prim(arr)
            self.assertEqual(val[0], 0)
            self.assertEqual(val[1], 0)

    def test_dominant_complementary_wavelengths(self):
        # gets tested in test_spectrum()
        pass

    def test_blackbody_normalized(self):

        # peak in visible range should be =1 for all temperatures, even if the peak is outside the visible region
        wl = color.wavelengths(100000)
        for T in [100, 1000, 3000, 500, 7800, 10000]:
            spec = color.normalized_blackbody(wl, T)
            self.assertAlmostEqual(np.max(spec), 1)

    def test_log_srgb(self):

        # log scale some image
        arr = np.array([[[0.2, 0.2, 0.1], [0.1, 0.1, 0.1]], [[1, 1, 1], [0, 0, 0]]])
        arr2 = color.log_srgb(arr)

        # check values
        self.assertFalse(np.all(arr2[0, 0] == arr[0, 0]))  # first value differs now
        self.assertTrue(np.all(arr2[0, 1] > 0))  # smallest value stays non-zero
        
        # test const image
        const = np.ones((100, 100, 3), dtype=np.float64)
        const2 = color.log_srgb(const)
        self.assertTrue(np.all(const == const2))

        # test zero image
        zero = np.zeros((100, 100, 3))
        zero2 = color.log_srgb(zero)
        self.assertTrue(np.all(zero == zero2))

        # generate srgb image in standard and logarithmic mode
        rgb = np.random.sample((1000, 1000, 3))
        luv0 = color.xyz_to_luv(color.srgb_to_xyz(rgb))
        lrgb = color.log_srgb(rgb)
        luv = color.xyz_to_luv(color.srgb_to_xyz(lrgb))

        # make sure logarithmic has a higher average lightness
        self.assertTrue(np.mean(luv0[:, :, 0]) < np.mean(luv[:, :, 0]))

        # make sure the saturation stayed roughly the same for all colors
        nz = luv[:, :, 0] > 0
        sat_quot = color.luv_saturation(luv0)[nz] / color.luv_saturation(luv)[nz]
        self.assertTrue(np.allclose(sat_quot, 1, rtol=1e-3))

    @pytest.mark.os
    def test_illuminant_whitepoint(self):

        # coordinates from https://en.wikipedia.org/wiki/Standard_illuminant#Illuminant_A
        coords = {
            "A": 	    (0.44757, 0.40745),
            "C": 	    (0.31006, 0.31616),
            "D50":      (0.34567, 0.35850),
            "D55":      (0.33242, 0.34743),
            "D65":      (0.31271, 0.32902),
            "D75":      (0.29902, 0.31485),
            "E":        (0.33333, 0.33333),
            "FL2":      (0.37208, 0.37529),
            "FL7":      (0.31292, 0.32933),
            "FL11":     (0.38052, 0.37713),
            "LED-B1":   (0.4560, 0.4078), 
            "LED-B2":   (0.4357, 0.4012), 
            "LED-B3":   (0.3756, 0.3723), 
            "LED-B4":   (0.3422, 0.3502), 
            "LED-B5":   (0.3118, 0.3236)
        }

        for ill in ot.presets.light_spectrum.standard:
            xyz = np.array([[[*ill.xyz()]]])
            coord = color.xyz_to_xyY(xyz)[0, 0]

            # higher delta, because we get different results depending on interpolation method of the illuminant
            self.assertAlmostEqual(coord[0], coords[ill.desc][0], delta=0.0003)
            self.assertAlmostEqual(coord[1], coords[ill.desc][1], delta=0.0003)

    def test_srgb_primaries(self):

        # text xy and uv positions of srgb primaries
        # also tests the corresponding constants in ot.tracer.color

        xy_coords = [color.SRGB_R_XY, color.SRGB_G_XY, color.SRGB_B_XY, color.WP_D65_XY]
        uv_coords = [color.SRGB_R_UV, color.SRGB_G_UV, color.SRGB_B_UV, color.WP_D65_UV]
        prec = 2e-5

        for ch, coord_xy, coord_uv in zip(ot.presets.light_spectrum.srgb, xy_coords, uv_coords):
            xyz = np.array([[[*ch.xyz()]]])
            xy = color.xyz_to_xyY(xyz)[0, 0]
            self.assertAlmostEqual(xy[0], coord_xy[0], delta=prec)
            self.assertAlmostEqual(xy[1], coord_xy[1], delta=prec)
            self.assertAlmostEqual(1-xy[0]-xy[1], 1-coord_xy[1]-coord_xy[0], delta=prec)

            uv = color.luv_to_u_v_l(color.xyz_to_luv(xyz))[0, 0, :2]
            self.assertAlmostEqual(uv[0], coord_uv[0], delta=prec)
            self.assertAlmostEqual(uv[1], coord_uv[1], delta=prec)

    def test_srgb_lines(self):

        # rgb lines have same dominant wavelength as rgb primaries
        # so with srgb conversion with mode "Absolute" we should get the same chromaticities as the primaries
        BGR = np.array(ot.presets.spectral_lines.rgb)
        XYZ = np.array([np.vstack((color.x_observer(BGR),
                                   color.y_observer(BGR),
                                   color.z_observer(BGR)))])
        XYZ = np.swapaxes(XYZ, 2, 1)
        XYZ2 = color.srgb_linear_to_xyz(color.xyz_to_srgb_linear(XYZ, rendering_intent="Absolute"))
        xyY = color.xyz_to_xyY(XYZ2)

        for xy0, xy1 in zip(xyY[0], [color.SRGB_B_XY, color.SRGB_G_XY, color.SRGB_R_XY]):
            self.assertAlmostEqual(xy0[0], xy1[0], delta=0.00001)
            self.assertAlmostEqual(xy0[1], xy1[1], delta=0.00001)

    def test_lines(self):
        # check type and value of spectral lines

        for lines in [ot.presets.spectral_lines.all_lines, *ot.presets.spectral_lines.all_line_combinations]:
            for line in lines:
                self.assertTrue(isinstance(line, float))
                self.assertTrue(line >= 380)
                self.assertTrue(line <= 780)

        # lines should be sorted by value
        lines = ot.presets.spectral_lines.all_lines
        self.assertTrue(np.all(np.diff(lines) > 0))


    def test_color_sat_scale(self):
        """test saturation scaling in 'Perceptual' rendering intent """

        # convert wavelength to XYZ color space
        def wl_to_xyz(wl):
            XYZ = np.column_stack((color.x_observer(wl),
                                   color.y_observer(wl),
                                   color.z_observer(wl)))
            return np.array([XYZ])

        # case 1: colors at gamut edge
        wl = color.wavelengths(1000)
        xyz = wl_to_xyz(wl)
        luv = color.xyz_to_luv(xyz)
        sf = color.get_saturation_scale(luv)
        self.assertAlmostEqual(sf, 0.3236, delta=1e-4)

        # case 2: srgb colors
        srgb = np.random.sample((100, 100, 3))
        xyz = color.srgb_to_xyz(srgb)
        luv = color.xyz_to_luv(xyz)
        sf = color.get_saturation_scale(luv)
        self.assertAlmostEqual(sf, 1.0, delta=1e-5)
        
        # case 3: black image
        srgb = np.zeros((100, 100, 3))
        xyz = color.srgb_to_xyz(srgb)
        luv = color.xyz_to_luv(xyz)
        sf = color.get_saturation_scale(luv)
        self.assertAlmostEqual(sf, 1.0, delta=1e-5)
        
        # case 4: white image
        srgb = np.ones((100, 100, 3))
        xyz = color.srgb_to_xyz(srgb)
        luv = color.xyz_to_luv(xyz)
        sf = color.get_saturation_scale(luv)
        self.assertAlmostEqual(sf, 1.0, delta=1e-5)

        # case 5: srgb colors with faint value outside srgb gamut
        srgb = np.random.sample((100, 100, 3))
        xyz = color.srgb_to_xyz(srgb)
        luv = color.xyz_to_luv(xyz)
        luv[0, 0] = [6.225, 4.216, -36.377]  # outside sRGB gamut
        luv[0, 1] = [100, 0, 0]  # brightest pixel
        sf = color.get_saturation_scale(luv)
        self.assertNotAlmostEqual(sf, 1.0, delta=1e-1)  # factor should be < 1
        
        # case 6: srgb colors with faint value outside srgb gamut, but below threshold
        # ignore values below 7% of peak brightness (which is 100, see above)
        sf = color.get_saturation_scale(luv, L_th=0.07)
        self.assertAlmostEqual(sf, 1.0, delta=1e-5)  # faint values are ignored, factor is 1

        # case 7: all colors are outside the visible gamut (for coverage testing)
        xyz = np.zeros((100, 100, 3))
        xyz[:, :, 1] = 1  # XYZ = (0, 1, 0) is a color left of the visible CIE xyz gamut
        luv = color.xyz_to_luv(xyz)
        sf = color.get_saturation_scale(luv)
        self.assertAlmostEqual(sf, 1.0, delta=1e-5)

        # case 8: gamut edge colors get scaled to sRGB edge and scaling factor is now 1
        wl = color.wavelengths(1000)
        xyz = wl_to_xyz(wl)
        srgb = color.xyz_to_srgb(xyz, rendering_intent="Perceptual")
        xyz2 = color.srgb_to_xyz(srgb)
        luv = color.xyz_to_luv(xyz2)
        sf = color.get_saturation_scale(luv)
        self.assertAlmostEqual(sf, 1, delta=1e-4)

        # case 9: like case #6, but check color.xyz_to_srgb function
        srgb = np.random.sample((100, 100, 3))
        xyz = color.srgb_to_xyz(srgb)
        luv = color.xyz_to_luv(xyz)
        luv[0, 0] = [6.225, 4.216, -36.377]  # outside sRGB gamut
        luv[0, 1] = [100, 0, 0]  # brightest pixel
        xyz2 = color.luv_to_xyz(luv)
        srgb2 = color.xyz_to_srgb(xyz2, rendering_intent="Perceptual", L_th=0.07, clip=False)
        xyz2 = color.srgb_to_xyz(srgb2)
        luv2 = color.xyz_to_luv(xyz2)
        sf = color.get_saturation_scale(luv2)
        self.assertNotAlmostEqual(sf, 1.0, delta=1e-1)  # factor should be < 1

        # case 10: manual scaling option
        wl = color.wavelengths(1000)
        xyz = wl_to_xyz(wl)
        srgb = color.xyz_to_srgb(xyz, rendering_intent="Perceptual", sat_scale=0.5, clip=False)
        xyz2 = color.srgb_to_xyz(srgb)
        luv = color.xyz_to_luv(xyz2)
        sf = color.get_saturation_scale(luv)
        self.assertAlmostEqual(sf, 2*0.3236, delta=1e-4)  # twice the value from case #1

    def test_spectral_color_map(self):

        # wl = np.linspace(100, 1000, 1000)
        wl = color.wavelengths(1000)
        rgba = color.spectral_colormap(wl)

        # check data range
        self.assertAlmostEqual(np.max(rgba[:, :3]), 1, delta=0.01)  # highest color is around 1
        self.assertAlmostEqual(np.min(rgba[:, :3]), 0, delta=0.00001)  # lowest color around 0
        self.assertTrue(np.allclose(rgba[:, 3], 1))  # alpha 100% everywhere

        # calculate hue
        srgb = np.array([rgba[:, :3]])
        srgbl = color.srgb_to_srgb_linear(srgb)
        xyz = color.srgb_linear_to_xyz(srgbl)
        luv = color.xyz_to_luv(xyz)
        hue = color.luv_hue(luv)

        # check hue range and direction
        mstep = np.mean(np.diff(hue[0, :]))
        self.assertTrue(mstep < 0)  # quotient below 0, as hue decreases
        self.assertAlmostEqual(np.abs(mstep)*wl.size/360, 0.75, delta=0.1)  # roughly covers 75% of the hue range
        
if __name__ == '__main__':
    unittest.main()

