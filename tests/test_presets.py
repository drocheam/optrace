#!/bin/env python3

import sys
sys.path.append('.')

import unittest
import numpy as np
import pytest
import matplotlib.pyplot as plt

import optrace as ot
import optrace.plots as otp
from optrace.tracer import color


class PresetTests(unittest.TestCase):

    def test_refraction_index_presets(self):

        wl = color.wavelengths(1000)

        # check presets
        for material in ot.presets.refraction_index.all_presets:
            n = material(wl)
            A0 = material.abbe_number()

            self.assertTrue(np.all((1 <= n) & (n <= 2.5)))  # sane refraction index everywhere
            self.assertTrue(A0 == np.inf or A0 < 150)  # sane Abbe Number

            # should have descriptions
            self.assertNotEqual(material.desc, "")
            self.assertNotEqual(material.long_desc, "")

            # real dispersive materials have a declining n
            if material.is_dispersive():
                self.assertTrue(np.all(np.diff(n) < 0)) # steadily declining

    @pytest.mark.os
    def test_light_spectrum_presets(self):
        # check presets
        wl = color.wavelengths(1000)
        for spec in ot.presets.light_spectrum.all_presets:

            if spec.is_continuous():
                spec(wl)

            spec.xyz()
            spec.color()
            spec.random_wavelengths(1000)

            # should have descriptions
            self.assertNotEqual(spec.desc, "")
            self.assertNotEqual(spec.long_desc, "")

    @pytest.mark.os
    def test_spectrum_presets(self):
        # check presets
        wl = color.wavelengths(1000)
        for spec in ot.presets.spectrum.xyz_observers:

            if spec.is_continuous():
                spec(wl)  # call
            
            # should have descriptions
            self.assertNotEqual(spec.desc, "")
            self.assertNotEqual(spec.long_desc, "")

    def test_spectrum_outside_definition(self):
        # check that spectrum presets have constant 0 beyond their definition
        for spec in [*ot.presets.light_spectrum.standard, *ot.presets.spectrum.xyz_observers]:
            if spec.desc != "E":
                self.assertEqual(spec(10), 0)
                self.assertEqual(spec(1000), 0)

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

    @pytest.mark.os
    @pytest.mark.slow
    def test_image_presets(self) -> None:

        for imgi in ot.presets.image.all_presets:
            RT = ot.Raytracer(outline=[-3, 3, -3, 3, 0, 6], silent=True)
            RSS = ot.RectangularSurface(dim=[6, 6])
            RS = ot.RaySource(RSS, pos=[0, 0, 0], image=imgi)
            RT.add(RS)
            RT.trace(500000)
            img = RT.source_image(256)
            
            otp.r_image_plot(img, "sRGB (Absolute RI)")

        plt.close('all')
    
    @pytest.mark.slow
    def test_geometry_eye_presets(self):

        def base_RT():
            RT = ot.Raytracer(outline=[-30, 30, -30, 30, -50, 200], silent=True)
            RSS = ot.CircularSurface(r=0.5)
            RS = ot.RaySource(RSS, pos=[0, 0, -50], spectrum=ot.presets.light_spectrum.d65)
            RT.add(RS)
            return RT

        # test example geometries without parameters
        for geom_func in ot.presets.geometry.eye_models:
            RT = base_RT()
            geom = geom_func()
            RT.add(geom)
            RT.trace(50000)

        # test parameter of arizona eye
        for A, r_det, P, pos in zip([0, 1, 2, 3, 4], [5, 8, 9, 10], [2, 3, 5, 5.7],
                                    [[0, 0, 0], [-1, 0, 5], [10, 7, -4], [8, 9, 10]]):
            RT = base_RT()
            geom = ot.presets.geometry.arizona_eye(adaptation=A, pupil=P, r_det=r_det, pos=pos)
            RT.add(geom)

            RT.trace(50000)

            self.assertFalse(RT.geometry_error)
            self.assertEqual(RT.detectors[-1].surface.r, r_det)
            self.assertEqual(RT.apertures[-1].surface.ri, P / 2)
            self.assertTrue(np.allclose(RT.lenses[0].pos - pos, 0))

        # test parameter of legrand eye
        for r_det, P, pos in zip([5, 8, 9, 10], [2, 3, 5, 5.7],
                                 [[0, 0, 0], [-1, 0, 5], [10, 7, -4], [8, 9, 10]]):
            RT = base_RT()
            geom = ot.presets.geometry.legrand_eye(pupil=P, r_det=r_det, pos=pos)
            RT.add(geom)
            RT.trace(50000)

            self.assertFalse(RT.geometry_error)
            self.assertEqual(RT.detectors[-1].surface.r, r_det)
            self.assertEqual(RT.apertures[-1].surface.ri, P / 2)
            self.assertTrue(np.allclose(RT.lenses[0].pos - pos - [0, 0, 0.25], 0))


    def test_0geometry_camera_preset(self):
        """check that ideal camera preset gets calculated correctly"""

        RT = ot.Raytracer(outline=[-20, 20, -20, 20, -2000, 2000], silent=True)

        for z_g in [-1000, -52, -3.59]:  # different object distances
            for b in [2, 125, 369]:  # different image distances
                for cam_pos in [[0, 0, 0], [-3, 2, 50]]:  # different camera positions

                    # create a point source
                    r_det = 10
                    div_angle = np.degrees(np.arctan(r_det/(cam_pos[2] - z_g))/2)
                    RS = ot.RaySource(ot.Point(), pos=[cam_pos[0], cam_pos[1], z_g],
                                      div_angle=div_angle, divergence="Isotropic")
                    RT.add(RS)

                    # add camera and aperture at image plane
                    RT.add(ot.presets.geometry.ideal_camera(z_g=z_g, cam_pos=cam_pos, b=b, r_det=r_det, r=r_det))
                    RT.add(ot.Aperture(ot.CircularSurface(r=8), pos=RT.detectors[0].pos))

                    RT.trace(10000)

                    # check that image distance is correct and that the image is a point
                    self.assertAlmostEqual(np.std(RT.rays.p_list[:, -1, :2] - cam_pos[:2]), 0, delta=1e-10)
                    self.assertAlmostEqual(RT.detectors[0].pos[2] - RT.lenses[0].pos[2], b)
                    
                    RT.clear()

        # check assignment of r_det and r
        G = ot.presets.geometry.ideal_camera(z_g=-1000, cam_pos=[0, 0, 5], b=10, r=6.3, r_det=5.4)
        self.assertEqual(G.lenses[0].front.r, 6.3)
        self.assertEqual(G.detectors[0].front.dim[0], 2*5.4)

        # for g = -inf, D of the lens is just 1/b (because of 1/f = 1/b + 1/g)
        G = ot.presets.geometry.ideal_camera(z_g=-np.inf, b=5, cam_pos=[0, 0, 0])
        self.assertAlmostEqual(G.lenses[0].D, 1000/5)

        # check if negative b and g are handled correctly
        self.assertRaises(ValueError, ot.presets.geometry.ideal_camera, z_g=5, b=5, cam_pos=[0, 0, 0])
        self.assertRaises(ValueError, ot.presets.geometry.ideal_camera, z_g=-5, b=-5, cam_pos=[0, 0, 0])

if __name__ == '__main__':
    unittest.main()
