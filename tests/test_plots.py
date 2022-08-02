#!/bin/env python3

import sys
sys.path.append('.')

import unittest

import optrace as ot
import optrace.plots as otp
import optrace.tracer.color as Color

import matplotlib.pyplot as plt


class PlotTests(unittest.TestCase):

    def tearDown(self) -> None:
        plt.close('all')

    def test_r_image_plots(self) -> None:
        
        RT = ot.Raytracer(outline=[-3, 3, -3, 3, 0, 6], silent=True)
        RSS = ot.Surface("Rectangle", dim=[6, 6])
        RS = ot.RaySource(RSS, pos=[0, 0, 0], image=ot.presets.Image.color_checker)
        RT.add(RS)

        RT.trace(200000)

        def check_plots(img):
            # image plots
            for modei in ot.RImage.display_modes:
                otp.r_image_plot(img, mode=modei, log=False, block=False)
                otp.r_image_plot(img, mode=modei, log=True, block=False)
            otp.r_image_plot(img, mode=ot.RImage.display_modes[0], flip=True, block=False)
            plt.close('all')

            # image cut plots
            for modei in ot.RImage.display_modes:
                otp.r_image_cut_plot(img, mode=modei, log=False, block=False, x=0.8)
                otp.r_image_cut_plot(img, mode=modei, log=True, y=-2.756, block=False)
            otp.r_image_cut_plot(img, mode=ot.RImage.display_modes[0], flip=True, block=False, x=0.5)
            plt.close('all')

        # cartesian plots
        img = RT.source_image(200)
        check_plots(img)

        # polar plots
        RT.add(ot.Detector(ot.Surface("Sphere", rho=-1/10), pos=[0, 0, 3]))
        img = RT.detector_image(200)
        check_plots(img)

        self.assertRaises(TypeError, otp.r_image_plot, [5, 5])  # invalid RImage
        self.assertRaises(TypeError, otp.r_image_cut_plot, [5, 5])  # invalid RImage
        self.assertRaises(ValueError, otp.r_image_plot, img, mode="4564")  # invalid RImage mode
        self.assertRaises(ValueError, otp.r_image_cut_plot, img, mode="4564")  # invalid RImage mode
        self.assertRaises(RuntimeError, otp.r_image_cut_plot, img)  # x and y missing

        # check zero image log plot
        RIm = ot.RImage(extent=[-1, 1, -1, 1])
        RIm.render()
        otp.r_image_cut_plot(RIm, x=0, log=True)  # log mode but zero image
        otp.r_image_plot(RIm, log=True)  # log mode but zero image

    def test_debug_plots(self) -> None:

        # RefractionIndexPlots
        otp.refraction_index_plot(ot.presets.RefractionIndex.misc)
        otp.refraction_index_plot(ot.presets.RefractionIndex.SF10)
        
        # Abbe Plot
        otp.abbe_plot(ot.presets.RefractionIndex.glasses, silent=True)

        # SpectrumPlot
        otp.spectrum_plot(ot.presets.light_spectrum.D50)
        otp.spectrum_plot(ot.presets.light_spectrum.standard)

        plt.close('all')
        
        RT = ot.Raytracer(outline=[-3, 3, -3, 3, 0, 6], silent=True)
        RSS = ot.Surface("Rectangle", dim=[6, 6])
        RS = ot.RaySource(RSS, pos=[0, 0, 0], image=ot.presets.Image.color_checker)
        RT.add(RS)
        RT.trace(200000)
        img = RT.source_image(200)
        
        # ChromacityPlots
        for normi in otp.chromacity_norms: 
            otp.chromacities_cie_1931(ot.presets.light_spectrum.D65, norm=normi)
            otp.chromacities_cie_1931(ot.presets.light_spectrum.standard, norm=normi)
            otp.chromacities_cie_1976(ot.presets.light_spectrum.D65, norm=normi)
            otp.chromacities_cie_1976(ot.presets.light_spectrum.standard, norm=normi)
            
            for RIi in Color.SRGB_RENDERING_INTENTS:
                otp.chromacities_cie_1931(img, norm=normi, rendering_intent=RIi)
                otp.chromacities_cie_1976(img, norm=normi, rendering_intent=RIi)
        
            plt.close('all')

        # AutoFocusDebug is tested elsewhere

    def test_image_presets(self) -> None:

        for imgi in ot.presets.Image.all_presets:
            RT = ot.Raytracer(outline=[-3, 3, -3, 3, 0, 6], silent=True)
            RSS = ot.Surface("Rectangle", dim=[6, 6])
            RS = ot.RaySource(RSS, pos=[0, 0, 0], image=imgi)
            RT.add(RS)
            RT.trace(200000)
            img = RT.source_image(200)
            
            otp.r_image_plot(img)

    def test_surface_profile_plot(self) -> None:

        SPP = otp.surface_profile_plot
        geom = ot.presets.Geometry.ArizonaEye()

        L = geom[0]  # cornea lens
        L.BackSurface.desc = "ABC"  # so plots shows desc as legend entry for this surface

        SPP(L.FrontSurface)
        SPP([L.FrontSurface, L.BackSurface])
        SPP([L.FrontSurface, L.BackSurface], remove_offset=True)
        SPP([L.FrontSurface, L.BackSurface], remove_offset=True, re=1)
        SPP([L.FrontSurface, L.BackSurface], remove_offset=True, r0=1)
        SPP([L.FrontSurface, L.BackSurface], remove_offset=True, r0=1, re=2)
        SPP([L.FrontSurface, L.BackSurface], title="Test", remove_offset=True, r0=1, re=2)


if __name__ == '__main__':
    unittest.main()
