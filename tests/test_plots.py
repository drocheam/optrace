#!/bin/env python3

import sys
sys.path.append('.')

import numpy as np
from scipy.optimize import OptimizeResult

import pytest
import unittest

import optrace as ot
import optrace.plots as otp
import optrace.tracer.color as color

import matplotlib.pyplot as plt

# Things that need to be checked by hand
# block=True actually blocks
# applying of title and labels
#######
# set PlotTests.manual = True for manual mode, where each window needs to controlled and closed by hand 


class PlotTests(unittest.TestCase):

    manual = False
    # manual = True

    def tearDown(self) -> None:
        plt.close('all')

    @pytest.mark.slow
    def test_r_image_plots(self) -> None:
        
        RT = ot.Raytracer(outline=[-3, 3, -3, 3, 0, 6], silent=True)
        RSS = ot.RectangularSurface(dim=[6, 6])
        RS = ot.RaySource(RSS, pos=[0, 0, 0], image=ot.presets.image.color_checker)
        RT.add(RS)

        RT.trace(200000)

        def check_plots(img):

            for i in np.arange(2):

                # image plots
                for modei in ot.RImage.display_modes:
                    otp.r_image_plot(img, mode=modei, log=False, block=self.manual)
                    otp.r_image_plot(img, mode=modei, log=True, block=self.manual)
                otp.r_image_plot(img, mode=ot.RImage.display_modes[0], flip=True, block=self.manual)
                plt.close('all')

                # image cut plots
                for modei in ot.RImage.display_modes:
                    otp.r_image_cut_plot(img, mode=modei, log=False, block=self.manual, x=0.1)
                    otp.r_image_cut_plot(img, mode=modei, log=True, y=-0.156, block=self.manual)
                otp.r_image_cut_plot(img, mode=ot.RImage.display_modes[0], flip=True, block=self.manual, x=0.3)
                plt.close('all')

                # make image empty and check if plots can handle this
                img.img *= 0

        # cartesian plots
        img = RT.source_image(200)
        check_plots(img)

        # polar plots with different projections
        for proj in ot.SphericalSurface.sphere_projection_methods:
            RT.add(ot.Detector(ot.SphericalSurface(r=3, R=-10), pos=[0, 0, 3]))
            img = RT.detector_image(200, projection_method=proj)
            check_plots(img)

        # check if user title gets applied
        otp.r_image_plot(img, mode=ot.RImage.display_modes[0], flip=True, block=self.manual, title="Test title")
        otp.r_image_cut_plot(img, mode=ot.RImage.display_modes[0], flip=True, block=self.manual, x=0.15, title="Test title")

        # exception tests
        self.assertRaises(TypeError, otp.r_image_plot, [5, 5])  # invalid RImage
        self.assertRaises(TypeError, otp.r_image_cut_plot, [5, 5])  # invalid RImage
        self.assertRaises(ValueError, otp.r_image_plot, img, mode="4564")  # invalid RImage mode
        self.assertRaises(ValueError, otp.r_image_cut_plot, img, mode="4564")  # invalid RImage mode
        self.assertRaises(TypeError, otp.r_image_cut_plot, img, mode=2)  # invalid mode type
        self.assertRaises(TypeError, otp.r_image_cut_plot, img, flip=2)  # invalid flip type
        self.assertRaises(TypeError, otp.r_image_cut_plot, img, title=2)  # invalid title type
        self.assertRaises(TypeError, otp.r_image_cut_plot, img, log=2)  # invalid log type
        self.assertRaises(TypeError, otp.r_image_cut_plot, img, block=2)  # invalid block type
        self.assertRaises(TypeError, otp.r_image_cut_plot, img, imc=2)  # invalid imc type
        self.assertRaises(TypeError, otp.r_image_plot, img, mode=2)  # invalid mode type
        self.assertRaises(TypeError, otp.r_image_plot, img, flip=2)  # invalid flip type
        self.assertRaises(TypeError, otp.r_image_plot, img, title=2)  # invalid title type
        self.assertRaises(TypeError, otp.r_image_plot, img, log=2)  # invalid log type
        self.assertRaises(TypeError, otp.r_image_plot, img, block=2)  # invalid block type
        self.assertRaises(TypeError, otp.r_image_plot, img, imc=2)  # invalid imc type
        self.assertRaises(RuntimeError, otp.r_image_cut_plot, img)  # x and y missing

        # check zero image log plot
        RIm = ot.RImage(extent=[-1, 1, -1, 1])
        RIm.render()
        otp.r_image_cut_plot(RIm, x=0, log=True, block=self.manual)  # log mode but zero image
        otp.r_image_plot(RIm, log=True, block=self.manual)  # log mode but zero image

    @pytest.mark.slow
    def test_chromacity_plots(self) -> None:

        RT = ot.Raytracer(outline=[-3, 3, -3, 3, 0, 6], silent=True)
        RSS = ot.RectangularSurface(dim=[6, 6])
        RS = ot.RaySource(RSS, pos=[0, 0, 0], image=ot.presets.image.color_checker)
        RT.add(RS)
        RT.trace(200000)
        img = RT.source_image(200)
        
        # ChromacityPlots
        # different paramter combinations
        for el in [ot.presets.light_spectrum.d65, ot.presets.light_spectrum.standard, img, []]:
            for i, cie in enumerate([otp.chromacities_cie_1931, otp.chromacities_cie_1976]):
                for normi in otp.chromacity_norms: 
                    title = None if not i else "Test title"  # sometimes set a different title
                    cie(el, norm=normi, block=self.manual)
                    if isinstance(el, ot.RImage):
                        for RIi in color.SRGB_RENDERING_INTENTS:
                            args = dict(title=title) if title is not None else {}
                            cie(el, norm=normi, rendering_intent=RIi, block=self.manual, **args)
                plt.close("all")

        # exception tests
        self.assertRaises(TypeError, otp.chromacities_cie_1931, ot.Point())  # invalid type
        self.assertRaises(TypeError, otp.chromacities_cie_1931, [ot.presets.light_spectrum.d65, 
                                                               ot.Point()])  # invalid type in list
        self.assertRaises(TypeError, otp.chromacities_cie_1976, ot.Point())  # invalid type
        self.assertRaises(TypeError, otp.chromacities_cie_1976, [ot.presets.light_spectrum.d65, 
                                                               ot.Point()])  # invalid type in list

    @pytest.mark.os 
    @pytest.mark.slow
    def test_spectrum_plots(self):

        self.assertRaises(RuntimeError, otp.spectrum_plot, ot.presets.light_spectrum.FDC)  # discrete type can't be plotted
        self.assertRaises(TypeError, otp.refraction_index_plot, ot.Point())
        self.assertRaises(TypeError, otp.spectrum_plot, ot.Point())
        self.assertRaises(TypeError, otp.refraction_index_plot, ot.presets.refraction_index.SF10, title=2)
        self.assertRaises(TypeError, otp.spectrum_plot, ot.presets.light_spectrum.d65, title=2)
        self.assertRaises(TypeError, otp.spectrum_plot, ot.presets.light_spectrum.d65, xlabel=2)
        self.assertRaises(TypeError, otp.spectrum_plot, ot.presets.light_spectrum.d65, ylabel=2)
        self.assertRaises(TypeError, otp.spectrum_plot, ot.presets.light_spectrum.d65, title=2)
        self.assertRaises(TypeError, otp.spectrum_plot, ot.presets.light_spectrum.d65, block=2)
        self.assertRaises(TypeError, otp.spectrum_plot, ot.presets.light_spectrum.d65, legend_off=2)
        self.assertRaises(TypeError, otp.spectrum_plot, ot.presets.light_spectrum.d65, labels_off=2)
        self.assertRaises(TypeError, otp.spectrum_plot, ot.presets.light_spectrum.d65, steps=[])
        self.assertRaises(TypeError, otp.spectrum_plot, ot.presets.light_spectrum.d65, color=2)
   
        # RefractionIndexPlots
        otp.refraction_index_plot(ot.presets.refraction_index.misc, block=self.manual)
        otp.refraction_index_plot(ot.presets.refraction_index.SF10, block=self.manual, title="Test title")
       
        # refraction_index_plot and spectrum_plot both call the underlying _spectrum_plot, without doing much else
        # so it is sufficient to test one of them

        # special case: SpectrumPlot, list of Histogram and normal spectrum
        N = 200000
        w = np.ones(N)
        wl = np.random.uniform(*color.WL_BOUNDS, N)
        rspec0 = ot.LightSpectrum.render(np.array([]), np.array([]))
        rspec1 = ot.LightSpectrum.render(wl, w)
        d65 = ot.presets.light_spectrum.d65

        for list_ in [ot.presets.light_spectrum.standard, d65, [d65], [], rspec0, [rspec0], rspec1,\
                      [rspec1, d65], [rspec0, rspec1]]:
            for color_ in [None, "r", ["g", "b"]]:
                lc = 1 if not isinstance(color_, list) else len(color_)
                ll = 1 if not isinstance(list_, list) else len(list_)
                if color_ is None or lc == ll:
                    for leg in [False, True]:
                        for lab in [False, True]:
                            for steps in [500, 5000]:
                                title = None if not lab else "abc"  # sometimes set a different title
                                args = dict(labels_off=lab, legend_off=leg, block=self.manual, color=color_)
                                args = args if title is None else (args | dict(title=title))
                                otp.spectrum_plot(list_, **args)

                    plt.close("all")

    def test_autofocus_cost_plot(self):

        # type checking
        args = (OptimizeResult(), dict())
        self.assertRaises(TypeError, otp.autofocus_cost_plot, [], args[1])  # not a OptimizeResult
        self.assertRaises(TypeError, otp.autofocus_cost_plot, args[0], [])  # incorrect afdict type, should be dict 
        self.assertRaises(TypeError, otp.autofocus_cost_plot, args[0], args[1], title=2)  # invalid title type
        self.assertRaises(TypeError, otp.autofocus_cost_plot, args[0], args[1], block=2)  # invalid block type

        # dummy data
        z = np.linspace(-2.5, 501, 200)
        cost = (z-250)**2 / 250**2
        afdict = dict(z=z, cost=cost)
        sci = OptimizeResult()
        sci.x = 250
        sci.fun = 0

        # calls
        otp.autofocus_cost_plot(sci, afdict, block=self.manual)
        otp.autofocus_cost_plot(sci, afdict, title="Test title", block=self.manual)

    def test_abbe_plot(self):

        # check type checking
        nl = ot.presets.refraction_index.misc
        self.assertRaises(TypeError, otp.abbe_plot, nl, silent=2)
        self.assertRaises(TypeError, otp.abbe_plot, nl, block=2)
        self.assertRaises(TypeError, otp.abbe_plot, nl, title=2)
        self.assertRaises(TypeError, otp.abbe_plot, nl, ri=2)
        self.assertRaises(TypeError, otp.abbe_plot, nl, lines=2)

        # test different paramter combinations
        for ri in [ot.presets.refraction_index.misc, [ot.presets.refraction_index.air,\
                   ot.presets.refraction_index.vacuum], [ot.presets.refraction_index.SF10], []]:
            for lines in [None, ot.presets.spectral_lines.rgb]:
                for sil in [False, True]:
                    title = None if not sil else "Test title"  # sometimes set a different title
                    args = dict(lines=lines, silent=sil) | (dict(title=title) if title is not None else {})
                    otp.abbe_plot(ri,  **args, block=self.manual)
            plt.close("all")

    def test_surface_profile_plot(self) -> None:

        SPP = otp.surface_profile_plot
        L = ot.presets.geometry.arizona_eye().elements[0]
        L.back.desc = "Test Legend"  # so plots shows desc as legend entry for this surface

        # check different paramter combinations
        for sl in [L.front, [L.front, L.back], []]:
            for pos in [[0, 0, 0], [1, -1, 5]]:
                L.move_to(pos)
                for ro in [False, True]:
                    for xb in [[None, None], [None, 1], [1, None], [-1, 2], [1, 2], [None, 12], [-10, 12], [15, 18]]:
                        SPP(sl, remove_offset=ro, silent=ro, x0=xb[0], xe=xb[1], block=self.manual)
                plt.close("all")

        # check type checking
        self.assertRaises(TypeError, SPP, L.front, title=2)
        self.assertRaises(TypeError, SPP, L.front, block=2)
        self.assertRaises(TypeError, SPP, L.front, remove_offset=2)
        self.assertRaises(TypeError, SPP, L.front, x0=[])
        self.assertRaises(TypeError, SPP, L.front, xe=[])
        self.assertRaises(TypeError, SPP, L.front, silent=2)
        self.assertRaises(TypeError, SPP, 5)


if __name__ == '__main__':
    unittest.main()
