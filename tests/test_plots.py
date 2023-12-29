#!/bin/env python3

import sys
sys.path.append('.')

import numpy as np
from scipy.optimize import OptimizeResult
import matplotlib.pyplot as plt

import os
import pytest
import unittest
import random

import optrace as ot
import optrace.plots as otp
import optrace.tracer.color as color
from optrace.tracer import misc as misc

import matplotlib.pyplot as plt

# Things that need to be checked by hand
# applying of title and labels
#######
# set PlotTests.manual = True for manual mode, where each window needs to controlled and closed by hand 


class PlotTests(unittest.TestCase):


    def tearDown(self) -> None:
        plt.close('all')

    @pytest.mark.slow
    def test_r_image_plots(self) -> None:
        
        RT = ot.Raytracer(outline=[-3, 3, -3, 3, 0, 6])
        RSS = ot.presets.image.color_checker([6, 6])
        RS = ot.RaySource(RSS, pos=[0, 0, 0])
        RT.add(RS)

        RT.trace(200000)

        def check_plots(img):

            for i in np.arange(2):

                log = random.choice([True, False])
                flip = random.choice([True, False])
                
                # image plots
                for modei in ot.RImage.display_modes:
                    otp.r_image_plot(img, mode=modei, log=log, flip=flip)
                    otp.r_image_cut_plot(img, modei, log=log, flip=flip, x=0.1)
                    otp.r_image_cut_plot(img, modei, log=log, flip=flip, y=-0.156)
                    plt.close('all')

                # make image empty and check if plots can handle this
                img.img *= 0

        # cartesian plots
        img = RT.source_image(200)
        check_plots(img)

        # polar plots with different projections
        RT.add(ot.Detector(ot.SphericalSurface(r=3, R=-10), pos=[0, 0, 3]))
        for proj in ot.SphericalSurface.sphere_projection_methods:
            img = RT.detector_image(200, projection_method=proj)
            check_plots(img)

        # check if user title gets applied
        otp.r_image_plot(img, mode=ot.RImage.display_modes[0], flip=True, title="Test title")
        otp.r_image_cut_plot(img, mode=ot.RImage.display_modes[0], flip=True, x=0.15, title="Test title")

        # check if 'limit' label works
        img._limit = 1
        otp.r_image_plot(img, mode=ot.RImage.display_modes[0], flip=True)
        
        # exception tests
        self.assertRaises(TypeError, otp.r_image_plot, [5, 5])  # invalid RImage
        self.assertRaises(TypeError, otp.r_image_cut_plot, [5, 5])  # invalid RImage
        self.assertRaises(ValueError, otp.r_image_plot, img, mode="4564")  # invalid RImage mode
        self.assertRaises(ValueError, otp.r_image_cut_plot, img, mode="4564")  # invalid RImage mode
        self.assertRaises(TypeError, otp.r_image_cut_plot, img, mode=2)  # invalid mode type
        self.assertRaises(TypeError, otp.r_image_cut_plot, img, flip=2)  # invalid flip type
        self.assertRaises(TypeError, otp.r_image_cut_plot, img, title=2)  # invalid title type
        self.assertRaises(TypeError, otp.r_image_cut_plot, img, log=2)  # invalid log type
        self.assertRaises(TypeError, otp.r_image_plot, img, mode=2)  # invalid mode type
        self.assertRaises(TypeError, otp.r_image_plot, img, flip=2)  # invalid flip type
        self.assertRaises(TypeError, otp.r_image_plot, img, title=2)  # invalid title type
        self.assertRaises(TypeError, otp.r_image_plot, img, log=2)  # invalid log type
        self.assertRaises(ValueError, otp.r_image_cut_plot, img)  # x and y missing

        # check zero image log plot
        RIm = ot.RImage(extent=[-1, 1, -1, 1])
        RIm.render()
        otp.r_image_cut_plot(RIm, x=0, log=True)  # log mode but zero image
        otp.r_image_plot(RIm, log=True)  # log mode but zero image

    @pytest.mark.slow
    def test_chromaticity_plots(self) -> None:

        RT = ot.Raytracer(outline=[-3, 3, -3, 3, 0, 6])
        RSS = ot.presets.image.color_checker([6, 6])
        RS = ot.RaySource(RSS, pos=[0, 0, 0])
        RT.add(RS)
        RT.trace(200000)
        img = RT.source_image(200)
        
        # ChromaticityPlots
        # different paramter combinations
        for el in [ot.presets.light_spectrum.d65, ot.presets.light_spectrum.standard, img, []]:
            for i, cie in enumerate([otp.chromaticities_cie_1931, otp.chromaticities_cie_1976]):
                for normi in otp.chromaticity_norms:
                    title = None if not i else "Test title"  # sometimes set a different title
                    cie(el, norm=normi)
                    if isinstance(el, ot.RImage):
                        for RIi in color.SRGB_RENDERING_INTENTS:
                            args = dict(title=title) if title is not None else {}
                            cie(el, norm=normi, rendering_intent=RIi, **args)
                plt.close("all")


        # test empty calls
        otp.chromaticities_cie_1931()
        otp.chromaticities_cie_1976()
        
        # exception tests
        self.assertRaises(TypeError, otp.chromaticities_cie_1931, ot.Point())  # invalid type
        self.assertRaises(TypeError, otp.chromaticities_cie_1931, [ot.presets.light_spectrum.d65,
                                                                   ot.Point()])  # invalid type in list
        self.assertRaises(TypeError, otp.chromaticities_cie_1976, ot.Point())  # invalid type
        self.assertRaises(TypeError, otp.chromaticities_cie_1976, [ot.presets.light_spectrum.d65,
                                                                   ot.Point()])  # invalid type in list
        self.assertRaises(TypeError, otp.chromaticities_cie_1931, ot.presets.light_spectrum.d65, title=[])  # invalid title type
        self.assertRaises(ValueError, otp.chromaticities_cie_1931, ot.presets.light_spectrum.d65, norm="abc")  # invalid norm
        self.assertRaises(ValueError, otp.chromaticities_cie_1931, ot.presets.light_spectrum.d65, rendering_intent="abc")  # invalid rendering_intent
        
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
        self.assertRaises(TypeError, otp.spectrum_plot, ot.presets.light_spectrum.d65, legend_off=2)
        self.assertRaises(TypeError, otp.spectrum_plot, ot.presets.light_spectrum.d65, labels_off=2)
        self.assertRaises(TypeError, otp.spectrum_plot, ot.presets.light_spectrum.d65, steps=[])
        self.assertRaises(TypeError, otp.spectrum_plot, ot.presets.light_spectrum.d65, color=2)
   
        # RefractionIndexPlots
        otp.refraction_index_plot(ot.presets.refraction_index.misc)
        otp.refraction_index_plot(ot.presets.refraction_index.SF10, title="Test title")
       
        # refraction_index_plot and spectrum_plot both call the underlying _spectrum_plot, without doing much else
        # so it is sufficient to test one of them

        # special case: SpectrumPlot, list of Histogram and normal spectrum
        N = 200000
        w = np.ones(N)
        wl = np.random.uniform(*ot.global_options.wavelength_range, N)
        rspec0 = ot.LightSpectrum.render(np.array([]), np.array([]))
        rspec1 = ot.LightSpectrum.render(wl, w)
        d65 = ot.presets.light_spectrum.d65

        for list_ in [ot.presets.light_spectrum.standard, d65, [d65], [], rspec0, [rspec0], rspec1,\
                      [rspec1, d65], [rspec0, rspec1]]:
            for color_ in [None, "r", ["g", "b"]]:
                lc = 1 if not isinstance(color_, list) else len(color_)
                ll = 1 if not isinstance(list_, list) else len(list_)
                if color_ is None or lc == ll:

                    for i in range(3):
                        leg = random.choice([True, False])
                        lab = random.choice([True, False])
                        cmap_func = lambda wl: plt.cm.viridis(wl/780)
                        ot.global_options.spectral_colormap = None if random.choice([True, False]) else cmap_func

                        title = None if not lab else "abc"  # sometimes set a different title
                        args = dict(labels_off=lab, legend_off=leg, color=color_)
                        args = args if title is None else (args | dict(title=title))
                        otp.spectrum_plot(list_, **args)

            plt.close("all")

        ot.global_options.spectral_colormap = None

    def test_autofocus_cost_plot(self):

        # type checking
        args = (OptimizeResult(), dict())
        self.assertRaises(TypeError, otp.autofocus_cost_plot, [], args[1])  # not a OptimizeResult
        self.assertRaises(TypeError, otp.autofocus_cost_plot, args[0], [])  # incorrect afdict type, should be dict 
        self.assertRaises(TypeError, otp.autofocus_cost_plot, args[0], args[1], title=2)  # invalid title type

        # dummy data
        sci, afdict = self.af_dummy()

        # calls
        otp.autofocus_cost_plot(sci, afdict)
        otp.autofocus_cost_plot(sci, afdict, title="Test title")

        # missing z, cost in afdict, possible with "Position Variance" but without return_cost = True
        otp.autofocus_cost_plot(sci, afdict | dict(z=None))
        otp.autofocus_cost_plot(sci, afdict | dict(cost=None))
        otp.autofocus_cost_plot(sci, afdict | dict(z=None))
        otp.autofocus_cost_plot(sci, afdict | dict(cost=None))

    @pytest.mark.os
    def test_abbe_plot(self):

        # check type checking
        nl = ot.presets.refraction_index.misc
        self.assertRaises(TypeError, otp.abbe_plot, nl, title=2)
        self.assertRaises(TypeError, otp.abbe_plot, nl, ri=2)
        self.assertRaises(TypeError, otp.abbe_plot, nl, lines=2)

        # test different paramter combinations
        for ri in [ot.presets.refraction_index.misc, [ot.presets.refraction_index.air,\
                   ot.presets.refraction_index.vacuum], [ot.presets.refraction_index.SF10], []]:
            for lines in [None, ot.presets.spectral_lines.rgb]:
                for sil in [False, True]:
                    title = None if not sil else "Test title"  # sometimes set a different title
                    args = dict(lines=lines) | (dict(title=title) if title is not None else {})
                    otp.abbe_plot(ri, **args)
            plt.close("all")

    @pytest.mark.slow
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
                        SPP(sl, remove_offset=ro, x0=xb[0], xe=xb[1])
                plt.close("all")

        # check type checking
        self.assertRaises(TypeError, SPP, L.front, title=2)
        self.assertRaises(TypeError, SPP, L.front, remove_offset=2)
        self.assertRaises(TypeError, SPP, L.front, x0=[])
        self.assertRaises(TypeError, SPP, L.front, xe=[])
        self.assertRaises(TypeError, SPP, 5)

    def test_image_plot(self):

        # check different images, extents and titles
        for flip in [True, False]:
            for s in [[3, 2], [1, 3]]:
                for title in ["", "abc"]:
                    otp.image_plot(ot.presets.image.color_checker(s), title=title, flip=flip)

        # check types
        self.assertRaises(TypeError, otp.image_plot, ot.presets.image.color_checker(s), title=5)  # invalid title
        self.assertRaises(TypeError, otp.image_plot, s)  # invalid img

        # coverage
        ########################################

        # plot empty array
        arr = np.zeros((100, 100, 3))
        otp.image_plot(ot.Image(arr, [3, 3]))

    def af_dummy(self):
        # dummy data for autofocus_cost_plot
        z = np.linspace(-2.5, 501, 200)
        cost = (z-250)**2 / 250**2
        af_dict = dict(z=z, cost=cost)
        af_sci = OptimizeResult()
        af_sci.x = 250
        af_sci.fun = 0

        return af_sci, af_dict

    @pytest.mark.os
    def test_saving(self) -> None:

        # test different file types (and file type detection)
        for path in ["figure.png", "figure.jpg", "figure.pdf", "figure.svg", "figure.tiff"]:
            assert not os.path.exists(path)
               
            otp.chromaticities_cie_1976([], path=path)
            self.assertTrue(os.path.exists(path))
            os.remove(path)

        # dummy RImage
        RIm = ot.RImage(extent=[-1, 1, -1, 1])
        RIm.render()

        # test different functions, they all must include a saving option and correctly handle the sargs parameter
        path = "figure.jpg"
        sargs = dict(pad_inches=10, dpi=120)
        for plot, args, kwargs in zip([otp.chromaticities_cie_1931, otp.chromaticities_cie_1976, otp.spectrum_plot, 
                                       otp.refraction_index_plot, otp.abbe_plot, otp.surface_profile_plot,
                                       otp.r_image_plot, otp.r_image_cut_plot, otp.image_plot, otp.autofocus_cost_plot], 
                                      [[[]], [[]], [[]], [[]], [[]], [[]], [RIm], [RIm], [ot.presets.image.color_checker(
                                                                                          [1, 2])], self.af_dummy()], 
                                      [{}, {}, {}, {}, {}, {}, {}, dict(x=0), {}, {}]):
            assert not os.path.exists(path)

            kwargs.update(sargs=sargs, path=path)
            plot(*args, **kwargs)

            self.assertTrue(os.path.exists(path))
            os.remove(path)

        # type error for path parameter
        self.assertRaises(TypeError, otp.chromaticities_cie_1976, [], path=2)

        # IOError if file could not be saved
        self.assertRaises(IOError, otp.chromaticities_cie_1976, [], path="./hjkhjkhkhjk/hjkhjkhk/jk")
       
if __name__ == '__main__':
    unittest.main()
