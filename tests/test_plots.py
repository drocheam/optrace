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

    def tearDown(self) -> None:
        plt.close('all')

    @pytest.mark.slow
    def test_r_image_plots(self) -> None:
        
        RT = ot.Raytracer(outline=[-3, 3, -3, 3, 0, 6], silent=True)
        RSS = ot.Surface("Rectangle", dim=[6, 6])
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
        for proj in ot.Surface.sphere_projection_methods:
            RT.add(ot.Detector(ot.Surface("Sphere", R=-10), pos=[0, 0, 3]))
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
        self.assertRaises(TypeError, otp.r_image_plot, img, mode=2)  # invalid mode type
        self.assertRaises(TypeError, otp.r_image_plot, img, flip=2)  # invalid flip type
        self.assertRaises(TypeError, otp.r_image_plot, img, title=2)  # invalid title type
        self.assertRaises(TypeError, otp.r_image_plot, img, log=2)  # invalid log type
        self.assertRaises(TypeError, otp.r_image_plot, img, block=2)  # invalid block type
        self.assertRaises(RuntimeError, otp.r_image_cut_plot, img)  # x and y missing

        # check zero image log plot
        RIm = ot.RImage(extent=[-1, 1, -1, 1])
        RIm.render()
        otp.r_image_cut_plot(RIm, x=0, log=True, block=self.manual)  # log mode but zero image
        otp.r_image_plot(RIm, log=True, block=self.manual)  # log mode but zero image

    @pytest.mark.slow
    def test_chromacity_plots(self) -> None:

        RT = ot.Raytracer(outline=[-3, 3, -3, 3, 0, 6], silent=True)
        RSS = ot.Surface("Rectangle", dim=[6, 6])
        RS = ot.RaySource(RSS, pos=[0, 0, 0], image=ot.presets.image.color_checker)
        RT.add(RS)
        RT.trace(200000)
        img = RT.source_image(200)
        
        # ChromacityPlots
        for normi in otp.chromacity_norms: 
            otp.chromacities_cie_1931(ot.presets.light_spectrum.d65, norm=normi, block=self.manual)
            otp.chromacities_cie_1931(ot.presets.light_spectrum.standard, norm=normi, block=self.manual)
            otp.chromacities_cie_1976(ot.presets.light_spectrum.d65, norm=normi, block=self.manual)
            otp.chromacities_cie_1976(ot.presets.light_spectrum.standard, norm=normi, block=self.manual)
            
            for RIi in color.SRGB_RENDERING_INTENTS:
                otp.chromacities_cie_1931(img, norm=normi, rendering_intent=RIi, block=self.manual)
                otp.chromacities_cie_1976(img, norm=normi, rendering_intent=RIi, block=self.manual)
        
            plt.close('all')

        # check if title gets applied
        otp.chromacities_cie_1931(ot.presets.light_spectrum.d65, norm=normi, block=self.manual, title="Test title")
        otp.chromacities_cie_1976(ot.presets.light_spectrum.d65, norm=normi, block=self.manual, title="Test title")

        # empty list
        otp.chromacities_cie_1931([], norm=normi, block=self.manual)
        otp.chromacities_cie_1976([], norm=normi, block=self.manual)

        # exception tests
        self.assertRaises(TypeError, otp.chromacities_cie_1931, ot.Point())  # invalid type
        self.assertRaises(TypeError, otp.chromacities_cie_1931, [ot.presets.light_spectrum.d65, 
                                                               ot.Point()])  # invalid type in list
        self.assertRaises(TypeError, otp.chromacities_cie_1976, ot.Point())  # invalid type
        self.assertRaises(TypeError, otp.chromacities_cie_1976, [ot.presets.light_spectrum.d65, 
                                                               ot.Point()])  # invalid type in list

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
        self.assertRaises(TypeError, otp.spectrum_plot, ot.presets.light_spectrum.d65, colors=2)
   
        # empty list
        otp.refraction_index_plot([], block=self.manual)
        otp.spectrum_plot([], block=self.manual)

        # additional coverage tests
        otp.refraction_index_plot(ot.presets.refraction_index.misc, legend_off=True, labels_off=False, block=self.manual)
        otp.refraction_index_plot(ot.presets.refraction_index.misc, legend_off=False, labels_off=True, block=self.manual)

        # RefractionIndexPlots
        otp.refraction_index_plot(ot.presets.refraction_index.misc, block=self.manual)
        otp.refraction_index_plot(ot.presets.refraction_index.SF10, block=self.manual, title="Test title")
        
        # SpectrumPlot
        otp.spectrum_plot(ot.presets.light_spectrum.d50, block=self.manual)
        otp.spectrum_plot(ot.presets.light_spectrum.standard, block=self.manual, title="Test title")

    def test_autofocus_cost_plot(self):

        # check type checking
        args = (OptimizeResult, dict())
        self.assertRaises(TypeError, otp.autofocus_cost_plot, args[0], args[1])
        self.assertRaises(TypeError, otp.autofocus_cost_plot, [], args[1])
        self.assertRaises(TypeError, otp.autofocus_cost_plot, args[0], [])
        self.assertRaises(TypeError, otp.autofocus_cost_plot, args[0], args[1], title=2)
        self.assertRaises(TypeError, otp.autofocus_cost_plot, args[0], args[1], block=2)

        z = np.linspace(-2.5, 501, 200)
        cost = (z-250)**2 / 250**2
        afdict = dict(z=z, cost=cost)

        sci = OptimizeResult()
        sci.x = 250
        sci.fun = 0
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

        # Abbe Plot
        otp.abbe_plot(ot.presets.refraction_index.misc, silent=True, block=self.manual)  # prints message for non-dispersive materials
        otp.abbe_plot(ot.presets.refraction_index.misc, silent=False, block=self.manual)  # prints no message for ...
        otp.abbe_plot(ot.presets.refraction_index.misc, title="Test title", block=self.manual)

        # special case: all elements non-dispersive
        otp.abbe_plot([ot.presets.refraction_index.air, ot.presets.refraction_index.vacuum], block=self.manual)

    def test_surface_profile_plot(self) -> None:

        SPP = otp.surface_profile_plot
        L = ot.presets.geometry.arizona_eye()[0]
        L.back.desc = "Test Legend"  # so plots shows desc as legend entry for this surface

        for pos in [[0, 0, 0], [1, -1, 5]]:
            L.move_to(pos)

            SPP(L.front, block=self.manual)
            SPP([L.front, L.back], block=self.manual)
            SPP([L.front, L.back], remove_offset=True, block=self.manual)
            SPP([L.front, L.back], remove_offset=True, xe=1, block=self.manual)
            SPP([L.front, L.back], remove_offset=True, x0=1, block=self.manual)
            SPP([L.front, L.back], remove_offset=True, x0=-1, xe=2, block=self.manual)
            SPP([L.front, L.back], title="Test Title", remove_offset=True, x0=1, xe=2, block=self.manual)

            # special cases
            SPP([L.front, L.back], remove_offset=True, xe=12, block=self.manual)  # part of the curve outside the surface
            SPP([L.front, L.back], remove_offset=True, x0=-10, xe=12, block=self.manual)  # some outside the curve
            SPP([L.front, L.back], remove_offset=True, x0=15, xe=18, block=self.manual, silent=True)  # all outside the curve
            SPP([L.front, L.back], remove_offset=True, x0=15, xe=18, block=self.manual, silent=False)  # all outside the curve
          
            plt.close('all')

        # empty list
        SPP([], block=self.manual)

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
