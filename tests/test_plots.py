#!/bin/env python3

import os
import unittest
import random
import concurrent
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import traceback
from contextlib import contextmanager  # context managers

import pytest
import numpy as np
from scipy.optimize import OptimizeResult
import matplotlib.pyplot as plt

import optrace as ot
import optrace.plots as otp
import optrace.tracer.color as color
from optrace.tracer import misc as misc



def catch_traceback(func, *args, **kwargs):

    try:
        func(*args, **kwargs)
    except Exception as e:
        return traceback.format_exc()
    finally:
        plt.close('all')

    return None


@contextmanager
def subprocessed_plot(func, workers = None) -> None:
    """plot in a subprocess. Returns a callable that mocks the provided function"""

    arg_list = []
    kwarg_list = []
    workers = workers or min(3, max(misc.cpu_count() - 2, 8)) 

    def get_args(*args, **kwargs):
        arg_list.append(args)
        kwarg_list.append(kwargs)

    try:
        yield get_args
    finally:
        
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as executor:
            future_to_res = [executor.submit(catch_traceback, func, *args, **kwargs)\
                             for args, kwargs in zip(arg_list, kwarg_list)]
            
            for future in concurrent.futures.as_completed(future_to_res):
                if (res := future.result()) is not None:
                    raise RuntimeError(res)


# Things that need to be checked by hand/eye:
# applying of title and labels

class PlotTests(unittest.TestCase):

    # random dark mode per test function
    # unfortunately does not get inherited for plotting in subprocesses
    def setUp(self) -> None:
        ot.global_options.plot_dark_mode = bool(np.random.choice([False, True]))

    def tearDown(self) -> None:
        plt.close('all')

    @staticmethod
    def image_plots(rimg, zero):

        for modei in ot.RenderImage.image_modes:
            img = rimg.get(modei, 200)
        
            # make image empty and check if plots can handle this
            if zero:
                rimg._data *= 0
            
            img.limit = None if random.choice([True, False]) else 5
            log = random.choice([True, False])
            flip = random.choice([True, False])
            title_dict = {} if random.choice([0, 1]) else dict(title="Test title")

            otp.image_plot(img, log=log, flip=flip, **title_dict)
            otp.image_profile_plot(img, log=log, flip=flip, x=0.1, **title_dict)
            otp.image_profile_plot(img, log=log, flip=flip, y=-0.156, **title_dict)
            plt.close('all')

    @pytest.mark.slow
    def test_image_plots(self) -> None:
        
        RT = ot.Raytracer(outline=[-3, 3, -3, 3, 0, 6])
        RSS = ot.presets.image.color_checker([6, 6])
        RS = ot.RaySource(RSS, pos=[0, 0, 0])
        RT.add(RS)
        RT.trace(200000)

        # plots with different projections
        RT.add(ot.Detector(ot.SphericalSurface(r=3, R=-10), pos=[0, 0, 3]))
        with subprocessed_plot(PlotTests.image_plots, workers=6) as check_plots_process:
            for proj in [None, *ot.SphericalSurface.sphere_projection_methods]:
                rimg = RT.detector_image(projection_method=proj)
                for zero in [False, True]:
                    check_plots_process(rimg, zero)

        # check zero image log plot
        RIm = ot.RenderImage(extent=[-1, 1, -1, 1])
        RIm.render()
        otp.image_profile_plot(RIm.get("Irradiance"), x=0, log=True)  # log mode but zero image
        otp.image_plot(RIm.get("Irradiance"), log=True)  # log mode but zero image

    def test_image_plots_exceptions(self) -> None:
        img = ot.presets.image.color_checker([6, 6])
        self.assertRaises(TypeError, otp.image_plot, [5, 5])  # invalid Image
        self.assertRaises(TypeError, otp.image_profile_plot, [5, 5])  # invalid Image
        self.assertRaises(TypeError, otp.image_profile_plot, img, flip=2)  # invalid flip type
        self.assertRaises(TypeError, otp.image_profile_plot, img, title=2)  # invalid title type
        self.assertRaises(TypeError, otp.image_profile_plot, img, log=2)  # invalid log type
        self.assertRaises(TypeError, otp.image_plot, img, flip=2)  # invalid flip type
        self.assertRaises(TypeError, otp.image_plot, img, title=2)  # invalid title type
        self.assertRaises(TypeError, otp.image_plot, img, log=2)  # invalid log type
        self.assertRaises(ValueError, otp.image_profile_plot, img)  # x and y missing

    @staticmethod
    def chromaticity_plots_process(list_, *args, **kwargs) -> None:
        otp.chromaticities_cie_1931(list_, *args, **kwargs)
        otp.chromaticities_cie_1976(list_, *args, **kwargs)

    @pytest.mark.slow
    def test_chromaticity_plots(self) -> None:

        RT = ot.Raytracer(outline=[-3, 3, -3, 3, 0, 6])
        RSS = ot.presets.image.color_checker([6, 6])
        RS = ot.RaySource(RSS, pos=[0, 0, 0])
        RT.add(RS)
        RT.trace(200000)
        img = RT.source_image()
        
        # different parameter combinations
        with subprocessed_plot(PlotTests.chromaticity_plots_process, workers=4) as chromaticity_plot:
            for el in [ot.presets.light_spectrum.d65, ot.presets.light_spectrum.standard, img,\
                       ot.presets.image.cell([1, 1]), []]:
                for normi in otp.chromaticity_norms:
                    title_dict = {} if random.choice([0, 1]) else dict(title="Test title")
                    chromaticity_plot(el, norm=normi, *title_dict)

        # test empty calls
        otp.chromaticities_cie_1931()
        otp.chromaticities_cie_1976()
        
    def test_chromaticity_plots_exceptions(self) -> None:

        chrom31, chrom76 =  otp.chromaticities_cie_1931,  otp.chromaticities_cie_1976
        self.assertRaises(TypeError, chrom31, ot.Point())  # invalid type
        self.assertRaises(TypeError, chrom31, [ot.presets.light_spectrum.d65, ot.Point()])  # invalid type in list
        self.assertRaises(TypeError, chrom76, ot.Point())  # invalid type
        self.assertRaises(TypeError, chrom76, [ot.presets.light_spectrum.d65, ot.Point()])  # invalid type in list
        self.assertRaises(TypeError, chrom31, ot.presets.light_spectrum.d65, title=[])  # invalid title type
        self.assertRaises(ValueError, chrom31, ot.presets.light_spectrum.d65, norm="abc") # invalid norm
    
    def test_spectrum_plots_exceptions(self) -> None:

        self.assertRaises(RuntimeError, otp.spectrum_plot, ot.presets.light_spectrum.FDC)
        # ^-- discrete type can't be plotted
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
        
    @pytest.mark.slow
    def test_spectrum_plots(self):

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

        # test different combinations
        with subprocessed_plot(otp.spectrum_plot) as spectrum_plot:

            for list_ in [ot.presets.light_spectrum.standard, d65, [d65], [], rspec0, [rspec0], rspec1,\
                          [rspec1, d65], [rspec0, rspec1]]:

                for color_ in [None, "r", ["g", "b"]]:

                    lc = 1 if not isinstance(color_, list) else len(color_)
                    ll = 1 if not isinstance(list_, list) else len(list_)

                    if color_ is None or lc == ll:

                        for i in range(3):
                            leg = random.choice([True, False])
                            lab = random.choice([True, False])
                            title_dict = {} if random.choice([0, 1]) else dict(title="Test title")
                            spectrum_plot(list_, labels_off=lab, legend_off=leg, color=color_, **title_dict)

        # check different spectral color map separately, as subprocess does not inherit the setting
        ot.global_options.spectral_colormap = lambda wl: plt.cm.viridis(wl/780)
        otp.refraction_index_plot(ot.presets.refraction_index.misc)
        ot.global_options.spectral_colormap = None

    def test_autofocus_cost_plot_exceptions(self):

        args = (OptimizeResult(), dict())
        self.assertRaises(TypeError, otp.focus_search_cost_plot, [], args[1])  # not a OptimizeResult
        self.assertRaises(TypeError, otp.focus_search_cost_plot, args[0], [])  # incorrect fsdict type, should be dict
        self.assertRaises(TypeError, otp.focus_search_cost_plot, args[0], args[1], title=2)  # invalid title type

    def test_autofocus_cost_plot(self):
    
        # dummy data
        sci, afdict = self.focus_search_dummy_data()

        # check different combinations
        with subprocessed_plot(otp.focus_search_cost_plot) as focus_plot:
            for afdict_overwrite in [dict(), dict(z=None), dict(cost=None)]:
                title_dict = {} if random.choice([0, 1]) else dict(title="Test title")
                focus_plot(sci, afdict | afdict_overwrite, **title_dict)

    def test_abbe_plot_exceptions(self):

        # check type checking
        nl = ot.presets.refraction_index.misc
        self.assertRaises(TypeError, otp.abbe_plot, nl, title=2)
        self.assertRaises(TypeError, otp.abbe_plot, nl, ri=2)
        self.assertRaises(TypeError, otp.abbe_plot, nl, lines=2)
    
    @pytest.mark.os
    def test_abbe_plot(self):

        # test different parameter combinations
        with subprocessed_plot(otp.abbe_plot) as abbe_plot:
            for ri in [ot.presets.refraction_index.misc, [ot.presets.refraction_index.air,\
                       ot.presets.refraction_index.vacuum], [ot.presets.refraction_index.SF10], []]:
                for lines in [None, ot.presets.spectral_lines.rgb]:
                    title_dict = {} if random.choice([0, 1]) else dict(title="Test title")
                    abbe_plot(ri, lines=lines, **title_dict)

    @pytest.mark.slow
    def test_surface_profile_plot(self) -> None:

        L = ot.presets.geometry.arizona_eye().elements[0]
        L.back.desc = "Test Legend"  # so plots shows desc as legend entry for this surface

        # check different paramter combinations
        with subprocessed_plot(otp.surface_profile_plot) as surface_plot:
            for sl in [L.front, [L.front, L.back], []]:
                for pos in [[0, 0, 0], [1, -1, 5]]:
                    L.move_to(pos)
                    for ro in [False, True]:
                        for xb in [[None, None], [None, 1], [1, None], [-1, 2], 
                                   [1, 2], [None, 12], [-10, 12], [15, 18]]:
                            surface_plot(sl, remove_offset=ro, x0=xb[0], xe=xb[1])

    def test_surface_profile_plot_exceptions(self) -> None:

        L = ot.presets.geometry.arizona_eye().elements[0]
        self.assertRaises(TypeError, otp.surface_profile_plot, L.front, title=2)
        self.assertRaises(TypeError, otp.surface_profile_plot, L.front, remove_offset=2)
        self.assertRaises(TypeError, otp.surface_profile_plot, L.front, x0=[])
        self.assertRaises(TypeError, otp.surface_profile_plot, L.front, xe=[])
        self.assertRaises(TypeError, otp.surface_profile_plot, 5)

    def focus_search_dummy_data(self):
        z = np.linspace(-2.5, 501, 200)
        cost = (z-250)**2 / 250**2
        af_dict = dict(z=z, cost=cost)
        af_sci = OptimizeResult()
        af_sci.x = 250
        af_sci.fun = 0

        return af_sci, af_dict

    @pytest.mark.os
    def test_saving_file_types(self) -> None:

        # test different file types (and file type detection)
        for path in ["figure.png", "figure.jpg", "figure.pdf", "figure.svg", "figure.tiff"]:
            assert not os.path.exists(path)
               
            otp.chromaticities_cie_1976([], path=path)
            self.assertTrue(os.path.exists(path))
            os.remove(path)
        
    @pytest.mark.os
    def test_saving_plot_types(self) -> None:
    
        # dummy RenderImage
        RIm = ot.RenderImage(extent=[-1, 1, -1, 1])
        RIm.render()

        # test different functions, they all must include a saving option and correctly handle the sargs parameter
        path = "figure.jpg"
        sargs = dict(pad_inches=10, dpi=120)
        for plot, args, kwargs in zip([otp.chromaticities_cie_1931, otp.chromaticities_cie_1976, otp.spectrum_plot,
                                       otp.refraction_index_plot, otp.abbe_plot, otp.surface_profile_plot,
                                       otp.image_plot, otp.image_profile_plot, otp.focus_search_cost_plot],
                                      [[[]], [[]], [[]], [[]], [[]], [[]], [RIm.get("Irradiance")],
                                        [RIm.get("sRGB (Absolute RI)")], self.focus_search_dummy_data()], 
                                      [{}, {}, {}, {}, {}, {}, {}, dict(x=0), {}]):
            assert not os.path.exists(path)

            kwargs.update(sargs=sargs, path=path)
            plot(*args, **kwargs)

            self.assertTrue(os.path.exists(path))
            os.remove(path)

    def test_saving_exceptions(self) -> None:
    
        # type error for path parameter
        self.assertRaises(TypeError, otp.chromaticities_cie_1976, [], path=2)

        # IOError if file could not be saved
        self.assertRaises(IOError, otp.chromaticities_cie_1976, [], path="./hjkhjkhkhjk/hjkhjkhk/jk")


if __name__ == '__main__':
    unittest.main()

