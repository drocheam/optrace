#!/bin/env python3

import sys
sys.path.append('.')

import unittest
import numpy as np
import warnings
import pytest

from scipy.special import erf

import optrace.tracer.misc as misc
import optrace.tracer.color as color
import optrace as ot



class SpectrumTests(unittest.TestCase):

    def test_spectrum(self):

        wl = color.wavelengths(100)
        sargs = dict(func=lambda x: x**2, lines=ot.presets.spectral_lines.FDC, line_vals=[0.5, 1., 5], 
                     wls=np.array([380, 580, 780]), vals=np.array([1, 0.5, 0]))

        for type_ in ot.Spectrum.spectrum_types:
            spec = ot.Spectrum(type_, **sargs)  # init
            if spec.is_continuous():
                spec(wl)  # call

        # call without parameters works
        ot.Spectrum()

    def test_spectrum_exceptions(self):

        spec = ot.Spectrum("Function")
        self.assertRaises(TypeError, spec, np.array([380., 780.]))  # no function
        spec = ot.Spectrum("Data")
        self.assertRaises(TypeError, spec, np.array([380., 780.]))  # wls or vals not specified
        
        self.assertRaises(ValueError, ot.Spectrum, "Monochromatic", wl=100)  # wavelength outside visible range
        self.assertRaises(ValueError, ot.Spectrum, "Rectangle", wl0=100, wl1=500)  # wavelength outside visible range
        self.assertRaises(ValueError, ot.Spectrum, "Rectangle", wl0=400, wl1=900)  # wavelength outside visible range
        self.assertRaises(ValueError, ot.Spectrum, "Blackbody", T=0)  # temperature <= 0
        self.assertRaises(ValueError, ot.Spectrum, "Gaussian", mu=100)  # wavelength outside visible range
        self.assertRaises(ValueError, ot.Spectrum, "Gaussian", sig=0)  # sigma <= 0
        self.assertRaises(ValueError, ot.Spectrum, "Gaussian", val=0)  # fact <= 0
        self.assertRaises(ValueError, ot.Spectrum, "Constant", val=-1)  # val < 0
        self.assertRaises(ValueError, ot.Spectrum, "Lines", lines=[400, 500], line_vals=[1, -1])  
        self.assertRaises(ValueError, ot.Spectrum, "Lines", lines=[400, np.inf], line_vals=[1, 1])
        # non finite values
        self.assertRaises(ValueError, ot.Spectrum, "Lines", lines=[400, 500], line_vals=[1, np.inf])  
        # non finite values
        self.assertRaises(ValueError, ot.Spectrum, "Lines", lines=[400, 500], line_vals=[1, -1])  
        # line weights below zero
        self.assertRaises(ValueError, ot.Spectrum, "Lines", lines=[100, 500], line_vals=[1, 2])  
        # wavelength outside visible range
        self.assertRaises(ValueError, ot.Spectrum, "AAAA")  # invalid type
        self.assertRaises(ValueError, ot.Spectrum, "Data", wls=[400, 500], vals=[5, -1])  # vals below 0
        self.assertRaises(ValueError, ot.Spectrum, "Data", wls=[300, 500], vals=[5, 5])  # wls below color.WL_MIN
        self.assertRaises(ValueError, ot.Spectrum, "Data", wls=[400, 800], vals=[5, 5])  # wls above color.WL_MAX
        self.assertRaises(ValueError, ot.Spectrum, "Data", wls=[400, np.nan], vals=[5, 5])  # invalid wls
        self.assertRaises(ValueError, ot.Spectrum, "Data", wls=[400, 500], vals=[5, np.nan])  # invalid vals
        self.assertRaises(ValueError, ot.Spectrum, "Data", wls=[400, 480, 500], vals=[5, 5, 5])  # wls steps not uniform
        self.assertRaises(ValueError, ot.Spectrum, "Data", wls=[500, 450, 400], vals=[5, 5, 5])  # wls steps negative
        self.assertRaises(ValueError, ot.Spectrum, "Data", wls=[400, 400, 400], vals=[5, 5, 5])  # wls steps zero

        self.assertRaises(TypeError, ot.Spectrum, "Function", func=1)  # invalid function type
        self.assertRaises(ValueError, ot.Spectrum, "Lines", lines=[], line_vals=[1, 2])  # lines empty
        self.assertRaises(ValueError, ot.Spectrum, "Lines", lines=[4, 5], line_vals=[])  # line_vals empty
        self.assertRaises(TypeError, ot.Spectrum, "Lines", lines=5, line_vals=[1, 2])  # invalid lines type
        self.assertRaises(TypeError, ot.Spectrum, "Lines", lines=[400, 500], line_vals=5)  # invalid line_vals type
        self.assertRaises(TypeError, ot.Spectrum, 5)  # invalid type
        self.assertRaises(TypeError, ot.Spectrum, "Constant", quantity=1)  # quantity not str
        self.assertRaises(TypeError, ot.Spectrum, "Constant", unit=1)  # unit not str
        self.assertRaises(TypeError, ot.Spectrum, "Data", wls=1, vals=[1, 2])  # wls not array-like
        self.assertRaises(TypeError, ot.Spectrum, "Data", wls=[400, 500], vals=3)  # vals not array-like

        self.assertRaises(RuntimeError, ot.Spectrum, "Function", func=lambda wl: 1-wl/550)  # func below 0
        self.assertRaises(AttributeError, spec.__setattr__, "aaa", 1)  # _new_lock active
        self.assertRaises(RuntimeError, ot.Spectrum("Monochromatic", wl=500), 500)  # call of discontinuous type
    
    def test_spectrum_curves(self):

        # some wavelengths
        wl = color.wavelengths(10000)

        # Monochromatic
        spec = ot.Spectrum("Monochromatic")
        self.assertFalse(spec.is_continuous())
        self.assertRaises(RuntimeError, spec, wl)
        
        # Lines
        spec = ot.Spectrum("Lines", lines=ot.presets.spectral_lines.rgb)
        self.assertFalse(spec.is_continuous())
        self.assertRaises(RuntimeError, spec, wl)

        # Constant
        val = 0.8123
        spec = ot.Spectrum("Constant", val=val)
        self.assertTrue(np.all(spec(wl) == val))
        
        # Rectangle
        wl0, wl1 = 450.123, 750.8987
        val = 0.8123
        spec = ot.Spectrum("Rectangle", val=val, wl0=wl0, wl1=wl1)
        ins = (wl >= wl0) & (wl <= wl1)
        vals = spec(wl)
        self.assertTrue(np.allclose(vals[ins]-val, 0))
        self.assertTrue(np.allclose(vals[~ins], 0))

        # Gauss lambda
        gauss = lambda x, mu, sig, fact: np.exp(-(x-mu)**2/2/sig**2) * fact

        # Gaussian
        mu, sig = 480.734, 120.456
        val = 12.3486
        spec = ot.Spectrum("Gaussian", mu=mu, sig=sig, val=val)
        vals = spec(wl)
        vals0 = gauss(wl, mu, sig, val)
        self.assertTrue(np.allclose(vals- vals0, 0))

        # Function, no func_args
        mu, sig = 480.734, 12.456
        val = 12.3486
        func = lambda x: gauss(x, mu, sig, val)
        spec = ot.Spectrum("Function", func=func)
        vals = spec(wl)
        vals0 = gauss(wl, mu, sig, val)
        self.assertTrue(np.allclose(vals - vals0, 0, atol=1e-4, rtol=0))
        
        # Function, with func_args
        mu, sig = 480.734, 12.456
        val = 12.3486
        spec = ot.Spectrum("Function", func=gauss, func_args=dict(mu=mu, sig=sig, fact=val))
        vals = spec(wl)
        vals0 = gauss(wl, mu, sig, val)
        self.assertTrue(np.allclose(vals - vals0, 0, atol=1e-4, rtol=0))

        # Data, full wavelength range
        mu, sig = 480.734, 12.456
        val = 12.3486
        vals = gauss(wl[::2], mu, sig, val)
        spec = ot.Spectrum("Data", wls=wl[::2], vals=vals)
        vals = spec(wl)
        vals0 = gauss(wl, mu, sig, val)
        self.assertTrue(np.allclose(vals - vals0, 0, atol=1e-3, rtol=0))
        
        # Data, partial wavelength range, values outside should be zero
        mu, sig = 480.734, 12.456
        val = 12.3486
        wl2 = np.linspace(mu-6*sig, mu+6*sig, 1000)
        vals2 = gauss(wl2, mu, sig, val)
        spec = ot.Spectrum("Data", wls=wl2, vals=vals2)
        vals = spec(wl)
        vals0 = gauss(wl, mu, sig, val)
        self.assertTrue(np.allclose(vals - vals0, 0, atol=1e-3, rtol=0))

    def test_transmission_spectrum(self):

        wl = color.wavelengths(100)
        sargs = dict(func=lambda x: 0.5, wls=np.array([380, 580, 780]), vals=np.array([1, 0.5, 0]))

        for type_ in ot.TransmissionSpectrum.spectrum_types:
            spec = ot.TransmissionSpectrum(type_, **sargs)  # init
            spec(wl)
            spec.get_xyz()
            spec.get_color()
           
        # call without parameters works
        ot.TransmissionSpectrum()
        
        self.assertRaises(ValueError, ot.TransmissionSpectrum, "Lines")  # invalid discrete type
        self.assertRaises(ValueError, ot.TransmissionSpectrum, "Monochromatic")  # invalid discrete type
        self.assertRaises(ValueError, ot.TransmissionSpectrum, "Gaussian", val=2)  # fact above 1 
        self.assertRaises(ValueError, ot.TransmissionSpectrum, "Constant", val=2)  # val above 1 
        self.assertRaises(ValueError, ot.TransmissionSpectrum, "Data", wls=[380, 500], vals=[0, 2])  # vals above 1
        self.assertRaises(RuntimeError, ot.TransmissionSpectrum, "Function", func=lambda wl: wl/550)  # func above 1 

    def test_light_spectrum(self):

        wl = color.wavelengths(100)
        sargs = dict(func=lambda x: x**2, lines=ot.presets.spectral_lines.FDC, line_vals=[0.5, 1., 5], 
                     wls=np.array([380, 580, 780]), vals=np.array([1, 0.5, 0]))
        sargs_hist = dict(func=lambda x: x**2, lines=ot.presets.spectral_lines.FDC, line_vals=[0.5, 1., 5], 
                     wls=np.array([380, 580, 780]), vals=np.array([1, 0.5]))

        for type_ in ot.LightSpectrum.spectrum_types:
            spec = ot.LightSpectrum(type_, **(sargs if type_ != "Histogram" else sargs_hist))
            if spec.is_continuous():
                spec(wl)
            spec.get_xyz()
            spec.get_color()
            spec.random_wavelengths(1000)

        # call without parameters works
        ot.LightSpectrum()

        self.assertRaises(ValueError, ot.LightSpectrum, "Constant", val=0)  # val <= 0

        # LightSpectrum has specialized functions compared to Spectrum, 
        # check if they handle missing properties correctly
        self.assertRaises(TypeError, ot.LightSpectrum("Data").get_xyz)  # wls or vals not specified
        self.assertRaises(TypeError, ot.LightSpectrum("Data").random_wavelengths, 100)  # wls or vals not specified
        self.assertRaises(TypeError, ot.LightSpectrum("Lines", line_vals=[1, 1, 1]).get_xyz)  
        # ^-- lines or line_vals not specified
        self.assertRaises(TypeError, ot.LightSpectrum("Lines", lines=[400, 500, 600]).get_xyz)  
        # ^-- lines or line_vals not specified
        self.assertRaises(TypeError, ot.LightSpectrum("Lines", line_vals=[1, 1, 1]).random_wavelengths, 100)  
        # ^-- lines or line_vals not specified
        self.assertRaises(TypeError, ot.LightSpectrum("Lines", lines=[400, 500, 600]).random_wavelengths, 100)  
        # ^-- lines or line_vals not specified
     
    def test_light_spectrum_color(self):

        # values from https://web.archive.org/web/20020606213045/http://www.videoessentials.com/jkp_facts.htm
        T = [1000, 2800, 5600, 7500, 15000]
        xy0 = [(0.653, 0.344), (0.452, 0.409), (0.330, 0.339), (0.300, 0.310), (0.264, 0.267)]

        wl = color.wavelengths(10000)

        for Ti, xy0i in zip(T, xy0):
            # blackbody spectrum
            spec = ot.LightSpectrum("Blackbody", T=Ti)

            # xy from srgb value (get_color())
            xyz = np.array([[[*spec.get_xyz()]]])
            xy1 = color.xyz_to_xyY(xyz)[0, 0, :2]
            self.assertTrue(np.allclose(xy0i - xy1, 0, atol=0.0005, rtol=0))
            
            # xy from xyz (get_xyz())
            rgb = np.array([[[*spec.get_color(rendering_intent="Ignore", clip=False)]]])
            xy2 = color.xyz_to_xyY(color.srgb_to_xyz(rgb))[0, 0, :2]
            self.assertTrue(np.allclose(xy0i - xy2, 0, atol=0.0005, rtol=0))

        # other spectra are tested in the presets test for standard illuminants
        
    def test_light_spectrum_random_wavelengths(self):

        # Monochromatic
        wl0 = 412.589
        wl = ot.LightSpectrum("Monochromatic", wl=wl0).random_wavelengths(1000)
        self.assertTrue(np.all(wl == wl0))  # only one valid value

        # Constant
        wl0, wl1 = color.WL_BOUNDS
        wl = ot.LightSpectrum("Constant").random_wavelengths(100000)
        self.assertAlmostEqual(np.mean(wl), (wl0+wl1)/2, delta=0.001)  # mean correct
        self.assertAlmostEqual(np.var(wl), (wl1-wl0)**2/12, delta=0.005)  # standard deviation correct
        # ^-- Variance for uniform distribution see https://en.wikipedia.org/wiki/Continuous_uniform_distribution

        # Rectangle
        wl0, wl1 = 572.5986, 752.69
        wl = ot.LightSpectrum("Rectangle", wl0=wl0, wl1=wl1).random_wavelengths(1000000)
        self.assertAlmostEqual(np.min(wl), wl0, delta=0.001)  # minimum near edge
        self.assertAlmostEqual(np.max(wl), wl1, delta=0.001)  # maximum near edge
        self.assertAlmostEqual(np.mean(wl), (wl1+wl0)/2, delta=0.001)  # mean correct
        self.assertAlmostEqual(np.var(wl), (wl1-wl0)**2/12, delta=0.005)  # variance correct

        # Lines
        lines = np.array(ot.presets.spectral_lines.F_eC_)
        vals = [0.2, 1., 3]
        spec = ot.LightSpectrum("Lines", lines=lines, line_vals=vals)
        wl = spec.random_wavelengths(100000)
        f32eps = 1000*np.finfo(np.float32).eps  # numerical epsilon for a wavelength of 1000
        self.assertTrue(np.all(np.any(np.abs(wl[:, np.newaxis] - lines) < f32eps, axis=1)))  
        # ^-- check that all wavelengths are one of "lines" with some tolerance which includes float number errors
        mean = np.sum(lines*vals)/np.sum(vals)
        self.assertAlmostEqual(np.mean(wl), mean, delta=0.005)
        # ^-- mean wavelength is correct
        std_val = np.sqrt(np.sum((lines - mean)**2 * vals/np.sum(vals))) # standard deviation
        self.assertAlmostEqual(np.std(wl), std_val, delta=0.005)
        # ^-- standard deviation for a discrete variable

        # gaussian function not truncated (=fully inside visible range)
        mu, sig = 572.568, 35.123
        spec = ot.LightSpectrum("Gaussian", mu=mu, sig=sig)
        wl = spec.random_wavelengths(100000)
        self.assertAlmostEqual(np.mean(wl), mu, delta=0.003)  # correct mean
        self.assertAlmostEqual(np.std(wl), sig, delta=0.003)  # correct standard deviation

        # pdf normal distribution
        def pdf(x, mu, sig): 
            return np.exp(-(x-mu)**2 / 2 / sig**2)/np.sqrt(2*np.pi)

        # cdf normal distribution
        def cdf(x, mu, sig):
            return 0.5*(1 + erf((x-mu)/(sig*np.sqrt(2))))
        
        # mean for truncated normal distribution, see
        # https://en.wikipedia.org/wiki/Truncated_normal_distribution
        def mean(mu, sig, wlm0, wlm1): 
            Z = cdf(wlm1, mu, sig) - cdf(wlm0, mu, sig)
            return mu + ((pdf(wlm0, mu, sig) - pdf(wlm1, mu, sig))/Z)*sig

        # standard deviation for truncated normal distribution, see
        # https://en.wikipedia.org/wiki/Truncated_normal_distribution
        def std(mu, sig, wlm0, wlm1): 
            Z = cdf(wlm1, mu, sig) - cdf(wlm0, mu, sig)
            var = sig**2*( 1 + ((wlm0-mu)/sig*pdf(wlm0, mu, sig) - (wlm1-mu)/sig*pdf(wlm1, mu, sig))/Z\
                  - ((pdf(wlm0, mu, sig) - pdf(wlm1, mu, sig))/Z)**2)
            return np.sqrt(var)
        
        # gaussian function right truncated
        mu, sig = 732.568, 65.123
        wlm0, wlm1 = color.WL_BOUNDS
        spec = ot.LightSpectrum("Gaussian", mu=mu, sig=sig)
        wl = spec.random_wavelengths(100000)
        self.assertAlmostEqual(mean(mu, sig, wlm0, wlm1), np.mean(wl), delta=0.005)
        self.assertAlmostEqual(std(mu, sig, wlm0, wlm1), np.std(wl), delta=0.005)

        # gaussian function left truncated
        mu, sig = 392.968, 25.123
        spec = ot.LightSpectrum("Gaussian", mu=mu, sig=sig)
        wl = spec.random_wavelengths(100000)
        self.assertAlmostEqual(mean(mu, sig, wlm0, wlm1), np.mean(wl), delta=0.005)
        self.assertAlmostEqual(std(mu, sig, wlm0, wlm1), np.std(wl), delta=0.005)

        # gaussian function both sides truncated
        mu, sig = 580.128, 190.123
        spec = ot.LightSpectrum("Gaussian", mu=mu, sig=sig)
        wl = spec.random_wavelengths(100000)
        self.assertAlmostEqual(mean(mu, sig, wlm0, wlm1), np.mean(wl), delta=0.005)
        self.assertAlmostEqual(std(mu, sig, wlm0, wlm1), np.std(wl), delta=0.005)
        
        # Function type
        mu, sig = 392.968, 25.123
        spec = ot.LightSpectrum("Function", func=lambda wl: pdf(wl, mu, sig))
        wl = spec.random_wavelengths(100000)
        self.assertAlmostEqual(mean(mu, sig, wlm0, wlm1), np.mean(wl), delta=0.005)
        self.assertAlmostEqual(std(mu, sig, wlm0, wlm1), np.std(wl), delta=0.005)
        
        # Data type, whole wavelength range
        # gaussian function both sides truncated
        mu, sig = 580.128, 190.123
        wls = color.wavelengths(1000)
        vals = pdf(wls, mu, sig)
        spec = ot.LightSpectrum("Data", wls=wls, vals=vals)
        wl = spec.random_wavelengths(100000)
        self.assertAlmostEqual(mean(mu, sig, wlm0, wlm1), np.mean(wl), delta=0.005)
        self.assertAlmostEqual(std(mu, sig, wlm0, wlm1), np.std(wl), delta=0.005)
        
        # Data type, only part of the wavelength range
        # gaussian function both sides truncated
        mu, sig = 580.128, 190.123
        wlm0, wlm1 = 400, 600
        wls = np.linspace(wlm0, wlm1, 1000)
        vals = pdf(wls, mu, sig)
        spec = ot.LightSpectrum("Data", wls=wls, vals=vals)
        wl = spec.random_wavelengths(100000)
        self.assertAlmostEqual(mean(mu, sig, wlm0, wlm1), np.mean(wl), delta=0.005)
        self.assertAlmostEqual(std(mu, sig, wlm0, wlm1), np.std(wl), delta=0.005)

    def test_light_spectrum_render(self):
        # check render
        wl = np.random.uniform(*ot.color.WL_BOUNDS, 10000)
        w = np.random.uniform(0, 1, 10000)
        P, N, desc = np.sum(w), 100, "ABC"
        spec = ot.LightSpectrum.render(wl, w, desc=desc)
        self.assertTrue(isinstance(spec, ot.LightSpectrum))  # check type
        self.assertAlmostEqual(P, np.sum(spec._vals)*(spec._wls[1]-spec._wls[0]))  
        # ^-- check power, convert from W/nm to W
        self.assertEqual(desc, spec.desc)  # additional properties set

        # special case: only one wavelength -> extend range
        spec = ot.LightSpectrum.render(np.array([500.]), np.array([1.]))
        self.assertFalse(spec._wls[0] == spec._wls[-1])
        
        # special case: minimal wavelength
        spec = ot.LightSpectrum.render(np.array([ot.color.WL_BOUNDS[0]]), np.array([1.]))
        self.assertFalse(spec._wls[0] < ot.color.WL_BOUNDS[0])
        
        # special case: maximal wavelength
        spec = ot.LightSpectrum.render(np.array([ot.color.WL_BOUNDS[1]]), np.array([1.]))
        self.assertFalse(spec._wls[-1] > ot.color.WL_BOUNDS[1])
        
        # special case: empty arrays
        spec = ot.LightSpectrum.render(np.array([]), np.array([]))

        # monochromatic source, position should be at correct wavelength
        for N in [100, 10000, 1000000]:
            wl = np.full(N, ot.presets.spectral_lines.d)
            w = np.ones_like(wl)
            spec = ot.LightSpectrum.render(wl, w)
            ind = spec._vals > 0
            self.assertEqual(np.count_nonzero(ind), 1)  # only one wavelength with power
            ind = np.where(ind)[0]
            wlp = spec._wls[ind][0] + (spec._wls[1] - spec._wls[0])/2  # convert bin edge to bin center
            self.assertAlmostEqual(wlp, ot.presets.spectral_lines.d)
        
        # check if spectrum is rendered correctly
        for N in [3000, 30000, 3000000]:
            wlr = misc.uniform(*color.WL_BOUNDS, N)
            w = color.d65_illuminant(wlr)
            spec = ot.LightSpectrum.render(wlr, w)

            # expected error due to noise and quantization, scales with 1/sqrt(N)
            delta = 0.02 / np.sqrt(N/30000)

            # check values at bin centers
            wls = spec._wls[:-1] + (spec._wls[1]-spec._wls[0])/2  # bin edges to center positions
            m = spec._vals/color.d65_illuminant(wls)  # ratio of rendered and correct spectrum
            self.assertAlmostEqual(np.std(m)/np.mean(m), 0, delta=delta/2) 
            
            # check values at arbitrary positions 
            wl = np.linspace(*color.WL_BOUNDS, 1000)
            m = spec(wl)/color.d65_illuminant(wl)
            wl, m = wl[1:-1], m[1:-1]
            self.assertAlmostEqual(np.std(m)/np.mean(m), 0, delta=delta) 

            # check if values are the same within a bin
            eps = 1e-7
            ds2 = (spec._wls[1] - spec._wls[0]) / 2
            self.assertAlmostEqual(np.max(np.abs(spec(spec._wls[:-1]+ds2) - spec._vals)), 0)  # at bin center
            self.assertAlmostEqual(np.max(np.abs(spec(spec._wls[:-1]+eps) - spec._vals)), 0)  # at bin start
            self.assertAlmostEqual(np.max(np.abs(spec(spec._wls[1:]-eps) - spec._vals)), 0)  # at bin end

        # special case: empty spectrum
        spec = ot.LightSpectrum.render(np.array([]), np.array([]))
        self.assertTrue(len(spec._wls) > 1)
        self.assertTrue(len(spec._vals) > 0)
        self.assertEqual(len(spec._wls), len(spec._vals)+1)
        self.assertTrue(np.all(spec._vals == 0))

if __name__ == '__main__':
    unittest.main()
