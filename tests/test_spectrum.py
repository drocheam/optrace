#!/bin/env python3

import sys
sys.path.append('.')

import unittest
import numpy as np
import warnings
import pytest

import optrace.tracer.misc as misc
import optrace.tracer.color as color
import optrace as ot



class SpectrumTests(unittest.TestCase):


    def test_spectrum(self):

        wl = color.wavelengths(100)
        sargs = dict(func=lambda x: x**2, lines=ot.presets.spectral_lines.FDC, line_vals=[0.5, 1., 5], 
                     wls=np.array([380, 400, 780]), vals=np.array([1, 0.5, 0]))

        for type_ in ot.Spectrum.spectrum_types:
            spec = ot.Spectrum(type_, **sargs)  # init
            if spec.is_continuous():
                spec(wl)  # call

        # call without parameters works
        ot.Spectrum()

        spec = ot.Spectrum("Function")
        self.assertRaises(RuntimeError, spec, np.array([380., 780.]))  # no function
        spec = ot.Spectrum("Data")
        self.assertRaises(RuntimeError, spec, np.array([380., 780.]))  # wls or vals not specified
        
        self.assertRaises(ValueError, ot.Spectrum, "Monochromatic", wl=100)  # wavelength outside visible range
        self.assertRaises(ValueError, ot.Spectrum, "Rectangle", wl0=100, wl1=500)  # wavelength outside visible range
        self.assertRaises(ValueError, ot.Spectrum, "Rectangle", wl0=400, wl1=900)  # wavelength outside visible range
        self.assertRaises(ValueError, ot.Spectrum, "Blackbody", T=0)  # temperature <= 0
        self.assertRaises(ValueError, ot.Spectrum, "Gaussian", mu=100)  # wavelength outside visible range
        self.assertRaises(ValueError, ot.Spectrum, "Gaussian", sig=0)  # sigma <= 0
        self.assertRaises(ValueError, ot.Spectrum, "Gaussian", fact=0)  # fact <= 0
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
       
        # equal operator
        self.assertTrue(ot.Spectrum("Constant") == ot.Spectrum("Constant"))
        self.assertFalse(ot.Spectrum("Constant", val=2) == ot.Spectrum("Constant", val=1))
        self.assertFalse(ot.Spectrum("Constant", val=2) == 1)  # comparison between different types
        self.assertTrue(ot.Spectrum("Data", wls=[400, 500], vals=[1, 2]) == ot.Spectrum("Data", 
                                                                                        wls=[400, 500], vals=[1, 2]))
        self.assertFalse(ot.Spectrum("Data", wls=[400, 500], vals=[1, 2]) == ot.Spectrum("Data", 
                                                                                         wls=[400, 500], vals=[1, 1]))
        self.assertFalse(ot.Spectrum("Data", wls=[400, 500], vals=[1, 2]) == ot.Spectrum("Data", 
                                                                                         wls=[400, 501], vals=[1, 2]))

    def test_transmission_spectrum(self):

        wl = color.wavelengths(100)
        sargs = dict(func=lambda x: 0.5, wls=np.array([380, 400, 780]), vals=np.array([1, 0.5, 0]))

        for type_ in ot.TransmissionSpectrum.spectrum_types:
            spec = ot.TransmissionSpectrum(type_, **sargs)  # init
            spec(wl)
            spec.get_xyz()
            spec.get_color()
           
        # call without parameters works
        ot.TransmissionSpectrum()
        
        self.assertRaises(ValueError, ot.TransmissionSpectrum, "Lines")  # invalid discrete type
        self.assertRaises(ValueError, ot.TransmissionSpectrum, "Monochromatic")  # invalid discrete type
        self.assertRaises(ValueError, ot.TransmissionSpectrum, "Gaussian", fact=2)  # fact above 1 
        self.assertRaises(ValueError, ot.TransmissionSpectrum, "Constant", val=2)  # val above 1 
        self.assertRaises(ValueError, ot.TransmissionSpectrum, "Data", wls=[380, 500], vals=[0, 2])  # vals above 1
        self.assertRaises(RuntimeError, ot.TransmissionSpectrum, "Function", func=lambda wl: wl/550)  # func above 1 

    def test_light_spectrum(self):

        wl = color.wavelengths(100)
        sargs = dict(func=lambda x: x**2, lines=ot.presets.spectral_lines.FDC, line_vals=[0.5, 1., 5], 
                     wls=np.array([380, 400, 780]), vals=np.array([1, 0.5, 0]))

        for type_ in ot.LightSpectrum.spectrum_types:
            spec = ot.LightSpectrum(type_, **sargs)  # init
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
        self.assertRaises(RuntimeError, ot.LightSpectrum("Data").get_xyz)  # wls or vals not specified
        self.assertRaises(RuntimeError, ot.LightSpectrum("Data").random_wavelengths, 100)  # wls or vals not specified
        self.assertRaises(RuntimeError, ot.LightSpectrum("Lines", line_vals=[1, 1, 1]).get_xyz)  
        # lines or line_vals not specified
        self.assertRaises(RuntimeError, ot.LightSpectrum("Lines", lines=[400, 500, 600]).get_xyz)  
        # lines or line_vals not specified
        self.assertRaises(RuntimeError, ot.LightSpectrum("Lines", line_vals=[1, 1, 1]).random_wavelengths, 100)  
        # lines or line_vals not specified
        self.assertRaises(RuntimeError, ot.LightSpectrum("Lines", lines=[400, 500, 600]).random_wavelengths, 100)  
        # lines or line_vals not specified
      
        # check render
        wl = np.random.uniform(ot.color.WL_MIN, ot.color.WL_MAX, 10000)
        w = np.random.uniform(0, 1, 10000)
        P = np.sum(w)  # overall power
        N = 100
        desc = "ABC"
        spec = ot.LightSpectrum.render(wl, w, N, desc=desc)
        self.assertTrue(isinstance(spec, ot.LightSpectrum))  # check type
        self.assertAlmostEqual(P, np.sum(spec._vals)*(spec._wls[1]-spec._wls[0]))  
        # check power, convert from W/nm to W
        self.assertEqual(N, spec._vals.shape[0])  # check array length
        self.assertEqual(desc, spec.desc)  # additionaly properties set

        # special case: only one wavelength -> extend range
        spec = ot.LightSpectrum.render(np.array([500.]), np.array([1.]))
        self.assertFalse(spec._wls[0] == spec._wls[-1])
        
        # special case: minimal wavelength
        spec = ot.LightSpectrum.render(np.array([ot.color.WL_MIN]), np.array([1.]))
        self.assertFalse(spec._wls[0] < ot.color.WL_MIN)
        
        # special case: maximal wavelength
        spec = ot.LightSpectrum.render(np.array([ot.color.WL_MAX]), np.array([1.]))
        self.assertFalse(spec._wls[-1] > ot.color.WL_MAX)
        
        # special case: empty arrays
        spec = ot.LightSpectrum.render(np.array([]), np.array([]))

if __name__ == '__main__':
    unittest.main()
