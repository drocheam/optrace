#!/bin/env python3

import sys
sys.path.append('.')

import unittest
import numpy as np
import pytest

import optrace as ot
from optrace.tracer import color



# TODO
class ConvolutionTests(unittest.TestCase):

    def test_exceptions(self):

        img = ot.presets.image.color_checker
        s_img = [5, 6]
        psf = ot.presets.image.ETDRS_chart_inverted
        s_psf = [0.1, 0.3]

        # check that call is correct, also tests threading
        for threading in [True, False]:
            ot.convolve(img, s_img, psf, s_psf, silent=True, threading=threading)

        # check invalid types
        self.assertRaises(TypeError, ot.convolve, 5, s_img, psf, s_psf)  # invalid img
        self.assertRaises(TypeError, ot.convolve, img, 5, psf, s_psf)  # invalid s_img
        self.assertRaises(TypeError, ot.convolve, img, s_img, 5, s_psf)  # invalid psf
        self.assertRaises(TypeError, ot.convolve, img, s_img, psf, 5)  # invalid s_psf
        self.assertRaises(TypeError, ot.convolve, img, s_img, psf, s_psf, k=[])  # invalid k
        self.assertRaises(TypeError, ot.convolve, img, s_img, psf, s_psf, silent=[])  # invalid silent
        self.assertRaises(TypeError, ot.convolve, img, s_img, psf, s_psf, threading=[])  # invalid threading
        

        # resolution tests
        ###########################################

        # low resolution psf
        psf2 = np.ones((5, 5, 3))
        self.assertRaises(ValueError, ot.convolve, img, s_img, psf2, s_psf)
        
        # low resolution image
        img2 = np.ones((5, 5, 3))
        self.assertRaises(ValueError, ot.convolve, img2, s_img, psf, s_psf)
        
        # huge resolution image
        img2 = np.ones((3000, 3000, 3))
        self.assertRaises(ValueError, ot.convolve, img2, s_img, psf, s_psf)
        
        # huge resolution psf
        psf2 = np.ones((3000, 3000, 3))
        self.assertRaises(ValueError, ot.convolve, img, s_img, psf2, s_psf)
        
        # psf larger than image
        s_psf2 = [s_img[0]+5, s_img[1]+5]
        self.assertRaises(ValueError, ot.convolve, img, s_img, psf, s_psf2)
       
        # shape tests
        ###############################################

        # invalid psf shape case 1
        psf2 = np.ones((300, 300, 2))
        self.assertRaises(ValueError, ot.convolve, img, s_img, psf2, s_psf)
        
        # invalid psf shape case 2
        psf2 = np.ones((300, 2))
        self.assertRaises(ValueError, ot.convolve, img, s_img, psf2, s_psf)
        
        # invalid image shape case 1
        img2 = np.ones((300, 300, 2))
        self.assertRaises(ValueError, ot.convolve, img2, s_img, psf, s_psf)
        
        # invalid image shape case 2
        img2 = np.ones((300, 2))
        self.assertRaises(ValueError, ot.convolve, img2, s_img, psf, s_psf)

        # coverage tests
        #############################################

        # low resolution psf warning
        psf2 = np.ones((30, 30, 3))
        for sil in [True, False]:
            ot.convolve(img, s_img, psf2, s_psf, silent=sil)
        
        # low resolution image warning
        img2 = np.ones((30, 30, 3))
        for sil in [True, False]:
            ot.convolve(img2, s_img, psf, s_psf, silent=sil)


    def test_point_psf(self):
        # test if convolution with a point produces the same image
        pass

    def test_behavior_basic(self):
        # test odd and even pixel counts, different pixel ratios
        pass

    def test_tracing_consistency(self):
        # check that tracing and convolution produce similar results
        # check with a dispersive source, so we are sure that colors are handled correctly
        pass

    def test_size_consistency(self):
        # test that different image/psf resolutions produce approximately the same result
        pass

    def test_coverage(self):
        # test silent, threading, inter_p_k parameter
        pass


if __name__ == '__main__':
    unittest.main()
