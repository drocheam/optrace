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
        # image/psf shape etc.
        pass

    def test_point_psf(self):
        # test if convolution with a point produces the same image
        pass

    def test_behavior_basic(self):
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

    def test_debug_parameters(self):
        # test behavior of debug, log, log_exp etc.
        pass

if __name__ == '__main__':
    unittest.main()
