#!/bin/env python3

import sys
sys.path.append('./src/')

import unittest
import warnings
import matplotlib.pyplot as plt
import time

import More_Complex_Example
import Image_RGB_Render_Example
import Image_RGB_Example
import Double_Prism_Example

class ExampleTests(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter("ignore")

    def test_Complex(self):
        More_Complex_Example.runExample(testrun=True)

    def test_RGBRender(self):
        Image_RGB_Render_Example.runExample(testrun=True, N=3e6)
        plt.close('all')

    def test_RGB(self):
        Image_RGB_Example.runExample(testrun=True)

    def test_Prism(self):
        Double_Prism_Example.runExample(testrun=True)

    def tearDown(self):
        warnings.simplefilter("default")
        time.sleep(1)


if __name__ == '__main__':
    unittest.main()
