#!/bin/env python3

import sys
sys.path.append('./src/')

import unittest
import os

class ExampleTests(unittest.TestCase):

    def test_0Complex(self):
        os.system("python ./src/More_Complex_Example.py")

    def test_3RGBRender(self):
        os.system("python ./src/Image_RGB_Render_Example.py")

    def test_2RGB(self):
        os.system("python ./src/Image_RGB_Example.py")

    def test_1Prism(self):
        os.system("python ./src/Double_Prism_Example.py")


if __name__ == '__main__':
    unittest.main()
