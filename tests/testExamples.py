#!/bin/env python3

import sys
sys.path.append(['./optrace/', './'])

import unittest
import subprocess

class ExampleTests(unittest.TestCase):

    def execute(self, str_, timeout=15):

        process = subprocess.Popen(str_)
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()

    def test_0Complex(self):
        self.execute(["python", "./examples/More_Complex_Example.py"])

    def test_3RGBRender(self):
        self.execute(["python", "./examples/Image_RGB_Render.py"], 90)

    def test_2RGB(self):
        self.execute(["python", "./examples/Image_RGB.py"])

    def test_1Prism(self):
        self.execute(["python", "./examples/Double_Prism.py"])


if __name__ == '__main__':
    unittest.main()
