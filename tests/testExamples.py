#!/bin/env python3

import sys
sys.path.append('./src/')

import unittest
import subprocess

class ExampleTests(unittest.TestCase):

    def execute(self, str_, timeout=10):

        process = subprocess.Popen(str_)
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()

    def test_0Complex(self):

        self.execute(["python", "./src/More_Complex_Example.py"])

    def test_3RGBRender(self):
        self.execute(["python", "./src/Image_RGB_Render_Example.py"], 90)

    def test_2RGB(self):
        self.execute(["python", "./src/Image_RGB_Example.py"])

    def test_1Prism(self):
        self.execute(["python", "./src/Double_Prism_Example.py"])


if __name__ == '__main__':
    unittest.main()
