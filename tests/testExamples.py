#!/bin/env python3

import sys
sys.path.append(['./optrace/', './'])

import unittest
import subprocess

class ExampleTests(unittest.TestCase):

    # execute, but kill after timeout, since everything should be automated
    # higher timeout for a human viewer to see if everythings working
    def execute(self, str_, timeout=20):

        process = subprocess.Popen(str_)
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()

    def test_0Complex(self):
        self.execute(["python", "./examples/More_Complex_Example.py"])

    def test_5RGBRender(self):
        self.execute(["python", "./examples/Image_RGB_Render.py"], 120)

    def test_4PresetsRefractionIndex(self):
        self.execute(["python", "./examples/RefractionIndex_Presets.py"])
    
    def test_3PresetsSpectrum(self):
        self.execute(["python", "./examples/Spectrum_Presets.py"])
    
    def test_2RGB(self):
        self.execute(["python", "./examples/Image_RGB.py"])

    def test_1Prism(self):
        self.execute(["python", "./examples/Double_Prism.py"])

if __name__ == '__main__':
    unittest.main()

