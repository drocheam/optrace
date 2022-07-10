#!/bin/env python3

import unittest
import subprocess
import warnings

# TODO make paths cross platform
class ExampleTests(unittest.TestCase):

    def setUp(self) -> None:
        # deactivate warnings
        warnings.simplefilter("ignore")
    
    def tearDown(self) -> None:
        # reset warnings
        warnings.simplefilter("default")
    
    # execute, but kill after timeout, since everything should be automated
    # higher timeout for a human viewer to see if everythings working
    def execute(self, str_, timeout=10):

        # start process. ignore warnings and redirect stdout to null
        process = subprocess.Popen(["python", "-W", "ignore", str_], stdout=subprocess.DEVNULL)
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()

    def test_0Complex(self):
        self.execute("./examples/More_Complex_Example.py")

    def test_5RGBRender(self):
        self.execute("./examples/Image_RGB_Render.py", 90)

    def test_4PresetsRefractionIndex(self):
        self.execute("./examples/RefractionIndex_Presets.py")
    
    def test_3PresetsSpectrum(self):
        self.execute("./examples/Spectrum_Presets.py")
    
    def test_2RGB(self):
        self.execute("./examples/Image_RGB.py")

    def test_1Prism(self):
        self.execute("./examples/Double_Prism.py")

if __name__ == '__main__':
    # deactivate warnings temporarily
    warnings.simplefilter("ignore")
    unittest.main()
    warnings.simplefilter("default")

