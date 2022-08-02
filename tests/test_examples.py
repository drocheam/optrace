#!/bin/env python3

import unittest
import subprocess
import warnings

from pathlib import Path

# TODO ensure installed version for examples is the same as the current development version

# add PYTHONPATH to env
import os
os.environ["PYTHONPATH"] = "."


class ExampleTests(unittest.TestCase):

    def setUp(self) -> None:
        # deactivate warnings
        warnings.simplefilter("ignore")
    
    def tearDown(self) -> None:
        # reset warnings
        warnings.simplefilter("default")
    
    # execute, but kill after timeout, since everything should be automated
    # higher timeout for a human viewer to see if everything is working
    def execute(self, name, timeout=10):

        path = str(Path.cwd() / "examples" / name)

        # start process. ignore warnings and redirect stdout to null
        process = subprocess.Popen(["python", "-W", "ignore", path], stdout=subprocess.DEVNULL, env=os.environ)
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()

    # tests are sorted alphabetically
    # ensure this gets run at the end by using zzzz prefix
    # zzzz, so it gets tested at the end
    def test_zzzz_rgb_render(self):
        self.execute("image_rgb_render.py", 90)

    def test_complex(self):
        self.execute("more_complex_example.py")
    
    def test_presets_refraction_index(self):
        self.execute("refraction_index_presets.py")
    
    def test_presets_spectrum(self):
        self.execute("spectrum_presets.py")
    
    def test_rgb(self):
        self.execute("image_rgb.py")

    def test_prism(self):
        self.execute("double_prism.py")

    def test_eye_model(self):
        self.execute("arizona_eye_model.py")
    
    def test_eye_model_imaging(self):
        self.execute("arizona_eye_model_imaging.py")


if __name__ == '__main__':
    # deactivate warnings temporarily
    warnings.simplefilter("ignore")
    unittest.main()
    warnings.simplefilter("default")

