#!/bin/env python3

from pathlib import Path

# add PYTHONPATH to env, so the examples can find optrace
import os

import unittest
import subprocess
import pytest  


if "TOX_ENV_DIR" not in os.environ:
    os.environ["PYTHONPATH"] = "."

class ExampleTests(unittest.TestCase):

    # execute, but kill after timeout, since everything should be automated
    # higher timeout for a human viewer to see if everything is working
    def execute(self, name, timeout=15):

        path = str(Path.cwd() / "examples" / name)

        # start process. ignore warnings and redirect stdout to null
        process = subprocess.Popen(["python", "-W", "ignore", path], stdout=subprocess.DEVNULL, env=os.environ)
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()

        # examples return exit code 0 or nothing (= None) since they are killed
        self.assertTrue(process.returncode in [None, 0])

    @pytest.mark.slow
    def test_rgb_render(self):
        self.execute("image_rgb_render.py", 70)

    @pytest.mark.slow
    def test_presets_refraction_index(self):
        self.execute("refraction_index_presets.py")
    
    @pytest.mark.slow
    def test_custom_surfaces(self):
        self.execute("custom_surfaces.py")
    
    @pytest.mark.slow
    def test_legrand_eye(self):
        self.execute("legrand_eye_model.py")
    
    @pytest.mark.slow
    def test_sphere_projections(self):
        self.execute("sphere_projections.py")
    
    @pytest.mark.slow
    def test_astigmatism(self):
        self.execute("astigmatism.py")
    
    @pytest.mark.slow
    def test_quickstart(self):
        self.execute("spherical_aberration.py")
    
    @pytest.mark.slow
    def test_achromat(self):
        self.execute("achromat.py")
    
    @pytest.mark.slow
    def test_microscope(self):
        self.execute("microscope.py")
    
    @pytest.mark.slow
    def test_presets_spectrum(self):
        self.execute("spectrum_presets.py")
    
    @pytest.mark.slow
    def test_rgb(self):
        self.execute("image_rgb.py")

    @pytest.mark.slow
    def test_prism(self):
        self.execute("double_prism.py")

    @pytest.mark.slow
    def test_brewster(self):
        self.execute("brewster_polarizer.py")
    
    @pytest.mark.slow
    def test_eye_model(self):
        self.execute("arizona_eye_model.py")
    

if __name__ == '__main__':
    unittest.main()

