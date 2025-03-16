#!/bin/env python3

from pathlib import Path  # path of this file
import os  # environment variables
import sys
import re

import unittest  # testing framework
import subprocess  # running processes
import pytest # testing framework


class ExampleTests(unittest.TestCase):

    def __init__(self, *args, **kwargs):

        env = os.environ.copy()

        if "TOX_ENV_DIR" not in os.environ:
            # add PYTHONPATH to env, so the examples can find optrace
            env["PYTHONPATH"] = str(Path.cwd())
        
        self.env = env

        super().__init__(*args, **kwargs)

    def execute(self, name, timeout=15):
        # execute, but kill after timeout, if something is hanging
        # example file content needs to be replaced to allow for a simple exit, originally the execution continues

        path = str(Path.cwd() / "examples" / name)

        # load file contents
        with open(path) as f:
            file = f.read()

        # remove blocking plot function
        file = file.replace("otp.block()", "")

        # make standard GUI runs exit after loading
        file = re.sub(r"((.+)\.run\()", r"\2._exit=True;\2.run(", file)

        # replace __file__, as we call python with -c command option
        file = file.replace("__file__", f"\"{path}\"")

        # inject exit of GUI for GUI automation example
        file = file.replace("sim.control(func=automated, args=(sim,))", 
                            "sim.control(func=lambda a: (automated(a), a.close()), args=(sim,))")

        # start process. ignore warnings and redirect stdout to null
        process = subprocess.Popen(["python", "-W", "ignore", "-c", file], stdout=subprocess.DEVNULL, env=self.env)
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()

        # examples return exit code 0 or nothing (= None) since they are killed
        # in cases of errors the exit code differs
        self.assertTrue(process.returncode in [None, 0])

    @pytest.mark.slow
    def test_image_render_many_rays(self):
        self.execute("image_render_many_rays.py", 75)

    def test_presets_refraction_index(self):
        self.execute("refraction_index_presets.py", 10)
    
    def test_legrand_eye(self):
        self.execute("legrand_eye_model.py", 10)
    
    def test_sphere_projections(self):
        self.execute("sphere_projections.py", 15)
    
    def test_cosine_surfaces(self):
        self.execute("cosine_surfaces.py", 15)
    
    def test_astigmatism(self):
        self.execute("astigmatism.py", 10)
    
    def test_spherical_aberration(self):
        self.execute("spherical_aberration.py", 10)
    
    @pytest.mark.slow
    def test_iol_pinhole_imaging(self):
        self.execute("IOL_pinhole_imaging.py", 15)
    
    def test_convolve(self):
        self.execute("psf_imaging.py", 10)
    
    def test_double_gauss(self):
        self.execute("double_gauss.py", 15)
    
    def test_achromat(self):
        self.execute("achromat.py", 10)
    
    @pytest.mark.slow
    @pytest.mark.os
    def test_microscope(self):
        self.execute("microscope.py", 45)
    
    def test_presets_spectrum(self):
        self.execute("spectrum_presets.py", 10)
    
    def test_image_render(self):
        self.execute("image_render.py", 10)
    
    @pytest.mark.slow
    def test_keratoconus(self):
        self.execute("keratoconus.py", 20)

    def test_prism(self):
        self.execute("prism.py", 10)
    
    @pytest.mark.slow
    def test_gui_automation(self):
        self.execute("gui_automation.py", 30)

    def test_brewster(self):
        self.execute("brewster_polarizer.py", 10)
    
    def test_eye_model(self):
        self.execute("arizona_eye_model.py", 15)
    

if __name__ == '__main__':
    unittest.main()

