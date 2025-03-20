#!/bin/env python3

from pathlib import Path  # path of this file
import os  # environment variables
import re

import unittest  # testing framework
import subprocess  # running processes
import pytest # testing framework


# Test examples and benchmarking file

class ExampleTests(unittest.TestCase):

    def _run_file(self, path, timeout=15):

        # load file contents
        with open(path) as f:
            cmd = f.read()

        # NOTE uncomment the next string replacement commands for manual, human-viewer review

        # remove blocking plot function
        cmd = cmd.replace("otp.block()", "")

        # make standard GUI runs exit after loading
        cmd = re.sub(r"((.+)\.run\()", r"\2._exit=True;\2.run(", cmd)

        # replace __file__, as we call python with -c command option
        cmd = cmd.replace("__file__", f"r\"{str(path)}\"")

        # inject exit of GUI for GUI automation example
        cmd = cmd.replace("sim.control(func=automated, args=(sim,))", 
                          "sim.control(func=lambda a: (automated(a), a.close()), args=(sim,))")

        # start process
        env = os.environ | {"PYTHONPATH": str(Path.cwd())}  # needed so example find optrace
        process = subprocess.run(["python", "-c", cmd], env=env, timeout=timeout)
        self.assertEqual(process.returncode, 0)

    @pytest.mark.slow
    def test_image_render_many_rays(self):
        self._run_file(Path.cwd() / "examples" / "image_render_many_rays.py", 90)

    def test_presets_refraction_index(self):
        self._run_file(Path.cwd() / "examples" / "refraction_index_presets.py")
    
    def test_legrand_eye(self):
        self._run_file(Path.cwd() / "examples" / "legrand_eye_model.py")
    
    def test_sphere_projections(self):
        self._run_file(Path.cwd() / "examples" / "sphere_projections.py")
    
    def test_cosine_surfaces(self):
        self._run_file(Path.cwd() / "examples" / "cosine_surfaces.py")
    
    def test_astigmatism(self):
        self._run_file(Path.cwd() / "examples" / "astigmatism.py")
    
    def test_spherical_aberration(self):
        self._run_file(Path.cwd() / "examples" / "spherical_aberration.py")
    
    @pytest.mark.slow
    def test_iol_pinhole_imaging(self):
        self._run_file(Path.cwd() / "examples" / "IOL_pinhole_imaging.py", 25)
    
    def test_convolve(self):
        self._run_file(Path.cwd() / "examples" / "psf_imaging.py")
    
    @pytest.mark.slow
    @pytest.mark.skip(reason="Don't really required as test, test on demand")
    def test_benchmark(self):
        self._run_file(Path.cwd() / "tests" / "benchmark.py", 45)
    
    def test_double_gauss(self):
        self._run_file(Path.cwd() / "examples" / "double_gauss.py")
   
    def test_achromat(self):
        self._run_file(Path.cwd() / "examples" / "achromat.py")
    
    @pytest.mark.slow
    @pytest.mark.os
    def test_microscope(self):
        self._run_file(Path.cwd() / "examples" / "microscope.py", 45)
    
    def test_presets_spectrum(self):
        self._run_file(Path.cwd() / "examples" / "spectrum_presets.py")
    
    def test_image_render(self):
        self._run_file(Path.cwd() / "examples" / "image_render.py")
    
    @pytest.mark.slow
    def test_keratoconus(self):
        self._run_file(Path.cwd() / "examples" / "keratoconus.py", 25)

    def test_prism(self):
        self._run_file(Path.cwd() / "examples" / "prism.py")
    
    @pytest.mark.slow
    def test_gui_automation(self):
        self._run_file(Path.cwd() / "examples" / "gui_automation.py", 35)

    def test_brewster(self):
        self._run_file(Path.cwd() / "examples" / "brewster_polarizer.py")
    
    @pytest.mark.os
    def test_eye_model(self):
        self._run_file(Path.cwd() / "examples" / "arizona_eye_model.py")
    

if __name__ == '__main__':
    unittest.main()

