#!/bin/env python3

from pathlib import Path  # path of this file
import os  # environment variables
import re

import unittest  # testing framework
import subprocess  # running processes
import pytest # testing framework


class ExampleTests(unittest.TestCase):

    def _run_file(self, name, timeout=15):
        # _run_file, but kill after timeout, if something is hanging
        # example file content needs to be replaced to allow for a simple exit, originally the execution continues

        # load file contents
        path = str(Path.cwd() / "examples" / name)
        with open(path) as f:
            cmd = f.read()

        # NOTE uncomment the next string replacement commands for manual, human-viewer review

        # remove blocking plot function
        cmd = cmd.replace("otp.block()", "")

        # make standard GUI runs exit after loading
        cmd = re.sub(r"((.+)\.run\()", r"\2._exit=True;\2.run(", cmd)

        # replace __file__, as we call python with -c command option
        cmd = cmd.replace("__file__", f"r\"{path}\"")

        # inject exit of GUI for GUI automation example
        cmd = cmd.replace("sim.control(func=automated, args=(sim,))", 
                            "sim.control(func=lambda a: (automated(a), a.close()), args=(sim,))")

        # start process
        env = os.environ | {"PYTHONPATH": str(Path.cwd())}  # needed so example find optrace
        process = subprocess.run(["python", "-c", cmd], env=env, timeout=timeout)
        self.assertEqual(process.returncode, 0)

    @pytest.mark.slow
    def test_image_render_many_rays(self):
        self._run_file("image_render_many_rays.py", 75)

    def test_presets_refraction_index(self):
        self._run_file("refraction_index_presets.py")
    
    def test_legrand_eye(self):
        self._run_file("legrand_eye_model.py")
    
    def test_sphere_projections(self):
        self._run_file("sphere_projections.py")
    
    def test_cosine_surfaces(self):
        self._run_file("cosine_surfaces.py")
    
    def test_astigmatism(self):
        self._run_file("astigmatism.py")
    
    def test_spherical_aberration(self):
        self._run_file("spherical_aberration.py")
    
    @pytest.mark.slow
    def test_iol_pinhole_imaging(self):
        self._run_file("IOL_pinhole_imaging.py", 25)
    
    def test_convolve(self):
        self._run_file("psf_imaging.py")
    
    def test_double_gauss(self):
        self._run_file("double_gauss.py")
   
    def test_achromat(self):
        self._run_file("achromat.py")
    
    @pytest.mark.slow
    @pytest.mark.os
    def test_microscope(self):
        self._run_file("microscope.py", 45)
    
    def test_presets_spectrum(self):
        self._run_file("spectrum_presets.py")
    
    def test_image_render(self):
        self._run_file("image_render.py")
    
    @pytest.mark.slow
    def test_keratoconus(self):
        self._run_file("keratoconus.py", 25)

    def test_prism(self):
        self._run_file("prism.py")
    
    @pytest.mark.slow
    def test_gui_automation(self):
        self._run_file("gui_automation.py", 35)

    def test_brewster(self):
        self._run_file("brewster_polarizer.py")
    
    @pytest.mark.os
    def test_eye_model(self):
        self._run_file("arizona_eye_model.py")
    

if __name__ == '__main__':
    unittest.main()

