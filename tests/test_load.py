#!/bin/env python3

import sys
sys.path.append('.')
import pathlib
import os

import pytest
import requests
import unittest

import optrace as ot


class LoadTests(unittest.TestCase):
    
    pytestmark = pytest.mark.random_order(disabled=True)


    def save_file(self, source, path):

        try:
            r = requests.get(source)

            # save to file
            with open(path,'wb') as f:
                f.write(r.content)

            return True

        except (ConnectionResetError, TimeoutError):
            return False

    # normally we would download the file in __init__ and delete in __del__
    # but using unittest multiple objects of this class are created
    
    # run first
    @pytest.mark.os
    def test_0_download_schott(self):
        self.save_file("https://raw.githubusercontent.com/nzhagen/zemaxglass/master/AGF_files/schott.agf", "schott.agf")

    # run last
    @pytest.mark.os
    def test_zzzzzzzzzz_delete_schott(self):
        os.remove("schott.agf")

    @pytest.mark.os
    def test_readlines(self):

        self.assertRaises(FileNotFoundError, ot.load._read_lines, "jiohunzuiznuiIz")  # no such file
        self.assertRaises(FileNotFoundError, ot.load._read_lines, ".")  # not a file

        # test different file encodings
        fnames = ["utf8.txt", "utf16.txt", "utf32.txt", "latin1.txt", "utf16le.txt", "utf16be.txt",
                  "utf32be.txt", "utf32le.txt", "windows1252.txt"]
        path0 = pathlib.Path(__file__).resolve().parent / "test_files"

        for i, fn in enumerate(fnames):
            path = str(path0 / fn)
            lines = ot.load._read_lines(path)
            if not i:
                lines0 = lines

            # all files have the same content, loaded lines should be the same for all
            self.assertEqual(lines, lines0)

    def test_agf_coverage(self):
        """load a file with materials with invalid formula mode number"""
        path = pathlib.Path(__file__).resolve().parent / "test_files" / "error.agf"

        for sil in [False, True]:  # muted and unmuted mode
            ns = ot.load.agf(str(path), silent=sil)
            self.assertEqual(len(ns), 1)  # one valid, two invalid

    @pytest.mark.os
    def test_agf_valid_files(self):

        # not sure about the copyright of these files, that's while download it only for testing
        
        url0 = "https://raw.githubusercontent.com/nzhagen/zemaxglass/master/AGF_files/"
        webfiles = ["archer.agf", "arton.agf", "birefringent.agf", "cdgm.agf", "corning.agf", "heraeus.agf",
                    "hikari.agf", "hoya.agf", "infrared.agf", "isuzu.agf", "liebetraut.agf", "lightpath.agf",
                    "lzos.agf", "misc.agf", "nikon.agf", "ohara.agf", "rad_hard.agf", "rpo.agf", "schott.agf",
                    "sumita.agf", "topas.agf", "umicore.agf", "zeon.agf"]
      
        webfiles = [url0+a for a in webfiles]
        webfiles += ["https://lweb.cfa.harvard.edu/~aas/pisco/model330/zemax_files/MY_I_LINE.AGF",
                     "https://lweb.cfa.harvard.edu/~aas/pisco/model330/zemax_files/SAFE.AGF",
                     "https://lweb.cfa.harvard.edu/~aas/pisco/model330/zemax_files/SCHOTT.AGF",
                     "https://usn.figshare.com/ndownloader/files/9478777"]

        for file in webfiles:

            # load web ressource
            print(file)
            temp_file = "temp.agf"
            success = self.save_file(file, temp_file)

            if not success:
                continue
  
            # count lines with NM, which is the number of different media
            lin = ot.load._read_lines(temp_file)
            lins = [li for li in lin if li.startswith("NM")]
            ncount = len(lins)

            # process file
            for sil in [False, True]:  # unmuted and muted mode
                ns = ot.load.agf(temp_file, silent=sil)
                self.assertTrue(len(ns)/ncount > 0.95)  # more than 95% detected
            # sometimes the files have media with the same name or media with invalid n below 1

            # remove file
            os.remove(temp_file)

    @pytest.mark.os
    def test_zmx_errors(self):

        path0 = pathlib.Path(__file__).resolve().parent / "test_files"
        self.assertRaises(RuntimeError, ot.load.zmx, str(path0 / "zmx_invalid_mode.zmx"))
        self.assertRaises(RuntimeError, ot.load.zmx, str(path0 / "zmx_invalid_unit.zmx"))
        self.assertRaises(RuntimeError, ot.load.zmx, str(path0 / "zmx_invalid_surface_type.zmx"))
        self.assertRaises(RuntimeError, ot.load.zmx, str(path0 / "zmx_invalid_material.zmx"))

    @pytest.mark.os
    def test_zmx_endoscope(self):

        RT = ot.Raytracer(outline=[-20, 20, -20, 20, -20, 200], silent=True)

        RS = ot.RaySource(ot.CircularSurface(r=0.05), spectrum=ot.presets.light_spectrum.d65, pos=[0, 0, -10])
        RT.add(RS)

        self.save_file("https://raw.githubusercontent.com/nzhagen/LensLibrary/main/zemax_files/Liang2006d.zmx", "temp.zmx") 
        G = ot.load.zmx("temp.zmx", dict(COC=ot.presets.refraction_index.COC, POLYSTYR=ot.presets.refraction_index.PS,
                                         BK7=ot.presets.refraction_index.BK7))
        RT.add(G)
        RT.trace(10000)
        self.assertFalse(RT.geometry_error)
        
        # test focal length. see
        # https://github.com/nzhagen/LensLibrary/blob/main/lens_properties_list.txt
        self.assertAlmostEqual(RT.tma().efl, 0.37, delta=0.005)
        
        os.remove("temp.zmx")
    
    def test_zmx_portrait_lens(self):

        RT = ot.Raytracer(outline=[-40, 40, -40, 40, -40, 200], silent=True)
        RS = ot.RaySource(ot.CircularSurface(r=7), spectrum=ot.presets.light_spectrum.d65, pos=[0, 0, -10])
        RT.add(RS)

        self.save_file("https://raw.githubusercontent.com/nzhagen/LensLibrary/main/zemax_files/1843519.zmx", "temp.zmx") 
        G = ot.load.zmx("temp.zmx")

        RT.add(G)
        RT.trace(10000)
        self.assertFalse(RT.geometry_error)
        
        # test focal length. see
        # https://github.com/nzhagen/LensLibrary/blob/main/lens_properties_list.txt
        self.assertAlmostEqual(RT.tma().efl, 100, delta=0.02)

        os.remove("temp.zmx")
    
    def test_zmx_camera_lens(self):

        RT = ot.Raytracer(outline=[-40, 40, -40, 40, -40, 200], silent=True)
        RS = ot.RaySource(ot.CircularSurface(r=0.7), spectrum=ot.presets.light_spectrum.d65, pos=[0, 0, -10])
        RT.add(RS)

        self.save_file("https://raw.githubusercontent.com/nzhagen/LensLibrary/main/zemax_files/7558005b.zmx", "temp.zmx") 
        G = ot.load.zmx("temp.zmx")

        RT.add(G)
        RT.trace(10000)
        self.assertFalse(RT.geometry_error)
        
        # test focal length. see
        # https://github.com/nzhagen/LensLibrary/blob/main/lens_properties_list.txt
        self.assertAlmostEqual(RT.tma(wl=587).efl, 4.55, delta=0.005)

        os.remove("temp.zmx")
    
    @pytest.mark.os
    def test_zmx_smith_tessar(self):

        RT = ot.Raytracer(outline=[-40, 40, -40, 40, -40, 200], silent=True)
        RS = ot.RaySource(ot.CircularSurface(r=7), spectrum=ot.presets.light_spectrum.d65, pos=[0, 0, -10])
        RT.add(RS)

        self.save_file("https://raw.githubusercontent.com/nzhagen/LensLibrary/main/zemax_files/Smith1998b.zmx", "temp.zmx") 

        n_schott = ot.load.agf("schott.agf")

        n_dict = dict(LAFN21=n_schott["N-LAF21"]) 
        G = ot.load.zmx("temp.zmx", n_schott | n_dict)

        RT.add(G)
        RT.trace(10000)
        self.assertFalse(RT.geometry_error)

        # test focal length. see
        # https://github.com/nzhagen/LensLibrary/blob/main/lens_properties_list.txt
        self.assertAlmostEqual(RT.tma().efl, 52.0, delta=0.1)

        os.remove("temp.zmx")

    def test_zmx_achromat(self):
        
        RT = ot.Raytracer(outline=[-40, 40, -40, 40, -40, 200], silent=True)
        RS = ot.RaySource(ot.CircularSurface(r=10), spectrum=ot.presets.light_spectrum.d65, pos=[0, 0, -10])
        RT.add(RS)

        self.save_file("https://www.edmundoptics.com/document/download/391148", "temp.zmx") 

        n_schott = ot.load.agf("schott.agf")

        for silent in [False, True]:
            G = ot.load.zmx("temp.zmx", n_schott, silent=silent)

        RT.add(G)
        RT.trace(10000)
        self.assertFalse(RT.geometry_error)

        # check properties, see
        # https://www.edmundoptics.com/p/25mm-dia-x-100mm-fl-vis-nir-coated-achromatic-lens/9763/
        tma = RT.tma(587.6)
        self.assertAlmostEqual(tma.d, 8.5, delta=1e-4)
        self.assertAlmostEqual(tma.bfl, 95.92, delta=0.05)
        self.assertAlmostEqual(tma.efl, 100, delta=0.05)

        os.remove("temp.zmx")

    def test_zmx_microscope_objective(self):

        RT = ot.Raytracer(outline=[-40, 40, -40, 40, -40, 200], silent=True)
        RS = ot.RaySource(ot.CircularSurface(r=0.8), spectrum=ot.presets.light_spectrum.d65, pos=[0, 0, -10])
        RT.add(RS)

        self.save_file("https://raw.githubusercontent.com/nzhagen/LensLibrary/main/zemax_files/4037934a.zmx", "temp.zmx") 

        n_schott = ot.load.agf("schott.agf")

        G = ot.load.zmx("temp.zmx", n_schott, no_marker=True)  # coverage test with no_marker

        RT.add(G)
        RT.trace(10000)
        self.assertFalse(RT.geometry_error)

        # test focal length. see
        # https://github.com/nzhagen/LensLibrary/blob/main/lens_properties_list.txt
        self.assertAlmostEqual(RT.tma().efl, 1, delta=0.002)

        os.remove("temp.zmx")
    
    @pytest.mark.os
    def test_zmx_plan_concave(self):
        """zmx of a single lens, so only two surfaces"""

        RT = ot.Raytracer(outline=[-40, 40, -40, 40, -40, 200], silent=True)
        RS = ot.RaySource(ot.CircularSurface(r=1.2), spectrum=ot.presets.light_spectrum.d65, pos=[0, 0, -10])
        RT.add(RS)

        self.save_file("https://www.edmundoptics.de/document/download/389048", "temp.zmx") 

        n_schott = ot.load.agf("schott.agf")

        G = ot.load.zmx("temp.zmx", n_schott, no_marker=True)  # coverage test with no_marker

        RT.add(G)
        RT.trace(10000)
        self.assertFalse(RT.geometry_error)

        # test focal length. see
        # https://www.edmundoptics.de/f/uncoated-plano-poncave-pcv-lenses/12263/
        tma = RT.tma(587.6)
        self.assertAlmostEqual(tma.efl, -6.00, delta=0.02)
        self.assertAlmostEqual(tma.bfl, -6.56, delta=0.02)
        self.assertAlmostEqual(tma.d, 1, delta=0.02)

        os.remove("temp.zmx")

    @pytest.mark.os
    def test_zmx_minimal(self):
        # minimal example from https://documents.pub/document/zemaxmanual.html?page=461
        # there should be no surfaces and lenses created, only a marker
        # which gets ignored with no_marker=True

        path = pathlib.Path(__file__).resolve().parent / "test_files" / "minimal.zmx"
        G = ot.load.zmx(str(path), no_marker=True)  # coverage test with no_marker

        # check if empty
        self.assertEqual(len(G.elements), 0)

if __name__ == '__main__':
    unittest.main()
