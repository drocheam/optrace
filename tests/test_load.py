#!/bin/env python3

import pathlib  # handling file paths
import os  # deleting files, getenv

import pytest  # testing framework
import unittest  # testing framework

import optrace as ot
import optrace.tracer.load as load


class LoadTests(unittest.TestCase):
    
    def __init__(self, *args, **kwargs):

        # load schott material file from examples
        schott_path = str(pathlib.Path(__file__).resolve().parent.parent\
                         / "examples" / "resources" / "materials" / "schott.agf")
        self.n_schott = ot.load_agf(schott_path)

        super().__init__(*args, **kwargs)


    @pytest.mark.os
    @pytest.mark.install
    def test_readlines(self):
        """test text loading for different encodings"""

        self.assertRaises(FileNotFoundError, load._read_lines, "jiohunzuiznuiIz")  # no such file
        self.assertRaises(FileNotFoundError, load._read_lines, ".")  # not a file

        # test different file encodings
        fnames = ["utf8.txt", "utf16.txt", "utf32.txt", "latin1.txt", "utf16le.txt", "utf16be.txt",
                  "utf32be.txt", "utf32le.txt", "windows1252.txt"]
        path0 = pathlib.Path(__file__).resolve().parent / "test_files" / "encoding"

        for i, fn in enumerate(fnames):
            path = str(path0 / fn)
            lines = load._read_lines(path)
            if not i:
                lines0 = lines

            # all files have the same content, loaded lines should be the same for all
            self.assertEqual(lines, lines0)


    def test_agf_coverage(self):
        """load a file with materials with invalid formula mode number"""
        path = pathlib.Path(__file__).resolve().parent / "test_files" / "edge_cases_agf" / "error.agf"

        ns = ot.load_agf(str(path))
        self.assertEqual(len(ns), 1)  # one valid, two invalid


    @pytest.mark.os
    @pytest.mark.install
    def test_agf_valid_files(self):
        """load different agf catalogues and check if everything is handled correctly"""

        path0 = pathlib.Path(__file__).resolve().parent / "test_files" / "zemaxglass" / "files"
        files0 = ["archer.agf", "arton.agf", "birefringent.agf", "cdgm.agf", "corning.agf", "heraeus.agf",
                  "hikari.agf", "hoya.agf", "infrared.agf", "isuzu.agf", "liebetraut.agf", "lightpath.agf",
                  "lzos.agf", "misc.agf", "nikon.agf", "ohara.agf", "rad_hard.agf", "rpo.agf", "schott.agf",
                  "sumita.agf", "topas.agf", "umicore.agf", "zeon.agf"]

        path1 = pathlib.Path(__file__).resolve().parent / "test_files" / "PISCO_zemax" / "files"
        files1 = ["MY_I_LINE.AGF", "SAFE.AGF", "SCHOTT.AGF"]
        
        path2 = pathlib.Path(__file__).resolve().parent / "test_files" / "eye_zemax" / "files" / "EYE.AGF"
        
        files = [path0 / file0 for file0 in files0] + [path1 / file1 for file1 in files1] + [path2] 

        for file in files:

            # count lines with NM, which is the number of different media
            lin = load._read_lines(str(file))
            lins = [li for li in lin if li.startswith("NM")]
            ncount = len(lins)

            # process file
            ns = ot.load_agf(str(file))
            self.assertTrue(len(ns)/ncount > 0.95)  # more than 95% detected
            # sometimes the files have media with the same name or media with invalid n below 1

    
    def test_zmx_special(self):
        """test zmx special cases: COAT provided, MATERIAL == _BLANK and MATERIAL unknown, 
        but index and Abbe provided"""
        path0 = pathlib.Path(__file__).resolve().parent / "test_files" / "edge_cases_zmx"
        ot.load_zmx(str(path0 / "zmx_special_cases.zmx"), self.n_schott)

    @pytest.mark.os
    def test_zmx_errors(self):
        """test runtime errors that occur with invalid zmx files"""

        path0 = pathlib.Path(__file__).resolve().parent / "test_files" / "edge_cases_zmx"
        self.assertRaises(RuntimeError, ot.load_zmx, str(path0 / "zmx_invalid_mode.zmx"))
        self.assertRaises(RuntimeError, ot.load_zmx, str(path0 / "zmx_invalid_unit.zmx"))
        self.assertRaises(RuntimeError, ot.load_zmx, str(path0 / "zmx_invalid_surface_type.zmx"))
        self.assertRaises(RuntimeError, ot.load_zmx, str(path0 / "zmx_invalid_material.zmx"))

    @pytest.mark.os
    def test_zmx_endoscope(self):
        """load an endoscope example"""

        RT = ot.Raytracer(outline=[-20, 20, -20, 20, -20, 200])

        RS = ot.RaySource(ot.CircularSurface(r=0.05), spectrum=ot.presets.light_spectrum.d65, pos=[0, 0, -10])
        RT.add(RS)

        zmx = str(pathlib.Path(__file__).resolve().parent / "test_files" / "LensLibrary" / "files" / "Liang2006d.zmx")
        G = ot.load_zmx(zmx, dict(COC=ot.presets.refraction_index.COC, POLYSTYR=ot.presets.refraction_index.PS,
                                         BK7=ot.presets.refraction_index.BK7))
        RT.add(G)
        RT.trace(10000)
        self.assertFalse(RT.geometry_error)
        
        # test focal length. see
        # https://github.com/nzhagen/LensLibrary/blob/main/lens_properties_list.txt
        self.assertAlmostEqual(RT.tma().efl, 0.37, delta=0.005)
        
    
    @pytest.mark.os
    def test_zmx_portrait_lens(self):
        """load an potrait lens example"""

        RT = ot.Raytracer(outline=[-40, 40, -40, 40, -40, 200])
        RS = ot.RaySource(ot.CircularSurface(r=7), spectrum=ot.presets.light_spectrum.d65, pos=[0, 0, -10])
        RT.add(RS)

        zmx = str(pathlib.Path(__file__).resolve().parent / "test_files" / "LensLibrary" / "files" / "1843519.zmx")
        G = ot.load_zmx(zmx)

        RT.add(G)
        RT.trace(10000)
        self.assertFalse(RT.geometry_error)
        
        # test focal length. see
        # https://github.com/nzhagen/LensLibrary/blob/main/lens_properties_list.txt
        self.assertAlmostEqual(RT.tma().efl, 100, delta=0.02)

    
    def test_zmx_camera_lens(self):
        """load a camera lens example"""

        RT = ot.Raytracer(outline=[-40, 40, -40, 40, -40, 200])
        RS = ot.RaySource(ot.CircularSurface(r=0.7), spectrum=ot.presets.light_spectrum.d65, pos=[0, 0, -10])
        RT.add(RS)

        zmx = str(pathlib.Path(__file__).resolve().parent / "test_files" / "LensLibrary" / "files" / "7558005b.zmx")
        G = ot.load_zmx(zmx)

        RT.add(G)
        RT.trace(10000)
        self.assertFalse(RT.geometry_error)
        
        # test focal length. see
        # https://github.com/nzhagen/LensLibrary/blob/main/lens_properties_list.txt
        self.assertAlmostEqual(RT.tma(wl=587).efl, 4.55, delta=0.005)

    
    @pytest.mark.os
    def test_zmx_smith_tessar(self):
        """load a tessar example"""

        RT = ot.Raytracer(outline=[-40, 40, -40, 40, -40, 200])
        RS = ot.RaySource(ot.CircularSurface(r=7), spectrum=ot.presets.light_spectrum.d65, pos=[0, 0, -10])
        RT.add(RS)

        zmx = str(pathlib.Path(__file__).resolve().parent / "test_files" / "LensLibrary" / "files" / "Smith1998b.zmx")
        n_dict = dict(LAFN21=self.n_schott["N-LAF21"]) 
        G = ot.load_zmx(zmx, self.n_schott | n_dict)

        RT.add(G)
        RT.trace(10000)
        self.assertFalse(RT.geometry_error)

        # test focal length. see
        # https://github.com/nzhagen/LensLibrary/blob/main/lens_properties_list.txt
        self.assertAlmostEqual(RT.tma().efl, 52.0, delta=0.1)


    def test_zmx_achromat(self):
        """load an achromat example"""

        # model from
        # https://www.edmundoptics.com/p/25mm-dia-x-100mm-fl-vis-nir-coated-achromatic-lens/9763/

        RT = ot.Raytracer(outline=[-40, 40, -40, 40, -40, 200])
        RS = ot.RaySource(ot.CircularSurface(r=10), spectrum=ot.presets.light_spectrum.d65, pos=[0, 0, -10])
        RT.add(RS)

        zmx = str(pathlib.Path(__file__).resolve().parent / "test_files" / "edmund_zmx" / "files" / "zmax_49360.zmx")
        G = ot.load_zmx(zmx, self.n_schott)

        RT.add(G)
        RT.trace(10000)

        # check properties
        tma = RT.tma(587.6)
        self.assertAlmostEqual(tma.d, 8.5, delta=1e-4)
        self.assertAlmostEqual(tma.bfl, 95.92, delta=0.05)
        self.assertAlmostEqual(tma.efl, 100, delta=0.05)


    def test_zmx_microscope_objective(self):
        """load an objective example"""

        RT = ot.Raytracer(outline=[-40, 40, -40, 40, -40, 200])
        RS = ot.RaySource(ot.CircularSurface(r=0.8), spectrum=ot.presets.light_spectrum.d65, pos=[0, 0, -10])
        RT.add(RS)

        zmx = str(pathlib.Path(__file__).resolve().parent / "test_files" / "LensLibrary" / "files" / "4037934a.zmx")
        G = ot.load_zmx(zmx, self.n_schott, no_marker=True)  # coverage test with no_marker

        RT.add(G)
        RT.trace(10000)
        self.assertFalse(RT.geometry_error)

        # test focal length. see
        # https://github.com/nzhagen/LensLibrary/blob/main/lens_properties_list.txt
        self.assertAlmostEqual(RT.tma().efl, 1, delta=0.002)

    
    def test_zmx_plan_concave(self):
        """zmx of a single lens, so only two surfaces"""

        # model from
        # https://www.edmundoptics.de/f/uncoated-plano-poncave-pcv-lenses/12263/

        RT = ot.Raytracer(outline=[-40, 40, -40, 40, -40, 200])
        RS = ot.RaySource(ot.CircularSurface(r=1.2), spectrum=ot.presets.light_spectrum.d65, pos=[0, 0, -10])
        RT.add(RS)

        zmx = str(pathlib.Path(__file__).resolve().parent / "test_files" / "edmund_zmx" / "files" / "zmax_45374.zmx")
        G = ot.load_zmx(zmx, self.n_schott, no_marker=True)  # coverage test with no_marker

        RT.add(G)
        RT.trace(10000)
        self.assertFalse(RT.geometry_error)

        # test focal length
        tma = RT.tma(587.6)
        self.assertAlmostEqual(tma.efl, -6.00, delta=0.02)
        self.assertAlmostEqual(tma.bfl, -6.56, delta=0.02)
        self.assertAlmostEqual(tma.d, 1, delta=0.02)


    def test_zmx_nikon_100x_objective(self):
        """this example tests:
        1) handling of multiple material dictionaries
        2) an ambient medium before a setup
        3) surfaces without diameter
        4) a long list of surfaces
        5) __BLANK materials, but with index and abbe number
        """
        n_path = pathlib.Path(__file__).resolve().parent / "test_files" / "zemaxglass" / "files"
        n_dict = self.n_schott | ot.load_agf(str(n_path / "ohara.agf"))
        n_dict = n_dict | ot.load_agf(str(n_path / "hikari.agf"))
        n_dict = n_dict | ot.load_agf(str(n_path / "hoya.agf"))
        n_dict["H-ZF7L"] = ot.RefractionIndex("Abbe", n=1.805180, V=25.46)

        # create tracer
        RT = ot.Raytracer(outline=[-2000, 2000, -2000, 2000, -50, 1200])

        # object
        RSS = ot.presets.image.cell([20e-3, 20e-3])
        RS = ot.RaySource(RSS, divergence="Lambertian",
                          pos=[0, 0, -0.00001], s=[0, 0, 1], div_angle=60, desc="Cell")
        RT.add(RS)
        
        zmx = str(pathlib.Path(__file__).resolve().parent / "test_files" / "Kurvits_zemax" / \
                  "files" / "Nikon_1p4NA_100x_US6519092B2_Embodiment_2_v2.zmx")
        G = ot.load_zmx(zmx, n_dict=n_dict)
        RT.add(G)
        
        det = ot.Detector(ot.RectangularSurface([200, 200]), pos=[0, 0, 1000])

        RT.trace(100000)

        img = RT.detector_image()
        self.assertTrue(img.power() > 0.35)

        
    @pytest.mark.os
    def test_zmx_minimal(self):
        """
        minimal example from https://documents.pub/document/zemaxmanual.html?page=461
        there should be no surfaces and lenses created, only a marker
        which gets ignored with no_marker=True
        """
        path = pathlib.Path(__file__).resolve().parent / "test_files" / "edge_cases_zmx" / "minimal.zmx"
        G = ot.load_zmx(str(path), no_marker=True)  # coverage test with no_marker

        # check if empty
        self.assertEqual(len(G.elements), 0)


if __name__ == '__main__':
    unittest.main()
