#!/bin/env python3

import sys
sys.path.append('.')

import unittest

import numpy as np
import optrace as ot
import optrace.plots as otp
import optrace.tracer.Color as Color

# from threading import Thread
import matplotlib.pyplot as plt

class PlotTests(unittest.TestCase):

    def test_RImagePlots(self) -> None:
        
        RT = ot.Raytracer(outline=[-3, 3, -3, 3, 0, 6], silent=True)
        RSS = ot.Surface("Rectangle", dim=[6, 6])
        RS = ot.RaySource(RSS, pos=[0, 0, 0], Image=ot.presets.Image.color_checker)
        RT.add(RS)

        RT.trace(200000)

        def checkPlots(img):
            # image plots
            for modei in ot.RImage.display_modes:
                otp.RImagePlot(img, mode=modei, log=False, block=False)
                otp.RImagePlot(img, mode=modei, log=True, block=False)
            otp.RImagePlot(img, mode=ot.RImage.display_modes[0], flip=True, block=False)
            plt.close('all')

            # image cut plots
            for modei in ot.RImage.display_modes:
                otp.RImageCutPlot(img, mode=modei, log=False, block=False, x=0.8)
                otp.RImageCutPlot(img, mode=modei, log=True, y=-2.756, block=False)
            otp.RImageCutPlot(img, mode=ot.RImage.display_modes[0], flip=True, block=False, x=0.5)
            plt.close('all')

        # cartesian plots
        img = RT.SourceImage(200)
        checkPlots(img)

        # polar plots
        RT.add(ot.Detector(ot.Surface("Sphere", rho=-1/10), pos=[0, 0, 3]))
        img = RT.DetectorImage(200)
        checkPlots(img)

        self.assertRaises(TypeError, otp.RImagePlot, [5, 5]) # invalid RImage
        self.assertRaises(TypeError, otp.RImageCutPlot, [5, 5]) # invalid RImage
        self.assertRaises(ValueError, otp.RImagePlot, img, mode="4564") # invalid RImage mode
        self.assertRaises(ValueError, otp.RImageCutPlot, img, mode="4564") # invalid RImage mode


    def test_DebugPlots(self) -> None:

        # RefractionIndexPlots
        otp.RefractionIndexPlot(ot.presets.RefractionIndex.misc)
        otp.RefractionIndexPlot(ot.presets.RefractionIndex.SF10)
        
        # Abbe Plot
        otp.AbbePlot(ot.presets.RefractionIndex.glasses, silent=True)

        # SpectrumPlot
        otp.SpectrumPlot(ot.presets.LightSpectrum.D50)
        otp.SpectrumPlot(ot.presets.LightSpectrum.standard)

        plt.close('all')
        
        RT = ot.Raytracer(outline=[-3, 3, -3, 3, 0, 6], silent=True)
        RSS = ot.Surface("Rectangle", dim=[6, 6])
        RS = ot.RaySource(RSS, pos=[0, 0, 0], Image=ot.presets.Image.color_checker)
        RT.add(RS)
        RT.trace(200000)
        img = RT.SourceImage(200)
        
        # ChromacityPlots
        for normi in otp.chromacity_norms: 
            otp.ChromacitiesCIE1931(ot.presets.LightSpectrum.D65, norm=normi)
            otp.ChromacitiesCIE1931(ot.presets.LightSpectrum.standard, norm=normi)
            otp.ChromacitiesCIE1976(ot.presets.LightSpectrum.D65, norm=normi)
            otp.ChromacitiesCIE1976(ot.presets.LightSpectrum.standard, norm=normi)
            
            for RIi in Color.sRGB_RI:
                otp.ChromacitiesCIE1931(img, norm=normi, RI=RIi)
                otp.ChromacitiesCIE1976(img, norm=normi, RI=RIi)
        
            plt.close('all')

        # AutoFocusDebug is tested elsewhere

    def test_Image_Presets(self) -> None:

        for imgi in ot.presets.Image.all_presets:
            RT = ot.Raytracer(outline=[-3, 3, -3, 3, 0, 6], silent=True)
            RSS = ot.Surface("Rectangle", dim=[6, 6])
            RS = ot.RaySource(RSS, pos=[0, 0, 0], Image=imgi)
            RT.add(RS)
            RT.trace(200000)
            img = RT.SourceImage(200)
            
            otp.RImagePlot(img)

        plt.close('all')


if __name__ == '__main__':
    unittest.main()

