#!/usr/bin/env python3

import optrace as ot
import optrace.plots as otp


otp.chromaticities_cie_1976(ot.presets.light_spectrum.standard_natural)
otp.spectrum_plot(ot.presets.light_spectrum.standard_natural, labels_off=False, title="CIE Standard Illuminants")
otp.spectrum_plot(ot.presets.light_spectrum.standard_led, labels_off=True, title="CIE Standard Illuminants LED")
otp.spectrum_plot(ot.presets.light_spectrum.standard_fl, labels_off=True,
                  title="CIE Standard Illuminants Fluorescent")

otp.chromaticities_cie_1931(ot.presets.light_spectrum.srgb, norm="Largest")
otp.spectrum_plot(ot.presets.light_spectrum.srgb, color=["r", "g", "b", "k"],
                  labels_off=True, title="sRGB Primaries")

otp.spectrum_plot(ot.presets.spectrum.xyz_observers, color=["r", "g", "b"],
                  title="CIE 1931 XYZ Standard Observer Color Matching Functions", labels_off=True)
otp.block() 
