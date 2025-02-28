#!/usr/bin/env python3

import optrace as ot
import optrace.plots as otp

# An example loading multiple light spectrum plots, including the sRGB primaries and standard illuminants.

otp.chromaticities_cie_1976(ot.presets.light_spectrum.standard_natural)
otp.spectrum_plot(ot.presets.light_spectrum.standard_natural, labels_off=False, title="CIE Standard Illuminants")
otp.spectrum_plot(ot.presets.light_spectrum.standard_led, labels_off=True, title="CIE Standard Illuminants LED")
otp.spectrum_plot(ot.presets.light_spectrum.standard_f, labels_off=True,
                  title="CIE Standard Illuminants Fluorescent")

ot.global_options.plot_dark_mode = False
otp.chromaticities_cie_1931(ot.presets.light_spectrum.srgb, norm="Euclidean")
otp.spectrum_plot(ot.presets.light_spectrum.srgb, color=["#f30", "#2b3", "#08e", "#999"],
                  labels_off=True, title="sRGB Primaries")

otp.spectrum_plot(ot.presets.spectrum.xyz_observers, color=["#f30", "#2b3", "#08e"],
                  title="CIE 1931 XYZ Standard Observer Color Matching Functions", labels_off=True)
otp.block() 
