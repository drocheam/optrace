#!/usr/bin/env python3

import optrace as ot
import optrace.plots

ot.plots.chromacities_cie_1976(ot.presets.light_spectrum.standard_natural)
ot.plots.spectrum_plot(ot.presets.light_spectrum.standard_natural, labels_off=False, title="Standard Illuminants")
ot.plots.spectrum_plot(ot.presets.light_spectrum.standard_led, labels_off=True, title="Standard Illuminants LED")
ot.plots.spectrum_plot(ot.presets.light_spectrum.standard_fl, labels_off=True, title="Standard Illuminants Fluorescent")

ot.plots.chromacities_cie_1931(ot.presets.light_spectrum.srgb)
ot.plots.spectrum_plot(ot.presets.light_spectrum.srgb, color=["r", "g", "b", "k"],
                       labels_off=True, title="sRGB Primaries")

ot.plots.spectrum_plot(ot.presets.spectrum.tristimulus, block=True, color=["r", "g", "b"],
                       title="Tristimulus Curves", labels_off=True)
        
