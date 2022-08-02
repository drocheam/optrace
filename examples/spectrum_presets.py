#!/usr/bin/env python3

import optrace as ot
import optrace.plots

ot.plots.chromacities_cie_1976(ot.presets.light_spectrum.standard)
ot.plots.spectrum_plot(ot.presets.light_spectrum.standard, labels_off=False, title="Standard Illuminants")

ot.plots.chromacities_cie_1931(ot.presets.light_spectrum.sRGB)
ot.plots.spectrum_plot(ot.presets.light_spectrum.sRGB, colors=["r", "g", "b", "k"],
                       labels_off=True, title="sRGB Primaries")

ot.plots.spectrum_plot(ot.presets.Spectrum.tristimulus, block=True, colors=["r", "g", "b"],
                       title="Tristimuli Curves", labels_off=True)
