#!/usr/bin/env python3

import sys
sys.path.append('.')

import optrace as ot
import optrace.plots


ot.plots.SpectrumPlot(ot.presets.LightSpectrum.standard, labels_off=True)
ot.plots.ChromacitiesCIE1976(ot.presets.LightSpectrum.standard)

ot.plots.SpectrumPlot(ot.presets.LightSpectrum.sRGB, colors=["r", "g", "b", "k"], labels_off=True)
ot.plots.ChromacitiesCIE1976(ot.presets.LightSpectrum.sRGB)

ot.plots.SpectrumPlot(ot.presets.Spectrum.tristimulus, block=True, colors=["r", "g", "b"], 
                      title="Tristimuli", labels_off=True)

