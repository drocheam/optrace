#!/usr/bin/env python3

import sys
sys.path.append('.')

import optrace as ot
import optrace.plots


ot.plots.SpectrumPlot(ot.presets_spec_standard, labels_off=True)
ot.plots.ChromacitiesCIE1976(ot.presets_spec_standard)

ot.plots.SpectrumPlot(ot.presets_spec_sRGB, colors=["r", "g", "b", "k"], labels_off=True)
ot.plots.ChromacitiesCIE1976(ot.presets_spec_sRGB)

ot.plots.SpectrumPlot(ot.presets_spec_tristimulus, block=True, colors=["r", "g", "b"], 
                      title="Tristimuli", labels_off=True)

