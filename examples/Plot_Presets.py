#!/usr/bin/env python3
# encoding: utf-8

import sys
sys.path.append('./')

import numpy as np
import optrace as ot
import optrace.plots


ot.plots.ChromacitiesCIE1976(ot.presets_spec_standard)
ot.plots.ChromacitiesCIE1931(ot.presets_spec_standard)

ot.plots.RefractionIndexPlot(ot.presets_n_glass, legend_off=True, title="Refraction Index Glass Presets")
ot.plots.RefractionIndexPlot(ot.presets_n_plastic, legend_off=True, title="Refraction Index Plastic Presets")
ot.plots.RefractionIndexPlot(ot.presets_n_misc, legend_off=True, title="Refraction Index Misc Presets")

ot.plots.SpectrumPlot(ot.presets_spec_sRGB, colors=["r", "g", "b", "k"], labels_off=True)

ot.plots.SpectrumPlot(ot.presets_spec_tristimulus, block=True, colors=["r", "g", "b"], 
                      title="Tristimuli", labels_off=True)

