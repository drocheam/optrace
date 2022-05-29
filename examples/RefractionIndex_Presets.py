#!/usr/bin/env python3
# encoding: utf-8

import sys
sys.path.append('./')

import optrace as ot
import optrace.plots


ot.plots.RefractionIndexPlot(ot.presets_n_glass, legend_off=True, title="Glass Presets")
ot.plots.RefractionIndexPlot(ot.presets_n_plastic, legend_off=True, title="Plastic Presets")
ot.plots.RefractionIndexPlot(ot.presets_n_misc, legend_off=True, block=False, title="Miscellaneous Presets")

ot.plots.AbbePlot(ot.presets_n_glass, title="Abbe Diagram for Glasses")
ot.plots.AbbePlot(ot.presets_n_plastic, title="Abbe Diagram for Plastics")
ot.plots.AbbePlot(ot.presets_n_misc, title="Abbe Diagram for Miscellaneous Materials", block=True)

