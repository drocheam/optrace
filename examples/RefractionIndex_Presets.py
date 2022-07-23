#!/usr/bin/env python3

import sys
sys.path.append('.')

import optrace as ot
import optrace.plots


ot.plots.RefractionIndexPlot(ot.presets.RefractionIndex.glasses, legend_off=True, title="Glass Presets")
ot.plots.RefractionIndexPlot(ot.presets.RefractionIndex.plastics, legend_off=True, title="Plastic Presets")
ot.plots.RefractionIndexPlot(ot.presets.RefractionIndex.misc, legend_off=True, block=False, title="Miscellaneous Presets")

ot.plots.AbbePlot(ot.presets.RefractionIndex.glasses, title="Abbe Diagram for Glasses")
ot.plots.AbbePlot(ot.presets.RefractionIndex.plastics, title="Abbe Diagram for Plastics")
ot.plots.AbbePlot(ot.presets.RefractionIndex.misc, title="Abbe Diagram for Miscellaneous Materials", block=True)

