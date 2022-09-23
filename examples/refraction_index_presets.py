#!/usr/bin/env python3

import optrace as ot
import optrace.plots

ot.plots.refraction_index_plot(ot.presets.refraction_index.glasses, legend_off=True, title="Glass Presets")
ot.plots.refraction_index_plot(ot.presets.refraction_index.plastics, legend_off=True, title="Plastic Presets")
ot.plots.refraction_index_plot(ot.presets.refraction_index.misc, legend_off=True, block=False,
                               title="Miscellaneous Presets")

ot.plots.abbe_plot(ot.presets.refraction_index.glasses, title="Abbe Diagram for Glasses")
ot.plots.abbe_plot(ot.presets.refraction_index.plastics, title="Abbe Diagram for Plastics")
ot.plots.abbe_plot(ot.presets.refraction_index.misc, title="Abbe Diagram for Miscellaneous Materials", block=True)
