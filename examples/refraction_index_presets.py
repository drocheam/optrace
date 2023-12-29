#!/usr/bin/env python3

import optrace as ot
import optrace.plots as otp

otp.refraction_index_plot(ot.presets.refraction_index.glasses, legend_off=True, title="Glass Presets")
otp.refraction_index_plot(ot.presets.refraction_index.plastics, legend_off=True, title="Plastic Presets")
otp.refraction_index_plot(ot.presets.refraction_index.misc, legend_off=True,
                               title="Miscellaneous Presets")

otp.abbe_plot(ot.presets.refraction_index.glasses, title="Abbe Diagram for Glasses")
otp.abbe_plot(ot.presets.refraction_index.plastics, title="Abbe Diagram for Plastics")
otp.abbe_plot(ot.presets.refraction_index.misc, title="Abbe Diagram for Miscellaneous Materials")
otp.block()
