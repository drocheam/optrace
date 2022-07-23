import optrace.tracer.presets.Lines as Lines  # spectral lines
# light spectrum and spectrum class
from optrace.tracer.spectrum import LightSpectrum
import optrace.tracer.Color as Color

# Standard Illuminants
#######################################################################################################################

_quan_unit = dict(quantity="Relative Radiant Power", unit="")

A: LightSpectrum = LightSpectrum("Function", func=lambda x: Color.Illuminant(x, "A"),
                                 desc="A", long_desc="Standard Illuminant A", **_quan_unit)
"""Standard Illuminant A"""

C: LightSpectrum = LightSpectrum("Function", func=lambda x: Color.Illuminant(x, "C"),
                                 desc="C", long_desc="Standard Illuminant C", **_quan_unit)
"""Standard Illuminant C"""

D50: LightSpectrum = LightSpectrum("Function", func=lambda x: Color.Illuminant(x, "D50"),
                                   desc="D50", long_desc="Standard Illuminant D50", **_quan_unit)
"""Standard Illuminant D50"""

D55: LightSpectrum = LightSpectrum("Function", func=lambda x: Color.Illuminant(x, "D55"),
                                   desc="D55", long_desc="Standard Illuminant D55", **_quan_unit)
"""Standard Illuminant D55"""

D65: LightSpectrum = LightSpectrum("Function", func=lambda x: Color.Illuminant(x, "D65"),
                                   desc="D65", long_desc="Standard Illuminant D65", **_quan_unit)
"""Standard Illuminant D65"""

D75: LightSpectrum = LightSpectrum("Function", func=lambda x: Color.Illuminant(x, "D75"),
                                   desc="D75", long_desc="Standard Illuminant D75", **_quan_unit)
"""Standard Illuminant D75"""

E: LightSpectrum = LightSpectrum("Function", func=lambda x: Color.Illuminant(x, "E"),
                                 desc="E", long_desc="Standard Illuminant E", **_quan_unit)
"""Standard Illuminant E"""

F2: LightSpectrum = LightSpectrum("Function", func=lambda x: Color.Illuminant(x, "F2"),
                                  desc="F2", long_desc="Standard Illuminant F2", **_quan_unit)
"""Standard Illuminant F2"""

F7: LightSpectrum = LightSpectrum("Function", func=lambda x: Color.Illuminant(x, "F7"),
                                  desc="F7", long_desc="Standard Illuminant F7", **_quan_unit)
"""Standard Illuminant F7"""

F11: LightSpectrum = LightSpectrum("Function", func=lambda x: Color.Illuminant(x, "F11"),
                                   desc="F11", long_desc="Standard Illuminant F11", **_quan_unit)
"""Standard Illuminant F11"""


standard: list = [A, C, D50, D55, D65, D75, E, F2, F7, F11]
"""standard illuminant presets in one list"""


# Possible sRGB Primaries
#######################################################################################################################

_quan_unit = dict(quantity="Relative Spectral Density", unit="")

sRGB_r: LightSpectrum = LightSpectrum("Function", func=Color._sRGB_r_primary,
                                      desc="R", long_desc="Possible sRGB R Primary", **_quan_unit)
"""Possible sRGB R Primary"""

sRGB_g: LightSpectrum = LightSpectrum("Function", func=Color._sRGB_g_primary,
                                      desc="G", long_desc="Possible sRGB G Primary", **_quan_unit)
"""Possible sRGB G Primary"""

sRGB_b: LightSpectrum = LightSpectrum("Function", func=Color._sRGB_b_primary,
                                      desc="B", long_desc="Possible sRGB B Primary", **_quan_unit)
"""Possible sRGB B Primary"""

sRGB_w: LightSpectrum = LightSpectrum("Function", func=lambda wl: Color._sRGB_r_primary(wl) +
                                                                  Color._sRGB_g_primary(wl) + Color._sRGB_b_primary(wl),
                                      desc="W", long_desc="Possible sRGB White Spectrum", **_quan_unit)
"""Possible sRGB White Spectrum"""


sRGB: list = [sRGB_r, sRGB_g, sRGB_b, sRGB_w]
"""sRGB channel and white presets in one list"""


# spectra for line combinations
#######################################################################################################################

FDC: LightSpectrum = LightSpectrum("Lines", lines=Lines.FDC, line_vals=[1, 1, 1],
                                   desc="Lines FDC", long_desc="Spectral Lines F, D, C")
"""Spectral Lines F, D, C"""

FdC: LightSpectrum = LightSpectrum("Lines", lines=Lines.FdC, line_vals=[1, 1, 1],
                                   desc="Lines FdC", long_desc="Spectral Lines F, d, C")
"""Spectral Lines F, d, C"""

FeC: LightSpectrum = LightSpectrum("Lines", lines=Lines.FeC, line_vals=[1, 1, 1],
                                   desc="Lines Fec", long_desc="Spectral Lines F, e, C")
"""Spectral Lines F, e, C"""

F_eC_: LightSpectrum = LightSpectrum("Lines", lines=Lines.F_eC_, line_vals=[1, 1, 1],
                                     desc="Lines F'eC'", long_desc="Spectral Lines F', e, C'")
"""Spectral Lines F', e, C'"""


lines: list = [FDC, FdC, FeC, F_eC_]
"""all lines spectrum presets in one list"""


# List of all spec presets
#######################################################################################################################

all_presets: list = [*standard, *lines, *sRGB]
"""all light spectrum presets in one list"""

