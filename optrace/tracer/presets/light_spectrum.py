from . import spectral_lines as Lines  # spectral lines
from ..spectrum import LightSpectrum
from .. import color

# Standard Illuminants
#######################################################################################################################

_quan_unit = dict(quantity="Relative Radiant Power", unit="")

a: LightSpectrum = LightSpectrum("Function", func=color.a_illuminant,
                                 desc="A", long_desc="Illuminant A", **_quan_unit)
"""Standard Illuminant A"""

c: LightSpectrum = LightSpectrum("Function", func=color.c_illuminant,
                                 desc="C", long_desc="Illuminant C", **_quan_unit)
"""Standard Illuminant C"""

d50: LightSpectrum = LightSpectrum("Function", func=color.d50_illuminant,
                                   desc="D50", long_desc="Illuminant D50", **_quan_unit)
"""Standard Illuminant D50"""

d55: LightSpectrum = LightSpectrum("Function", func=color.d55_illuminant,
                                   desc="D55", long_desc="Illuminant D55", **_quan_unit)
"""Standard Illuminant D55"""

d65: LightSpectrum = LightSpectrum("Function", func=color.d65_illuminant,
                                   desc="D65", long_desc="Illuminant D65", **_quan_unit)
"""Standard Illuminant D65"""

d75: LightSpectrum = LightSpectrum("Function", func=color.d75_illuminant,
                                   desc="D75", long_desc="Illuminant D75", **_quan_unit)
"""Standard Illuminant D75"""

e: LightSpectrum = LightSpectrum("Function", func=color.e_illuminant,
                                  desc="E", long_desc="Illuminant E", **_quan_unit)
"""Standard Illuminant E"""

fl2: LightSpectrum = LightSpectrum("Function", func=color.fl2_illuminant,
                                   desc="FL2", long_desc="Illuminant FL2", **_quan_unit)
"""Standard Illuminant FL2"""

fl7: LightSpectrum = LightSpectrum("Function", func=color.fl7_illuminant,
                                   desc="FL7", long_desc="Illuminant FL7", **_quan_unit)
"""Standard Illuminant FL7"""

fl11: LightSpectrum = LightSpectrum("Function", func=color.fl11_illuminant,
                                    desc="FL11", long_desc="Illuminant FL11", **_quan_unit)
"""Standard Illuminant FL11"""

led_b1: LightSpectrum = LightSpectrum("Function", func=color.led_b1_illuminant,
                                      desc="LED-B1", long_desc="Illuminant LED-B1", **_quan_unit)
"""Standard Illuminant LED-B1"""

led_b2: LightSpectrum = LightSpectrum("Function", func=color.led_b2_illuminant,
                                      desc="LED-B2", long_desc="Illuminant LED-B2", **_quan_unit)
"""Standard Illuminant LED-B2"""

led_b3: LightSpectrum = LightSpectrum("Function", func=color.led_b3_illuminant,
                                      desc="LED-B3", long_desc="Illuminant LED-B3", **_quan_unit)
"""Standard Illuminant LED-B3"""

led_b4: LightSpectrum = LightSpectrum("Function", func=color.led_b4_illuminant,
                                      desc="LED-B4", long_desc="Illuminant LED-B4", **_quan_unit)
"""Standard Illuminant LED-B4"""

led_b5: LightSpectrum = LightSpectrum("Function", func=color.led_b5_illuminant,
                                      desc="LED-B5", long_desc="Illuminant LED-B5", **_quan_unit)
"""Standard Illuminant LED-B5"""

standard_natural: list = [a, c, d50, d55, d65, d75, e]
"""natural standard illuminants. Includes illuminant A and E as well as daylight illuminants"""

standard_fl: list = [fl2, fl7, fl11]
"""standard illuminants for fluorescent lamp light"""

standard_led: list = [led_b1, led_b2, led_b3, led_b4, led_b5]
"""standard illuminants for LED light"""

standard: list = [*standard_natural, *standard_fl, *standard_led]
"""standard illuminant presets in one list"""


# Possible sRGB Primaries
#######################################################################################################################

_quan_unit = dict(quantity="Relative Spectral Density", unit="")

srgb_r: LightSpectrum = LightSpectrum("Function", func=color.srgb_r_primary,
                                      desc="R", long_desc="sRGB R Primary", **_quan_unit)
"""Possible sRGB R Primary"""

srgb_g: LightSpectrum = LightSpectrum("Function", func=color.srgb_g_primary,
                                      desc="G", long_desc="sRGB G Primary", **_quan_unit)
"""Possible sRGB G Primary"""

srgb_b: LightSpectrum = LightSpectrum("Function", func=color.srgb_b_primary,
                                      desc="B", long_desc="sRGB B Primary", **_quan_unit)
"""Possible sRGB B Primary"""

srgb_w: LightSpectrum = LightSpectrum("Function", func=lambda wl: color.srgb_r_primary(wl) +
                                                                  color.srgb_g_primary(wl) + color.srgb_b_primary(wl),
                                      desc="W", long_desc="sRGB White Spectrum", **_quan_unit)
"""Possible sRGB White Spectrum"""


srgb: list = [srgb_r, srgb_g, srgb_b, srgb_w]
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

rgb_lines: LightSpectrum = LightSpectrum("Lines", lines=Lines.rgb, line_vals=[0.5745000, 0.5985758, 0.3895581],
                                         desc="RGB Lines'", long_desc="Spectral Lines 450, 550, 650nm")
"""Spectral Lines 450, 550, 650nm with a power ratio producing neutral D65 white"""

lines: list = [FDC, FdC, FeC, F_eC_, rgb_lines]
"""all lines spectrum presets in one list"""


# List of all spec presets
#######################################################################################################################

all_presets: list = [*standard, *lines, *srgb]
"""all light spectrum presets in one list"""
