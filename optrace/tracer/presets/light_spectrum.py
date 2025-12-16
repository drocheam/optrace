from . import spectral_lines as Lines  # spectral lines
from ..spectrum import LightSpectrum
from .. import color

# Standard Illuminants
#######################################################################################################################

a: LightSpectrum = LightSpectrum("Function", func=color.a_illuminant,
                                 desc="A", long_desc="Illuminant A")
"""Standard Illuminant A. Typical, domestic, tungsten-filament lighting. Color temperature of 2856K."""

c: LightSpectrum = LightSpectrum("Function", func=color.c_illuminant,
                                 desc="C", long_desc="Illuminant C")
"""Standard Illuminant C. Obsolete, average / north sky daylight. Color temperature of 6774K."""

d50: LightSpectrum = LightSpectrum("Function", func=color.d50_illuminant,
                                   desc="D50", long_desc="Illuminant D50")
"""Standard Illuminant D50. Horizon light. Color temperature of 5003K."""

d55: LightSpectrum = LightSpectrum("Function", func=color.d55_illuminant,
                                   desc="D55", long_desc="Illuminant D55")
"""Standard Illuminant D55. Mid-morning/mid-afternoon daylight. Color temperature of 5503K."""

d65: LightSpectrum = LightSpectrum("Function", func=color.d65_illuminant,
                                   desc="D65", long_desc="Illuminant D65")
"""Standard Illuminant D65. Noon daylight. Color temperature of 6504K"""

d75: LightSpectrum = LightSpectrum("Function", func=color.d75_illuminant,
                                   desc="D75", long_desc="Illuminant D75")
"""Standard Illuminant D75. North sky daylight. Color temperature of 7504K."""

e: LightSpectrum = LightSpectrum("Function", func=color.e_illuminant,
                                  desc="E", long_desc="Illuminant E")
"""Standard Illuminant E. Equal energy radiator with a color temperature of 5455K."""

f2: LightSpectrum = LightSpectrum("Function", func=color.f2_illuminant,
                                   desc="F2", long_desc="Illuminant F2")
"""Standard Illuminant F2. Fluorescent lamp with two semi-broadband emissions. Color temperature of 4230K."""

f7: LightSpectrum = LightSpectrum("Function", func=color.f7_illuminant,
                                   desc="F7", long_desc="Illuminant F7")
"""Standard Illuminant F7. Broadband fluorescent lamp with multiple phosphors. Color temperature of 6500K."""

f11: LightSpectrum = LightSpectrum("Function", func=color.f11_illuminant,
                                    desc="F11", long_desc="Illuminant F11")
"""Standard Illuminant F11. Narrowband triband fluorescent lamp in R, G, B regions. Color temperature of 4000K."""

led_b1: LightSpectrum = LightSpectrum("Function", func=color.led_b1_illuminant,
                                      desc="LED-B1", long_desc="Illuminant LED-B1")
"""Standard Illuminant LED-B1. Blue excited phosphor type LED with a color temperature of 2733K."""

led_b2: LightSpectrum = LightSpectrum("Function", func=color.led_b2_illuminant,
                                      desc="LED-B2", long_desc="Illuminant LED-B2")
"""Standard Illuminant LED-B2. Blue excited phosphor type LED with a color temperature of 2998K."""

led_b3: LightSpectrum = LightSpectrum("Function", func=color.led_b3_illuminant,
                                      desc="LED-B3", long_desc="Illuminant LED-B3")
"""Standard Illuminant LED-B3. Blue excited phosphor type LED with a color temperature of 4103K."""

led_b4: LightSpectrum = LightSpectrum("Function", func=color.led_b4_illuminant,
                                      desc="LED-B4", long_desc="Illuminant LED-B4")
"""Standard Illuminant LED-B4. Blue excited phosphor type LED with a color temperature of 5109K."""

led_b5: LightSpectrum = LightSpectrum("Function", func=color.led_b5_illuminant,
                                      desc="LED-B5", long_desc="Illuminant LED-B5")
"""Standard Illuminant LED-B5. Blue excited phosphor type LED with a color temperature of 6598K."""

led_bh1: LightSpectrum = LightSpectrum("Function", func=color.led_bh1_illuminant,
                                      desc="LED-BH1", long_desc="Illuminant LED-BH1")
"""Standard Illuminant LED-BH1. Hybrid type white LED with added red. Color temperature of 2851K."""

led_rgb1: LightSpectrum = LightSpectrum("Function", func=color.led_rgb1_illuminant,
                                      desc="LED-RGB1", long_desc="Illuminant LED-RGB1")
"""Standard Illuminant LED-RGB1. Tri-led RGB source with a color temperature of 2840K."""

led_v1: LightSpectrum = LightSpectrum("Function", func=color.led_v1_illuminant,
                                      desc="LED-V1", long_desc="Illuminant LED-V1")
"""Standard Illuminant LED-V1. Violet enhanced blue excited phosphor LED with a color temperature of 2724K."""

led_v2: LightSpectrum = LightSpectrum("Function", func=color.led_v2_illuminant,
                                      desc="LED-V2", long_desc="Illuminant LED-V2")
"""Standard Illuminant LED-V2. Violet enhanced blue excited phosphor LED with a color temperature of 4070K."""

standard_natural: list = [a, c, d50, d55, d65, d75, e]
"""natural standard illuminants. Includes illuminant A and E as well as daylight illuminants"""

standard_f: list = [f2, f7, f11]
"""standard illuminants for fluorescent lamp light"""

standard_led: list = [led_b1, led_b2, led_b3, led_b4, led_b5, led_bh1, led_rgb1, led_v1, led_v2]
"""standard illuminants for LED light"""

#: :meta hide-value:
standard: list = [*standard_natural, *standard_f, *standard_led]
"""standard illuminant presets in one list"""


# Possible sRGB Primaries
#######################################################################################################################


srgb_r: LightSpectrum = LightSpectrum("Function", func=color.srgb_r_primary,
                                      desc="R", long_desc="sRGB R Primary")
"""Exemplary sRGB R Primary Spectrum"""

srgb_g: LightSpectrum = LightSpectrum("Function", func=color.srgb_g_primary,
                                      desc="G", long_desc="sRGB G Primary")
"""Exemplary sRGB G Primary Spectrum"""

srgb_b: LightSpectrum = LightSpectrum("Function", func=color.srgb_b_primary,
                                      desc="B", long_desc="sRGB B Primary")
"""Exemplary sRGB B Primary Spectrum"""

srgb_w: LightSpectrum = LightSpectrum("Function", func=lambda wl: color.srgb_r_primary(wl) +
                                                                  color.srgb_g_primary(wl) + color.srgb_b_primary(wl),
                                      desc="W", long_desc="sRGB White Spectrum")
"""Exemplary sRGB White Spectrum"""

srgb_r_power_factor, srgb_g_power_factor, srgb_b_power_factor = color.SRGB_PRIMARY_POWER_FACTORS
"""Power ratios for white mixing from sRGB primaries"""

#: :meta hide-value:
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
                                         desc="RGB Lines'", long_desc="sRGB Primary Dominant Wavelengths")
"""Spectral Lines 450, 550, 650nm with a power ratio producing neutral D65 white"""

#: :meta hide-value:
lines: list = [FDC, FdC, FeC, F_eC_, rgb_lines]
"""all lines spectrum presets in one list"""


# List of all spec presets
#######################################################################################################################

all_presets: list = [*standard, *lines, *srgb]
"""all light spectrum presets in one list"""

