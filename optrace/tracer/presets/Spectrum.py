
import optrace.tracer.presets.Lines as Lines  # spectral lines
import optrace.tracer.Color as Color  # tristimulus curves

from optrace.tracer.spectrum import Spectrum, LightSpectrum  # light spectrum and spectrum class

# Standard Illuminants
#######################################################################################################################

_quan_unit = dict(quantity="Relative Radiant Power", unit="")

preset_spec_A: LightSpectrum = LightSpectrum("Function", func=lambda x: Color.Illuminant(x, "A"),
                                             desc="A", long_desc="Standard Illuminant A", **_quan_unit)
"""Standard Illuminant A"""

preset_spec_C: LightSpectrum = LightSpectrum("Function", func=lambda x: Color.Illuminant(x, "C"),    
                                             desc="C", long_desc="Standard Illuminant C", **_quan_unit)
"""Standard Illuminant C"""

preset_spec_D50: LightSpectrum = LightSpectrum("Function", func=lambda x: Color.Illuminant(x, "D50"),  
                                               desc="D50", long_desc="Standard Illuminant D50", **_quan_unit)
"""Standard Illuminant D50"""

preset_spec_D55: LightSpectrum = LightSpectrum("Function", func=lambda x: Color.Illuminant(x, "D55"),  
                                               desc="D55", long_desc="Standard Illuminant D55", **_quan_unit)
"""Standard Illuminant D55"""

preset_spec_D65: LightSpectrum = LightSpectrum("Function", func=lambda x: Color.Illuminant(x, "D65"),
                                               desc="D65", long_desc="Standard Illuminant D65", **_quan_unit)
"""Standard Illuminant D65"""

preset_spec_D75: LightSpectrum = LightSpectrum("Function", func=lambda x: Color.Illuminant(x, "D75"),  
                                               desc="D75", long_desc="Standard Illuminant D75", **_quan_unit)
"""Standard Illuminant D75"""

preset_spec_E: LightSpectrum = LightSpectrum("Function", func=lambda x: Color.Illuminant(x, "E"),    
                                             desc="E", long_desc="Standard Illuminant E", **_quan_unit)
"""Standard Illuminant E"""

preset_spec_F2: LightSpectrum = LightSpectrum("Function", func=lambda x: Color.Illuminant(x, "F2"),  
                                              desc="F2", long_desc="Standard Illuminant F2", **_quan_unit)
"""Standard Illuminant F2"""

preset_spec_F7: LightSpectrum = LightSpectrum("Function", func=lambda x: Color.Illuminant(x, "F7"),   
                                              desc="F7", long_desc="Standard Illuminant F7", **_quan_unit)
"""Standard Illuminant F7"""

preset_spec_F11: LightSpectrum = LightSpectrum("Function", func=lambda x: Color.Illuminant(x, "F11"),
                                               desc="F11", long_desc="Standard Illuminant F11", **_quan_unit)
"""Standard Illuminant F11"""


presets_spec_standard: list = [preset_spec_A, preset_spec_C, preset_spec_D50, preset_spec_D55, 
                               preset_spec_D65, preset_spec_D75, preset_spec_E,  preset_spec_F2,  
                               preset_spec_F7, preset_spec_F11]
"""standard illuminant presets in one list"""


# Tristimulus Curves
#######################################################################################################################

preset_spec_X: Spectrum = Spectrum("Function", func=lambda x: Color.Tristimulus(x, "X"),
                                   desc="X", long_desc="Tristimulus X Curve", quantity="Relative Response", unit="")
"""Tristimulus X curve"""

preset_spec_Y: Spectrum = Spectrum("Function", func=lambda x: Color.Tristimulus(x, "Y"),
                                   desc="Y", long_desc="Tristimulus Y Curve", quantity="Relative Response", unit="")
"""Tristimulus Y curve"""

preset_spec_Z: Spectrum = Spectrum("Function", func=lambda x: Color.Tristimulus(x, "Z"),
                                   desc="Z", long_desc="Tristimulus Z Curve", quantity="Relative Response", unit="")
"""Tristimulus Z curve"""


presets_spec_tristimulus: list = [preset_spec_X, preset_spec_Y, preset_spec_Z]
"""tristimulus curve spectrum presets in one list"""


# Possible sRGB Primaries
#######################################################################################################################

_quan_unit = dict(quantity="Relative Spectral Density", unit="")

preset_spec_sRGB_r: LightSpectrum = LightSpectrum("Function", func=Color._sRGB_r_primary,
                                                  desc="R", long_desc="Possible sRGB R Primary", **_quan_unit)
"""Possible sRGB R Primary"""

preset_spec_sRGB_g: LightSpectrum = LightSpectrum("Function", func=Color._sRGB_g_primary,
                                                  desc="G", long_desc="Possible sRGB G Primary", **_quan_unit)
"""Possible sRGB G Primary"""
                             
preset_spec_sRGB_b: LightSpectrum = LightSpectrum("Function", func=Color._sRGB_b_primary,
                                                  desc="B", long_desc="Possible sRGB B Primary", **_quan_unit)
"""Possible sRGB B Primary"""
                             
preset_spec_sRGB_w: LightSpectrum = LightSpectrum("Function", func=lambda wl: Color._sRGB_r_primary(wl) +\
                                                          Color._sRGB_g_primary(wl) + Color._sRGB_b_primary(wl),
                                                  desc="W", long_desc="Possible sRGB White Spectrum", **_quan_unit)
"""Possible sRGB White Spectrum"""


presets_spec_sRGB: list = [preset_spec_sRGB_r, preset_spec_sRGB_g, preset_spec_sRGB_b, preset_spec_sRGB_w]
"""sRGB channel and white presets in one list"""


# spectra for line combinations
#######################################################################################################################

preset_spec_FDC: LightSpectrum = LightSpectrum("Lines", lines=Lines.preset_lines_FDC, line_vals=[1, 1, 1],
                                               desc="Lines FDC", long_desc="Spectral Lines F, D, C")
"""Spectral Lines F, D, C"""

preset_spec_FdC: LightSpectrum = LightSpectrum("Lines", lines=Lines.preset_lines_FdC, line_vals=[1, 1, 1],
                                               desc="Lines FdC", long_desc="Spectral Lines F, d, C")
"""Spectral Lines F, d, C"""

preset_spec_FeC: LightSpectrum = LightSpectrum("Lines", lines=Lines.preset_lines_FeC, line_vals=[1, 1, 1], 
                                               desc="Lines Fec", long_desc="Spectral Lines F, e, C")
"""Spectral Lines F, e, C"""

preset_spec_F_eC_: LightSpectrum = LightSpectrum("Lines", lines=Lines.preset_lines_F_eC_, line_vals=[1, 1, 1],
                                                 desc="Lines F'eC'", long_desc="Spectral Lines F', e, C'")
"""Spectral Lines F', e, C'"""


presets_spec_lines: list = [preset_spec_FDC, preset_spec_FdC, preset_spec_FeC, preset_spec_F_eC_]
"""all lines spectrum presets in one list"""


# List of all spec presets
#######################################################################################################################

presets_spec_light: list = [*presets_spec_standard, *presets_spec_lines, *presets_spec_sRGB]
"""all light spectrum presets in one list"""

