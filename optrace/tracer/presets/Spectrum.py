from optrace.tracer.Spectrum import *
from optrace.tracer.presets.Lines import *

# Standard Illuminants
#######################################################################################################################

preset_spec_A = Spectrum("Function", func=lambda x: Color.Illuminant(x, "A"),
                         desc="A", long_desc="Standard Illuminant A", quantity="Relative Radiant Power")

preset_spec_C = Spectrum("Function", func=lambda x: Color.Illuminant(x, "C"),    
                         desc="C", long_desc="Standard Illuminant C", quantity="Relative Radiant Power")

preset_spec_D50 = Spectrum("Function", func=lambda x: Color.Illuminant(x, "D50"),  
                            desc="D50", long_desc="Standard Illuminant D50", quantity="Relative Radiant Power")

preset_spec_D55 = Spectrum("Function", func=lambda x: Color.Illuminant(x, "D55"),  
                           desc="D55", long_desc="Standard Illuminant D55", quantity="Relative Radiant Power")

preset_spec_D65 = Spectrum("Function", func=lambda x: Color.Illuminant(x, "D65"),
                           desc="D65", long_desc="Standard Illuminant D65", quantity="Relative Radiant Power")

preset_spec_D75 = Spectrum("Function", func=lambda x: Color.Illuminant(x, "D75"),  
                           desc="D75", long_desc="Standard Illuminant D75", quantity="Relative Radiant Power")

preset_spec_E = Spectrum("Function", func=lambda x: Color.Illuminant(x, "E"),    
                         desc="E", long_desc="Standard Illuminant E", quantity="Relative Radiant Power")

preset_spec_F2 = Spectrum("Function", func=lambda x: Color.Illuminant(x, "F2"),  
                          desc="F2", long_desc="Standard Illuminant F2", quantity="Relative Radiant Power")

preset_spec_F7 = Spectrum("Function", func=lambda x: Color.Illuminant(x, "F7"),   
                          desc="F7", long_desc="Standard Illuminant F7", quantity="Relative Radiant Power")

preset_spec_F11 = Spectrum("Function", func=lambda x: Color.Illuminant(x, "F11"),
                           desc="F11", long_desc="Standard Illuminant F11", quantity="Relative Radiant Power")

presets_spec_standard = [preset_spec_A, preset_spec_C, preset_spec_D50, preset_spec_D55, 
                         preset_spec_D65, preset_spec_D75, preset_spec_E,  preset_spec_F2,  
                         preset_spec_F7, preset_spec_F11]

# Tristimulus Curves
#######################################################################################################################

preset_spec_X = Spectrum("Function", func=lambda x: Color.Tristimulus(x, "X"),
                         desc="X", long_desc="Tristimulus X Curve", quantity="Relative Response")

preset_spec_Y = Spectrum("Function", func=lambda x: Color.Tristimulus(x, "Y"),
                         desc="Y", long_desc="Tristimulus Y Curve", quantity="Relative Response")

preset_spec_Z = Spectrum("Function", func=lambda x: Color.Tristimulus(x, "Z"),
                         desc="Z", long_desc="Tristimulus Z Curve", quantity="Relative Response")

presets_spec_tristimulus = [preset_spec_X, preset_spec_Y, preset_spec_Z]


# Possible sRGB Primaries
#######################################################################################################################

preset_spec_sRGB_r = Spectrum("Function", func=Color._sRGB_r_primary,
                              desc="R", long_desc="Possible sRGB R Primary", quantity="Relative Spectral Density")
                             
preset_spec_sRGB_g = Spectrum("Function", func=Color._sRGB_g_primary,
                              desc="G", long_desc="Possible sRGB G Primary", quantity="Relative Spectral Density")
                             
preset_spec_sRGB_b = Spectrum("Function", func=Color._sRGB_b_primary,
                              desc="B", long_desc="Possible sRGB B Primary", quantity="Relative Spectral Density")
                             
preset_spec_sRGB_w = Spectrum("Function", func=lambda wl: Color._sRGB_r_primary(wl) +\
                                                          Color._sRGB_g_primary(wl) + Color._sRGB_b_primary(wl),
                              desc="W", long_desc="Possible sRGB White Spectrum", quantity="Relative Spectral Density")

presets_spec_sRGB = [preset_spec_sRGB_r, preset_spec_sRGB_g, preset_spec_sRGB_b, preset_spec_sRGB_w]

# spectra for line combinations
#######################################################################################################################

preset_spec_FDC = Spectrum("Lines", lines=preset_lines_FDC, line_vals=[1, 1, 1],
                           desc="Lines FDC", long_desc="Spectral Lines F, D, C") 

preset_spec_FdC = Spectrum("Lines", lines=preset_lines_FdC, line_vals=[1, 1, 1],
                           desc="Lines FdC", long_desc="Spectral Lines F, d, C") 

preset_spec_FeC = Spectrum("Lines", lines=preset_lines_FeC, line_vals=[1, 1, 1], 
                           desc="Lines Fec", long_desc="Spectral Lines F, e, C") 

preset_spec_F_eC_ = Spectrum("Lines", lines=preset_lines_F_eC_, line_vals=[1, 1, 1],
                             desc="Lines F'eC'", long_desc="Spectral Lines F', e, C'")

presets_spec_lines = [preset_spec_FDC, preset_spec_FdC,  
                      preset_spec_FeC, preset_spec_F_eC_]


# List of all spec presets
#######################################################################################################################

presets_spec = [*presets_spec_standard, *presets_spec_lines, *presets_spec_tristimulus, *presets_spec_sRGB]

