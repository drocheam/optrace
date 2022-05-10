from optrace.tracer.Spectrum import *
from optrace.tracer.presets.Lines import *

# Illuminants
preset_spec_A   = Spectrum("Function", func=lambda x: Color.Illuminant(x, "A"))
preset_spec_C   = Spectrum("Function", func=lambda x: Color.Illuminant(x, "C"))
preset_spec_D50 = Spectrum("Function", func=lambda x: Color.Illuminant(x, "D50"))
preset_spec_D55 = Spectrum("Function", func=lambda x: Color.Illuminant(x, "D55"))
preset_spec_D65 = Spectrum("Function", func=lambda x: Color.Illuminant(x, "D65"))
preset_spec_D75 = Spectrum("Function", func=lambda x: Color.Illuminant(x, "D75"))
preset_spec_E   = Spectrum("Function", func=lambda x: Color.Illuminant(x, "E"))
preset_spec_F2  = Spectrum("Function", func=lambda x: Color.Illuminant(x, "F2"))
preset_spec_F7  = Spectrum("Function", func=lambda x: Color.Illuminant(x, "F7"))
preset_spec_F11 = Spectrum("Function", func=lambda x: Color.Illuminant(x, "F11"))

# spectra for line combinations
preset_spec_FDC   = Spectrum("Lines", lines=preset_lines_FDC,   line_vals=[1, 1, 1]) 
preset_spec_FdC   = Spectrum("Lines", lines=preset_lines_FdC,   line_vals=[1, 1, 1]) 
preset_spec_FeC   = Spectrum("Lines", lines=preset_lines_FeC,   line_vals=[1, 1, 1]) 
preset_spec_F_eC_ = Spectrum("Lines", lines=preset_lines_F_eC_, line_vals=[1, 1, 1])

# neutral density filter transmission curves
preset_spec_ND01 = Spectrum("Constant", val=10**(-0.1))
preset_spec_ND02 = Spectrum("Constant", val=10**(-0.2))
preset_spec_ND03 = Spectrum("Constant", val=10**(-0.3))
preset_spec_ND04 = Spectrum("Constant", val=10**(-0.4))
preset_spec_ND05 = Spectrum("Constant", val=10**(-0.5))
preset_spec_ND06 = Spectrum("Constant", val=10**(-0.6))
preset_spec_ND07 = Spectrum("Constant", val=10**(-0.7))
preset_spec_ND08 = Spectrum("Constant", val=10**(-0.8))
preset_spec_ND09 = Spectrum("Constant", val=10**(-0.9))
preset_spec_ND10 = Spectrum("Constant", val=10**(-1.0))
preset_spec_ND13 = Spectrum("Constant", val=10**(-1.3))
preset_spec_ND15 = Spectrum("Constant", val=10**(-1.5))
preset_spec_ND20 = Spectrum("Constant", val=10**(-2.0))
preset_spec_ND30 = Spectrum("Constant", val=10**(-3.0))
preset_spec_ND40 = Spectrum("Constant", val=10**(-4.0))
preset_spec_ND50 = Spectrum("Constant", val=10**(-5.0))


presets_spec = [preset_spec_A, preset_spec_C, preset_spec_D50, preset_spec_D55, 
                preset_spec_D65, preset_spec_D75, preset_spec_E,  preset_spec_F2,  
                preset_spec_F7, preset_spec_F11, preset_spec_FDC, preset_spec_FdC,  
                preset_spec_FeC, preset_spec_F_eC_, preset_spec_ND01, preset_spec_ND02, 
                preset_spec_ND03, preset_spec_ND04, preset_spec_ND05, preset_spec_ND06, 
                preset_spec_ND07, preset_spec_ND08, preset_spec_ND09, preset_spec_ND10, 
                preset_spec_ND13, preset_spec_ND15, preset_spec_ND20, preset_spec_ND30, 
                preset_spec_ND40, preset_spec_ND50]

