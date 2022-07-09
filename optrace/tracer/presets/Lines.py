
# spectral lines
#######################################################################################################################
"""all lines specified with name, element, color"""

# spectral lines from https://en.wikipedia.org/wiki/Abbe_number
# preset_line_i: float  = 365.01 	#: i    Hg 	UV-A # can't simulate this one
preset_line_h:   float = 404.66 	#: h 	Hg 	violet
preset_line_g:   float = 435.84 	#: g 	Hg 	blue
preset_line_F_:  float = 479.99 	#: F' 	Cd 	blue
preset_line_F:   float = 486.13 	#: F 	H 	blue
preset_line_e:   float = 546.07 	#: e 	Hg 	green
preset_line_d:   float = 587.56     #: d 	He 	yellow
preset_line_D:   float = 589.3      #: D 	Na 	yellow
preset_line_C_:  float = 643.85 	#: C' 	Cd 	red
preset_line_C:   float = 656.27 	#: C 	H 	red
preset_line_r:   float = 706.52 	#: r 	He 	red
preset_line_A_:  float = 768.2 	    #: A' 	K 	IR-A
# preset_line_s: float  = 852.11 	#: s 	Cs 	IR-A # can't simulate this one
# preset_line_t: float  = 1013.98   #: t 	Hg 	IR-A  # can't simulate this one

presets_line: list = [preset_line_h, preset_line_g, preset_line_F_, preset_line_F, preset_line_e,
                      preset_line_d, preset_line_D, preset_line_C, preset_line_C_,
                      preset_line_r, preset_line_A_]
"""all line presets in one list"""


# line combinations, used for the calculation of Abbe numbers
#######################################################################################################################

preset_lines_FDC: list = [preset_line_F,  preset_line_D, preset_line_C]    #: FDC line combination
preset_lines_FdC: list = [preset_line_F,  preset_line_d, preset_line_C]    #: FdC line combination
preset_lines_FeC: list = [preset_line_F,  preset_line_e, preset_line_C]    #: FeC line combination
preset_lines_F_eC_: list = [preset_line_F_, preset_line_e, preset_line_C_] #: F'eC' line combination

presets_lines: list = [preset_lines_FDC, preset_lines_FdC, preset_lines_FeC, preset_lines_F_eC_]
"""all line combination presets in one list"""

