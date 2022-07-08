
# spectral lines
#######################################################################################################################

# spectral lines from https://en.wikipedia.org/wiki/Abbe_number
# preset_line_i  = 365.01 	# i     Hg 	UV-A # can't simulate this one
preset_line_h  = 404.66 	# h 	Hg 	violet
preset_line_g  = 435.84 	# g 	Hg 	blue
preset_line_F_ = 479.99 	# F' 	Cd 	blue
preset_line_F  = 486.13 	# F 	H 	blue
preset_line_e  = 546.07 	# e 	Hg 	green
preset_line_d  = 587.56 	# d 	He 	yellow
preset_line_D  = 589.3 	    # D 	Na 	yellow
preset_line_C_ = 643.85 	# C' 	Cd 	red
preset_line_C  = 656.27 	# C 	H 	red
preset_line_r  = 706.52 	# r 	He 	red
preset_line_A_ = 768.2 	    # A' 	K 	IR-A
# preset_line_s  = 852.11 	# s 	Cs 	IR-A # can't simulate this one
# preset_line_t  = 1013.98    # t 	Hg 	IR-A  # can't simulate this one

presets_line = [# preset_line_i, 
                preset_line_h, preset_line_g, preset_line_F_, preset_line_F, preset_line_e,
                preset_line_d, preset_line_D, preset_line_C, preset_line_C_,
                preset_line_r, preset_line_A_, 
                # preset_line_s, preset_line_t
                ]

# line combinations, used for the calculation of Abbe numbers
#######################################################################################################################

preset_lines_FDC   = [preset_line_F,  preset_line_D, preset_line_C]
preset_lines_FdC   = [preset_line_F,  preset_line_d, preset_line_C]
preset_lines_FeC   = [preset_line_F,  preset_line_e, preset_line_C]
preset_lines_F_eC_ = [preset_line_F_, preset_line_e, preset_line_C_]

presets_lines = [preset_lines_FDC, preset_lines_FdC, preset_lines_FeC, preset_lines_F_eC_]

