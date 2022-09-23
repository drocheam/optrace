# spectral lines
#######################################################################################################################
"""all lines specified with name, element, color"""

# spectral lines from https://en.wikipedia.org/wiki/Abbe_number
# i: float  = 365.01 	#: i    Hg 	UV-A  # can't simulate this one
h:   float = 404.66 	#: h 	Hg 	violet
g:   float = 435.84 	#: g 	Hg 	blue
F_:  float = 479.99 	#: F' 	Cd 	blue
F:   float = 486.13 	#: F 	H 	blue
e:   float = 546.07 	#: e 	Hg 	green
d:   float = 587.56     #: d 	He 	yellow
D:   float = 589.3      #: D 	Na 	yellow
C_:  float = 643.85 	#: C' 	Cd 	red
C:   float = 656.27 	#: C 	H 	red
r:   float = 706.52 	#: r 	He 	red
A_:  float = 768.2 	    #: A' 	K 	IR-A
# s: float  = 852.11 	#: s 	Cs 	IR-A  # can't simulate this one
# t: float  = 1013.98   #: t 	Hg 	IR-A  # can't simulate this one


all_lines: list = [h, g, F_, F, e, d, D, C_, C, r, A_]
"""all line presets in one list. Order by value."""


# line combinations, used for the calculation of Abbe numbers
#######################################################################################################################

FDC:   list = [F,  D, C]   #: FDC line combination
FdC:   list = [F,  d, C]   #: FdC line combination
FeC:   list = [F,  e, C]   #: FeC line combination
F_eC_: list = [F_, e, C_]  #: F'eC' line combination

# With the help of
# http://www.brucelindbloom.com/index.html?ColorCalcHelp.html
# and some finetuning
rgb:   list =  [464.3118, 549.1321, 611.2826]
"""dominant wavelengths of sRGB primaries,
 with order b, g, r. With image render mode 'Absolute' we get the primaries."""

all_line_combinations: list = [FDC, FdC, FeC, F_eC_, rgb]
"""all line combination presets in one list"""
