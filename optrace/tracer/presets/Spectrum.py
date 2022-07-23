import optrace.tracer.Color as Color  # tristimulus curves
from optrace.tracer.spectrum import Spectrum  # spectrum class


# Tristimulus Curves
#######################################################################################################################

X: Spectrum = Spectrum("Function", func=lambda x: Color.Tristimulus(x, "X"),
                       desc="X", long_desc="Tristimulus X Curve", quantity="Relative Response", unit="")
"""tristimulus X curve"""

Y: Spectrum = Spectrum("Function", func=lambda x: Color.Tristimulus(x, "Y"),
                       desc="Y", long_desc="Tristimulus Y Curve", quantity="Relative Response", unit="")
"""tristimulus Y curve"""

Z: Spectrum = Spectrum("Function", func=lambda x: Color.Tristimulus(x, "Z"),
                       desc="Z", long_desc="Tristimulus Z Curve", quantity="Relative Response", unit="")
"""tristimulus Z curve"""


tristimulus: list = [X, Y, Z]
"""tristimulus curve spectrum presets in one list"""


#######################################################################################################################

all_presets: list = [*tristimulus]
"""all spectrum presets in one list"""
