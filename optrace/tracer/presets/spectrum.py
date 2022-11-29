from .. import color  # xyz observers
from ..spectrum import Spectrum  # spectrum class


# CIE 1931 XYZ observer color matching functions
#######################################################################################################################

x: Spectrum = Spectrum("Function", func=color.x_observer,
                       desc="x", long_desc="x observer", quantity="Relative Response", unit="")
"""CIE 1931 x curve"""

y: Spectrum = Spectrum("Function", func=color.y_observer,
                       desc="y", long_desc="y observer", quantity="Relative Response", unit="")
"""CIE 1931 y curve"""

z: Spectrum = Spectrum("Function", func=color.z_observer,
                       desc="z", long_desc="z observer", quantity="Relative Response", unit="")
"""CIE 1931 z curve"""


xyz_observers: list = [x, y, z]
"""x, y, z observer curve spectrum presets in one list"""


#######################################################################################################################

all_presets: list = [*xyz_observers]
"""all spectrum presets in one list"""
