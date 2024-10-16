from .. import color  # xyz observers
from ..spectrum import Spectrum  # spectrum class


# CIE 1931 XYZ observer color matching functions
#######################################################################################################################

x: Spectrum = Spectrum("Function", func=color.x_observer,
                       desc="x", long_desc="CIE 1931 2° x observer", quantity="Relative Response", unit="")
"""CIE 1931 2° colorimetric standard observer x curve"""

y: Spectrum = Spectrum("Function", func=color.y_observer,
                       desc="y", long_desc="CIE 1931 2° y observer", quantity="Relative Response", unit="")
"""CIE 1931 2° colorimetric standard observer y curve"""

z: Spectrum = Spectrum("Function", func=color.z_observer,
                       desc="z", long_desc="CIE 1931 2° z observer", quantity="Relative Response", unit="")
"""CIE 1931 2° colorimetric standard observer z curve"""


xyz_observers: list = [x, y, z]
"""x, y, z observer curve spectrum presets in one list"""


#######################################################################################################################

all_presets: list = [*xyz_observers]
"""all spectrum presets in one list"""
