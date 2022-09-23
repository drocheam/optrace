from .. import color  # tristimulus curves
from ..spectrum import Spectrum  # spectrum class


# Tristimulus Curves
#######################################################################################################################

x: Spectrum = Spectrum("Function", func=color.x_tristimulus,
                       desc="x", long_desc="Tristimulus x Curve", quantity="Relative Response", unit="")
"""tristimulus x curve"""

y: Spectrum = Spectrum("Function", func=color.y_tristimulus,
                       desc="y", long_desc="Tristimulus y Curve", quantity="Relative Response", unit="")
"""tristimulus y curve"""

z: Spectrum = Spectrum("Function", func=color.z_tristimulus,
                       desc="z", long_desc="Tristimulus z Curve", quantity="Relative Response", unit="")
"""tristimulus z curve"""


tristimulus: list = [x, y, z]
"""tristimulus curve spectrum presets in one list"""


#######################################################################################################################

all_presets: list = [*tristimulus]
"""all spectrum presets in one list"""
