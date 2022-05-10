
"""
Filter class:
A filter is a surface with wavelength dependent or constant transmittance.
Useful for color filters or apertures.
"""

import numpy as np

from optrace.tracer.SObject import *
from optrace.tracer.Spectrum import *
from optrace.tracer.Surface import *  # for the Filter surface

class Filter(SObject):

    abbr = "F"

    def __init__(self, 
                 Surface:       Surface, 
                 pos:           (list | np.ndarray),
                 spectrum:      Spectrum = None,
                 **kwargs)\
            -> None:
        """
        Create a Filter object.

        :param Surface: Surface object
        :param pos: 3D position of Filter center (numpy array or list)
        :param filter_type: "Constant" or "Function" (string)
        :param tau: transmittance (float between 0 and 1), used for filter_type="Constant"
        :param func: transmittance function, used for filter_type="Function"
        """
        super().__init__(Surface, pos, **kwargs)

        self.spectrum = spectrum

        self._new_lock = True

    def __call__(self, wl: np.ndarray) -> np.ndarray:        
        """
        Return filter transmittance in range [0, 1] for specified wavelengths.

        :param wl: wavelengths
        :return: transmittance
        """
        return self.spectrum(wl)

    def getColor(self) -> tuple[float, float, float, float]:
        """
        Get sRGB color tuple from filter transmission curve

        :return: sRGB color tuple, with each channel in range [0, 1]
        """
        return self.spectrum.getColor(illu="D65")

    def __setattr__(self, key, val):
      
        if key == "spectrum" and not isinstance(val, Spectrum):
            raise TypeError("spectrum needs to be of type Spectrum.")

        super().__setattr__(key, val)

