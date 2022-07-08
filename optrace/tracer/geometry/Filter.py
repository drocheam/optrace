
"""
Filter class:
A filter is a surface with wavelength dependent or constant transmittance.
Useful for color filters or apertures.
"""

import numpy as np  # ndarray type

from optrace.tracer.geometry.SObject import SObject  # parent class
from optrace.tracer.spectrum.TransmissionSpectrum import TransmissionSpectrum  # for transmission
from optrace.tracer.geometry.Surface import Surface  # Surface type


class Filter(SObject):

    abbr = "F"  # object abbreviation
    _allow_non_2D = False  # don't allow points or lines as surfaces

    def __init__(self, 
                 Surface:       Surface, 
                 pos:           (list | np.ndarray),
                 spectrum:      TransmissionSpectrum,
                 **kwargs)\
            -> None:
        """
        Create a Filter object.

        :param Surface: Surface object
        :param pos: 3D position of Filter center
        :param spectrum: transmission spectrum. Output range needs be inside [0, 1].
        """
        super().__init__(Surface, pos, **kwargs)

        self.spectrum = spectrum
        self._new_lock = True # new properties can't be assigned after this

    def __call__(self, wl: np.ndarray) -> np.ndarray:        
        """
        Return filter transmittance for specified wavelengths.

        :param wl: wavelengths (1D np.ndarray)
        :return: transmittance (1D np.ndarray)
        """
        return self.spectrum(wl)

    def getColor(self) -> tuple[float, float, float, float]:
        """
        Get Filter color under daylight (D65).

        :return: sRGBA color tuple, with each channel in range [0, 1]
        """
        return self.spectrum.getColor()

    def __setattr__(self, key: str, val) -> None:
      
        if key == "spectrum":
            self._checkType(key, val, TransmissionSpectrum)

        super().__setattr__(key, val)

