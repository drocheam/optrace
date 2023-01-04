
import numpy as np  # ndarray type
from typing import Any  # Any type

from .element import Element  # parent class
from .surface import Surface  # Surface type
from ..spectrum.transmission_spectrum import TransmissionSpectrum  # for transmission
from ..misc import PropertyChecker as pc  # check types and values


class Filter(Element):

    abbr: str = "F"  #: object abbreviation
    _allow_non_2D: bool = False  # don't allow points or lines as surfaces

    def __init__(self,
                 surface:       Surface,
                 pos:           (list | np.ndarray),
                 spectrum:      TransmissionSpectrum,
                 **kwargs)\
            -> None:
        """
        Create a Filter object. A filter is a surface with wavelength dependent or constant transmittance.
        Useful for color filters or apertures.


        :param Surface: Surface object
        :param pos: 3D position of Filter center
        :param spectrum: transmission spectrum. Output range needs be inside [0, 1].
        :param kwargs: additional keyword arguments for parent classes
        """
        super().__init__(surface, pos, **kwargs)

        self.spectrum = spectrum
        self._new_lock = True  # new properties can't be assigned after this

    def __call__(self, wl: np.ndarray) -> np.ndarray:
        """
        Return filter transmittance for specified wavelengths,
        pass-through of spectrum.__call__, see this method for details
        """
        return self.spectrum(wl)

    def get_color(self, rendering_intent="Absolute", clip=True) -> tuple[float, float, float, float]:
        """
        Get Filter color under daylight (D65).
        pass-through of spectrum.get_color, see this method for details
        """
        return self.spectrum.get_color(rendering_intent, clip)

    def __setattr__(self, key: str, val: Any) -> None:
        """
        Assigns the value of an attribute.
        Also performs type checking.

        :param key: attribute name
        :param val: value to assign
        """
        if key == "spectrum":
            pc.check_type(key, val, TransmissionSpectrum)

        super().__setattr__(key, val)
