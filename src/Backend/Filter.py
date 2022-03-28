
"""
Filter class:
A filter is a surface with wavelength dependent or constant transmittance.
Useful for color filters or apertures.
"""


import numpy as np

from Backend.Surface import Surface  # for the Filter surface
import Backend.Color as Color  # for the calculation of the filter color 
from Backend.SObject import SObject

from typing import Callable  # for function type hints


class Filter(SObject):

    def __init__(self, 
                 Surface:       Surface, 
                 pos:           (list | np.ndarray),
                 filter_type:   str = "Constant",
                 tau:           float = 0.,
                 func:          Callable[[np.ndarray], np.ndarray] = None)\
            -> None:
        """
        Create a Filter object.

        :param Surface: Surface object
        :param pos: 3D position of Filter center (numpy array or list)
        :param filter_type: "Constant" or "Function" (string)
        :param tau: transmittance (float between 0 and 1), used for filter_type="Constant"
        :param func: transmittance function, used for filter_type="Function"
        """

        # use a Surface copy, since we change its position in 3D space
        # self.Surface = Surface.copy()
        super().__init__(Surface, pos)

        self.filter_type = filter_type
        self.tau = float(tau)
        self.func = func

        match filter_type:
            case "Constant": 
                if not (0 <= tau <= 1):
                    raise ValueError("Transmittance tau needs to be inside range [0, 1]")

            case "Function":
                if func is None:
                    raise ValueError("filter_type='Function', but Function not specified")

            case _:
                raise ValueError(f"Invalid filter_type '{filter_type}'.")


    def __call__(self, wl: np.ndarray) -> np.ndarray:        
        """
        Return filter transmittance in range [0, 1] for specified wavelengths.

        :param wl: wavelengths
        :return: transmittance
        """
        match self.filter_type:
            case "Constant":
                return np.full_like(wl, self.tau, dtype=np.float32)

            case "Function":
                return self.func(wl)

            case _:
                raise ValueError(f"filter_type '{filter_type}' not implemented.")

    def getColor(self) -> tuple[float, float, float]:
        """
        Get sRGB color tuple from filter transmission curve

        :return: sRGB color tuple, with each channel in range [0, 1]
        """
        return tuple(Color.ColorUnderDaylight(self.__call__))

