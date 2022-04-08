
"""
Filter class:
A filter is a surface with wavelength dependent or constant transmittance.
Useful for color filters or apertures.
"""


import numpy as np
from typing import Callable  # for function type hints

from optrace.tracer.SObject import *
from optrace.tracer.Surface import *  # for the Filter surface
import optrace.tracer.Color as Color  # for the calculation of the filter color 


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

        super().__init__(Surface, pos)

        self.name = "Filter"
        self.short_name = "F"
        
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

        self._new_lock = True

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
        # Estimate color of an object under daylight using the transmittance/reflectance spectrum.

        wl      = Color.wavelengths(1000)
        bb_spec = Color.Illuminant(wl, "D65")
        prod    = bb_spec * self.__call__(wl)

        Y0 = np.sum(bb_spec * Color.Tristimulus(wl, "Y"))

        Xc = np.sum(prod * Color.Tristimulus(wl, "X"))
        Yc = np.sum(prod * Color.Tristimulus(wl, "Y"))
        Zc = np.sum(prod * Color.Tristimulus(wl, "Z"))

        XYZn = np.array([[[Xc, Yc, Zc]]] / Y0)

        RGBL = Color.XYZ_to_sRGBLinear(XYZn, normalize=False)
        RGB = Color.sRGBLinear_to_sRGB(RGBL, normalize=False)[0, 0]

        # 1 - Yc/Y0 is the ratio of visble ambient light coming through the filter
        # gamma correct for non-linear human vision
        alpha = (1 - Yc/Y0) ** (1/2.2)

        return tuple(RGB), alpha


    def crepr(self):

        """

        """

        return [self.FrontSurface.crepr(), self.filter_type, self.tau, id(self.func)]
