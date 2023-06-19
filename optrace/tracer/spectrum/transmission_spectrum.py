
from typing import Any  # Any type

import numpy as np  # calculations

from .spectrum import Spectrum  # parent class
from .. import color  # color conversion and illuminants
from ..misc import PropertyChecker as pc  # check types and values


class TransmissionSpectrum(Spectrum):

    # don't allow all types of class Spectrum, especially not types "Lines" and "Monochromatic"
    spectrum_types: list[str] = ["Constant", "Data", "Rectangle", "Gaussian", "Function"]

    def __init__(self,
                 spectrum_type:     str = "Gaussian",
                 inverse:           bool = False,
                 **sargs):
        """
        Define a TransmissionSpectrum object.

        :param spectrum_type: spectrum type, one of spectrum_types
        :param inverse: if the function is inversed, meaning subtracted from 1. 
            A transmittance function becomes an absorptance function.
        :param sargs: additional parameters (See the Spectrum() constuctor)
        """

        self.inverse = inverse
    
        quantity: str = "Transmission T"  #: physical quantity
        unit: str = ""  #: physical unit

        super().__init__(spectrum_type, unit=unit, quantity=quantity, **sargs)

    def xyz(self) -> np.ndarray:
        """
        Get the Spectrum XYZ Color under daylight D65

        :return: 3 element XYZ color array
        """
        # illuminate the filter with daylight, get color of resulting spectrum
        wl = color.wavelengths(5000)
        spec = color.d65_illuminant(wl) * self(wl)
        return color.xyz_from_spectrum(wl, spec)

    def color(self, rendering_intent="Absolute", clip=True, L_th=0, sat_scale=None) -> tuple[float, float, float, float]:
        """
        Get the Spectrum sRGB color and the opacity.

        :param rendering_intent: rendering_intent for sRGB conversion
        :param clip: if values are clipped towards the sRGB data range
        :return: tuple of R, G, B, and the opacity, all with data range  [0, 1]
        """
        XYZ = self.xyz()
       
        # Y of daylight spectrum
        wl = color.wavelengths(5000)
        Y0 = color.xyz_from_spectrum(wl, color.d65_illuminant(wl))[1]

        # 1 - Yc/Y0 is the ratio of visible ambient light coming through the filter
        # gamma correct for non-linear human vision
        alpha = (1 - XYZ[1]/Y0) ** (1/2.4)
        XYZ /= Y0

        XYZ = np.array([[[*XYZ]]])  # needs to be 3D
        RGB = color.xyz_to_srgb(XYZ, rendering_intent=rendering_intent, clip=clip, L_th=L_th, sat_scale=sat_scale)[0, 0]

        return RGB[0], RGB[1], RGB[2], alpha

    def __call__(self, wl: list | np.ndarray | float) -> np.ndarray:
        """
        Get the spectrum values

        :param wl: wavelength array
        :return: values at provided wavelengths
        """

        if not self.inverse:
            return super().__call__(wl)
        else:
            return 1.0 - super().__call__(wl)

    def __setattr__(self, key: str, val: Any) -> None:
        """
        assigns the value of an attribute
        :param key: attribute name
        :param val: value to assign
        """
        # transmission needs to be 0 <= T <= 1, case "0 <=" already handled in super class
        if key in ["fact", "val"] and isinstance(val, int | float):
            pc.check_not_above(key, val, 1)

        if key == "_vals" and isinstance(val, list | np.ndarray):
            if np.max(val) > 1:
                raise ValueError("all elements in vals need to be in range [0, 1].")

        if key == "inverse":
            pc.check_type(key, val, bool)

        if key == "func" and callable(val):
            wls = color.wavelengths(1000)
            T = val(wls)
            if np.any(T > 1):
                raise RuntimeError("Function func needs to return values in range [0, 1] over the visible range.")

        super().__setattr__(key, val)
