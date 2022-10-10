
from typing import Any  # Any type

import numpy as np  # calculations

from .spectrum import Spectrum  # parent class
from .. import color  # color conversion and illuminants
from ..misc import PropertyChecker as pc  # check types and values


class TransmissionSpectrum(Spectrum):

    # don't allow all types of class Spectrum, especially not types "Lines" and "Monochromatic"
    spectrum_types: list[str] = ["Constant", "Data", "Rectangle", "Gaussian", "Function"]

    quantity: str = "Transmission T"
    unit: str = ""

    def get_xyz(self) -> np.ndarray:
        """

        :return:
        """
        # illuminate the filter with daylight, get color of resulting spectrum
        wl = color.wavelengths(5000)
        spec = color.d65_illuminant(wl) * self(wl)
        return color.xyz_from_spectrum(wl, spec)

    def get_color(self, rendering_intent="Absolute", clip=True) -> tuple[float, float, float, float]:
        """

        :param rendering_intent:
        :param clip:
        :return:
        """
        XYZ = self.get_xyz()
       
        # Y of daylight spectrum
        wl = color.wavelengths(5000)
        Y0 = color.xyz_from_spectrum(wl, color.d65_illuminant(wl))[1]

        # 1 - Yc/Y0 is the ratio of visible ambient light coming through the filter
        # gamma correct for non-linear human vision
        alpha = (1 - XYZ[1]/Y0) ** (1/2.4)
        XYZ /= Y0

        XYZ = np.array([[[*XYZ]]])  # needs to be 3D
        RGB = color.xyz_to_srgb(XYZ, rendering_intent=rendering_intent, clip=clip)[0, 0]

        return RGB[0], RGB[1], RGB[2], alpha

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

        if key == "func" and callable(val):
            wls = color.wavelengths(1000)
            T = val(wls)
            if np.any(T > 1):
                raise RuntimeError("Function func needs to return values in range [0, 1] over the visible range.")

        super().__setattr__(key, val)
