
from optrace.tracer.spectrum.spectrum import Spectrum  # parent class
from optrace.tracer.spectrum.light_spectrum import LightSpectrum  # illumination

import optrace.tracer.color as Color  # color conversion and illuminants
import optrace.tracer.presets.light_spectrum as presets_spectrum  # D65 LightSpectrum

import numpy as np  # calculations
from optrace.tracer.misc import PropertyChecker as pc  # check types and values


class TransmissionSpectrum(Spectrum):

    # don't allow all types of class Spectrum, especially not types "Lines" and "Monochromatic"
    spectrum_types = ["Constant", "Data", "Rectangle", "Gaussian", "Function"]

    quantity = "Transmission T"
    unit = ""

    def get_xyz(self) -> np.ndarray:
        """

        :return:
        """
        # illuminate the filter with daylight, get color of resulting spectrum
        func1 = lambda wl: Color.illuminant(wl, "D65") * self(wl)
        lspec = LightSpectrum("Function", func=func1)

        return lspec.get_xyz()

    def get_color(self) -> tuple[float, float, float, float]:
        """

        :return:
        """
        XYZ = self.get_xyz()
        Y0 = presets_spectrum.D65.get_xyz()[0, 0, 1]

        # 1 - Yc/Y0 is the ratio of visible ambient light coming through the filter
        # gamma correct for non-linear human vision
        alpha = (1 - XYZ[0, 0, 1]/Y0) ** (1/2.2)
        XYZ /= Y0
    
        RGB = Color.XYZ_to_sRGB(XYZ)[0, 0]

        return RGB[0], RGB[1], RGB[2], alpha

    def __setattr__(self, key, val) -> None:
        """"""
        # transmission needs to be 0 <= T <= 1, case "0 <=" already handled in super class
        if key in ["fact", "val"] and isinstance(val, int | float):
            pc.checkNotAbove(key, val, 1)

        if key == "_vals" and isinstance(val, list | np.ndarray):
            if np.max(val) > 1:
                raise ValueError("all elements in vals need to be in range [0, 1].")

        if key == "func" and callable(val):
            wls = Color.wavelengths(1000)
            T = val(wls)
            if np.max(T) > 1:
                raise RuntimeError("Function func needs to return values in range [0, 1] over the visible range.")

        super().__setattr__(key, val)
