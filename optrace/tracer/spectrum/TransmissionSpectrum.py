
from optrace.tracer.spectrum.Spectrum import Spectrum
from optrace.tracer.spectrum.LightSpectrum import LightSpectrum

import optrace.tracer.Color as Color
import optrace.tracer.presets.LightSpectrum as presets_spectrum

import numpy as np


class TransmissionSpectrum(Spectrum):

    # don't allow all types of class Spectrum, especially not types "Lines" and "Monochromatic"
    spectrum_types = ["Constant", "Data", "Rectangle", "Gaussian", "Function"]

    quantity = "Transmission T"
    unit = ""

    def getXYZ(self):

        # illuminate the filter with daylight, get color of resulting spectrum
        func1 = lambda wl: Color.Illuminant(wl, "D65") * self(wl)
        lspec = LightSpectrum("Function", func=func1)

        return lspec.getXYZ()

    def getColor(self):

        XYZ = self.getXYZ()
        Y0 = presets_spectrum.D65.getXYZ()[0, 0, 1]

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
            self._checkNotAbove(key, val, 1)

        if key == "_vals" and isinstance(val, list | np.ndarray):
            if np.max(val) > 1:
                raise ValueError("all elements in vals need to be in range [0, 1].")

        if key == "func" and callable(val):
            wls = Color.wavelengths(1000)
            T = val(wls)
            if np.max(T) > 1:
                raise RuntimeError("Function func needs to return values in range [0, 1] over the visible range.")

        super().__setattr__(key, val)

