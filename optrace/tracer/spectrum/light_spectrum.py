

import numpy as np  # calculations
import numexpr as ne  # faster calculations
import scipy.special  # error function and inverse
from typing import Any  # Any type

from .spectrum import Spectrum  # parent class
from .. import color # color conversions
from .. import misc  # random_from_distribution
from ..misc import PropertyChecker as pc  # check types and values


class LightSpectrum(Spectrum):

    quantity: str = "Spectral Power Density"
    unit: str = "W/nm"

    spectrum_types: list[str] = [*Spectrum.spectrum_types, "Blackbody"]
    """possible spectrum types"""

    def __init__(self, 
                 spectrum_type: str = "Blackbody", 
                 T:             float = 5500, 
                 **sargs):
        """
        
        :param spectrum_type:
        :param T:
        :param sargs:
        """
        self.T = T
        super().__init__(spectrum_type, **sargs)

    @staticmethod
    def render(wl:        np.ndarray,
               w:         np.ndarray,
               N:         int = 3000,
               **kwargs)\
            -> 'LightSpectrum':
        """
         LightSpectrum has unit W/nm

        :param wl:
        :param w:
        :param N:
        :param kwargs:
        :return:
        """
        spec = LightSpectrum("Data", **kwargs)

        # empty spectrum
        if not wl.shape[0]:
            spec._wls = np.linspace(color.WL_MIN, color.WL_MAX, N)
            spec._vals = np.zeros(N, dtype=np.float64)
            return spec
        
        wl0, wl1 = np.min(wl), np.max(wl)

        # bounds equal => add +-1nm, but make sure it is inside visible range
        if wl0 == wl1:
            wl0, wl1 = max(wl0-1, color.WL_MIN), min(wl0+1, color.WL_MAX)

        spec._vals, spec._wls = np.histogram(wl, bins=N, weights=w, range=[wl0, wl1])
        spec._wls = spec._wls[:-1]
        spec._vals /= (spec._wls[1] - spec._wls[0])  # scale by delta lambda for W/nm

        return spec

    def random_wavelengths(self, N: int) -> np.ndarray:
        """

        :param N:
        :return:
        """

        match self.spectrum_type:

            case "Monochromatic":
                wl = np.full(N, self.wl, dtype=np.float32)

            case ("Constant" | "Rectangle"):
                wl0 = color.WL_MIN if self.spectrum_type == "Constant" else self.wl0
                wl1 = color.WL_MAX if self.spectrum_type == "Constant" else self.wl1

                wl = misc.uniform(wl0, wl1, N)

            case "Lines":
                if self.lines is None:
                    raise RuntimeError("Spectrum lines not defined.")
                if self.line_vals is None:
                    raise RuntimeError("Spectrum line_vals not defined.")

                wl = misc.random_from_distribution(self.lines, self.line_vals, N, kind="discrete")

            case "Data":
                if self._wls is None or self._vals is None:
                    raise RuntimeError("spectrum_type='Data' but wls or vals not specified")

                wl = misc.random_from_distribution(self._wls, self._vals, N)

            case "Gaussian":
                # don't use the whole [0, 1] range for our random variable,
                # since we only simulate wavelengths in [380, 780], but gaussian pdf is unbound
                # therefore calculate minimal and maximal bound for the new random variable interval
                Xl = (1 + scipy.special.erf((color.WL_MIN - self.mu)/(np.sqrt(2)*self.sig)))/2
                Xr = (1 + scipy.special.erf((color.WL_MAX - self.mu)/(np.sqrt(2)*self.sig)))/2
                X = misc.uniform(Xl, Xr, N)

                # icdf of gaussian pdf
                wl = self.mu + np.sqrt(2)*self.sig * scipy.special.erfinv(2*X-1)

            # sadly there is no easy way to generate a icdf from a planck blackbody curve pdf
            # there are expressions for the integral (=cdf) of the curve, like in
            # https://www.researchgate.net/publication/346307633_Theoretical_Basis_of_Blackbody_Radiometry
            # Equation 3.49, but they involve infinite sums, that can't be inverted easily

            case ("Blackbody" | "Function"):
                cnt = 10000 if self.spectrum_type == "Function" else 4000

                wlr = color.wavelengths(cnt)
                wl = misc.random_from_distribution(wlr, self(wlr), N)

        return wl

    def __call__(self, wl: list | np.ndarray | float) -> np.ndarray:
        """

        :param wl:
        :return:
        """
        if self.spectrum_type == "Blackbody":
            wl_ = np.asarray_chkfinite(wl, dtype=np.float32)
            return color.blackbody(wl_, T=self.T)
        else:
            return super().__call__(wl)

    def get_xyz(self) -> np.ndarray:
        """

        :return:
        """
        match self.spectrum_type:

            case "Monochromatic":
                wl = np.array([self.wl])
                spec = np.array([1.])

            case "Lines":
                if self.lines is None:
                    raise RuntimeError("Spectrum lines not defined.")

                if self.line_vals is None:
                    raise RuntimeError("Spectrum line_vals not defined.")

                wl = self.lines
                spec = self.line_vals

            case _:
                cnt = 10000 if self.spectrum_type in ["Function", "Data"] else 4000
                wl = color.wavelengths(cnt)
                spec = self(wl)

        xyz = np.array([[[np.sum(spec * color.x_tristimulus(wl)),
                          np.sum(spec * color.y_tristimulus(wl)),
                          np.sum(spec * color.z_tristimulus(wl))]]])

        return xyz

    def get_color(self) -> tuple[float, float, float]:
        """

        :return:
        """

        XYZ = self.get_xyz()
        RGB = color.xyz_to_srgb(XYZ)[0, 0]
        return RGB[0], RGB[1], RGB[2]

    def __setattr__(self, key: str, val: Any) -> None:
        """
        assigns the value of an attribute
        :param key: attribute name
        :param val: value to assign
        """
        # "Constant" with val == 0 not allowed for light spectrum
        if key == "val" and isinstance(val, int | float):
            pc.check_above(key, val, 0)

        if key == "T":
            pc.check_type(key, val, int | float)
            val = float(val)
            pc.check_above(key, val, 0)

        super().__setattr__(key, val)
