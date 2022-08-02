

from optrace.tracer.spectrum.spectrum import Spectrum  # parent class
import optrace.tracer.color as Color  # color conversions
import optrace.tracer.misc as misc  # faster calculations
from optrace.tracer.misc import PropertyChecker as pc  # check types and values

import numpy as np  # calculations
import numexpr as ne  # faster calculations
import scipy.special  # error function and inverse


class LightSpectrum(Spectrum):

    quantity = "Spectral Power Density"
    unit = "W/nm"

    @staticmethod
    def render(wl:        np.ndarray,
               w:         np.ndarray,
               N:         int = 2000,
               **kwargs)\
            -> 'LightSpectrum':
        """

        :param wl:
        :param w:
        :param N:
        :param kwargs:
        :return:
        """
        spec = LightSpectrum("Data", **kwargs)

        wl0, wl1 = np.min(wl), np.max(wl)

        # bounds equal => add +-1nm, but make sure it is inside visible range
        if wl0 == wl1:
            wl0, wl1 = max(wl0-1, Color.WL_MIN), min(wl0+1, Color.WL_MAX)

        spec._wls = np.linspace(wl0, wl1, N)
        spec._vals = np.zeros(N, dtype=np.float64)

        wli = ne.evaluate("N/(wl1-wl0)*(wl-wl0)")
       
        # handle case where wavelength is exactly at the range edge (">=" for float errors)
        wli[wli >= N] -= 1

        np.add.at(spec._vals, wli.astype(int), w)
        spec._vals *= N/(wl1-wl0)

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
                wl0 = Color.WL_MIN if self.spectrum_type == "Constant" else self.wl0
                wl1 = Color.WL_MAX if self.spectrum_type == "Constant" else self.wl1

                wl = np.random.uniform(wl0, wl1, N)

            case "Lines":
                if self.lines is None:
                    raise RuntimeError("Spectrum lines not defined.")
                if self.line_vals is None:
                    raise RuntimeError("Spectrum line_vals not defined.")

                wl = np.random.choice(self.lines, N, p=self.line_vals/np.sum(self.line_vals))

            case "Data":
                if self._wls is None or self._vals is None:
                    raise RuntimeError("spectrum_type='Data' but wls or vals not specified")

                wl = misc.random_from_distribution(self._wls, self._vals, N)

            case "Gaussian":
                # don't use the whole [0, 1] range for our random variable,
                # since we only simulate wavelengths in [380, 780], but gaussian pdf is unbound
                # therefore calculate minimal and maximal bound for the new random variable interval
                Xl = (1 + scipy.special.erf((Color.WL_MIN - self.mu)/(np.sqrt(2)*self.sig)))/2
                Xr = (1 + scipy.special.erf((Color.WL_MAX - self.mu)/(np.sqrt(2)*self.sig)))/2
                X = np.random.uniform(Xl, Xr, N)
               
                # icdf of gaussian pdf
                wl = self.mu + np.sqrt(2)*self.sig * scipy.special.erfinv(2*X-1)

            # sadly there is no easy way to generate a icdf from a planck blackbody curve pdf
            # there are expressions for the integral (=cdf) of the curve, like in
            # https://www.researchgate.net/publication/346307633_Theoretical_Basis_of_Blackbody_Radiometry
            # Equation 3.49, but they involve infinite sums, that can't be inverted easily

            case ("Blackbody" | "Function"):
                cnt = 10000 if self.spectrum_type == "Function" else 4000

                wlr = Color.wavelengths(cnt)
                wl = misc.random_from_distribution(wlr, self(wlr), N)

            case _:
                raise RuntimeError(f"spectrum_type '{self.spectrum_type}' not handled.")

        return wl

    def get_xyz(self) -> np.ndarray:
        """"""
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
                wl = Color.wavelengths(cnt)
                spec = self(wl)

        XYZ = np.array([[[np.sum(spec * Color.tristimulus(wl, "X")),
                          np.sum(spec * Color.tristimulus(wl, "Y")),
                          np.sum(spec * Color.tristimulus(wl, "Z"))]]])

        return XYZ

    def get_color(self) -> tuple[float, float, float]:
        """

        :return:
        """

        XYZ = self.get_xyz()
        RGB = Color.XYZ_to_sRGB(XYZ)[0, 0]
        return RGB[0], RGB[1], RGB[2]

    def __setattr__(self, key, val) -> None:
        """"""

        # "Constant" with val == 0 not allowed for light spectrum
        if key == "val" and isinstance(val, int | float):
            pc.checkAbove(key, val, 0)

        super().__setattr__(key, val)
