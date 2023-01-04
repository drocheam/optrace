
import numpy as np  # calculations
import scipy.special  # error function and inverse
from typing import Any  # Any type

from .spectrum import Spectrum  # parent class
from .. import color # color conversions
from .. import misc  # random_from_distribution
from ..misc import PropertyChecker as pc  # check types and values


class LightSpectrum(Spectrum):

    quantity: str = "Spectral Power Density"   #: physical quantity
    unit: str = "W/nm"  #: physical unit

    spectrum_types: list[str] = [*Spectrum.spectrum_types, "Blackbody", "Histogram"]
    """possible spectrum types"""

    def __init__(self, 
                 spectrum_type: str = "Blackbody", 
                 T:             float = 5500, 
                 **sargs):
        """
        Create a LightSpectrum, compared to the Spectrum parent class this provides
        rendering of spectra and the generation of random wavelengths.
        As well as two additional types (Blackbody and Histogram)

        :param spectrum_type: spectrum type, one of "spectrum_types"
        :param T: blackbody temperature in Kelvin
        :param sargs:
        """
        self.T = T
        super().__init__(spectrum_type, **sargs)

    @staticmethod
    def render(wl:        np.ndarray,
               w:         np.ndarray,
               **kwargs)\
            -> 'LightSpectrum':
        """
        Render a LightSpectrum from a list of wavelengths and powers.
        The resulting LightSpectrum has type "Histogram" and unit W/nm

        :param wl: wavelength array
        :param w: weight array
        :param kwargs:  additional keyword arguments when creating the LightSpectrum
        :return: the LightSpectrum object
        """
        # init light spectrum
        spec = LightSpectrum("Histogram", **kwargs)

        # set bins to at least 101, scale using sqrt(N) above that
        # make sure it is an odd int, so we have a bin for the range center
        N = max(101, np.sqrt(np.count_nonzero(w)))
        N = 1 + 2 * (int(N) // 2)  # make an odd int

        # empty spectrum
        if not wl.shape[0]:
            spec._wls = color.wavelengths(N+1)
            spec._vals = np.zeros(N, dtype=np.float64)
        
        else:
            # wavelength range
            wl0, wl1 = np.min(wl), np.max(wl)

            # add +-1nm to range, but make sure it is inside visible range
            if np.abs(wl0 - wl1) < 1:
                wl0, wl1 = max(wl0-1, color.WL_BOUNDS[0]), min(wl0+1, color.WL_BOUNDS[1])

            spec._vals, spec._wls = np.histogram(wl, bins=N, weights=w, range=[wl0, wl1])
            spec._vals /= (spec._wls[1] - spec._wls[0])  # scale by delta lambda for W/nm

        return spec

    def random_wavelengths(self, N: int) -> np.ndarray:
        """
        Generate random wavelengths following the spectral distribution.

        :param N: number of wavelengths
        :return: wavelength array, shape (N,)
        """

        match self.spectrum_type:

            case "Monochromatic":
                wl = np.full(N, self.wl, dtype=np.float32)

            case ("Constant" | "Rectangle"):
                wl0 = color.WL_BOUNDS[0] if self.spectrum_type == "Constant" else self.wl0
                wl1 = color.WL_BOUNDS[1] if self.spectrum_type == "Constant" else self.wl1

                wl = misc.uniform(wl0, wl1, N)

            case "Lines":
                pc.check_type("LightSpectrum.lines", self.lines, np.ndarray | list)
                pc.check_type("LightSpectrum.line_vals", self.line_vals, np.ndarray | list)
                wl = misc.random_from_distribution(self.lines, self.line_vals, N, kind="discrete")

            case "Data":
                pc.check_type("LightSpectrum.wls", self._wls, np.ndarray | list)
                pc.check_type("LightSpectrum.vals", self._vals, np.ndarray | list)
                wl = misc.random_from_distribution(self._wls, self._vals, N)

            case "Gaussian":
                # although scipy.stats.truncnorm exists, we want to implement this ourselves 
                # so misc.uniform can be used as random generator

                # don't use the whole [0, 1] range for our random variable,
                # since we only simulate wavelengths in [380, 780], but gaussian pdf is unbound
                # therefore calculate minimal and maximal bound for the new random variable interval
                Xl = (1 + scipy.special.erf((color.WL_BOUNDS[0] - self.mu)/(np.sqrt(2)*self.sig)))/2
                Xr = (1 + scipy.special.erf((color.WL_BOUNDS[1] - self.mu)/(np.sqrt(2)*self.sig)))/2
                X = misc.uniform(Xl, Xr, N)

                # icdf of gaussian pdf
                wl = self.mu + np.sqrt(2)*self.sig * scipy.special.erfinv(2*X-1)

            # sadly there is no easy way to generate a icdf from a planck blackbody curve pdf
            # there are expressions for the integral (=cdf) of the curve, like in
            # https://www.researchgate.net/publication/346307633_Theoretical_Basis_of_Blackbody_Radiometry
            # Equation 3.49, but they involve infinite sums, that can't be inverted easily

            case ("Blackbody" | "Function" | "Histogram"):
                cnt = 4000 if self.spectrum_type == "Blackbody" else 10000

                wlr = color.wavelengths(cnt)
                wl = misc.random_from_distribution(wlr, self(wlr), N)

        return wl

    def __call__(self, wl: list | np.ndarray | float) -> np.ndarray:
        """
        Get the spectrum values

        :param wl: wavelength array
        :return: values at provided wavelengths
        """
        if self.spectrum_type == "Blackbody":
            wl_ = np.asarray_chkfinite(wl, dtype=np.float64)
            return color.blackbody(wl_, T=self.T)

        elif self.spectrum_type == "Histogram":
            pc.check_type("wls", self._wls, np.ndarray)
            pc.check_type("vals", self._vals, np.ndarray)
            assert len(self._wls) == len(self._vals) + 1

            # check values
            wl_ = np.asarray_chkfinite(wl, dtype=np.float64)

            # get bin indices, exclude ones that are zero or len(bins)
            ind = np.digitize(wl_, self._wls)
            ins = (ind > 0) & (ind < self._wls.shape[0])
            
            res = np.zeros_like(wl_)
            res[ins] = self._vals[ind[ins]-1]
           
            return res
        else:
            return super().__call__(wl)

    def get_xyz(self) -> np.ndarray:
        """
        Get the XYZ color of the spectrum

        :return: 3 element array of XYZ values
        """
        match self.spectrum_type:

            case "Monochromatic":
                wl = np.array([self.wl])
                spec = np.array([self.val])

            case "Lines":
                pc.check_type("LightSpectrum.lines", self.lines, np.ndarray | list)
                pc.check_type("LightSpectrum.line_vals", self.line_vals, np.ndarray | list)

                wl = self.lines
                spec = self.line_vals

            case _:
                cnt = 10000 if self.spectrum_type in ["Function", "Data", "Histogram"] else 4000
                wl = color.wavelengths(cnt)
                spec = self(wl)

        return color.xyz_from_spectrum(wl, spec)

    def get_color(self, rendering_intent="Ignore", clip=False) -> tuple[float, float, float]:
        """
        Get the sRGB color of the spectrum

        :param rendering_intent: rendering intent for the sRGB conversion
        :param clip: if values are clipped to the [0, 1] data range
        :return: tuple of 3 sRGB values
        """
        XYZ = np.array([[[*self.get_xyz()]]])
        RGB = color.xyz_to_srgb(XYZ, rendering_intent=rendering_intent, clip=clip)[0, 0]
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
