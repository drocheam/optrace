from __future__ import annotations

import numpy as np  # calculations
import scipy.special  # error function and inverse
from typing import Any  # Any type

from .spectrum import Spectrum  # parent class
from .. import color # color conversions
from .. import misc  # random_from_distribution
from ..misc import PropertyChecker as pc  # check types and values


class LightSpectrum(Spectrum):

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

        line_spec = spectrum_type in ["Monochromatic", "Lines"]
        unit = "W" if line_spec else "W/nm"
        quantity = "Spectral Power" if line_spec else "Spectral Power Density"

        super().__init__(spectrum_type, unit=unit, quantity=quantity, **sargs)

    @staticmethod
    def render(wl:        np.ndarray,
               w:         np.ndarray,
               **kwargs)\
            -> LightSpectrum:
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
            return self.val * color.normalized_blackbody(wl_, T=self.T)

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

    def xyz(self) -> np.ndarray:
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

    def color(self, rendering_intent="Ignore", clip=False) -> tuple[float, float, float]:
        """
        Get the sRGB color of the spectrum

        :param rendering_intent: rendering intent for the sRGB conversion
        :param clip: if values are clipped to the [0, 1] data range
        :return: tuple of 3 sRGB values
        """
        XYZ = np.array([[[*self.xyz()]]])
        RGB = color.xyz_to_srgb(XYZ, rendering_intent=rendering_intent, clip=clip)[0, 0]
        return RGB[0], RGB[1], RGB[2]

    def dominant_wavelength(self) -> float:
        """
        Dominant wavelength of the spectrum, that is the wavelength with the same hue.

        :return: dominant wavelength in nm if any exists, else np.nan
        """
        return color.dominant_wavelength(self.xyz())

    def complementary_wavelength(self) -> float:
        """
        Complementary wavelength of the spectrum, that is the wavelength with the opposite hue.

        :return: complementary wavelength in nm if any exists, else np.nan
        """
        return color.complementary_wavelength(self.xyz())

    def centroid_wavelength(self) -> float:
        """
        The centroid wavelength. This is the center of mass wavelength according to its spectral power.
        Another name would be "power-weighted average wavelength".

        :return: centroid wavelength in nm
        """
        # Calculation: see https://de.wikipedia.org/wiki/Schwerpunktwellenl%C3%A4nge
        match self.spectrum_type:

            case "Monochromatic":
                return self.wl

            case "Lines":
                lamb = np.array(self.lines)
                s = np.array(self.line_vals)
                return np.sum(s*lamb) / np.sum(s)
     
            case "Rectangle":
                return np.mean([self.wl0, self.wl1])
            
            case "Constant":
                return np.mean(color.WL_BOUNDS)

            # Blackbody, Function, Data, Gaussian
            case _ :
                wl = color.wavelengths(100000)
                s = self(wl)
                return np.trapz(wl*s) / np.trapz(s) if np.any(s > 0) else np.mean(color.WL_BOUNDS)

    def peak(self) -> float:
        """
        Peak value of the spectrum. 

        :return: peak value in self.unit
        """
       
        match self.spectrum_type:

            case ("Monochromatic" | "Gaussian" | "Rectangle" | "Constant"| "Blackbody"):
                return self.val

            case "Lines":
                return np.max(self.line_vals)

            case ("Histogram" | "Data"):
                return np.max(self._vals)

            case _:
                wl = color.wavelengths(100000)
                return np.max(self(wl))

    def peak_wavelength(self) -> float:
        """
        Peak wavelength of the spectrum. 
        For a spectrum with a constant maximum region (Constant, Rectangle) or multiple maxima the first one is returned.

        :return: peak wavelength in nm
        """
       
        match self.spectrum_type:

            case "Monochromatic":
                return self.wl

            case "Lines":
                return self.lines[np.argmax(self.line_vals)]

            case "Rectangle":
                return self.wl0

            case "Constant":
                return color.WL_BOUNDS[0]

            case "Gaussian":
                return self.mu  # True because it is enforced that mu is inside the visible range

            # also includes "Blackbody", as the peak might not lie inside the visible range
            case _:
                wl = color.wavelengths(100000)
                return wl[np.argmax(self(wl))]

    def fwhm(self) -> float:
        """
        Return the full width half maximum (FWHM).
        The smallest distance to the half height is calculated on both sides of the highest peak in the spectrum.

        :return: FWHM
        """
        match self.spectrum_type:

            case ("Monochromatic" | "Lines"):
                return 0

            case "Rectangle":
                return self.wl1 - self.wl0

            case "Constant":
                return color.WL_BOUNDS[1] - color.WL_BOUNDS[0]

            # default case (Function, Data, Histogram, Gaussian)
            # Gaussian is included here because it could be truncated
            case _:
                # find the peak
                wl = color.wavelengths(100000)
                spec = self(wl)
                ind = np.argmax(spec)
                specp = spec[ind]

                # right half crossing
                specr = spec[ind:]
                br = specr < 0.5*specp
                indr = ind + np.argmax(br) if np.any(br) else spec.shape[0] - 1
                
                # left half crossing
                specl = np.flip(spec[:ind])
                bl = specl < 0.5*specp
                indl = ind - np.argmax(bl) if np.any(bl) else 0

                # FWHM is the difference between crossings
                return wl[indr] - wl[indl]

    def _power(self, sensitivity) -> float:
        """
        Function for calculating the power of a spectrum with a wavelength dependent sensitivity function.

        :return: power
        """
        match self.spectrum_type:

            case "Monochromatic":
                return sensitivity(self.wl)*self.val

            case "Lines":
                return np.sum(sensitivity(self.lines)*self.line_vals)

            case "Histogram":
                dl = self._wls[1] - self._wls[0]
                wl2 = self._wls[:-1] + dl/2
                return np.sum(sensitivity(wl2)*self._vals)*dl
            
            case _:
                wl = color.wavelengths(100000)
                return np.trapz(sensitivity(wl)*self(wl)) * (wl[1] - wl[0])

    def power(self) -> float:
        """
        Power in W of the spectrum
        
        :return: power
        """
        return self._power(lambda x: np.ones_like(x))
    
    def luminous_power(self) -> float:
        """
        Luminous power in lm of the spectrum

        :return: luminous power
        """
        return self._power(lambda x: 683.0*color.y_observer(x))

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

        # a user defined data spectrum can't be constantly zero
        if key == "_vals" and val is not None and self.spectrum_type != "Histogram":
            vals = np.asarray_chkfinite(val)
            if np.any(vals < 0):
                raise ValueError("Values below zero in LightSpectrum.")

            if not np.any(vals > 0):
                raise ValueError("LightSpectrum can't be constantly zero.")

        super().__setattr__(key, val)
