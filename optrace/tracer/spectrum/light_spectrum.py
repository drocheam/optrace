
import numpy as np  # calculations
import scipy.special  # error function and inverse
import scipy.integrate
from typing import Any, Self, assert_never

from .spectrum import Spectrum  # parent class
from .. import color # color conversions
from .. import random  # inverse_transform_sampling
from ...property_checker import PropertyChecker as pc  # check types and values
from ...global_options import global_options as go


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
            -> Self:
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

        # set bins to at least 51, scale using sqrt(N) above that (since mean SNR grows with sqrt(N))
        # make sure it is an odd int, so we have a bin for the range center
        N = max(51, np.sqrt(np.count_nonzero(w))/2)
        N = 1 + 2 * (int(N) // 2)  # make an odd int

        # empty spectrum
        if not wl.shape[0]:
            spec._wls = color.wavelengths(N+1)
            spec._vals = np.zeros(N, dtype=np.float64)
        
        else:
            # wavelength range
            wl0, wl1 = wl.min(), wl.max()

            # add +-1nm to range, but make sure it is inside visible range
            if np.abs(wl0 - wl1) < 1:
                wl0, wl1 = max(wl0-1, go.wavelength_range[0]), min(wl0+1, go.wavelength_range[1])

            spec._vals, spec._wls = np.histogram(wl, bins=N, weights=w, range=[wl0, wl1]) # TODO use built-in method from np.histogram_bin_edges
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
                wl = np.broadcast_to(np.float32(self.wl), N)

            case ("Constant" | "Rectangle"):
                wl0 = go.wavelength_range[0] if self.spectrum_type == "Constant" else self.wl0
                wl1 = go.wavelength_range[1] if self.spectrum_type == "Constant" else self.wl1

                wl = random.stratified_interval_sampling(wl0, wl1, N)

            case "Lines":
                pc.check_type("LightSpectrum.lines", self.lines, np.ndarray | list)
                pc.check_type("LightSpectrum.line_vals", self.line_vals, np.ndarray | list)
                wl = random.inverse_transform_sampling(self.lines, self.line_vals, N, kind="discrete")

            case "Data":
                pc.check_type("LightSpectrum.wls", self._wls, np.ndarray | list)
                pc.check_type("LightSpectrum.vals", self._vals, np.ndarray | list)
                wl = random.inverse_transform_sampling(self._wls, self._vals, N)

            case "Gaussian":
                # although scipy.stats.truncnorm exists, we want to implement this ourselves 
                # so random.stratified_interval_sampling can be used as random generator

                # don't use the whole [0, 1] range for our random variable,
                # since we only simulate wavelengths in [380, 780], but gaussian pdf is unbound
                # therefore calculate minimal and maximal bound for the new random variable interval
                Xl = (1 + scipy.special.erf((go.wavelength_range[0] - self.mu)/(np.sqrt(2)*self.sig)))/2
                Xr = (1 + scipy.special.erf((go.wavelength_range[1] - self.mu)/(np.sqrt(2)*self.sig)))/2
                X = random.stratified_interval_sampling(Xl, Xr, N)

                # icdf of gaussian pdf
                wl = self.mu + np.sqrt(2)*self.sig * scipy.special.erfinv(2*X-1)

            # sadly there is no easy way to generate a icdf from a planck blackbody curve pdf
            # there are expressions for the integral (=cdf) of the curve, like in
            # https://www.researchgate.net/publication/346307633_Theoretical_Basis_of_Blackbody_Radiometry
            # Equation 3.49, but they involve infinite sums, that can't be inverted easily

            case ("Blackbody" | "Function" | "Histogram"):
                cnt = 4000 if self.spectrum_type == "Blackbody" else 10000

                wlr = color.wavelengths(cnt)
                wl = random.inverse_transform_sampling(wlr, self(wlr), N)

            case _:
                assert_never(self.spectrum_type)

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

    def color(self, 
              rendering_intent:     str = "Ignore", 
              clip:                 bool = False, 
              L_th:                 float = 0.0, 
              chroma_scale:         float = 0.0)\
            -> tuple[float, float, float]:
        """
        Get the sRGB color of the spectrum

        :param rendering_intent: rendering intent for the sRGB conversion
        :param clip: if values are clipped to the [0, 1] data range
        :return: tuple of 3 sRGB values
        """
        XYZ = np.array([[[*self.xyz()]]])
        RGB = color.xyz_to_srgb(XYZ, rendering_intent=rendering_intent, 
                                clip=clip, L_th=L_th, chroma_scale=chroma_scale)[0, 0]
        return float(RGB[0]), float(RGB[1]), float(RGB[2])

    def dominant_wavelength(self) -> float:
        """
        Dominant wavelength of the spectrum, that is the wavelength with the same hue.

        :return: dominant wavelength in nm if any exists, else np.nan
        """
        return float(color.dominant_wavelength(self.xyz()))

    def complementary_wavelength(self) -> float:
        """
        Complementary wavelength of the spectrum, that is the wavelength with the opposite hue.

        :return: complementary wavelength in nm if any exists, else np.nan
        """
        return float(color.complementary_wavelength(self.xyz()))

    def centroid_wavelength(self) -> float:
        """
        The centroid wavelength. This is the center of mass wavelength according to its spectral power.
        Another name would be "power-weighted average wavelength".

        :return: centroid wavelength in nm
        """
        # Calculation: see https://de.wikipedia.org/wiki/Schwerpunktwellenl%C3%A4nge
        match self.spectrum_type:

            case "Monochromatic":
                res = self.wl

            case "Lines":
                lamb = np.array(self.lines)
                s = np.array(self.line_vals)
                res = np.sum(s*lamb) / np.sum(s)
     
            case "Rectangle":
                res = np.mean([self.wl0, self.wl1])
            
            case "Constant":
                res = np.mean(go.wavelength_range)

            # Blackbody, Function, Data, Gaussian
            case _ :
                wl = color.wavelengths(100000)
                s = self(wl)
                res = scipy.integrate.trapezoid(wl*s) / scipy.integrate.trapezoid(s)\
                        if np.any(s > 0) else np.mean(go.wavelength_range)
    
        return float(res)

    def peak(self) -> float:
        """
        Peak value of the spectrum. 

        :return: peak value in self.unit
        """
       
        match self.spectrum_type:

            case ("Monochromatic" | "Gaussian" | "Rectangle" | "Constant"| "Blackbody"):
                res = self.val

            case "Lines":
                res = self.line_vals.max()

            case ("Histogram" | "Data"):
                res = self._vals.max()

            case _:
                wl = color.wavelengths(100000)
                res = self(wl).max()

        return float(res)

    def peak_wavelength(self) -> float:
        """
        Peak wavelength of the spectrum. 
        For a spectrum with a constant maximum region (Constant, Rectangle)
        or multiple maxima the first one is returned.

        :return: peak wavelength in nm
        """
        match self.spectrum_type:

            case "Monochromatic":
                res = self.wl

            case "Lines":
                res = self.lines[np.argmax(self.line_vals)]

            case "Rectangle":
                res = self.wl0

            case "Constant":
                res = go.wavelength_range[0]

            case "Gaussian":
                res = self.mu  # True because it is enforced that mu is inside the visible range

            # also includes "Blackbody", as the peak might not lie inside the visible range
            case _:
                wl = color.wavelengths(100000)
                res = wl[np.argmax(self(wl))]

        return float(res)

    def fwhm(self) -> float:
        """
        Return the full width half maximum (FWHM).
        The smallest distance to the half height is calculated on both sides of the highest peak in the spectrum.

        :return: FWHM
        """
        match self.spectrum_type:

            case ("Monochromatic" | "Lines"):
                res = 0

            case "Rectangle":
                res = self.wl1 - self.wl0

            case "Constant":
                res = go.wavelength_range[1] - go.wavelength_range[0]

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
                res = wl[indr] - wl[indl]

        return float(res)

    def _power(self, sensitivity) -> float:
        """
        Function for calculating the power of a spectrum with a wavelength dependent sensitivity function.

        :return: power
        """
        match self.spectrum_type:

            case "Monochromatic":
                res = sensitivity(self.wl)*self.val

            case "Lines":
                res = np.sum(sensitivity(self.lines)*self.line_vals)

            case "Histogram":
                dl = self._wls[1] - self._wls[0]
                wl2 = self._wls[:-1] + dl/2
                res = np.sum(sensitivity(wl2)*self._vals)*dl
            
            case _:
                wl = color.wavelengths(100000)
                res = scipy.integrate.trapezoid(sensitivity(wl)*self(wl)) * (wl[1] - wl[0])
        
        return float(res)

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
            vals = np.asarray_chkfinite(val, dtype=np.float64)

            if np.any(vals < 0):
                raise ValueError("Values below zero in LightSpectrum.")

            if not np.any(vals > 0):
                raise ValueError("LightSpectrum can't be constantly zero.")
        
            super().__setattr__(key, vals)
            return

        super().__setattr__(key, val)
