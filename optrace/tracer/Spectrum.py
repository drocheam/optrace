
"""

"""

from typing import Callable
import numpy as np
import scipy.special

import optrace.tracer.Color as Color
import optrace.tracer.Misc as misc
from optrace.tracer.BaseClass import *

# TODO multiple gaussians? Specify mu, sig, val as list
class Spectrum(BaseClass):

    spectrum_types = ["Monochromatic", "Blackbody", "Constant", "Data", "Lines", "Rectangle", "Gaussian", "Function"]

    def __init__(self, 
                 spectrum_type:     str, 
                 lines:             list[float] | np.ndarray = None,
                 line_vals:         list[float] | np.ndarray = None,
                 wl:                float = 550.,
                 wl0:               float = 400.,
                 wl1:               float = 600.,
                 val:               float = 1.,
                 wls:               np.ndarray = None,
                 vals:              np.ndarray = None,
                 func:              Callable[[np.ndarray], np.ndarray] = None,
                 mu:                float = 550.,
                 sig:               float = 50.,
                 fact:              float = 1.,
                 T:                 float = 6504.,
                 unit:              str = "",
                 quantity:          str = "",
                 **kwargs):
        """"""
    
        self.spectrum_type = spectrum_type
        self.lines = lines
        self.line_vals = line_vals
        self.func = func

        self.wl, self.wl0, self.wl1 = wl, wl0, wl1
        self.val, self.fact, self.mu, self.sig, self.T = val, fact, mu, sig, T

        self._wls, self._vals = misc.uniform(wls, vals, 5000)
       
        # hold infos for plotting etc.
        self.quantity = quantity
        self.unit = unit

        super().__init__(**kwargs)
        self._new_lock = True

    def __call__(self, wl: np.ndarray) -> np.ndarray:
        """"""

        # don't enable evaluating a function with discontinuities or where only some discrete values amount to anything
        # float errors would ruin anything anyway
        if not self.isContinuous():
            raise RuntimeError(f"Can't call discontinuous spectrum_type '{self.spectrum_type}'")

        match self.spectrum_type:

            case "Blackbody":
                res = Color.Blackbody(wl, T=self.T)

            case "Constant":
                res = np.full_like(wl, self.val, dtype=np.float64)
       
            case "Data":
                res = np.interp(wl, self._wls, self._vals)

            case "Rectangle":
                res = np.zeros(wl.shape, dtype=np.float64)
                res[(self.wl0 < wl) & (wl < self.wl1)] = self.val

            case "Gaussian":
                fact, mu, sig = self.fact, self.mu, self.sig
                res = misc.calc("fact*exp(-(wl-mu)**2/(2*sig**2))")

            case "Function":
                if self.func is None:
                    raise RuntimeError("spectrum_type='Function' but parameter func not specified")
                res = self.func(wl)

            case _:
                raise RuntimeError(f"spectrum_type '{self.spectrum_type}' not handled.")

        return res

    def getXYZ(self, illu: str = None):
        """"""
        match self.spectrum_type:

            case "Monochromatic":
                wl = np.array([self.wl])
                spec = np.array([1.])

            case "Lines":
                wl = np.array(self.lines)
                spec = np.ones_like(wl)

            case _:
                cnt = 10000 if self.spectrum_type in ["Function", "Data"] else 4000
                wl = Color.wavelengths(cnt)
                spec = self(wl)

        if illu is not None:
            illu_spec = Color.Illuminant(wl, illu)
            Y0 = np.sum(illu_spec * Color.Tristimulus(wl, "Y"))
            spec *= illu_spec
        else:
            Y0 = None

        XYZ = np.array([[[np.sum(spec * Color.Tristimulus(wl, "X")),\
                          np.sum(spec * Color.Tristimulus(wl, "Y")),\
                          np.sum(spec * Color.Tristimulus(wl, "Z"))]]])
        return XYZ, Y0

    def getColor(self, illu: str = None) -> tuple[float, float, float, float]:
        """"""

        XYZ, Y0 = self.getXYZ(illu)

        if illu is not None:
            # 1 - Yc/Y0 is the ratio of visble ambient light coming through the filter
            # gamma correct for non-linear human vision
            alpha = (1 - XYZ[0, 0, 1]/Y0) ** (1/2.2)
            XYZ /= Y0
        else:
            alpha = 1

        RGB = Color.XYZ_to_sRGB(XYZ)[0, 0]

        return RGB[0], RGB[1], RGB[2], alpha

    def randomWavelengths(self, N: int) -> np.ndarray:
        """"""

        match self.spectrum_type:

            case "Monochromatic":
                wl = np.full(N, self.wl, dtype=np.float32)

            case ("Constant" | "Rectangle"):
                wl0 = Color.WL_MIN if self.spectrum_type == "Constant" else self.wl0
                wl1 = Color.WL_MAX if self.spectrum_type == "Constant" else self.wl1

                wl = np.random.uniform(wl0, wl1, N)

            case "Lines":
                wl = np.random.choice(self.lines, N, p=self.line_vals/np.sum(self.line_vals))

            case "Data":
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

    @staticmethod
    def makeSpectrum(wl:        np.ndarray,
                     w:         np.ndarray, 
                     N:         int = 2000, 
                     unit:      str = "W/nm", 
                     quantity:  str = "Spectral Density", 
                     **kwargs)\
            -> 'Spectrum':
        """"""
        spec = Spectrum("Data", unit=unit, quantity=quantity, **kwargs)
        
        wl0, wl1 = np.min(wl), np.max(wl)
        spec._wls = np.linspace(wl0, wl1, N)
        spec._vals = np.zeros(N, dtype=np.float64)

        wli = misc.calc("N/(wl1-wl0)*(wl-wl0)")
        
        # handle case where wavelength is exactly at the range edge (">=" for float errors)
        wli[wli >= N] -= 1

        np.add.at(spec._vals, wli.astype(int), w)

        return spec

    def isContinuous(self) -> bool:
        """"""
        return self.spectrum_type not in ["Lines", "Monochromatic"]

    def __setattr__(self, key, val0):
      
        # work on copies of ndarray and list
        val = val0.copy() if isinstance(val0, list | np.ndarray) else val0
        
        match key:
           
            case "spectrum_type":
                self._checkType(key, val, str)
                self._checkIfIn(key, val, self.spectrum_types)

            case ("lines" | "line_vals") if val is not None:
                self._checkType(key, val, list | np.ndarray)
                val = np.array(val, dtype=np.float32)

                if val.shape[0] == 0:
                    raise ValueError(f"'{key}' can't be empty.")

                if key == "lines" and ((wlo := np.min(val)) < Color.WL_MIN or (wlo := np.max(val)) > Color.WL_MAX):
                    raise ValueError(f"'lines' need to be inside visible range [{Color.WL_MIN}nm, {Color.WL_MAX}nm]"\
                                     f", but got a value of {wlo}nm.")

                if key == "line_vals" and (lmin := np.min(val))  < 0:
                    raise ValueError(f"line_vals must be all positive, but one value is {lmin}")

            case ("quantity" | "unit"):
                self._checkType(key, val, str)

            case "func":
                self._checkNoneOrCallable(key, val)

            case ("wl", "wl0", "wl", "T", "mu", "sig", "val", "fact"):
                self._checkType(key, val, int | float)
                val = float(val)

                if key in ["wl", "wl0", "wl1", "mu"]:
                    self._checkNotBelow(key, val, Color.WL_MIN)
                    self._checkNotAbove(key, val, Color.WL_MAX)

                if key in ["val", "fact", "sig"]:
                    self._checkNotBelow(key, val, 0)

        super().__setattr__(key, val)

