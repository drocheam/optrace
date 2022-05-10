
"""

"""

from typing import Callable
import copy
import numpy as np

import optrace.tracer.Color as Color
import optrace.tracer.Misc as misc
from optrace.tracer.BaseClass import *

# TODO multiple gaussians? Specify mu, sig, val as list
class Spectrum(BaseClass):

    spectrum_types = ["Monochromatic", "Blackbody", "Constant", "Lines", "Rectangle", "Gaussian", "Function"]

    def __init__(self, 
                 spectrum_type:     str, 
                 lines:             list[float] | np.ndarray = None,
                 line_vals:         list[float] | np.ndarray = None,
                 wl:                float = 550.,
                 wl0:               float = 400.,
                 wl1:               float = 600.,
                 val:               float = 1.,
                 func:              Callable[[np.ndarray], np.ndarray] = None,
                 mu:                float = 550.,
                 sig:               float = 50.,
                 fact:              float = 1.,
                 T:                 float = 6504.):
        """"""
    
        self.spectrum_type = spectrum_type
        self.lines = lines
        self.line_vals = line_vals
        self.func = func

        self.wl, self.wl0, self.wl1 = wl, wl0, wl1
        self.val, self.mu, self.sig, self.T = val, mu, sig, T

        self._new_lock = True

    def __call__(self, wl: np.ndarray) -> np.ndarray:
        """"""

        match self.spectrum_type:

            case "Monochromatic":
                res = np.zeros(wl.shape, dtype=np.float64)
                res[wl == self.wl] = self.val
                
            case "Blackbody":
                res = Color.Blackbody(wl, T=self.T)

            case "Constant":
                res = np.full_like(wl, self.val, dtype=np.float64)
        
            case "Rectangle":
                res = np.zeros(wl.shape, dtype=np.float64)
                res[(self.wl0 < wl) & (wl < self.wl1)] = self.val

            case "Gaussian":
                val, mu, sig = self.val, self.mu, self.sig
                res = misc.calc("val*exp(-(wl-mu)**2/(2*sig**2))")

            case "Function":
                if self.func is None:
                    raise RuntimeError("spectrum_type='Function' but parameter func not specified")
                res = self.func(wl)

            case "Lines":
                res = np.full_like(wl, self.val, dtype=np.float64)

                for i, line in enumerate(self.lines):
                    res[wl == line] = self.line_vals[i]

            case _:
                raise RuntimeError(f"spectrum_type '{self.spectrum_type}' not handled.")

        return res

    def getColor(self, illu: str = None) -> tuple[float, float, float, float]:
        """"""
        match self.spectrum_type:

            case "Monochromatic":
                wl = np.array([self.wl])
                spec = np.array([1.])

            case "Lines":
                wl = np.array(self.lines)
                spec = np.ones_like(wl)

            case ("Blackbody" | "Function" | "Gaussian" | "Rectangle" | "Constant"):
                cnt = 10000 if self.spectrum_type == "Function" else 4000
                wl = Color.wavelengths(cnt)
                spec = self(wl)

            case _:
                raise RuntimeError(f"spectrum_type '{self.spectrum_type}' not handled in getColor()")


        if illu is not None:
            illu_spec = Color.Illuminant(wl, illu)
            Y0 = np.sum(illu_spec * Color.Tristimulus(wl, "Y"))
            spec *= illu_spec

        XYZ = np.array([[[np.sum(spec * Color.Tristimulus(wl, "X")),\
                          np.sum(spec * Color.Tristimulus(wl, "Y")),\
                          np.sum(spec * Color.Tristimulus(wl, "Z"))]]])

        if illu is not None:
            # 1 - Yc/Y0 is the ratio of visble ambient light coming through the filter
            # gamma correct for non-linear human vision
            alpha = (1 - XYZ[0, 0, 1]/Y0) ** (1/2.2)
            XYZ /= Y0
        else:
            alpha = 1

        RGB = Color.XYZ_to_sRGB(XYZ)[0, 0]

        return RGB[0], RGB[1], RGB[2], alpha

    # TODO use erf for Gaussian
    def randomWavelengths(self, N: int) -> np.ndarray:
        """"""

        match self.spectrum_type:

            case "Monochromatic":
                wavelengths = np.full(N, self.wl, dtype=np.float32)

            case ("Constant" | "Rectangle"):
                wl0 = Color.WL_MIN if self.spectrum_type == "Constant" else self.wl0
                wl1 = Color.WL_MAX if self.spectrum_type == "Constant" else self.wl1

                wavelengths = np.random.uniform(wl0, wl1, N)

            case "Lines":
                wavelengths = np.random.choice(self.lines, N, p=self.line_vals/np.sum(self.line_vals))

            case ("Blackbody" | "Function" | "Gaussian"):
                cnt = 10000 if self.spectrum_type == "Function" else 4000

                wl = Color.wavelengths(cnt)
                wavelengths = misc.random_from_distribution(wl, self(wl), N)
            
            case _:
                raise RuntimeError(f"spectrum_type '{self.spectrum_type}' not handled.")
      
        return wavelengths

    def __setattr__(self, key, val0):
      
        # work on copies of ndarray and list
        val = val0.copy() if isinstance(val0, list | np.ndarray) else val0
        
        match key:
            
            case "lines" if val is not None:
                if not isinstance(val, list | np.ndarray):
                    raise TypeError("lines needs to be of type list or np.ndarray")

                val = np.array(val, dtype=np.float32)
                if val.shape[0] == 0:
                    raise ValueError("'lines' can't be empty.")

                if (wlo := np.min(val)) < Color.WL_MIN or (wlo := np.max(val)) > Color.WL_MAX:
                    raise ValueError(f"'lines' need to be inside visible range [{Color.WL_MIN}nm, {Color.WL_MAX}nm]"\
                                     f", but got a value of {wlo}nm.")

        super().__setattr__(key, val)

