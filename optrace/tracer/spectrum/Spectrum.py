
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
    unit = ""
    quantity = ""

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
                 unit:              str = None,
                 quantity:          str = None,
                 **kwargs):
        """"""
    
        self.spectrum_type = spectrum_type
        self.lines = lines
        self.line_vals = line_vals
        self.func = func

        self.wl, self.wl0, self.wl1 = wl, wl0, wl1
        self.val, self.fact, self.mu, self.sig, self.T = val, fact, mu, sig, T

        if wls is not None:
            self._checkType("wls", wls, list | np.ndarray)
        if vals is not None:
            self._checkType("vals", vals, list | np.ndarray)

        self._wls, self._vals = misc.uniform_resample(wls, vals, 5000) if wls is not None and vals is not None else (wls, vals)
       
        # hold infos for plotting etc. Defined in each subclass.
        self.unit = unit if unit is not None else self.unit
        self.quantity = quantity if quantity is not None else self.quantity

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
                if self._wls is None or self._vals is None:
                    raise RuntimeError("spectrum_type='Data' but wls or vals not specified")
                res = np.interp(wl, self._wls, self._vals, left=0, right=0)

            case "Rectangle":
                res = np.zeros(wl.shape, dtype=np.float64)
                res[(self.wl0 <= wl) & (wl <= self.wl1)] = self.val

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

    def isContinuous(self) -> bool:
        """"""
        return self.spectrum_type not in ["Lines", "Monochromatic"]

    def __eq__(self, other: 'Spectrum') -> bool:
        """Equal operator. Compares self to 'other'."""

        if self is other or self.crepr() == other.crepr():
            return True

        elif self.spectrum_type == "Data" and other.spectrum_type == "Data":
            if self._wls == other._wls and self._vals == other._vals\
               and self.quantity == other.quantity and self.unit == other.unit:
                return True

        return False

    def __ne__(self, other: 'Spectrum') -> bool:
        """Not equal operator. Compares self to 'other'."""
        return not self.__eq__(other)

    def getDesc(self):
        """"""
        fallback = str(self.val) if self.spectrum_type == "Constant" else self.spectrum_type
        return super().getDesc(fallback=fallback)
    
    def __setattr__(self, key, val):
      
        match key:
           
            case "spectrum_type":
                self._checkType(key, val, str)
                self._checkIfIn(key, val, self.spectrum_types)

            case ("lines" | "line_vals") if val is not None:
                self._checkType(key, val, list | np.ndarray)
                val2 = np.array(val, dtype=np.float32)

                if val2.shape[0] == 0:
                    raise ValueError(f"'{key}' can't be empty.")

                if key == "lines" and ((wlo := np.min(val2)) < Color.WL_MIN or (wlo := np.max(val2)) > Color.WL_MAX):
                    raise ValueError(f"'lines' need to be inside visible range [{Color.WL_MIN}nm, {Color.WL_MAX}nm]"\
                                     f", but got a value of {wlo}nm.")

                if key == "line_vals" and (lmin := np.min(val2))  < 0:
                    raise ValueError(f"line_vals must be all positive, but one value is {lmin}")

                super().__setattr__(key, val2)
                return

            case ("quantity" | "unit"):
                self._checkType(key, val, str)

            case "func":
                self._checkNoneOrCallable(key, val)
                if val is not None:
                    wls = Color.wavelengths(1000)
                    T = val(wls)
                    if np.min(T) < 0 or np.max(T) <= 0:
                        raise RuntimeError("Function func needs to return positive values over the visible range.")

            case "_vals" if val is not None:
                if (lmin := np.min(val))  < 0:
                    raise ValueError(f"vals must be all positive, but one value is {lmin}")

            case ("wl" | "wl0" | "wl1" | "T" | "mu" | "sig" | "val" | "fact"):
                self._checkType(key, val, int | float)
                val = float(val)

                if key in ["wl", "wl0", "wl1", "mu"]:
                    self._checkNotBelow(key, val, Color.WL_MIN)
                    self._checkNotAbove(key, val, Color.WL_MAX)

                if key in ["val"]:
                    self._checkNotBelow(key, val, 0)
                
                if key in ["fact", "sig", "T"]:
                    self._checkAbove(key, val, 0)

        super().__setattr__(key, val)

