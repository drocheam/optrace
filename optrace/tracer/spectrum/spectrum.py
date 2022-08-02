
from typing import Callable  # Callable type
import numpy as np  # calculations
import numexpr as ne  # faster calculations
import copy  # deepcopy for dicts

import optrace.tracer.color as Color  # color conversions
import optrace.tracer.misc as misc  # faster calculations
from optrace.tracer.base_class import BaseClass  # parent class
from optrace.tracer.misc import PropertyChecker as pc  # check types and values


# TODO multiple gaussians? Specify mu, sig, val as list
class Spectrum(BaseClass):

    spectrum_types = ["Monochromatic", "Blackbody", "Constant", "Data", "Lines", "Rectangle", "Gaussian", "Function"]
    unit = ""
    quantity = ""

    def __init__(self, 
                 spectrum_type:     str, 
                 val:               float = 1.,
                 lines:             list[float] | np.ndarray = None,
                 line_vals:         list[float] | np.ndarray = None,
                 wl:                float = 550.,
                 wl0:               float = 400.,
                 wl1:               float = 600.,
                 wls:               np.ndarray = None,
                 vals:              np.ndarray = None,
                 func:              Callable[[np.ndarray], np.ndarray] = None,
                 mu:                float = 550.,
                 sig:               float = 50.,
                 fact:              float = 1.,
                 T:                 float = 6504.,
                 unit:              str = None,
                 quantity:          str = None,
                 func_args:         dict = None,
                 **kwargs):
        """

        :param spectrum_type:
        :param val:
        :param lines:
        :param line_vals:
        :param wl:
        :param wl0:
        :param wl1:
        :param wls:
        :param vals:
        :param func:
        :param mu:
        :param sig:
        :param fact:
        :param T:
        :param unit:
        :param quantity:
        :param func_args:
        :param kwargs:
        """
    
        self.spectrum_type = spectrum_type
        self.lines = lines
        self.line_vals = line_vals
        self.func_args = func_args if func_args is not None else dict()
        self.func = func  # make sure this comes after func_args, so func is called correctly

        self.wl, self.wl0, self.wl1 = wl, wl0, wl1
        self.val, self.fact, self.mu, self.sig, self.T = val, fact, mu, sig, T

        if wls is not None:
            pc.checkType("wls", wls, list | np.ndarray)
        if vals is not None:
            pc.checkType("vals", vals, list | np.ndarray)

        self._wls, self._vals = misc.uniform_resample(wls, vals, 5000) if wls is not None and vals is not None \
            else (wls, vals)
       
        # hold infos for plotting etc. Defined in each subclass.
        self.unit = unit if unit is not None else self.unit
        self.quantity = quantity if quantity is not None else self.quantity

        super().__init__(**kwargs)
        self._new_lock = True

    def __call__(self, wl: np.ndarray | list | float) -> np.ndarray:
        """

        :param wl:
        :return:
        """
        # don't enable evaluating a function with discontinuities or where only some discrete values amount to anything
        # float errors would ruin anything anyway
        if not self.is_continuous():
            raise RuntimeError(f"Can't call discontinuous spectrum_type '{self.spectrum_type}'")

        wl_ = wl if isinstance(wl, np.ndarray) else np.array(wl, dtype=np.float32)

        match self.spectrum_type:

            case "Blackbody":
                res = Color.blackbody(wl_, T=self.T)

            case "Constant":
                res = np.full_like(wl_, self.val, dtype=np.float64)
       
            case "Data":
                if self._wls is None or self._vals is None:
                    raise RuntimeError("spectrum_type='Data' but wls or vals not specified")
                res = np.interp(wl_, self._wls, self._vals, left=0, right=0)

            case "Rectangle":
                res = np.zeros_like(wl, dtype=np.float64)
                res[(self.wl0 <= wl_) & (wl_ <= self.wl1)] = self.val

            case "Gaussian":
                fact, mu, sig = self.fact, self.mu, self.sig
                res = ne.evaluate("fact*exp(-(wl-mu)**2/(2*sig**2))")

            case "Function":
                if self.func is None:
                    raise RuntimeError("spectrum_type='Function' but parameter func not specified")
                res = self.func(wl_, **self.func_args)

            case _:
                raise RuntimeError(f"spectrum_type '{self.spectrum_type}' not handled.")

        return res

    def is_continuous(self) -> bool:
        """"""
        return self.spectrum_type not in ["Lines", "Monochromatic"]

    def __eq__(self, other: 'Spectrum') -> bool:
        """Equal operator. Compares self to 'other'."""

        if type(self) is not type(other):
            return False

        elif self is other or (self.spectrum_type != "Data" and self.crepr() == other.crepr()):
            return True

        elif self.spectrum_type == "Data" and other.spectrum_type == "Data":
            if np.all(self._wls == other._wls) and np.all(self._vals == other._vals)\
               and self.quantity == other.quantity and self.unit == other.unit:
                return True

        return False

    def __ne__(self, other: 'Spectrum') -> bool:
        """Not equal operator. Compares self to 'other'."""
        return not self.__eq__(other)

    def get_desc(self) -> str:
        """"""
        fallback = str(self.val) if self.spectrum_type == "Constant" else self.spectrum_type
        return super().get_desc(fallback=fallback)
    
    def __setattr__(self, key, val):
      
        match key:
           
            case "spectrum_type":
                pc.checkType(key, val, str)
                pc.checkIfIn(key, val, self.spectrum_types)

            case ("lines" | "line_vals") if val is not None:
                pc.checkType(key, val, list | np.ndarray)
                val2 = np.array(val, dtype=np.float32)

                if val2.shape[0] == 0:
                    raise ValueError(f"'{key}' can't be empty.")

                if key == "lines" and ((wlo := np.min(val2)) < Color.WL_MIN or (wlo := np.max(val2)) > Color.WL_MAX):
                    raise ValueError(f"'lines' need to be inside visible range [{Color.WL_MIN}nm, {Color.WL_MAX}nm]"
                                     f", but got a value of {wlo}nm.")

                if key == "line_vals" and (lmin := np.min(val2)) < 0:
                    raise ValueError(f"line_vals must be all positive, but one value is {lmin}")

                super().__setattr__(key, val2)
                return

            case "func_args":
                pc.checkType(key, val, dict)
                super().__setattr__(key, copy.deepcopy(val))  # enforce deepcopy of dict
                return
                
            case ("quantity" | "unit"):
                pc.checkType(key, val, str)

            case "func":
                pc.checkNoneOrCallable(key, val)
                if val is not None:
                    wls = Color.wavelengths(1000)
                    T = val(wls, **self.func_args)
                    if np.min(T) < 0 or np.max(T) <= 0:
                        raise RuntimeError("Function func needs to return positive values over the visible range.")

            case "_vals" if val is not None:
                if (lmin := np.min(val)) < 0:
                    raise ValueError(f"vals must be all positive, but one value is {lmin}")

            case ("wl" | "wl0" | "wl1" | "T" | "mu" | "sig" | "val" | "fact"):
                pc.checkType(key, val, int | float)
                val = float(val)

                if key in ["wl", "wl0", "wl1", "mu"]:
                    pc.checkNotBelow(key, val, Color.WL_MIN)
                    pc.checkNotAbove(key, val, Color.WL_MAX)

                if key in ["val"]:
                    pc.checkNotBelow(key, val, 0)
                
                if key in ["fact", "sig", "T"]:
                    pc.checkAbove(key, val, 0)

        super().__setattr__(key, val)
