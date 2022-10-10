
import copy  # deepcopy for dicts
from typing import Callable, Any  # Callable and Any type

import numpy as np  # calculations
import numexpr as ne  # faster calculations

from .. import color  # color conversions
from .. import misc  # faster calculations
from ..base_class import BaseClass  # parent class
from ..misc import PropertyChecker as pc  # check types and values



class Spectrum(BaseClass):

    spectrum_types: list[str] = ["Monochromatic", "Constant", "Data", "Lines",
                                 "Rectangle", "Gaussian", "Function"]
    """possible spectrum types. Can be changed by subclasses"""

    unit: str = ""
    """spectrum unit"""

    quantity: str = ""
    """spectrum quantity"""

    def __init__(self,
                 spectrum_type:     str = "Gaussian",
                 val:               float = 1.,
                 lines:             np.ndarray | list = None,
                 line_vals:         np.ndarray | list = None,
                 wl:                float = 550.,
                 wl0:               float = 400.,
                 wl1:               float = 600.,
                 wls:               np.ndarray = None,
                 vals:              np.ndarray = None,
                 func:              Callable[[np.ndarray], np.ndarray] = None,
                 mu:                float = 550.,
                 sig:               float = 50.,
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
        :param unit:
        :param quantity:
        :param func_args:
        :param kwargs:
        """

        self.spectrum_type = spectrum_type
        self.lines = lines
        self.line_vals = line_vals
        self.func_args = func_args if func_args is not None else {}
        self.func = func  # make sure this comes after func_args, so func is called correctly

        self.wl, self.wl0, self.wl1 = wl, wl0, wl1
        self.val, self.mu, self.sig = val, mu, sig
        self._wls, self._vals = wls, vals

        # hold infos for plotting etc. Defined in each subclass.
        self.unit = unit if unit is not None else self.unit
        self.quantity = quantity if quantity is not None else self.quantity

        super().__init__(**kwargs)
        self._new_lock = True

    def __call__(self, wl: list | np.ndarray | float) -> np.ndarray:
        """

        :param wl:
        :return:
        """
        # don't enable evaluating a function with discontinuities or where only some discrete values amount to anything
        # float errors would ruin anything anyway
        if not self.is_continuous():
            raise RuntimeError(f"Can't call discontinuous spectrum_type '{self.spectrum_type}'")

        wl_ = np.asarray_chkfinite(wl, dtype=np.float64)

        match self.spectrum_type:

            case "Constant":
                res = np.full_like(wl_, self.val, dtype=np.float64)

            case "Data":
                pc.check_type("Spectrum.wls", self._wls, np.ndarray | list)
                pc.check_type("Spectrum.vals", self._vals, np.ndarray | list)
                res = np.interp(wl_, self._wls, self._vals, left=0, right=0)

            case "Rectangle":
                res = np.zeros_like(wl, dtype=np.float64)
                res[(self.wl0 <= wl_) & (wl_ <= self.wl1)] = self.val

            case "Gaussian":
                val, mu, sig = self.val, self.mu, self.sig
                res = ne.evaluate("val*exp(-(wl-mu)**2/(2*sig**2))")

            case "Function":  # pragma: no branch
                pc.check_callable("Spectrum.func", self.func)
                res = self.func(wl_, **self.func_args)

        return res

    def is_continuous(self) -> bool:
        """:return: if the spectrum is continuous, thus not discrete"""
        return self.spectrum_type not in ["Lines", "Monochromatic"]

    def get_desc(self, fallback: str = None) -> str:
        """
        get description
        :param fallback: unused parameter
        :return:
        """
        fallback = str(self.val) if self.spectrum_type == "Constant" else self.spectrum_type
        return super().get_desc(fallback=fallback)

    def __setattr__(self, key: str, val: Any) -> None:
        """
        assigns the value of an attribute
        :param key: attribute name
        :param val: value to assign
        """
        match key:

            case "spectrum_type":
                pc.check_type(key, val, str)
                pc.check_if_element(key, val, self.spectrum_types)

            case ("lines" | "line_vals") if val is not None:
                pc.check_type(key, val, list | np.ndarray)
                val2 = np.asarray_chkfinite(val, dtype=np.float32)

                if val2.shape[0] == 0:
                    raise ValueError(f"'{key}' can't be empty.")

                if key == "lines" and ((wlo := np.min(val2)) < color.WL_BOUNDS[0] or (wlo := np.max(val2)) > color.WL_BOUNDS[1]):
                    raise ValueError(f"'lines' need to be inside visible range [{color.WL_BOUNDS[0]}nm, {color.WL_BOUNDS[1]}nm]"
                                     f", but got a value of {wlo}nm.")

                if key == "line_vals" and (lmin := np.min(val2)) < 0:
                    raise ValueError(f"line_vals must be all positive, but one value is {lmin}.")

                super().__setattr__(key, val2)
                return

            case "func_args":
                pc.check_type(key, val, dict)
                super().__setattr__(key, copy.deepcopy(val))  # enforce deepcopy of dict
                return

            case ("quantity" | "unit"):
                pc.check_type(key, val, str)

            case "func":
                pc.check_none_or_callable(key, val)
                if val is not None:
                    wls = color.wavelengths(10000)
                    T = val(wls, **self.func_args)
                    if np.min(T) < 0 or np.max(T) <= 0:
                        raise RuntimeError("Function func needs to return positive values over the visible range.")

            case ("_wls" | "_vals") if val is not None:
                
                pc.check_type(key, val, list | np.ndarray)
                val2 = np.asarray_chkfinite(val, dtype=np.float64)

                if key == "_wls":
                    pc.check_not_below("wls[0]", val[0], color.WL_BOUNDS[0])
                    pc.check_not_above("wls[-1]", val[-1], color.WL_BOUNDS[1])

                    if np.std(np.diff(val2)) > 1e-4 or np.any(np.diff(val2) <= 0) or (val[1]-val[0] < 1e-6):
                        raise ValueError("wls needs to be monotonically increasing with the same step size.")
                else: 
                    if (lmin := np.min(val2)) < 0:
                        raise ValueError(f"vals must be all positive, but one value is {lmin}")

                super().__setattr__(key, val2)
                return

            case ("wl" | "wl0" | "wl1" | "mu" | "sig" | "val"):
                pc.check_type(key, val, int | float)
                val = float(val)

                if key in ["wl", "wl0", "wl1", "mu"]:
                    pc.check_not_below(key, val, color.WL_BOUNDS[0])
                    pc.check_not_above(key, val, color.WL_BOUNDS[1])

                if key in ["val"]:
                    pc.check_above(key, val, 0)

                if key in ["sig"]:
                    pc.check_above(key, val, 0)

        super().__setattr__(key, val)
