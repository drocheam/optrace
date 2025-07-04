
import copy  # deepcopy for dicts
from typing import Callable, Any, assert_never

import numpy as np  # calculations

from .. import color  # color conversions
from ..base_class import BaseClass  # parent class
from ...property_checker import PropertyChecker as pc  # check types and values
from ...global_options import global_options as go


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
                 func_args:         dict = {},
                 **kwargs):
        """
        Create a Spectrum object. 
        Most of the time this class isn't used, as there are specific subclasses.

        :param spectrum_type: one of "spectrum_types"
        :param val: factor for types "Rectangle", "Constant", "Gaussian" etc.
        :param lines: line wavelengths for "Lines" type
        :param line_vals: line values for "Lines" type
        :param wl: wavelength for "Monochromatic" type
        :param wl0: lower wavelength for "Rectangle" type
        :param wl1: upper wavelength for "Rectangle" type
        :param wls: wavelength list for "List" type
        :param vals: values for "List" type
        :param mu: center wavelength for "Gaussian" type
        :param sig: standard deviation for "Gaussian" type
        :param unit: unit string 
        :param quantity: quantity string
        :param func: spectrum function, must take a wavelength vector and return float values
        :param func_args: dict of optional keyword arguments for func
        :param kwargs: additional keyword arguments for the parent class
        """

        self.spectrum_type = spectrum_type
        self.lines = lines
        self.line_vals = line_vals
        self.func_args = func_args
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
        get spectrum values

        :param wl: wavelength array
        :return: values at provided wavelengths
        """
        # don't enable evaluating a function with discontinuities or where only some discrete values amount to anything
        # float errors would ruin anything anyway
        if not self.is_continuous():
            raise RuntimeError(f"Can't call discontinuous spectrum_type '{self.spectrum_type}'")

        wl_ = np.asarray_chkfinite(wl, dtype=np.float64)

        match self.spectrum_type:

            case "Constant":
                return np.broadcast_to(self.val, wl_.shape)

            case "Data":
                pc.check_type("Spectrum.wls", self._wls, np.ndarray | list)
                pc.check_type("Spectrum.vals", self._vals, np.ndarray | list)
                return np.interp(wl_, self._wls, self._vals, left=0, right=0)

            case "Rectangle":
                res = np.zeros_like(wl, dtype=np.float64)
                res[(self.wl0 <= wl_) & (wl_ <= self.wl1)] = self.val
                return res

            case "Gaussian":
                val, mu, sig = self.val, self.mu, self.sig
                return val*np.exp(-(wl-mu)**2/(2*sig**2))

            case "Function":
                pc.check_callable("Spectrum.func", self.func)
                return self.func(wl_, **self.func_args)

            case _:
                assert_never(self.spectrum_type)

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

                if key == "lines" and ((wlo := np.min(val2)) < go.wavelength_range[0]
                                       or (wlo := np.max(val2)) > go.wavelength_range[1]):
                    raise ValueError(f"'lines' need to be inside visible range [{go.wavelength_range[0]}nm, "
                                     f"{go.wavelength_range[1]}nm]"
                                     f", but got a value of {wlo}nm.")

                if key == "line_vals" and (lmin := np.min(val2)) < 0:
                    raise ValueError(f"line_vals must be all positive, but one value is {lmin}.")

                if key == "lines" and len(np.unique(val)) != len(val):
                    raise ValueError("All elements inside of 'lines' must be unique.")

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
                    pc.check_not_below("wls[0]", val[0], go.wavelength_range[0])
                    pc.check_not_above("wls[-1]", val[-1], go.wavelength_range[1])

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
                    pc.check_not_below(key, val, go.wavelength_range[0])
                    pc.check_not_above(key, val, go.wavelength_range[1])

                if key in ["val"]:
                    pc.check_above(key, val, 0)

                if key in ["sig"]:
                    pc.check_above(key, val, 0)

        super().__setattr__(key, val)
