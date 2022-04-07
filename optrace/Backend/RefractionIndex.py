
"""
RefractionIndex class:
Provides the creation and computation of constant or wavelength depended refraction indices
"""

import numpy as np
import scipy.interpolate
import optrace.Backend.Misc as misc
import optrace.Backend.Color as Color

from typing import Callable

# TODO copy method

class RefractionIndex:

    n_types = ["Constant", "Cauchy", "List", "Function", "SiO2","BK7", "K5", "BaK4", "BaF10", "SF10"]
        
    def __init__(self,
                 n_type: str,
                 n:     float = 1.0,
                 wls:   np.ndarray = None,
                 ns:    np.ndarray = None,
                 func:  Callable[[np.ndarray], np.ndarray] = None,
                 A:     float = 1.0,
                 B:     float = 0.0,
                 C:     float = 0.0,
                 D:     float = 0.0)\
            -> None:
        """
        Create a RefractionIndex object of type "ntype".

        :param n_type: "Constant", "Cauchy", "List", "Function" or preset "SiO2","BK7", "K5", "BaK4", "BaF10", "SF10" (string)
        :param n: refraction index for ntype="Constant" (float)
        :param wls: wavelength list in nm for ntype="List" (numpy 1D array)
        :param ns: refraction index list for ntype="List" (numpy 1D array)
        :param func: function for ntype="Function", input needs to be in nm
        :param A: A coefficient for ntype="Cauchy"
        :param B: B coefficient for ntype="Cauchy" in 1/µm^2
        :param C: C coefficient for ntype="Cauchy" in 1/µm^4
        :param D: D coefficient for ntype="Cauchy" in 1/µm^6
        """

        # self._new_lock = False

        self.n_type = n_type
        self.n = n
        self.func = func
        self.wls, self.ns = wls, ns
        self.A, self.B, self.C, self.D = A, B, C, D

        match n_type:

            # presets from https://en.wikipedia.org/wiki/Cauchy%27s_equation
            ##########################################################################
            case "SiO2":    self.A, self.B, self.C, self.D  = 1.4580, 0.00354, 0., 0.
            case "BK7":     self.A, self.B, self.C, self.D  = 1.5046, 0.00420, 0., 0.
            case "K5":      self.A, self.B, self.C, self.D  = 1.5220, 0.00459, 0., 0.
            case "BaK4":    self.A, self.B, self.C, self.D  = 1.5690, 0.00531, 0., 0.
            case "BaF10":   self.A, self.B, self.C, self.D  = 1.6700, 0.00743, 0., 0.
            case "SF10":    self.A, self.B, self.C, self.D  = 1.7280, 0.01342, 0., 0.
            ##########################################################################

        self._new_lock = True

    def __call__(self, wl: np.ndarray | list | float) -> np.ndarray:
        """
        Returns the refractive index at specified wavelengths.
        Call on obj using obj(wavelengths).

        :param wl: wavelengths in nm (numpy 1D array)
        :return: array of refraction indices
        """
        wl_ = wl if isinstance(wl, np.ndarray) else np.array(wl, dtype=np.float32)

        match self.n_type:

            case ("SiO2" | "BK7" | "K5" | "BaK4" | "BaF10" | "SF10" | "Cauchy"):
                # parameters are specified in 1/µm^n, so convert nm wavelengths to µm with factor 1e-3
                return misc.calc("A + B/l**2 + C/l**4 + D/l**6", l=wl_*1e-3, A=self.A, B=self.B, C=self.C, D=self.D)

            case "List":
                if self.ns is None or self.wls is None:
                    raise RuntimeError("n_type='List', but ns or wls not specified.")
                func = scipy.interpolate.interp1d(self.wls, self.ns, bounds_error=True)
                return func(wl_)

            case "Constant":
                return np.full_like(wl_, self.n, dtype=np.float32)

            case "Function":
                if self.func is None:
                    raise RuntimeError("n_type='Function', but func not specified.")
                return self.func(wl_)

            case _:
                raise RuntimeError(f"{self.n_type} not handled here.")

    def __setattr__(self, key, val):
       
        if key == "n_type" and (not isinstance(val, str) or val not in self.n_types):
            raise ValueError(f"n_type needs to be one of {self.n_types}")

        if key in ["n", "A", "B", "C", "D"]:
            if not isinstance(val, int | float):
                raise TypeError(f"Parameter {key} needs to be of type int or float")

            val = float(val)

            if key in ["n", "A"] and val < 1:
                raise ValueError(f"Parameter {key} needs to be >= 1.0, but is {val}.")

        if key in ["wls", "ns"]:

            if not isinstance(val, list | np.ndarray | None):
                raise TypeError(f"{key} needs to be of type np.ndarray or list")

            if val is not None:
                val = np.array(val, dtype=np.float32)

                if key == "wls":
                    if (wlo := np.min(val)) < Color.WL_MIN or (wlo := np.max(val)) > Color.WL_MAX:
                        raise ValueError(f"Got wavelength value {wlo}nm outside visible range [{Color.WL_MIN}nm, {Color.WL_MAX}nm]."
                                         "Make sure to pass the list in nm values, not in m values.")

                if key == "ns":
                    if (nsm := np.min(val)) < 1.0:
                        raise ValueError(f"Refraction indices ns need to be >= 1.0, but minimum is {nsm}")

        if "_new_lock" in self.__dict__ and self._new_lock and key not in self.__dict__:
            raise ValueError(f"Invalid property {key}")

        self.__dict__[key] = val


    def __eq__(self, other: 'RefractionIndex') -> bool:
        """
        Returns if two RefractionIndex Objects are equal.
        Call using operator ==, e.g. obj1 == obj2.

        :param other: RefractionIndex Object to compare to
        :return: bool value
        """

        # other is a reference
        if self is other:
            return True

        if self.n_type == other.n_type:
            
            match self.n_type:

                case "Constant":
                    return self.n == other.n

                case ("SiO2" | "BK7" | "K5" | "BaK4" | "BaF10" | "SF10" | "Cauchy"):
                    return self.A == other.A and self.B == other.B and self.C == other.C and self.D == other.D

                case "List":
                    return self.wls == other.wls and self.ns == other.ns

                # other.func is a reference to the same function
                case "Function":
                    return self.func is other.func

                case _:
                    raise RuntimeError(f"{self.n_type} not handled here.")

        return False

    def __ne__(self, other: 'RefractionIndex') -> bool:
        """
        Not equal operator.
        Call using operator !=, e.g. obj1 != obj2.

        :param other: RefractionIndex Object to compare to
        :return: bool value
        """
        return not self.__eq__(other)

    def crepr(self):
        """
        """
        return [self.n, self.A, self.B, self.C, self.D, self.n_type,
                     id(self.wls), id(self.ns), id(self.func)]

    def __str__(self):
        """gets called when print(obj) or repr(obj) gets called"""
        return f"{self.__class__.__name__} at {hex(id(self))} with {self.__dict__}"
