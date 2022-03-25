
"""
RefractionIndex class:
Provides the creation and computation of constant or wavelength depended refraction indices
"""

import numpy as np
import numexpr as ne
import scipy.interpolate

from typing import Callable

# TODO copy method

class RefractionIndex:

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

        self.n_type = n_type
        self.n = float(n)
        self.func = func
        self.wls = np.array(wls, dtype=np.float32)
        self.ns = np.array(ns, dtype=np.float32)
        self.A, self.B, self.C, self.D = float(A), float(B), float(C), float(D)

        match n_type:

            # presets from https://en.wikipedia.org/wiki/Cauchy%27s_equation
            ################################################################
            case "SiO2":
                self.A, self.B, self.C, self.D  = 1.4580, 0.00354, 0., 0.

            case "BK7":
                self.A, self.B, self.C, self.D  = 1.5046, 0.00420, 0., 0.

            case "K5":
                self.A, self.B, self.C, self.D  = 1.5220, 0.00459, 0., 0.

            case "BaK4":
                self.A, self.B, self.C, self.D  = 1.5690, 0.00531, 0., 0.

            case "BaF10":
                self.A, self.B, self.C, self.D  = 1.6700, 0.00743, 0., 0.

            case "SF10":
                self.A, self.B, self.C, self.D  = 1.7280, 0.01342, 0., 0.
            ################################################################

            case "List":
                if ns is None or wls is None:
                    raise ValueError("n_type='List', but ns or wls not specified.")

            case "Function":
                if func is None:
                    raise ValueError("n_type='Function', but func not specified.")

            case _ if n_type not in ["Cauchy", "Constant"]:
                raise ValueError(f"Invalid refraction index type '{n_type}'")


        # check parameters

        if A < 1:
            raise ValueError(f"Parameter A needs to be >= 1.0, but is {A}.")

        if n < 1:
            raise ValueError(f"Refraction index n needs to be >= 1.0, but is {n}.")
   
        if wls is not None:
            if (wlo := np.min(wls)) < 380. or (wlo := np.max(wls)) > 780.:
                raise ValueError(f"Got wavelength value {wlo}nm outside visible range [380nm, 780nm]."
                                 "Make sure to pass the list in nm values, not in m values.")

        if ns is not None:
            if (nsm := np.min(ns)) < 1.0:
                raise ValueError(f"Refraction indices ns need to be >= 1.0, but minimum is {nsm}")

    def __call__(self, wl: np.ndarray | list | float) -> np.ndarray:
        """
        Returns the refractive index at specified wavelengths.
        Call on obj using obj(wavelengths).

        :param wl: wavelengths in nm (numpy 1D array)
        :return: array of refraction indices
        """
        wl_ = wl if isinstance(wl, np.ndarray) else np.array(wl)

        match self.n_type:

            case ("SiO2" | "BK7" | "K5" | "BaK4" | "BaF10" | "SF10" | "Cauchy"):
                l = wl_ * 1e-3  # parameters are specified in 1/µm^n, so convert nm wavelengths to µm
                A, B, C, D = self.A, self.B, self.C, self.D
                return ne.evaluate("A + B/l**2 + C/l**4 + D/l**6")

            case "List":
                func = scipy.interpolate.interp1d(self.wls, self.ns, bounds_error=True)
                return func(wl_)

            case "Constant":
                return np.full_like(wl_, self.n, dtype=np.float32)

            case "Function":
                return self.func(wl_)

            case _:
                raise ValueError(f"RefractionIndex function not defined for n_type '{n_type}'")


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

        return False

    def __ne__(self, other: 'RefractionIndex') -> bool:
        """
        Not equal operator.
        Call using operator !=, e.g. obj1 != obj2.

        :param other: RefractionIndex Object to compare to
        :return: bool value
        """
        return not self.__eq__(other)

