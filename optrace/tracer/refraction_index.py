
"""
RefractionIndex class:
Provides the creation and computation of constant or wavelength depended refraction indices
"""
from typing import Any  # Any type

import numpy as np  # calculations
import numexpr as ne  # faster calculations

from .presets import spectral_lines as Lines  # spectral lines for Abbe number
from .spectrum import Spectrum  # parent class
from . import color  # for visible wavelength range
from .misc import PropertyChecker as pc  # check types and values


# TODO how to handle wavelengths outside the range of data provided
# absorb?


class RefractionIndex(Spectrum):

    # Refraction Index Models:
    # see https://doc.comsol.com/5.5/doc/com.comsol.help.roptics/roptics_ug_optics.6.46.html

    n_types: list[str] = ["Abbe", "Cauchy", "Conrady", "Sellmeier", "Constant", "Data", "Function"]
    spectrum_types: list[str] = n_types  # alias

    quantity: str = "Refraction Index n"
    unit: str = ""

    def __init__(self,
                 n_type:    str = "Constant",
                 n:         float = 1.0,
                 coeff:     list = None,
                 lines:     np.ndarray | list = None,
                 V:         float = None,
                 **kwargs)\
            -> None:
        """
        Create a RefractionIndex object of type "n_type".

        See https://doc.comsol.com/5.5/doc/com.comsol.help.roptics/roptics_ug_optics.6.46.html
        for the model equations.

        Cauchy coefficients are specified in order [A, B, C, D] with units µm^n with n = 0, 2, 4, 6
        Sellmeier are specified in order [A1, B1, A2, B2, A3, B3, A4, B4], Cs are specified in µm^2
        Conrady coefficients are specified as [A, B, C] with units 1, µm, µm**3.5

        In Abbe mode a curve is estimated using Abbe number V, center refractive index n and 3 spectral lines 'lines'.
        n at center wavelength lines[1].

        :param n_type: "Constant", "Cauchy", "Sellmeier", "Function" or presets RefractionIndex.materials
        :param n: refraction index for ntype="Constant" (float)
        :param func: function for n_type="Function", input needs to be in nm
        :param V: Abbe number for n_type="Abbe"
        :param lines: spectral lines to use for n_type="Abbe",
                      list of 3 wavelengths [short wavelength, center wavelength, long wavelength]
        """
        self.spectrum_type = n_type  # needs to be first so coeff gets set correctly
        self.coeff = coeff
        self.V = V

        lines = lines if lines is not None else Lines.FDC

        super().__init__(n_type, val=n, lines=lines, **kwargs)

        self._new_lock = True

    def __call__(self, wl: list | np.ndarray | float) -> np.ndarray:
        """
        Returns the refractive index at specified wavelengths.
        Call on obj using obj(wavelengths).

        :param wl: wavelengths in nm (numpy 1D array)
        :return: array of refraction indices
        """
        wl_ = np.asarray_chkfinite(wl, dtype=np.float64)

        if self.spectrum_type in ["Cauchy", "Conrady", "Sellmeier"]:
            pc.check_type("RefractionIndex.coeff", self.coeff, np.ndarray | list)

        match self.spectrum_type:

            case "Cauchy":
                # parameters are specified in 1/µm^n, so convert nm wavelengths to µm with factor 1e-3
                l = wl_*1e-3
                A, B, C, D, = tuple(self.coeff)[:4]
                ns = ne.evaluate("A + B/l**2 + C/l**4 + D/l**6")

            case "Abbe":
                # estimate a refractive index curve from abbe number
                # note: many different curves can have the same number, these is just an estimation for a possible one

                # use wl in um
                l = 1e-3*np.array(self.lines)
                nc = self.val

                # compromise between Cauchy (d=0) and Hetzberger (d=0.028)
                d = 0.014

                # solve for B and A from Abbe Number and center refraction index
                B = 1/self.V * (nc - 1) / (1/(l[0]**2-d) - 1/(l[2]**2-d))
                A = nc - B/(l[1]**2-d)

                ns = A + B/((1e-3*wl_)**2-d)

            case "Conrady":
                l = wl_*1e-3
                A, B, C = tuple(self.coeff)[:3]
                ns = ne.evaluate("A + B/l + C/l**3.5")

            case "Sellmeier":
                wl2 = ne.evaluate("(wl_*1e-3)**2") # since Cs are specified in µm, not in nm
                A1, B1, A2, B2, A3, B3, A4, B4 = tuple(self.coeff)
                ns = ne.evaluate("sqrt(1 + A1*wl2/(wl2-B1) + A2*wl2/(wl2-B2) + A3*wl2/(wl2-B3) + A4*wl2/(wl2-B4))")

            case _:
                ns = super().__call__(wl_)

        if (nm := np.min(ns)) < 1:
            raise RuntimeError(f"RefractionIndex below 1 with value {nm}.")

        return ns

    def __eq__(self, other: Any) -> bool:
        """
        Equal operator. Compares self to 'other'.
        :param other:
        :return:
        """

        if type(self) is not type(other):
            return False

        elif self is other or (self.spectrum_type != "Data" and self.crepr() == other.crepr()):
            return True

        elif self.spectrum_type == "Data" and other.spectrum_type == "Data":
            if np.all(self._wls == other._wls) and np.all(self._vals == other._vals)\
               and self.quantity == other.quantity and self.unit == other.unit:
                return True

        return False

    def __ne__(self, other: Any) -> bool:
        """Not equal operator. Compares self to 'other'.
        :param other:
        :return:
        """
        return not self.__eq__(other)

    def __setattr__(self, key: str, val: Any) -> None:
        """
        assigns the value of an attribute
        :param key: attribute name
        :param val: value to assign
        """
        match key:

            case "val" if isinstance(val, int | float):
                pc.check_not_below(key, val, 1)

            case "coeff" if val is not None:

                pc.check_type(key, val, list)

                match self.spectrum_type:
                    case "Cauchy":      cnt = 4
                    case "Conrady":     cnt = 3
                    case "Sellmeier":   cnt = 8
                    case _:             cnt = 8

                if len(val) > cnt:
                    raise ValueError(f"{key} needs to be a list with maximum {cnt} numeric coefficients")

                # pad to 8 coeffs
                val2 = val.copy()
                val2 += [0] * (8 - len(val))

                super().__setattr__(key, val2)
                return

                # validity of coeffs is checked in __call__
                # otherwise it would be possible that the coeffs seem invalid,
                # but the n_type is changed afterwards, making them valid

            case "_vals" if val is not None:
                if np.min(val) < 1:
                    raise ValueError("all vals values needs to be at least 1.")

            case "lines" if isinstance(val, list | np.ndarray):

                if len(val) != 3:
                    raise ValueError("Property 'lines' for n_type='Abbe' needs to have exactly 3 elements")

                if not val[0] < val[1] < val[2]:
                    raise ValueError("The values of property 'lines' need to be ascending.")

            case "func" if callable(val):
                wls = color.wavelengths(1000)
                n = val(wls, **self.func_args)
                if np.min(n) < 1:
                    raise ValueError("Function func needs to output values >= 1 over the whole visible range.")

            case "V" if val is not None:
                pc.check_type(key, val, float | int)
                pc.check_above(key, val, 0)
                np.asarray_chkfinite(val)

        super().__setattr__(key, val)

    def get_abbe_number(self, lines: list = None) -> float:
        """
        Calculates the Abbe Number. The spectral lines can be overwritten with the parameter.
        Otherwise the RefractionIndex.lines parameter is used from its initialization, which defaults to FDC lines.

        :param lines: list of 3 wavelengths [short, center, long]
        :return:
        """
        lines = lines if lines is not None else self.lines  # default to FDC spectral lines
        ns, nc, nl = tuple(self(lines))
        return (nc - 1) / (ns - nl) if ns != nl else np.inf

    def is_dispersive(self) -> bool:
        """:return: if the refractive index is dispersive using the Abbe Number"""
        return self.get_abbe_number() != np.inf
