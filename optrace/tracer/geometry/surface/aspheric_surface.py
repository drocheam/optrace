from typing import Any  # "Any" type

import numpy as np  # calculations

from .function_surface_1d import FunctionSurface1D  # parent class
from ....property_checker import PropertyChecker as pc  # check values and types


class AsphericSurface(FunctionSurface1D):

    rotational_symmetry: bool = True  #: has the surface rotation symmetry?

    def __init__(self,
                 r:                 float,
                 R:                 float,
                 k:                 float,
                 coeff:             list | np.ndarray,
                 **kwargs)\
            -> None:
        """
        Define an aspheric surface, which is a ConicSurface with a
        n additional polynomial component a_0*r^2 + a_1*r^4 + ...
        There is no upper bound on number of coefficients

        :param r: surface radius
        :param R: curvature circle
        :param k: conic constant
        :param coeff: coefficients for orders r^2, r^4, r^6, ... as list, where units are mm^-1, mm^-3, ...
        :param kwargs: additional keyword arguments for parent classes
        """
        self._lock = False

        self.k = k
        self.R = R
        self.coeff = coeff
        
        # the paraxial curvature circle radius at r=0 is just the inverse of the second derivative,
        # for the asphere case the second derivative is 1/R + 2*coeff0 at r=0
        parax_roc = 1 / (1/self.R + 2*self.coeff[0])

        super().__init__(r, func=self._asph, deriv_func=self._deriv, parax_roc=parax_roc, **kwargs)

        self.lock()

    @property
    def info(self) -> str:
        """info string, characterizing the surface"""
        return super().info + f", R = {self.R:.5g} mm, k = {self.k:.5g}\n"\
            f"coeff = {self.coeff}"

    def _asph(self, r: np.ndarray) -> np.ndarray:
        """
        asphere function

        :param r: radial values, 1D array
        :return: surface values
        """
        # conic section function
        rho, k = 1/self.R, self.k
        z = rho * r**2 /(1 + np.sqrt(1 - (k+1) * rho**2 * r**2))

        # polynomial part
        z += np.polyval(self._np_coeff, r)

        return z

    def _deriv(self, r: np.ndarray) -> np.ndarray:
        """
        derivative of the aspheric function

        :param r: radial values, 1D array
        :return: derivative values
        """
        # derivative of conic section in regards to r
        k, rho = self.k, 1/self.R
        fr = r*rho / np.sqrt(1 - (k+1) * rho**2 *r**2)

        # add derivative of polynomial part
        der_coeff = np.polyder(self._np_coeff)
        fr += np.polyval(der_coeff, r)

        return fr

    def flip(self) -> None:
        """flip the surface around the x-axis"""

        # override super() method, so we instead negate curvature radius and coefficients

        self._lock = False
        
        self.R *= -1
        self.coeff.flags.writeable = True
        self.coeff *= -1
        self.parax_roc *= -1
        a = self.pos[2] - (self.z_max - self.pos[2])
        b = self.pos[2] + (self.pos[2] - self.z_min)
        self.z_min, self.z_max = a, b

        self.lock()

    @property
    def _np_coeff(self) -> np.ndarray:
        """
        convert asphere coefficients to polynomial coefficients for numpy.polynomial.
        While our coefficients are [a2, a4, a6, ...] for r^2, r^4, r^6, ...
        the numpy coefficients must be given as [a6, a5, a4, ... a0]
        """
        np_coeff = np.zeros(2*len(self.coeff) + 1, dtype=np.float64)
        np_coeff[2::2] = self.coeff
        return np.flip(np_coeff)

    def __setattr__(self, key: str, val: Any) -> None:
        """
        assigns the value of an attribute
        :param key: attribute name
        :param val: value to assign
        """
        if key in ["R", "k"]:
            pc.check_type(key, val, float | int)
            val = float(val)

            if key == "R" and (val == 0 or not np.isfinite(val)):
                raise ValueError("R needs to be non-zero and finite. Use planar surface types for planar surfaces.")

        elif key == "coeff":
            pc.check_type(key, val, list | np.ndarray)
            coeff = np.asarray_chkfinite(val, dtype=np.float64)

            # special case: empty coefficients
            if not len(coeff):
                raise ValueError("Empty coeff list. Provide coefficients or use ConicSurface instead.")

            super().__setattr__(key, coeff)
            return

        super().__setattr__(key, val)
