from typing import Any

import numpy as np  # calculations
import numexpr as ne  # faster calculations

from .function_surface import FunctionSurface
from ...misc import PropertyChecker as pc



class AsphericSurface(FunctionSurface):

    rotational_symmetry: bool = True


    def __init__(self,
                 r:                 float,
                 R:                 float,
                 k:                 float,
                 coeff:             list | np.ndarray,
                 **kwargs)\
            -> None:
        """

        no upper bound on number of coefficients

        :param r:
        :param R:
        :param k:
        :param coeff: coefficients for orders r^2, r^4, r^6, ... as list, where units are mm^-1, mm^-3, ...
        :return:
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
        return super().info + f", R = {self.R:.5g} mm, k = {self.k:.5g}\n"\
            f"coeff = {self.coeff}"

    def _asph(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:

        r = ne.evaluate("sqrt(x**2 + y**2)")

        # conic section function
        rho, k = 1/self.R, self.k
        z = ne.evaluate("rho * r**2 /(1 + sqrt(1 - (k+1) * rho**2 * r**2))")

        # polynomial part
        z += np.polyval(self._np_coeff, r)

        return z

    def _deriv(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

        # class FunctionSurface expects a function that takes x, y, relative to its center

        r = ne.evaluate("sqrt(x**2 + y**2)")
        phi = ne.evaluate("arctan2(y, x)")

        # derivative of conic section in regards to r
        k, rho = self.k, 1/self.R
        fr = ne.evaluate("r*rho / sqrt(1 - (k+1) * rho**2 *r**2)")

        # add derivative of polynomial part
        der_coeff = np.polyder(self._np_coeff)
        fr += np.polyval(der_coeff, r)

        # x and y components are just the radial component rotated by phi
        return fr*np.cos(phi), fr*np.sin(phi)

    def flip(self) -> None:

        # but now we have simply -1 as factor,
        # instead we'd like a factor of 1 and to negate R and the coefficients
        # so we reset the sign and negate those properties

        self._lock = False
        
        self.R *= -1
        self.coeff.flags.writeable = True
        self.coeff *= -1
        self.parax_roc *= -1
        a = self.pos[2] - (self.z_max - self.pos[2])
        b = self.pos[2] + (self.pos[2] - self.z_min)
        self.z_min, self.z_max = a, b

        self.lock()

    # override rotate function of FunctionSurface parent class
    def rotate(self, angle: float) -> None:
        pass

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
