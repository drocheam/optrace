
from typing import Any  # Any type

import numpy as np  # calculations

from .presets import spectral_lines as Lines  # spectral lines for Abbe number
from .spectrum import Spectrum  # parent class
from . import color  # for visible wavelength range
from .misc import PropertyChecker as pc  # check types and values


class RefractionIndex(Spectrum):

    # Refraction Index Models:
    # see https://doc.comsol.com/5.5/doc/com.comsol.help.roptics/roptics_ug_optics.6.46.html

    coeff_count = {"Cauchy": 4, "Conrady": 3, "Sellmeier1": 6, "Sellmeier2": 5, "Sellmeier3": 8,
                    "Sellmeier4": 5, "Sellmeier5": 10, "Herzberger": 6, "Extended": 8, "Extended2": 8,
                    "Handbook of Optics 1": 4, "Handbook of Optics 2": 4, "Schott": 6, "Extended3": 9}
    """number of coefficients for the different refractive index models"""
    
    n_types: list[str] = ["Abbe", "Cauchy", "Conrady", "Constant", "Data", "Extended", "Extended2",
                          "Extended3", "Function", "Handbook of Optics 1", "Handbook of Optics 2",
                          "Sellmeier1", "Sellmeier2", "Sellmeier3", "Sellmeier4",
                          "Sellmeier5", "Herzberger", "Schott"]
    """possible refractive index types"""

    spectrum_types: list[str] = n_types  # alias

    quantity: str = "Refraction Index n"  #: physical quantity
    unit: str = ""  #: physical unit

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
        See the documentation on information on how to provide all parameters correctly.

        :param n_type: one of "n_types"
        :param n: refraction index for n_type="Constant"
        :param V: Abbe number for n_type="Abbe"
        :param coeff: list of coefficients for the model, see the documentation
        :param lines: spectral lines to use for n_type="Abbe",
                      list of 3 wavelengths [short wavelength, center wavelength, long wavelength]
        """
        self.spectrum_type = n_type  # needs to be first so coeff gets set correctly
        self.coeff = coeff
        self.V = V

        lines = lines if lines is not None else Lines.FdC

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
            
        if self.coeff is not None:
            c = self.coeff

        # check that they are provided
        elif self.spectrum_type not in ["Constant", "Data", "Function", "Abbe"]:
            raise TypeError(f"coefficient variable 'coeff' needs to be provided for n_type='{self.spectrum_type}'.")
        
        # most formula use lambda^2 in µm^2
        if self.spectrum_type not in ["Constant", "Function", "Conrady"]:
            wl2 = (wl_*1e-3)**2

        match self.spectrum_type:

            case "Abbe":
                # estimate a refractive index curve from abbe number
                # note: many different curves can have the same number, these is just an estimation for a possible one

                # use wl in um
                l = 1e-3*np.array(self.lines)
                nc = self.val

                # compromise between Cauchy (d=0) and Herzberger (d=0.028)
                d = 0.014

                # solve for B and A from Abbe Number and center refraction index
                B = 1/self.V * (nc - 1) / (1/(l[0]**2-d) - 1/(l[2]**2-d))
                A = nc - B/(l[1]**2-d)

                ns = A + B/(wl2-d)

            case "Conrady":
                l = wl_*1e-3
                ns = c[0] + c[1]/l + c[2]/l**3.5

            case "Cauchy":
                ns = c[0] + c[1]/wl2 + c[2]/wl2**2 + c[3]/wl2**3
            
            case "Sellmeier1":
                ns = np.sqrt(1 + c[0]*wl2/(wl2-c[1]) + c[2]*wl2/(wl2-c[3]) + c[4]*wl2/(wl2-c[5]))
            
            case "Sellmeier2":
                ns = np.sqrt(1 + c[0] + c[1]*wl2/(wl2-c[2]**2) + c[3]/(wl2-c[4]**2))
            
            case "Sellmeier3":
                ns = np.sqrt(1 + c[0]*wl2/(wl2-c[1]) + c[2]*wl2/(wl2-c[3]) + c[4]*wl2/(wl2-c[5]) + c[6]*wl2/(wl2-c[7]))
            
            case "Sellmeier4":
                ns = np.sqrt(c[0] + c[1]*wl2/(wl2-c[2]) + c[3]*wl2/(wl2-c[4]))
            
            case "Sellmeier5":
                ns = np.sqrt(1 + c[0]*wl2/(wl2-c[1]) + c[2]*wl2/(wl2-c[3]) + c[4]*wl2/(wl2-c[5])
                               + c[6]*wl2/(wl2-c[7])+ c[8]*wl2/(wl2-c[9]))

            case "Schott":
                ns = np.sqrt(c[0] + c[1]*wl2 + c[2]/wl2 + c[3]/wl2**2 + c[4]/wl2**3 + c[5]/wl2**4)
            
            case "Herzberger":
                L = 1/(wl2 - 0.028)
                ns = c[0] + c[1]*L + c[2]*L**2 + c[3]*wl2 + c[4]*wl2**2 + c[5]*wl2**3

            case "Handbook of Optics 1":
                ns = np.sqrt(c[0] + c[1]/(wl2 - c[2]) - c[3]*wl2)
            
            case "Handbook of Optics 2":
                ns = np.sqrt(c[0] + c[1]*wl2/(wl2 - c[2]) - c[3]*wl2)
            
            case "Extended":
                ns = np.sqrt(c[0] + c[1]*wl2 + c[2]/wl2 + c[3]/wl2**2 + c[4]/wl2**3
                             + c[5]/wl2**4 + c[6]/wl2**5 + c[7]/wl2**6)

            case "Extended2":
                ns = np.sqrt(c[0] + c[1]*wl2 + c[2]/wl2 + c[3]/wl2**2 + c[4]/wl2**3 + c[5]/wl2**4
                             + c[6]*wl2**2 + c[7]*wl2**3)

            case "Extended3":
                ns = np.sqrt(c[0] + c[1]*wl2 + c[2]*wl2**2 + c[3]/wl2 + c[4]/wl2**2 + c[5]/wl2**3
                             + c[6]*wl2**4 + c[7]*wl2**5 + c[8]/wl2**6)

            case "Data" if self._wls is not None:
                # no extrapolation in "Data" Mode
                wlmin = np.min(wl_)
                wlmax = np.max(wl_)

                if wlmin < self._wls[0] or wlmax > self._wls[-1]:
                    raise RuntimeError(f"Wavelength range [{wlmin:.5g}, {wlmax:.5g}] larger than data range"
                                       f" [{self._wls[0]}, {self._wls[-1]}] for this material.")

                ns = super().__call__(wl_)
            
            case _:
                ns = super().__call__(wl_)


        wlb = np.argmin(ns)
        if (nm := ns.flat[wlb]) < 1:
            raise RuntimeError(f"Refraction index below 1 with value {nm:.4g} at {wl_.flat[wlb]:.4g}nm.")

        return ns

    def __eq__(self, other: Any) -> bool:
        """
        Equal operator. Compares self to 'other'.

        :param other: object to compare to
        :return: if both are equal
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
        """
        Not-equal operator. Compares self to 'other'.

        :param other: object to compare to
        :return: if both are not equal
        """
        return not self.__eq__(other)

    def __setattr__(self, key: str, val: Any) -> None:
        """
        assigns the value of an attribute
        :param key: attribute name
        :param val: value to assign
        """
        match key:

            case "val":
                pc.check_type(key, val, int | float)
                np.asarray_chkfinite(val)
                pc.check_not_below(key, val, 1)

            case "coeff" if val is not None:

                pc.check_type(key, val, list)
                cnt = self.coeff_count[self.spectrum_type]

                if len(val) != cnt:
                    raise ValueError(f"{key} needs to be a list with exactly {cnt} numeric coefficients for mode "
                                     f"{self.spectrum_type}, but got {len(val)}.")

                super().__setattr__(key, val.copy())
                return

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

    def abbe_number(self, lines: list = None) -> float:
        """
        Calculates the Abbe Number. The spectral lines can be overwritten with the parameter.
        Otherwise the RefractionIndex.lines parameter is used from its initialization, which defaults to FDC lines.

        :param lines: list of 3 wavelengths [short, center, long]
        :return:
        """
        lines = lines if lines is not None else self.lines  # default to FDC spectral lines
        ns, nc, nl = tuple(self(lines))
        return float((nc - 1) / (ns - nl) if ns != nl else np.inf)

    def is_dispersive(self) -> bool:
        """:return: if the refractive index is dispersive, determined by the Abbe Number"""
        return self.abbe_number() != np.inf
