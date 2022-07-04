
"""
RefractionIndex class:
Provides the creation and computation of constant or wavelength depended refraction indices
"""

import copy
import numpy as np
import optrace.tracer.Misc as misc
import optrace.tracer.presets.Lines as Lines
from optrace.tracer.spectrum.Spectrum import Spectrum as Spectrum
from optrace.tracer.BaseClass import *
import optrace.tracer.Color as Color


class RefractionIndex(Spectrum):

    # Refraction Index Models:
    # see https://doc.comsol.com/5.5/doc/com.comsol.help.roptics/roptics_ug_optics.6.46.html

    n_types = ["Abbe", "Cauchy", "Conrady", "Sellmeier", "Constant", "Data", "Function"]
    spectrum_types = n_types # alias
   
    quantity = "Refraction Index n"
    unit = ""

    def __init__(self,
                 n_type:    str = "Constant",
                 n:         float = 1.0,
                 coeff:     list = [0, 0, 0, 0, 0, 0, 0, 0],
                 lines:     list = Lines.preset_lines_FDC,
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
        # self._new_lock = False
        self.spectrum_type = n_type # needs to be here so coeff gets set correctly
        self.coeff = coeff
        self.V = V
        
        super().__init__(n_type, val=n, lines=lines, **kwargs)

        self._new_lock = True

    def __call__(self, wl: np.ndarray | list | float) -> np.ndarray:
        """
        Returns the refractive index at specified wavelengths.
        Call on obj using obj(wavelengths).

        :param wl: wavelengths in nm (numpy 1D array)
        :return: array of refraction indices
        """
        wl_ = wl if isinstance(wl, np.ndarray) else np.array(wl, dtype=np.float32)

        match self.spectrum_type:

            case "Cauchy":
                # parameters are specified in 1/µm^n, so convert nm wavelengths to µm with factor 1e-3
                A, B, C, D, = tuple(self.coeff)[:4]
                ns = misc.calc("A + B/l**2 + C/l**4 + D/l**6", l=wl_*1e-3)
            
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
                A, B, C = tuple(self.coeff)[:3]
                ns = misc.calc("A + B/l + C/l**3.5", l=wl_*1e-3)

            case "Sellmeier":
                wl2 = misc.calc("(wl_*1e-3)**2") # since Cs are specified in µm, not in nm
                A1, B1, A2, B2, A3, B3, A4, B4 = tuple(self.coeff)
                ns = misc.calc("sqrt(1 + A1*wl2/(wl2-B1) + A2*wl2/(wl2-B2) + A3*wl2/(wl2-B3) + A4*wl2/(wl2-B4))")

            case _:
                ns = super().__call__(wl_)

        if (nm := np.min(ns)) < 1:
            raise RuntimeError(f"RefractionIndex below 1 with value {nm}.")

        return ns

    def __setattr__(self, key, val0):
      
        # work on copies of ndarray and list
        val = val0.copy() if isinstance(val0, list | np.ndarray) else val0

        match key:
        
            case "val" if isinstance(val, int | float):
                self._checkNotBelow(key, val, 1)

            case "coeff":

                self._checkType(key, val, list)
                
                match self.spectrum_type:
                    case "Cauchy":      cnt = 4
                    case "Conrady":     cnt = 3
                    case "Sellmeier":   cnt = 8
                    case _:             cnt = 8

                if len(val) > cnt:
                    raise ValueError(f"{key} needs to be a list with maximum {cnt} numeric coefficients")

                # pad to 8 coeffs
                val += [0] * (8 - len(val))

                # validity of coeffs is checked in __call__
                # otherwise it would be possible that the coeffs seem invalid,
                # but the n_type is changed afterwards, making them valid

            case "_vals" if val is not None:
                if np.min(val) < 1:
                    raise ValueError("all vals values needs to be at least 1.")

            case "lines" if isinstance(val, list | np.ndarray):
                val2 = np.array(val)

                if val2.shape[0] != 3:
                    raise ValueError("Property 'lines' for n_type='Abbe' needs to have exactly 3 elements")

                if not (val2[0] < val2[1] and val2[1] < val2[2]):
                    raise ValueError("The values of property 'lines' need to be ascending.")

            case "func" if callable(val):
                wls = Color.wavelengths(1000)
                n = val(wls)
                if np.min(n) < 1:
                    raise ValueError("Function func needs to output values >= 1 over the whole visible range.")

            case "V" if val is not None:
                self._checkType(key, val, float | int)
                self._checkAbove(key, val, 0)

        super().__setattr__(key, val)

    def getAbbeNumber(self, lines=Lines.preset_lines_FDC) -> float:
        """
        Calculates the Abbe Number.

        :param lines: list of 3 wavelengths [short, center, long]
        :return:
        """
        ns, nc, nl = tuple(self(lines))
        return (nc - 1) / (ns - nl) if ns != nl else np.inf

    def isDispersive(self) -> bool:
        """Checks if dispersive using the Abbe Number"""
        return self.getAbbeNumber() != np.inf

