
"""
Surface class:
Provides the functionality for the creation of numerical or analytical surfaces.
The class contains methods for interpolation, surface masking and normal calculation
"""

from typing import Any, Callable  # Any type
import copy

import numpy as np  # calculations

from ... import misc  # calculations
from ...misc import PropertyChecker as pc  # check types and values
from .surface import Surface


class FunctionSurface:
    pass


class FunctionSurface(Surface):

    def __init__(self,
                 r:                 float,
                 func:              Callable[[np.ndarray, np.ndarray], np.ndarray],
                 mask_func:         Callable[[np.ndarray, np.ndarray], np.ndarray] = None,
                 deriv_func:        Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]] = None,
                 hit_func:          Callable[[np.ndarray, np.ndarray], np.ndarray] = None,
                 func_args:         dict = None,
                 mask_args:         dict = None,
                 deriv_args:        dict = None,
                 hit_args:          dict = None,
                 z_min:             float = None,
                 z_max:             float = None,
                 parax_roc:         float = None,
                 **kwargs)\
            -> None:
        """
        Create a surface object.
        The z-coordinate in the pos array is irrelevant if the surface is used for a lens, since it wil be
        adapted inside the lens class

        """
        self._lock = False

        super().__init__(r, **kwargs)

        # sign used for reversing the surface
        self._sign = 1 
        
        # funcs
        self.func = func
        self.mask_func = mask_func
        self.deriv_func = deriv_func
        self.hit_func = hit_func

        # use empty dict or provided dict, if not empty
        self._func_args = func_args or {}
        self._mask_args = mask_args or {}
        self._deriv_args = deriv_args or {}
        self._hit_args = hit_args or {}
        
        self._offset = 0  # provisional offset
        self._offset = self._get_values(np.array([0]), np.array([0]))[0]
        self.z_min, self.z_max = self._find_bounds()
        self.parax_roc = parax_roc

        if z_max is not None and z_min is not None:
            z_range_probed = self.z_max - self.z_min
            z_range_provided = z_max - z_min
            
            if z_range_probed and z_range_provided + self.N_EPS < z_range_probed:
                self.print(f"Provided a z-extent of {z_range_provided} for surface {repr(self)},"
                           f"but measured range is at least {z_range_probed}, an increase of at "
                           f"least {100*(z_range_probed - z_range_provided)/z_range_probed:.5g}."
                           f" I will use the measured values for now.")
            else:

                range_factor = 1.2
                if z_range_provided > range_factor*z_range_probed:
                    self.print(f"WARNING: Provided z-range is more than {(range_factor-1)*100:.5g}% "
                               f"larger than measured z-range")

                z_max_ = self.z_max + self._offset
                z_min_ = self.z_min + self._offset

                if z_max + self.N_EPS < z_max_:
                    self.print(f"WARNING: Provided z_max={z_max} lower than measured value of {z_max_}."
                               f" Using the measured values for now")
                
                elif z_min - self.N_EPS > z_min_:
                    self.print(f"WARNING: Provided z_min={z_min} higher than measured value of {z_min_}."
                               f" Using the measured values for now")
                else:
                    self.z_min, self.z_max = z_min - self._offset, z_max - self._offset

        elif z_max is None and z_min is None:
            self.print(f"Estimated z-bounds of surface {repr(self)}: [{self._offset+self.z_min}, "
                       f"{self._offset+self.z_max}], provide actual values to make it more exact.")
        
        else:
            raise ValueError(f"z_max and z_min need to be both None or both need a value")

        self.lock()

    def _get_values(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Get surface values but in relative coordinate system to surface center.
        And without masking out values beyond the surface extent

        :param x: x-coordinate array (numpy 1D or 2D array)
        :param y: y-coordinate array (numpy 1D or 2D array)
        :return: z-coordinate array (numpy 1D or 2D array, depending on shape of x and y)
        """
        # additional mask function is handled in get_values()
        return self._sign*(self.func(x, y, **self._func_args) - self._offset)

    def get_mask(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Get surface mask values. A value of 1 means the surface is defined here.

        :param x: x-coordinate array (numpy 1D or 2D array)
        :param y: y-coordinate array (numpy 1D or 2D array)
        :return: z-coordinate array (numpy 1D or 2D array, depending on shape of x and y)
        """
        mask = super().get_mask(x, y)

        # additionally evaluate SurfaceFunction mask if defined
        if self.mask_func is not None:
            mask = mask & self.mask_func(x - self.pos[0], y - self.pos[1], **self._mask_args)

        return mask

    def get_normals(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Get normal vectors of the surface.

        :param x: x-coordinates (numpy 1D array)
        :param y: y-coordinates (numpy 1D array)
        :return: normal vector array of shape (x.shape[0], 3), components in second dimension (numpy 2D array)
        """

        if self.deriv_func is None:
            return super().get_normals(x, y)
    
        else:
            n = np.tile([0., 0., 1.], (x.shape[0], 1))

            # coordinates actually on surface
            m = self.get_mask(x, y)
            xm, ym = x[m], y[m]

            nxn, nyn = self.deriv_func(xm - self.pos[0], ym - self.pos[1], **self._deriv_args)
            n[m, 0] = -nxn*self._sign
            n[m, 1] = -nyn*self._sign
            n[m] = misc.normalize(n[m])

            return n

    def find_hit(self, p: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """

        :param p:
        :param s:
        :return:
        """
        # numerical hit finding when no hit_func is defined
        # or when the function is reversed
        if self.hit_func is None or self._sign == -1:
            return super().find_hit(p, s)

        else:
            # spatial 3D shift so we have coordinates relative to the surface center
            # _offset is removed, since it was introduced by Surface and is unknown to the SurfaceFunction object
            dp = self.pos - [0, 0, self._offset]

            p_hit0 = self.hit_func(p - dp, s, **self._hit_args)
            is_hit = self.mask_func(p_hit0[:, 0], p_hit0[:, 1], **self._mask_args)

            return p_hit0 + dp, is_hit  # transform to standard coordinates

    def reverse(self) -> FunctionSurface:
       
        # to spare ourselves the costly init phase we copy the current object and invert some properties

        # make copy and move to origin
        S = self.copy()
        S.move_to([0, 0, 0])
       
        # unlock
        S._lock = False

        # invert sign
        S._sign *= -1

        # invert curvature circle if given
        S.parax_roc = self.parax_roc if self.parax_roc is None else -self.parax_roc

        # assign new values for z_min, z_max. Both are negated and switched
        S.z_min = -(self.z_max - self.pos[2])
        S.z_max = -(self.z_min - self.pos[2])

        # lock
        S.lock()

        if S._sign == -1 and S.hit_func is not None:
            self.print(f"WARNING: Neglecting hit_func parameter, since the function surface is now reversed/negated.")

        return S

    def __setattr__(self, key: str, val: Any) -> None:
        """
        assigns the value of an attribute
        :param key: attribute name
        :param val: value to assign
        """
        match key:

            case ("z_max" | "z_min"):
                pc.check_type(key, val, float | int)
                val = float(val)

            case ("deriv_func" | "hit_func" | "mask_func"):
                pc.check_none_or_callable(key, val)

            case ("func"):
                pc.check_callable(key, val)
            
            case ("_deriv_args" | "_func_args" | "_hit_args" | "_mask_args"):
                pc.check_type(key, val, dict)
                super().__setattr__(key, copy.deepcopy(val))  # enforce deepcopy of dict
                return

        super().__setattr__(key, val)
