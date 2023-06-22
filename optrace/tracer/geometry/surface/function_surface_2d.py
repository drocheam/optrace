
from typing import Any, Callable  # Any and Callable type
import copy  # for

import numpy as np  # calculations
import numexpr as ne  # faster calculations

from ... import misc  # calculations
from ...misc import PropertyChecker as pc  # check types and values
from .surface import Surface  # parent class


class FunctionSurface2D(Surface):
    
    rotational_symmetry: bool = False  #: has the surface rotational symmetry
    _1D: bool = False  #: defined by a 1D vector?

    def __init__(self,
                 r:                 float,
                 func:              Callable[[np.ndarray, np.ndarray], np.ndarray],
                 mask_func:         Callable[[np.ndarray, np.ndarray], np.ndarray] = None,
                 deriv_func:        Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]] = None,
                 func_args:         dict = None,
                 mask_args:         dict = None,
                 deriv_args:        dict = None,
                 z_min:             float = None,
                 z_max:             float = None,
                 parax_roc:         float = None,
                 **kwargs)\
            -> None:
        """
        Create a surface object defined by a mathematical function.

        Most of the time the automatic detection of the surface extent values z_min, z_max works correctly,
        assign the values manually otherwise.

        Providing parax_roc only makes sense, if the surface center can be locally described by a spherical surface.

        :param r: surface radius
        :param func: surface function, must take two 1D arrays (x and y positions) and return one 1D float array
        :param mask_func: mask function (optional), must take two 1D arrays (x and y positions)
                          and return one boolean 1D array, where the surface is defined
        :param deriv_func: derivative function (optional), must take two 1D arrays (x and y positions)
                          and return two float 1D arrays with partial derivatives in x and y direction
        :param func_args: optional dict for keyword arguments for parameter func
        :param mask_args: optional dict for keyword arguments for parameter mask_func
        :param deriv_args: optional dict for keyword arguments for parameter deriv_func
        :param parax_roc: optional paraxial radius of curvature
        :param z_min: exact value for the maximum surface z-extent, optional
        :param z_max: exact value for the minimum surface z-extent, optional
        :param kwargs: additional keyword arguments for parent classes
        """
        self._lock = False

        super().__init__(r, **kwargs)

        # sign used for reversing the surface
        self._sign = 1 
       
        # angle for rotation of the surface
        self._angle = 0

        # funcs
        self.func = func
        self.mask_func = mask_func
        self.deriv_func = deriv_func

        # use empty dict or provided dict, if not empty
        self._func_args = func_args or {}
        self._mask_args = mask_args or {}
        self._deriv_args = deriv_args or {}
        
        self._offset = 0  # provisional offset
        self._offset = self._values(np.array([0.]), np.array([0.]))[0]
        self.parax_roc = parax_roc

        self.__set_zmin_zmax(z_min, z_max)

        self.lock()

    def __set_zmin_zmax(self, z_min: float, z_max: float) -> None:
        """
        Assign zmin, zmax depending on find bounds or user provided values.
        Check for plausibility and output messages.
        """       
        if self._1D:
            rn = np.linspace(0, self.r, 10000)
            zn = self._values(rn, np.zeros_like(rn))
            mn = self.mask(rn, np.zeros_like(rn))
            self.z_min, self.z_max = np.min(zn[mn]), np.max(zn[mn])

        else:
            self.z_min, self.z_max = self._find_bounds()

        if z_max is not None and z_min is not None:
            z_range_probed = self.z_max - self.z_min
            z_range_provided = z_max - z_min
            
            if z_range_probed and z_range_provided + self.N_EPS < z_range_probed:
                self.print(f"Provided a z-extent of {z_range_provided} for surface {self.get_desc(hex(id(self)))},"
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
            self.print(f"Estimated z-bounds of surface {self.get_desc(hex(id(self)))}: [{self._offset+self.z_min:.9g}, "
                       f"{self._offset+self.z_max:.9g}], provide actual values to make it more exact.")
        
        else:
            raise ValueError(f"z_max and z_min need to be both None or both need a value")

    def _values(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Get surface values but in relative coordinate system to surface center.
        And without masking out values beyond the surface extent or own additional mask

        :param x: x-coordinate array (numpy 1D or 2D array)
        :param y: y-coordinate array (numpy 1D or 2D array)
        :return: z-coordinate array (numpy 1D or 2D array, depending on shape of x and y)
        """
        if self._1D:
            r = ne.evaluate("sqrt(x**2 + y**2)")
            vals = self.func(r, **self._func_args)

        else:
            x_, y_ = self._rotate_rc(x, y, -self._angle)
            vals = self.func(x_, self._sign*y_, **self._func_args)

        assert isinstance(vals, np.ndarray), "func must return a np.ndarray"
        assert not vals.shape[0] or isinstance(vals[0], np.float64),\
                "Elements of return value of func must be of type np.float64"

        return self._sign*(vals - self._offset)

    def mask(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Get surface mask values. A value of 1 means the surface is defined here.

        :param x: x-coordinate array (numpy 1D or 2D array)
        :param y: y-coordinate array (numpy 1D or 2D array)
        :return: z-coordinate array (numpy 1D or 2D array, depending on shape of x and y)
        """
        mask = super().mask(x, y)

        # additionally evaluate FunctionSurface2D mask if defined
        if self.mask_func is not None:
            
            # relative coordinates
            xm = x - self.pos[0]
            ym = y - self.pos[1]
       
            if self._1D:
                rm = ne.evaluate("sqrt(xm**2 + ym**2)")
                maskf = self.mask_func(rm, **self._mask_args)

            else:
                x_, y_ = self._rotate_rc(xm, ym, -self._angle)
                maskf = self.mask_func(x_, self._sign*y_, **self._mask_args)
                
            assert isinstance(maskf, np.ndarray), "mask_func must return a np.ndarray"
            assert not maskf.shape[0] or isinstance(maskf[0], bool | np.bool_),\
                    f"Elements of return value of mask_func must be of type bool"

            mask = mask & maskf

        return mask

    def normals(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Get normal vectors of the surface.

        :param x: x-coordinates (numpy 1D array)
        :param y: y-coordinates (numpy 1D array)
        :return: normal vector array of shape (x.shape[0], 3), components in second dimension (numpy 2D array)
        """

        if self.deriv_func is None:
            return super().normals(x, y)
    
        else:

            n = np.tile([0., 0., 1.], (x.shape[0], 1))

            # coordinates actually on surface
            m = self.mask(x, y)
            
            # relative coordinates
            xm = x[m] - self.pos[0]
            ym = y[m] - self.pos[1]
        
            if self._1D:
                phi = ne.evaluate("arctan2(ym, xm)")
                rm = ne.evaluate("sqrt(xm**2 + ym**2)")
                nr = self._sign*self.deriv_func(rm, **self._deriv_args)
                
                assert isinstance(nr, np.ndarray), "deriv_func must return a np.ndarray"
                assert not nr.shape[0] or isinstance(nr[0], np.float64), "values of deriv_func must be np.float64"
                
                nxn, nyn = ne.evaluate("nr*cos(phi)"), ne.evaluate("nr*sin(phi)")

            else:
                # rotate surface
                x_, y_ = self._rotate_rc(xm, ym, -self._angle)

                # rotating [x, y, z] around [1, 0, 0] by pi gives us [x, -y, -z]
                # we need to negate this, so the vector points in +z direction
                # -> [-x, y, z]
            
                nxn, nyn = self.deriv_func(xm, self._sign*ym, **self._deriv_args)
                # ^-- y is negative, since surface is flipped
                
                assert isinstance(nxn, np.ndarray), "deriv_func must return two np.ndarray"
                assert isinstance(nyn, np.ndarray), "deriv_func must return two np.ndarray"
                assert not nxn.shape[0] or isinstance(nxn[0], np.float64),\
                        "Elements of return value of deriv_func must be of type np.float64"
                assert not nyn.shape[0] or isinstance(nyn[0], np.float64),\
                        "Elements of return value of deriv_func must be of type np.float64"

                nxn, nyn = self._rotate_rc(nxn*self._sign, nyn, self._angle)

            n[m, 0] = -nxn
            n[m, 1] = -nyn
            n[m] = misc.normalize(n[m])

            return n

    def flip(self) -> None:
        """flip the surface around the x-axis"""

        # unlock
        self._lock = False

        # invert sign
        self._sign *= -1

        # invert curvature circle if given
        self.parax_roc = self.parax_roc if self.parax_roc is None else -self.parax_roc

        # assign new values for z_min, z_max. Both are negated and switched
        a = self.pos[2] - (self.z_max - self.pos[2])
        b = self.pos[2] - (self.z_min - self.pos[2])
        self.z_min, self.z_max = a, b

        # lock
        self.lock()

    def rotate(self, angle: float) -> None:
        """
        rotate the surface around the z-axis

        :param angle: rotation angle in degrees
        """
        if not self._1D:
            self._lock = False
            self._angle += np.deg2rad(angle)
            self.lock()
       
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

            case ("deriv_func" | "mask_func"):
                pc.check_none_or_callable(key, val)

            case ("func"):
                pc.check_callable(key, val)
            
            case ("_deriv_args" | "_func_args" | "_mask_args"):
                pc.check_type(key, val, dict)
                super().__setattr__(key, copy.deepcopy(val))  # enforce deepcopy of dict
                return

        super().__setattr__(key, val)
