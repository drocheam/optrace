
import numpy as np   # calculations
from typing import Callable  # Callable typing hint
import warnings  # print warnings
import copy  # for deepcopy of dicts

from optrace.tracer.base_class import BaseClass  # parent class
from optrace.tracer.misc import PropertyChecker as pc  # check types and values


class SurfaceFunction(BaseClass):

    def __init__(self,
                 func:          Callable[[np.ndarray, np.ndarray], np.ndarray],
                 r:             float,
                 func_args:     dict = None,
                 mask_func:     Callable[[np.ndarray, np.ndarray], np.ndarray] = None,
                 mask_args:     dict = None,
                 deriv_func:    Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]] = None,
                 deriv_args:    dict = None,
                 hit_func:      Callable[[np.ndarray, np.ndarray], np.ndarray] = None,
                 hit_args:      dict = None,
                 zmin:          float = None,
                 zmax:          float = None,
                 **kwargs)\
            -> None:
        """

        :param func:
        :param r:
        :param mask:
        :param derivative:
        :param hits:
        :param zmin:
        :param zmax:
        """
        self.func = func
        self.r = r

        self.mask_func = mask_func
        self.deriv_func = deriv_func
        self.hit_func = hit_func

        self.func_args = dict() if func_args is None else func_args
        self.mask_args = dict() if mask_args is None else mask_args
        self.deriv_args = dict() if deriv_args is None else deriv_args
        self.hit_args = dict() if hit_args is None else hit_args

        super().__init__(**kwargs)
        
        # get offset at (0, 0), gets removed later
        self.off = self.func(np.array([0.]), np.array([0.]), **self.func_args)[0]

        if zmax is None or zmin is None:
            if not self.silent:
                warnings.warn("WARNING: zmin or zmax missing, the values will be determined automatically."
                              "This is however less accurate than specifying them.")
            self.zmin, self.zmax = self.__find_bounds()
            self.zmin, self.zmax = self.zmin - self.off, self.zmax - self.off
        else:
            self.zmin, self.zmax = zmin - self.off, zmax - self.off
        
        self._lock = True
        self._new_lock = True

    def has_derivative(self) -> bool:
        """returns if a derivative function is implemented"""
        return self.deriv_func is not None

    def has_hits(self) -> bool:
        """returns if a hit fining function is implemented"""
        return self.hit_func is not None

    def get_hits(self, p: np.ndarray, s: np.ndarray) -> np.ndarray:
        """

        :param p: support vector array, shape (N, 3)
        :param s: direction vector array, shape (N, 3)
        :return: hit position vector array, shape (N, 3)
        """
        return self.hit_func(p, s, **self.hit_args)

    def get_derivative(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """

        :param x: x coordinate array
        :param y: y coordinate array
        :return: partial derivative in x direction and y direction
        """
        return self.deriv_func(x, y, **self.deriv_args)

    def get_mask(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """

        :param x: x coordinate array
        :param y: y coordinate array
        :return: bool array
        """
        # values outside circle are masked out
        m = np.zeros_like(x, dtype=bool)
        inside = x**2 + y**2 <= self.r**2

        m[inside] = self.mask_func(x[inside], y[inside], **self.mask_args) if self.mask_func is not None else True

        return m

    def get_values(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Calculate the Values on the Surface.

        :param x: x coordinate array
        :param y: y coordinate array
        :return: z coordinate array
        """
        inside = x**2 + y**2 <= self.r**2

        z = np.full_like(x, self.zmax, dtype=np.float64)
        z[inside] = self.func(x[inside], y[inside], **self.func_args) - self.off
    
        return z

    def __find_bounds(self) -> tuple[float, float]:
        """
        Estimate min and max z-value on Surface by sampling dozen values.

        :return: min and max z-value on Surface
        """
        # how to regularly sample a circle area, while sampling 
        # as much different phi and r values as possible?
        # => sunflower sampling for surface area
        # see https://stackoverflow.com/a/44164075

        N = 10000
        ind = np.arange(0, N, dtype=np.float64) + 0.5

        r = np.sqrt(ind/N) * self.r
        phi = 2*np.pi * (1 + 5**0.5)/2 * ind

        vals = self.func(r*np.cos(phi), r*np.sin(phi), **self.func_args)
        
        # mask out invalid values
        mask = self.get_mask(r * np.cos(phi), r * np.sin(phi))
        vals[~mask] = np.nan
      
        # in many cases the minimum and maximum are at the center or edge of the surface
        # => sample them additionally

        # values at surface edge
        phi2 = np.linspace(0, 2*np.pi, 1001)  # N is odd, since step size is 1/(N-1) * 2*pi, 
        r2 = np.full_like(phi2, self.r, dtype=np.float64)
        vals2 = self.func(r2*np.cos(phi2), r2*np.sin(phi2), **self.func_args)
        mask = self.get_mask(r2 * np.cos(phi2), r2 * np.sin(phi2))
        vals2[~mask] = np.nan

        # surface center
        vals3 = self.func(np.array([0.]), np.array([0.]), **self.func_args)
        if not self.get_mask(np.array([0.]), np.array([0.])):
            vals3 = np.nan

        # add all surface values into one array
        vals = np.concatenate((vals, vals2, vals3))

        # find minimum and maximum value
        zmin = np.nanmin(vals)
        zmax = np.nanmax(vals)

        return zmin, zmax

    def __setattr__(self, key, val):

        match key:
            case ("r" | "off" | "zmax" | "zmin"):
                pc.checkType(key, val, float | int)
                val = float(val)

            case ("deriv_func" | "func" | "hit_func" | "mask_func"):
                pc.checkNoneOrCallable(key, val)

            case ("deriv_args" | "func_args" | "hit_args" | "mask_args"):
                pc.checkType(key, val, dict)
                super().__setattr__(key, copy.deepcopy(val))  # enforce deepcopy of dict
                return

        super().__setattr__(key, val)
