
from typing import Callable  # Callable type

import numpy as np  # np.ndarray type

from .function_surface_2d import FunctionSurface2D  # parent class


class FunctionSurface1D(FunctionSurface2D):
    
    rotational_symmetry: bool = True  #: has the surface rotational symmetry?
    _1D: bool = True  #: definied by 1D vector?

    def __init__(self,
                 r:                 float,
                 func:              Callable[[np.ndarray], np.ndarray],
                 mask_func:         Callable[[np.ndarray], np.ndarray] = None,
                 deriv_func:        Callable[[np.ndarray], np.ndarray] = None,
                 func_args:         dict = {},
                 mask_args:         dict = {},
                 deriv_args:        dict = {},
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
        :param func: surface function, must take one 1D array (r positions) and return one 1D float array
        :param mask_func: mask function (optional), must take one 1D array (r positions)
                          and return one boolean 1D array, where the surface is defined
        :param deriv_func: derivative function (optional), must take one 1D array (r positions)
                          and return one float 1D array with the derivative in r-direction
        :param func_args: optional dict for keyword arguments for parameter func
        :param mask_args: optional dict for keyword arguments for parameter mask_func
        :param deriv_args: optional dict for keyword arguments for parameter deriv_func
        :param parax_roc: optional paraxial radius of curvature
        :param z_min: exact value for the maximum surface z-extent, optional
        :param z_max: exact value for the minimum surface z-extent, optional
        :param kwargs: additional keyword arguments for parent classes
        """
        super().__init__(r, func, mask_func, deriv_func, func_args, mask_args, deriv_args,\
                         z_min, z_max, parax_roc, **kwargs)
