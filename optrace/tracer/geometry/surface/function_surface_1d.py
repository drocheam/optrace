
from typing import Any, Callable  # Any type
import copy

import numpy as np  # calculations

from .function_surface_2d import FunctionSurface2D


class FunctionSurface1D(FunctionSurface2D):
    
    rotational_symmetry: bool = True
    _1D: bool = True

    def __init__(self,
                 r:                 float,
                 func:              Callable[[np.ndarray], np.ndarray],
                 mask_func:         Callable[[np.ndarray], np.ndarray] = None,
                 deriv_func:        Callable[[np.ndarray], np.ndarray] = None,
                 func_args:         dict = None,
                 mask_args:         dict = None,
                 deriv_args:        dict = None,
                 z_min:             float = None,
                 z_max:             float = None,
                 parax_roc:         float = None,
                 **kwargs)\
            -> None:
        """

        """

        super().__init__(r, func, mask_func, deriv_func, func_args, mask_args, deriv_args,\
                         z_min, z_max, parax_roc, **kwargs)
