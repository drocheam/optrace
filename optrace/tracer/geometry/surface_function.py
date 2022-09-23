import copy  # deepcopy of dicts
from typing import Callable, Any  # Callable and Any typing hints

import numpy as np   # ndarray type

from ..base_class import BaseClass  # parent class
from ..misc import PropertyChecker as pc  # check types and values


class SurfaceFunction(BaseClass):

    def __init__(self,
                 func:          Callable[[np.ndarray, np.ndarray], np.ndarray],
                 mask_func:     Callable[[np.ndarray, np.ndarray], np.ndarray] = None,
                 deriv_func:    Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]] = None,
                 hit_func:      Callable[[np.ndarray, np.ndarray], np.ndarray] = None,
                 func_args:     dict = None,
                 mask_args:     dict = None,
                 deriv_args:    dict = None,
                 hit_args:      dict = None,
                 **kwargs)\
            -> None:
        """

        :param func:
        :param mask_func:
        :param deriv_func:
        :param hit_func:
        :param func_args:
        :param mask_args:
        :param deriv_args:
        :param hit_args:
        """
        self.func = func
        self.mask_func = mask_func
        self.deriv_func = deriv_func
        self.hit_func = hit_func

        # use empty dict or provided dict, if not empty
        self._func_args = func_args or {}
        self._mask_args = mask_args or {}
        self._deriv_args = deriv_args or {}
        self._hit_args = hit_args or {}

        super().__init__(**kwargs)
        self._lock = True
        self._new_lock = True

    def get_hits(self, p: np.ndarray, s: np.ndarray) -> np.ndarray:
        """

        :param p: support vector array, shape (N, 3)
        :param s: direction vector array, shape (N, 3)
        :return: hit position vector array, shape (N, 3)
        """
        return self.hit_func(p, s, **self._hit_args)

    def get_derivative(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """

        :param x: x coordinate array
        :param y: y coordinate array
        :return: partial derivative in x direction and y direction
        """
        return self.deriv_func(x, y, **self._deriv_args)

    def get_mask(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """

        :param x: x coordinate array
        :param y: y coordinate array
        :return: bool array
        """
        return self.mask_func(x, y, **self._mask_args) 

    def get_values(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Calculate the Values on the Surface.

        :param x: x coordinate array
        :param y: y coordinate array
        :return: z coordinate array
        """
        return self.func(x, y, **self._func_args)

    def __setattr__(self, key: str, val: Any) -> None:
        """
        assigns the value of an attribute
        :param key: attribute name
        :param val: value to assign
        """
        match key:
            case ("deriv_func" | "hit_func" | "mask_func"):
                pc.check_none_or_callable(key, val)

            case ("func"):
                pc.check_callable(key, val)
            
            case ("_deriv_args" | "_func_args" | "_hit_args" | "_mask_args"):
                pc.check_type(key, val, dict)
                super().__setattr__(key, copy.deepcopy(val))  # enforce deepcopy of dict
                return

        super().__setattr__(key, val)
