from typing import Any
import numpy as np

from .base_image import BaseImage
from ...property_checker import PropertyChecker as pc
from .. import color


class ScalarImage(BaseImage):

    def __init__(self,
                 data:              np.ndarray | str,
                 s:                 list | np.ndarray = None,
                 extent:            (list | np.ndarray) = None,
                 **kwargs)\
            -> None:
        """
        Init a ScalarImage object.

        When provided as numpy array, the image data must be non-negative and have only one channel (simple 2D array).
        When provided as image path, the image must have no color information.

        When parameter 'data' is provided as array, 
        element [0, 0] defines the lower left corner (negative x, negative y)

        :param data: Image filepath or two dimensional numpy array.
        :param s: image side lengths, x-dimension first
        :param extent: image extent in the form [xs, xe, ys, ye]
        :param kwargs: additional keyword arguments for the BaseImage and BaseClass class. 
         These include the limit and quantity option.
        """
        self._new_lock = False

        super().__init__(data, s, extent, **kwargs)
        self._new_lock = True

    def __setattr__(self, key: str, val: Any) -> None:
        """
        assigns the value of an attribute
        :param key: attribute name
        :param val: value to assign
        """
        match key:

            case "_data":
                pc.check_type(key, val, np.ndarray)
                val2 = np.asarray_chkfinite(val, dtype=np.float64)

                if val2.ndim == 3:
                    raise ValueError("Image can't have color information. "
                                     "Either use a RGBImage or remove color information.")

                elif val2.ndim != 2:
                    raise ValueError(f"Image needs to have two dimensions but has shape {val2.shape}.")

                if (min_ := np.min(val2)) < 0.0:
                    raise ValueError(f"There is a negative value of {min_} inside the image. "
                                     "Make sure all image data is non-negative.")

                super().__setattr__(key, val2)
                return

        super().__setattr__(key, val)

