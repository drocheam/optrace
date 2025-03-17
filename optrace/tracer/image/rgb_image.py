from __future__ import annotations
from typing import Any  # Callable and Any type

import numpy as np  # calculations

from .base_image import BaseImage
from ...property_checker import PropertyChecker as pc  # check types and values



class RGBImage(BaseImage):

    def __init__(self,
                 data:        numpy.ndarray | str,
                 s:           list | numpy.ndarray = None,
                 extent:      (list | numpy.ndarray) = None,
                 **kwargs)\
            -> None:
        """
        Init a RGBImage object.

        When provided as numpy array, the image data must have values in range [0, 1] and three channels
        in the third dimension.
        When provided as image path, the image is loaded with three channels regardless of colorspace.

        When parameter 'data' is provided as array, 
        element [0, 0] defines the lower left corner (negative x, negative y)

        :param data: Image filepath or three dimensional numpy array.
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

                if val2.ndim != 3 or val2.shape[2] != 3:
                    raise ValueError(f"Image needs to have three dimensions with 3 elements"
                                     f" (RGB) in the third dimension, but has shape {val2.shape}.")

                if (min_ := np.min(val2)) < 0.0:
                    raise ValueError(f"There is a negative value of {min_} inside the image. "
                                     "Make sure all image data is in the range [0, 1].")
                
                if (max_ := np.max(val2)) > 1.0:
                    raise ValueError(f"There is a value of {max_} inside the image. "
                                     "Make sure all image data is in the range [0, 1].")

                super().__setattr__(key, val2)
                return

        super().__setattr__(key, val)

