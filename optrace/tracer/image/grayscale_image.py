from typing import Any
import numpy as np

from .scalar_image import ScalarImage
from ...property_checker import PropertyChecker as pc
from .. import color



class GrayscaleImage(ScalarImage):

    def __init__(self,
                 data:              np.ndarray | str,
                 s:                 list | np.ndarray = None,
                 extent:            (list | np.ndarray) = None,
                 **kwargs)\
            -> None:
        """
        Init a GrayscaleImage object.
        Subclass of ScalarImage, but communicates an image in sRGB grayscale with gamma compression.

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

    def to_rgb_image(self) -> 'RGBImage':
        """
        Create an RGBImage from this object.
        :return: RGBImage
        """
        from .rgb_image import RGBImage
        return RGBImage(np.repeat(self._data[:, :, np.newaxis], 3, axis=2), extent=self.extent, desc=self.desc, 
                        long_desc=self.long_desc, quantity=self.quantity, projection=self.projection, limit=self.limit)

    def __setattr__(self, key: str, val: Any) -> None:
        """
        assigns the value of an attribute
        :param key: attribute name
        :param val: value to assign
        """
        match key:

            case "_data":
                pc.check_type(key, val, np.ndarray)
                
                if (max_ := val.max()) > 1.0:
                    raise ValueError(f"There is a value of {max_} inside the image. "
                                     "Make sure all image data is in the range [0, 1].")

        super().__setattr__(key, val)

