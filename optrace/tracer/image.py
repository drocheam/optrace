from __future__ import annotations
from typing import Any  # Callable and Any type

import cv2  # image loading and saving
import numpy as np  # calculations
import os

from . import color  # xyz_observers curves and sRGB conversions
from .base_class import BaseClass  # parent class
from .misc import PropertyChecker as pc  # check types and values



class Image(BaseClass):

    def __init__(self,
                 image:             numpy.ndarray | str,
                 s:                 list | numpy.ndarray,
                 **kwargs)\
            -> None:
        """
        Init an Image object.
        This object stores a sRGB image as well as side length information.

        When parameter image is provided as array, 
        element [0, 0] defines the lower left corner (negative x, negative y)

        :param image: Three-dimensional numpy array with data range [0, 1] or a file path to an image
        :param s: image side lengths, x-dimension first
        """
        self._new_lock = False

        self._data = self._load_image(image) if isinstance(image, str) else image
        self.s = s

        super().__init__(**kwargs)
        self._new_lock = True

    def _load_image(self, path: str) -> numpy.ndarray:
        """
        Loads an image file and converts it into a numpy array.
        
        :param path: path to the file
        :return: image numpy array, three dimensions, three channels, value range 0-1
        """
        if not cv2.haveImageReader(path):
            raise IOError(f"Can't find/process file {path}")

        image = cv2.imread(path, flags=cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
        return np.flipud(image)  # flip so element 0 is in the lower left corner

    @property
    def extent(self) -> list[float, float, float, float]:
        """extent list of the image. Provided as [x0, x1, y0, y1]"""
        return [-self.s[0]/2, self.s[0]/2, -self.s[1]/2, self.s[1]/2]

    @property
    def shape(self) -> tuple[int, int, int]:
        """data shape of the image. y-dimension first"""
        return self._data.shape

    @property
    def data(self) -> numpy.ndarray:
        """image data array"""
        return self._data.copy()

    def save(self, 
             path:    str,
             params:  list = [],
             flip:    bool = False)\
            -> None:
        """
        Save the image data as image file.
        The image is rescaled (and therefore interpolated) so we have square pixels before the export.

        :param path: path with file ending to save the image to
        :param params: additional parameters for cv2.imwrite, see cv ImwriteFlags
        :param flip: if image should be flipped (rotated 180 degrees)
        """
        # check if folder of path exists and check if valid format
        folder = os.path.split(path)[0]
        if not (folder == "" or os.path.isdir(folder)) or not cv2.haveImageWriter(path):
            raise IOError(f"Can't create/write file {path}")
        
        # approximate size we need to rescale to to make square pixels
        if self.s[0] > self.s[1]:
            siz = (int(self.shape[0]*self.s[0]/self.s[1]), self.shape[0])
        else:
            siz = (self.shape[1], int(self.shape[1]*self.s[0]/self.s[1]))
     
        # resize in sRGB linear color space
        img_rgb_lin = color.srgb_to_srgb_linear(self._data)
        img_r = cv2.resize(img_rgb_lin, siz, interpolation=cv2.INTER_LINEAR)
        img_r_rgb = color.srgb_linear_to_srgb(img_r)
        
        # normalize and convert to 8bit cv2 format
        img2 = (255*img_r_rgb).astype(np.uint8)
        img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
        img2 = np.flipud(img2) # flip back so it is saved correctly

        if flip:
            img2 = np.fliplr(np.flipud(img2))

        # save
        cv2.imwrite(path, img2, params)

    def __setattr__(self, key: str, val: Any) -> None:
        """
        assigns the value of an attribute
        :param key: attribute name
        :param val: value to assign
        """
        match key:

            case "s":
                pc.check_type(key, val, list | tuple | np.ndarray)
                val2 = np.asarray_chkfinite(val, dtype=np.float64)

                if val2.shape[0] != 2:
                    raise ValueError("s needs to have 2 elements.")
                
                pc.check_above("s[0]", val[0], 0)
                pc.check_above("s[1]", val[1], 0)

                super().__setattr__(key, val2)
                return
           
            case "_data":
                pc.check_type(key, val, np.ndarray)
                val2 = np.asarray_chkfinite(val)

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

