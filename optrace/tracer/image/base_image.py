from __future__ import annotations
from typing import Any  # Callable and Any type

import cv2  # image loading and saving
import numpy as np  # calculations
import os

from ..base_class import BaseClass  # parent class
from ...property_checker import PropertyChecker as pc  # check types and values
from ..geometry.surface import SphericalSurface



class BaseImage(BaseClass):

    def __init__(self,
                 data:              numpy.ndarray | str,
                 s:                 (list | numpy.ndarray) = None,
                 extent:            (list | numpy.ndarray) = None,
                 projection:        str = None,
                 quantity:          str = "",
                 limit:             float = None,
                 **kwargs)\
            -> None:
        """
        Init an Image object. Parent class of ScalarImage, GrayscaleImage and RGBImage

        Image dimensions should be either provided by the s or extent parameter.

        When parameter 'data' is provided as array, 
        element [0, 0] defines the lower left corner (negative x, negative y)

        :param image: Three-dimensional numpy array with data range [0, 1] or a file path to an image
        :param s: image side lengths, x-dimension first
        :param extent: image extent in the form [xs, xe, ys, ye]
        :param quantity: name of the stored image quantity/property
        :param limit: resolution limit information. Single value in micrometers.
        :param kwargs: additional keyword arguments for the BaseClass
        """
        self._new_lock = False

        self._data = self._load_image(data) if isinstance(data, str) else data

        if extent is None and s is None:
            raise ValueError("Either s or extent need to be provided for Images")

        elif extent is None:
                
            pc.check_type("s", s, list | tuple | np.ndarray)
            s2 = np.asarray_chkfinite(s, dtype=np.float64)

            if s2.shape[0] != 2:
                raise ValueError("s needs to have 2 elements.")
            
            pc.check_above("s[0]", s2[0], 0)
            pc.check_above("s[1]", s2[1], 0)

            self.extent = [-s2[0]/2, s2[0]/2, -s2[1]/2, s2[1]/2]
        else:
            self.extent = extent

        self.quantity = quantity
        self.projection = projection
        self.limit = limit

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
        image = np.flipud(image) #  # flip so element 0 is in the lower left corner

        if self.__class__.__name__ == "RGBImage":
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
        else:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) / 255.0

    @property
    def shape(self) -> tuple[int, int, int]:
        """data shape of the image. y-dimension first"""
        return self._data.shape

    @property
    def data(self) -> numpy.ndarray:
        """image data array"""
        return self._data.copy()

    @property 
    def s(self) -> list[float, float]:
        """image side lengths, x-dimension first"""
        return [float(self.extent[1] - self.extent[0]), float(self.extent[3] - self.extent[2])]

    @property
    def Apx(self) -> float:
        """area per pixel in mm^2"""
        return float(self.s[0] * self.s[1] / (self.shape[1] * self.shape[0]))

    # TODO different logic for grayscale image ?
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
    
        # resize so pixel are square. NOTE: For sRGB this should have actually place in linear sRGB
        img = cv2.resize(self._data, siz, interpolation=cv2.INTER_LINEAR)
      
        # if single channel image / linear image: make three channels and normalize
        if self._data.ndim == 2:
            if (maxi := np.max(img)):
                img /= maxi
            img = np.broadcast_to(img[:, :, np.newaxis], [img.shape[0], img.shape[1], 3])

        # normalize and convert to 8bit cv2 format
        img2 = (255*img).astype(np.uint8)
        img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR) 
        img2 = np.flipud(img2) # flip back so it is saved correctly

        if flip:
            img2 = np.fliplr(np.flipud(img2))

        # save
        cv2.imwrite(path, img2, params)
    
    def profile(self,
                x:      float = None,
                y:      float = None)\
            -> tuple[np.ndarray, list[np.ndarray]]:
        """
        Create an image profile.
        Only specify one of parameters x or y.

        There is no interpolation, the nearest pixel value is chosen (nearest neighbor 'interpolation')
        
        :param x: x-value for the profile
        :param y: y-value for the profile
        :return: bin edge array, list of image cuts (one for linear images, three for rgb images)
        """
        img = self._data

        if x is not None:

            if not self.extent[0] <= x <= self.extent[1]:
                raise ValueError(f"Position x={x} is outside the image x-extent of {self.extent[:2]}")

            bins = np.linspace(self.extent[2], self.extent[3], self.shape[0]+1)
            ind = int((x - self.extent[0]) / self.s[0] * self.shape[1] * (1 - 1e-12))
            iml = [img[:, ind]] if img.ndim == 2 else [img[:, ind, 0], img[:, ind, 1], img[:, ind, 2]]
        
        elif y is not None:

            if not self.extent[2] <= y <= self.extent[3]:
                raise ValueError(f"Position y={y} is outside the image y-extent of {self.extent[2:]}")
            
            bins = np.linspace(self.extent[0], self.extent[1], self.shape[1]+1)
            ind = int((y - self.extent[2]) / self.s[1] * self.shape[0] * (1 - 1e-12))
            iml = [img[ind]] if img.ndim == 2 else [img[ind, :, 0], img[ind, :, 1], img[ind, :, 2]]
        
        else:
            raise ValueError("Either x or y parameter must be provided.")

        return bins, iml

    def __setattr__(self, key: str, val: Any) -> None:
        """
        assigns the value of an attribute
        :param key: attribute name
        :param val: value to assign
        """
        match key:
            
            case "extent":
                pc.check_type(key, val, list | tuple | np.ndarray)
                val2 = np.asarray_chkfinite(val, dtype=np.float64)

                if val2.shape[0] != 4:
                    raise ValueError("Extent needs to have 4 elements.")

                if val2[0] > val[1] or val2[2] > val2[3]:
                    raise ValueError("Extent needs to be an array with [x0, x1, y0, y1] with x0 < x1 and y0 < y1.")

                super().__setattr__(key, val2)
                return
           
            case "_data":
                pc.check_type(key, val, np.ndarray)
                val2 = np.asarray_chkfinite(val, dtype=np.float64)
                super().__setattr__(key, val2)
                return
            
            case "limit" if val is not None:
                pc.check_type(key, val, float | int)
                pc.check_above(key, val, 0)
                val = float(val)

            case "quantity":
                pc.check_type(key, val, str)
            
            case "projection" if val is not None:
                pc.check_type(key, val, str)
                pc.check_if_element(key, val, SphericalSurface.sphere_projection_methods)

        super().__setattr__(key, val)

