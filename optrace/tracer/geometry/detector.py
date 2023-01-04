from typing import Any  # Any type

import numpy as np  # for ndarray type

from .element import Element  # parent class
from .surface import Surface, DataSurface1D, DataSurface2D,\
        FunctionSurface1D, FunctionSurface2D  # supported surface types


class Detector(Element):

    abbr: str = "DET"  #: object abbreviation
    _allow_non_2D: bool = False  # don't allow points or lines as surfaces

    def __init__(self,
                 surface:             Surface,
                 pos:                 (list | np.ndarray),
                 **kwargs)\
            -> None:
        """
        Create a Detector object.

        :param surface: the Detector surface
        :param pos: position in 3D space
        :param kwargs: additional keyword arguments for parent classes
        """
        super().__init__(surface, pos, **kwargs)
        self._new_lock = True  # no new properties after this

    def __setattr__(self, key: str, val: Any) -> None:
        """
        Assigns the value of an attribute.
        Also performs type checking.

        :param key: attribute name
        :param val: value to assign
        """

        if key == "front" and isinstance(val, DataSurface2D | DataSurface1D | FunctionSurface1D | FunctionSurface2D):
            raise RuntimeError("Classes and subclasses of DataSurface1D, DataSurface2D, FunctionSurface2D"\
                               " are not supported as Detector surfaces.")

        super().__setattr__(key, val)
