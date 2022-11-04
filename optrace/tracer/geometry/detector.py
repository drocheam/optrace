
"""
Detector class:
rectangular plane perpendicular to optical axis for creation of Detector Images
"""

import numpy as np  # for ndarray type
import numexpr as ne  # faster calculations

from .element import Element  # parent class
from ..misc import PropertyChecker as pc  # check types and values
from .surface import Surface, RectangularSurface, CircularSurface, SphericalSurface


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
        """
        super().__init__(surface, pos, **kwargs)
        self._new_lock = True  # no new properties after this

    def __setattr__(self, key, val):

        if key == "front" and isinstance(val, Surface):
            if not isinstance(val, SphericalSurface | RectangularSurface | CircularSurface):
                raise RuntimeError(f"Only surfaces with type Sphere, Rectangle, Circle are supported for detectors")

        super().__setattr__(key, val)
