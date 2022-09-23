
"""
Detector class:
rectangular plane perpendicular to optical axis for creation of Detector Images
"""

import numpy as np  # for ndarray type
import numexpr as ne  # faster calculations

from .surface import Surface  # for the Detector surface
from .element import Element  # parent class
from ..misc import PropertyChecker as pc  # check types and values



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
            if not val.has_hit_finding():
                raise RuntimeError(f"surface_type '{val.surface_type}' has no hit finding functionality.")
            if val.surface_type not in ["Sphere", "Rectangle", "Circle"]:
                raise RuntimeError(f"Only surfaces with type Sphere, Rectangle, Circle are supported for detectors")

        super().__setattr__(key, val)
