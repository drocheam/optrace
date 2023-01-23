from typing import Any  # Any type

import numpy as np  # for ndarray type

from ..misc import PropertyChecker as pc  # check types and values
from .element import Element  # parent class
from .line import Line  # front object of the Marker


class LineMarker(Element):

    def __init__(self,
                 r:             float,
                 pos:           (list | np.ndarray),
                 desc:          str = "",
                 angle:         float = 0,
                 text_factor:   float = 1.,
                 line_factor:   float = 1.,
                 **kwargs)\
            -> None:
        """
        Create a new LineMarker.

        A LineMarker is an Element, where the front surface is a Line
        LineMarkers are used as line and text annotations in the tracing geometry

        :param r: radius of the line
        :param desc: text to display
        :param angle: angle of the line in xy-plane
        :param pos: position of marker
        :param text_factor: text scaling factor
        :param line_factor: marker scaling factor
        :param kwargs: additional keyword args for class Element and BaseClass
        """
        self.text_factor = text_factor
        self.line_factor = line_factor
        
        super().__init__(Line(r=r, angle=angle), pos, desc=desc, **kwargs)

        # lock object
        self._geometry_lock = True
        self._new_lock = True

    def __setattr__(self, key: str, val: Any) -> None:
        """
        Assigns the value of an attribute.
        Also performs type checking.

        :param key: attribute name
        :param val: value to assign
        """

        if key in ["text_factor", "line_factor"]:
                pc.check_type(key, val, float | int)
            
        super().__setattr__(key, val)

