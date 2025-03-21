from typing import Any  # Any type

import numpy as np  # for ndarray type

from ....property_checker import PropertyChecker as pc  # check types and values
from ..element import Element  # parent class
from ..point import Point  # front object of the Marker


class PointMarker(Element):

    def __init__(self,
                 desc:          str,
                 pos:           (list | np.ndarray),
                 text_factor:   float = 1.,
                 marker_factor: float = 1.,
                 label_only:    bool = False,
                 **kwargs)\
            -> None:
        """
        Create a new PointMarker.

        A PointMarker is an Element, where the front surface is a point
        Markers are used as point and/or text annotations in the tracing geometry
        text only: provide a text string and set the parameter label_only=True
        marker/point only: leave the text empty

        :param desc: text to display
        :param pos: position of marker
        :param text_factor: text scaling factor
        :param marker_factor: marker scaling factor
        :param label_only: don't plot marker, only text
        :param kwargs: additional keyword args for class Element and BaseClass
        """
        self.marker_factor = marker_factor
        self.text_factor = text_factor
        self.label_only = label_only
        
        super().__init__(Point(), pos, desc=desc, **kwargs)

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

        match key:

            case ("text_factor" | "marker_factor"):
                pc.check_type(key, val, float | int)
            
            case "label_only":
                pc.check_type(key, val, bool)

        super().__setattr__(key, val)

