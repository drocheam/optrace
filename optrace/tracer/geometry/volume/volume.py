
from typing import Any  # Any type

import numpy as np  # for ndarray type

from ...misc import PropertyChecker as pc  # check types and values
from ..element import Element
from ..surface.surface import Surface


class Volume(Element):

    def __init__(self,
                 front,
                 back, 
                 pos,
                 d1,
                 d2,
                 color:         tuple[float] = None,
                 opacity:       float = 0.3,
                 **kwargs)\
            -> None:
        """
        Create a volume object with two surfaces and thickness.
        A volume object is an Element that displays some 3D object and does not interact with rays.

        :param front: front surface
        :param back: back surface
        :param pos: position
        :param d1: thickness between front and position
        :param d2: thickness between position and back
        :param color: sRGB color tuple, optional
        :param opacity: plotting opacity, value range 0.0 - 1.0
        :param kwargs: additional keyword args for class Element and BaseClass
        """
        self.opacity = opacity
        self.color = color

        super().__init__(front, pos, back, d1=d1, d2=d2, **kwargs)

    def __setattr__(self, key: str, val: Any) -> None:
        """
        Assigns the value of an attribute.
        Also performs type checking.

        :param key: attribute name
        :param val: value to assign
        """

        match key:

            case "opacity":
                pc.check_type(key, val, float | int)
                pc.check_above(key, val, 0)
                pc.check_not_above(key, val, 1)

            case "color" if val is not None:
                pc.check_type(key, val, list | tuple)
                if len(val) != 3:
                    raise ValueError("Volume color needs to be a tuple of 3 float values")

        super().__setattr__(key, val)

