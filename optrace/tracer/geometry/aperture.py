
import numpy as np  # for ndarray type

from .element import Element  # parent class
from .surface import Surface  # for Surfaces


class Aperture(Element):

    abbr: str = "AP"  #: object abbreviation
    _allow_non_2D: bool = False  # don't allow points or lines as surfaces

    def __init__(self,
                 surface:       Surface,
                 pos:           (list | np.ndarray),
                 **kwargs)\
            -> None:
        """
        Create a Aperture object. Used for absorption of rays.

        :param surface: Surface object
        :param pos: 3D position of Aperture center (numpy array or list)
        :param kwargs: keyword parameters for parent classes
        """
        super().__init__(surface, pos, **kwargs)
        self._new_lock = True  # no new properties after this
