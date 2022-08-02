"""
Aperture class:
"""

import numpy as np  # for ndarray type

from optrace.tracer.geometry.s_object import SObject  # parent class
from optrace.tracer.geometry.surface import Surface  # for Surface type


class Aperture(SObject):

    abbr = "AP"  # object abbreviation
    _allow_non_2D = False  # don't allow points or lines as surfaces

    def __init__(self,
                 surface:       Surface,
                 pos:           (list | np.ndarray),
                 **kwargs)\
            -> None:
        """
        Create a Aperture object.

        :param surface: Surface object
        :param pos: 3D position of Aperture center (numpy array or list)
        :param kwargs:
        """

        super().__init__(surface, pos, **kwargs)

        self._new_lock = True  # no new properties after this
