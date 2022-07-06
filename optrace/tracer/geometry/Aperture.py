
"""
Aperture class:
"""

import numpy as np

from optrace.tracer.geometry.SObject import SObject
from optrace.tracer.geometry.Surface import Surface


class Aperture(SObject):

    abbr = "AP"  # object abbreviation 
    _allow_non_2D = False  # don't allow points or lines as surfaces

    def __init__(self, 
                 Surface:       Surface, 
                 pos:           (list | np.ndarray),
                 **kwargs)\
            -> None:
        """
        Create a Aperture object.

        :param Surface: Surface object
        :param pos: 3D position of Aperture center (numpy array or list)
        """

        super().__init__(Surface, pos, **kwargs)
     
        self._new_lock = True  # no new properties after this

