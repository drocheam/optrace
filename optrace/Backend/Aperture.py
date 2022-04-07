
"""
Aperture class:
"""


import numpy as np
from typing import Callable  # for function type hints

from optrace.Backend.SObject import *
from optrace.Backend.Surface import *  # for the Aperture surface


class Aperture(SObject):

    def __init__(self, 
                 Surface:       Surface, 
                 pos:           (list | np.ndarray))\
            -> None:
        """
        Create a Aperture object.

        :param Surface: Surface object
        :param pos: 3D position of Aperture center (numpy array or list)
        """

        super().__init__(Surface, pos)

        self.name = "Aperture"
        self.short_name = "AP"
      
        self._new_lock = True

