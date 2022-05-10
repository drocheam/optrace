
"""
Aperture class:
"""

import numpy as np
from typing import Callable  # for function type hints

from optrace.tracer.SObject import *
from optrace.tracer.Surface import *  # for the Aperture surface


class Aperture(SObject):

    abbr = "AP"

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
      
        self._new_lock = True

