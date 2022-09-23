
"""
"""

import numpy as np  # for ndarray type

from ..misc import PropertyChecker as pc
from ..base_class import BaseClass


class Marker(BaseClass):


    def __init__(self,
                 desc:          str,
                 pos:           (list | np.ndarray),
                 text_factor:   float = 1.,
                 marker_factor: float = 1.,
                 **kwargs)\
            -> None:
        """
        """
        self.marker_factor = marker_factor
        self.text_factor = text_factor
        self.pos = pos

        super().__init__(desc=desc, **kwargs)
        self._new_lock = True  # no new properties after this

    # so we have the same function for Markers and elements
    def move_to(self, pos: np.ndarray | list):
        self.pos = pos

    @property
    def extent(self) -> tuple[float, float, float, float, float, float]:
        """"""
        return self.pos.repeat(2)

    def __setattr__(self, key, val):
        """"""

        match key:

            case ("text_factor" | "marker_factor"):
                pc.check_type(key, val, float | int)

            case "pos":
                pc.check_type(key, val, list | np.ndarray)
                val2 = np.asarray_chkfinite(val, dtype=np.float64)
                super().__setattr__(key, val2)
                return

        super().__setattr__(key, val)

