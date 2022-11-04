
from typing import Any  # Any type

import numpy as np  # calculations

from ..base_class import BaseClass  # parent class


class Point:
    pass

class Point(BaseClass):

    def __init__(self,
                 **kwargs)\
            -> None:
        """
        """
        self._lock = False
                
        self.pos = np.array([0., 0., 0.], dtype=np.float64)
        self.z_min = self.z_max = self.pos[2]

        super().__init__(**kwargs)
        self.lock()

    def move_to(self, pos: (list | np.ndarray)) -> None:
        """
        Moves the surface in 3D space.

        :param pos: 3D position to move to (list or numpy 1D array)
        """

        self._lock = False

        # update position
        self.pos = np.asarray_chkfinite(pos, dtype=np.float64)
        self.z_min = pos[2]
        self.z_max = pos[2]

        self.lock()

    def reverse(self) -> Point:
        P = self.copy()
        P.move_to([0, 0, 0])
        return P

    @property
    def extent(self) -> tuple[float, float, float, float, float, float]:
        """
        """
        return tuple(self.pos.repeat(2))

    def get_random_positions(self, N: int) -> np.ndarray:
        """

        :param N:
        :return:
        """
        return np.tile(self.pos, (N, 1))

