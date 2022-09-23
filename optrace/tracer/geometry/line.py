

from typing import Any  # Any type

import numpy as np  # calculations

from ..base_class import BaseClass  # parent class
from ..misc import PropertyChecker as pc  # check types and values
from .. import misc


class Line(BaseClass):

    def __init__(self,
                 r:             float = 3.,
                 angle:         float = 0,
                 **kwargs)\
            -> None:
        """

        :param r: radial size
        :param ang: in degrees
        """
        self._lock = False

        self.pos = np.asarray_chkfinite([0., 0., 0.], dtype=np.float64)
        self.r = r
        self.angle = angle 
        self.z_min = self.z_max = self.pos[2]

        super().__init__(**kwargs)
        self.lock()

    def move_to(self, pos: (list | np.ndarray)) -> None:
        """

        :param pos: 3D position to move to (list or numpy 1D array)
        """

        self._lock = False

        # update position
        self.pos = np.asarray_chkfinite(pos, dtype=np.float64)
        self.z_min = pos[2]
        self.z_max = pos[2]

        self.lock()

    @property
    def extent(self) -> tuple[float, float, float, float, float, float]:
        """
        """
        return self.pos[0] - self.r * np.cos(self.angle),\
               self.pos[0] + self.r * np.cos(self.angle),\
               self.pos[1] - self.r * np.sin(self.angle),\
               self.pos[1] + self.r * np.sin(self.angle),\
               self.z_min,\
               self.z_max

    def get_random_positions(self, N: int) -> np.ndarray:
        """

        :param N:
        :return:
        """
        p = np.zeros((N, 3), dtype=np.float64, order='F')

        t = misc.uniform(-self.r, self.r, N)
        p[:, 0] = self.pos[0] + np.cos(self.angle)*t
        p[:, 1] = self.pos[1] + np.sin(self.angle)*t
        p[:, 2] = self.pos[2]

        return p

    def __setattr__(self, key: str, val: Any) -> None:
        """
        assigns the value of an attribute
        :param key: attribute name
        :param val: value to assign
        """

        if key in ["r", "angle"]:
            pc.check_type(key, val, float | int)
            val = float(val)

            if key == "r":
                pc.check_above(key, val, 0)
            else:  # convert to radians
                val = val / 180 * np.pi

        super().__setattr__(key, val)
