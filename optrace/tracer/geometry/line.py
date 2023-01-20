
from typing import Any  # Any type

import numpy as np  # calculations

from ..base_class import BaseClass  # parent class
from ..misc import PropertyChecker as pc  # check types and values
from .. import misc  # for misc.uniform


class Line(BaseClass):

    def __init__(self,
                 r:             float = 3.,
                 angle:         float = 0,
                 **kwargs)\
            -> None:
        """
        Create a Line object. A Line lies in a plane perpendicular to the z-axis.

        :param r: radial size
        :param angle: axis angle in xy-plane, value in degrees
        :param kwargs: additional keyword arguments for parent classes
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
        Move the line in 3D space.

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
        Line extent, values for a smallest box encompassing all of the surface

        :return: tuple of x0, x1, y0, y1, z0, z1
        """
        ang = np.deg2rad(self.angle)
        return self.pos[0] - self.r * np.cos(ang),\
               self.pos[0] + self.r * np.cos(ang),\
               self.pos[1] - self.r * np.sin(ang),\
               self.pos[1] + self.r * np.sin(ang),\
               self.z_min,\
               self.z_max

    def flip(self) -> None:
        """flip the line around the x-axis"""
        self._lock = False
        self.angle *= -1  
        self.lock()

    def rotate(self, angle: float) -> None:
        """
        rotate the line around the z-axis

        :param angle: rotation angle in degrees
        """
        self._lock = False
        self.angle += angle 
        self.lock()

    def random_positions(self, N: int) -> np.ndarray:
        """
        Get random 3D positions on the line, uniformly distributed

        :param N: number of positions
        :return: position array, shape (N, 3)
        """
        p = np.zeros((N, 3), dtype=np.float64, order='F')

        ang = np.deg2rad(self.angle)
        t = misc.uniform(-self.r, self.r, N)
        p[:, 0] = self.pos[0] + np.cos(ang)*t
        p[:, 1] = self.pos[1] + np.sin(ang)*t
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

        super().__setattr__(key, val)
