
import numpy as np  # calculations

from ..base_class import BaseClass  # parent class


class Point(BaseClass):

    def __init__(self,
                 **kwargs)\
            -> None:
        """
        Create a Point object. This is just a position in 3D space

        :param kwargs: additional keyword arguments for parent classes
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

    def flip(self) -> None:
        """flip the point around the x-axis"""
        pass
    
    def rotate(self, angle: float) -> None:
        """
        rotate the point around the z-axis

        :param angle: rotation angle in degrees
        """
        pass

    @property
    def extent(self) -> tuple[float, float, float, float, float, float]:
        """
        Point extent, values for a smallest box encompassing all of the surface

        :return: tuple of x0, x1, y0, y1, z0, z1
        """
        return tuple(self.pos.repeat(2))

    def random_positions(self, N: int) -> np.ndarray:
        """
        Get random positions, for a point that is just the list of its position.

        :param N: number of positions
        :return: positions, shape (N, 3)
        """
        return np.tile(self.pos, (N, 1))

