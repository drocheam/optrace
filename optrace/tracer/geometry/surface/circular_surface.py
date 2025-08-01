
import numpy as np  # calculations

from ... import random  # calculations
from .surface import Surface  # parent class


class CircularSurface(Surface):

    rotational_symmetry: bool = True  # has the surface rotational symmetry?
    
    def __init__(self,
                 r:        float,
                 **kwargs)\
            -> None:
        """
        Create a surface object.
        The z-coordinate in the pos array is irrelevant if the surface is used for a lens, since it wil be
        adapted inside the lens class

        :param r: radial size for surface_type="Conic", "Sphere", "Circle" or "Ring" (float)
        :param kwargs: additional keyword arguments for parent classes
        """
        self._lock = False

        super().__init__(r, **kwargs)

        self.parax_roc = np.inf
        self.z_min = self.z_max = self.pos[2]
        self.lock()

    def random_positions(self, N: int) -> np.ndarray:
        """
        Get 3D random positions on the surface, uniformly distributed.

        :param N: number of positions
        :return: position array, shape (N, 3)
        """
        p = np.zeros((N, 3), dtype=np.float64, order='F')
        p[:, 0], p[:, 1] = random.stratified_ring_sampling(0, self.r, N)
        p += self.pos

        return p

