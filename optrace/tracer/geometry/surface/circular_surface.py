
"""
Surface class:
Provides the functionality for the creation of numerical or analytical surfaces.
The class contains methods for interpolation, surface masking and normal calculation
"""

import numpy as np  # calculations
import numexpr as ne  # faster calculations

from ... import misc  # calculations
from .surface import Surface



class CircularSurface(Surface):

    rotational_symmetry: bool = True
    

    def __init__(self,
                 r:        float,
                 **kwargs)\
            -> None:
        """
        Create a surface object.
        The z-coordinate in the pos array is irrelevant if the surface is used for a lens, since it wil be
        adapted inside the lens class

        :param r: radial size for surface_type="Conic", "Sphere", "Circle" or "Ring" (float)
        """
        self._lock = False

        super().__init__(r, **kwargs)

        self.parax_roc = np.inf
        self.z_min = self.z_max = self.pos[2]
        self.lock()

    def get_random_positions(self, N: int) -> np.ndarray:
        """

        :param N:
        :return:
        """
        # x, y = misc.ring_uniform(0, self.r, N)
        x, y = misc.ring_uniform(0, self.r, N)

        p = np.tile(self.pos, (N, 1))
        p[:, 0] += x
        p[:, 1] += y

        return p

