
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
        p = np.zeros((N, 3), dtype=np.float64, order='F')

        # weight with square root to get equally distributed points
        r, theta = misc.uniform2(0, self.r**2, 0, 2*np.pi, N)
        r = np.sqrt(r)

        x0, y0 = self.pos[0], self.pos[1]

        p[:, 0] = ne.evaluate("x0 + r*cos(theta)")
        p[:, 1] = ne.evaluate("y0 + r*sin(theta)")
        p[:, 2] = self.pos[2]

        return p
