
"""
Surface class:
Provides the functionality for the creation of numerical or analytical surfaces.
The class contains methods for interpolation, surface masking and normal calculation
"""

from typing import Any  # Any type

import numpy as np  # calculations
import numexpr as ne  # faster calculations


from ... import misc  # calculation
from ...misc import PropertyChecker as pc  # check types and values
from .surface import Surface


class RingSurface(Surface):
    
    rotational_symmetry: bool = True


    def __init__(self,
                 r:                 float,
                 ri:                float,
                 **kwargs)\
            -> None:
        """
        Create a surface object.
        The z-coordinate in the pos array is irrelevant if the surface is used for a lens, since it wil be
        adapted inside the lens class

        :param r: radial size for surface_type="Conic", "Sphere", "Circle" or "Ring" (float)
        :param ri: radius of inner circle for surface_type="Ring" (float)
        """
        self._lock = False
        
        super().__init__(r, **kwargs)
        
        self.r, self.ri = r, ri
        self.parax_roc = np.inf
        
        self.z_min = self.z_max = self.pos[2]
        if ri >= r:
            raise ValueError("ri needs to be smaller than r.")

        self.lock()

    @property
    def info(self) -> str:
        return super().info + ", ri = {self.ri:.5g} mm"

    def get_plotting_mesh(self, N: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get 2D plotting mesh. Note that the values are not gridded, the distance can be arbitrary.
        The only guarantee is that neighbouring array values are neighbouring values in 3D space.

        :param N: number of grid values in each dimension (int)
        :return: X, Y, Z coordinate array (all numpy 2D array)
        """
        X0, Y0, Z = super().get_plotting_mesh(N)
        X, Y = X0 - self.pos[0], Y0 - self.pos[1]

        # convert to polar coordinates
        R = np.sqrt(X**2 + Y**2)
        Phi = np.arctan2(Y, X)
        
        # extra precautions to plot the inner ring
        # otherwise the inner circle or the ring could be too small to resolve, depending on the plotting resolution

        # move points near inner edge towards the edge line
        # create two circles, one slightly outside the edge (mask5)
        # and one slightly below (mask4)

        # ring larger
        if self.ri < self.r/2:
            rr = self.r - self.ri  # diameter of ring area
            mask4 = R <= (self.ri + rr/3)
            mask5 = (R > (self.ri + rr/3)) & (R < (self.ri + 2/3*rr))
        # diameter of inner circle larger
        else:
            mask4 = (R < self.ri/2)
            mask5 = (R < self.ri) & (R >= self.ri/2)

        # move points onto the two circles
        # we need to shift by more than eps, since values with eps are still on the surface
        X[mask4] = (self.ri - 4*self.N_EPS) * np.cos(Phi[mask4])
        Y[mask4] = (self.ri - 4*self.N_EPS) * np.sin(Phi[mask4])
        Z[mask4] = np.nan
        X[mask5] = (self.ri + 4*self.N_EPS) * np.cos(Phi[mask5])
        Y[mask5] = (self.ri + 4*self.N_EPS) * np.sin(Phi[mask5])
        Z[mask5] = self.pos[2]

        return X+self.pos[0], Y+self.pos[1], Z

    def get_mask(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Get surface mask values. A value of 1 means the surface is defined here.

        :param x: x-coordinate array (numpy 1D or 2D array)
        :param y: y-coordinate array (numpy 1D or 2D array)
        :return: z-coordinate array (numpy 1D or 2D array, depending on shape of x and y)
        """
        x0, y0, z0 = self.pos
        r2 = ne.evaluate("(x - x0) ** 2 + (y - y0) ** 2")
        return ((self.ri - self.N_EPS)**2 <= r2) & (r2 <= (self.r + self.N_EPS)**2)

    def get_random_positions(self, N: int) -> np.ndarray:
        """

        :param N:
        :return:
        """
        p = np.zeros((N, 3), dtype=np.float64, order='F')

        # weight with square root to get equally distributed points
        r, theta = misc.uniform2(self.ri**2, self.r**2, 0, 2*np.pi, N)
        r = np.sqrt(r)

        x0, y0 = self.pos[0], self.pos[1]

        p[:, 0] = ne.evaluate("x0 + r*cos(theta)")
        p[:, 1] = ne.evaluate("y0 + r*sin(theta)")
        p[:, 2] = self.pos[2]

        return p

    def __setattr__(self, key: str, val: Any) -> None:
        """
        assigns the value of an attribute
        :param key: attribute name
        :param val: value to assign
        """
        if key == "ri":
            pc.check_type(key, val, float | int)
            val = float(val)
            pc.check_above(key, val, 0)

        super().__setattr__(key, val)
