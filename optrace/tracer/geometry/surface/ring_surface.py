
from typing import Any  # Any type

import numpy as np  # calculations

from ... import random  # calculation
from ....property_checker import PropertyChecker as pc  # check types and values
from .surface import Surface  # parent class

# TODO Hurb test

class RingSurface(Surface):
    
    rotational_symmetry: bool = True  #: has the surface rotational symmetry?

    def __init__(self,
                 r:                 float,
                 ri:                float,
                 **kwargs)\
            -> None:
        """
        Create a ring surface, an area between two concentric circles (also known as annulus)

        :param r: radial size for surface_type="Conic", "Sphere", "Circle" or "Ring" (float)
        :param ri: radius of inner circle for surface_type="Ring" (float)
        :param kwargs: additional keyword arguments for parent classes
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
        """info message describing the surface"""
        return super().info + ", ri = {self.ri:.5g} mm"

    def plotting_mesh(self, N: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get 2D plotting mesh. Note that the values are not gridded, the distance can be arbitrary.
        The only guarantee is that neighbouring array values are neighbouring values in 3D space.

        :param N: number of grid values in each dimension (int)
        :return: X, Y, Z coordinate array (all numpy 2D array)
        """
        X0, Y0, Z = super().plotting_mesh(N)
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

    def hurb_props(self, x: np.ndarray, y: np.ndarray)\
            -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculates the properties for Heisenberg Uncertainty Ray Bending.

        :param x: ray position at surface, x-coordinate
        :param y: ray position at surface, y-coordinate
        :return: distances axis 1, distances axis 2, minor axis vector, ray mask for rays to bend
        """
        # polar coordinates
        r = np.sqrt((x - self.pos[0])**2 + (y - self.pos[1])**2)
        theta = np.atan2(y - self.pos[1], x - self.pos[0])

        # fitting of largest ellipse inside circle
        # see STRAY LIGHT SIMULATION WITH ADVANCED MONTE CARLO TECHNIQUES*, Dr. Barry K. Likeness 
        # -> minor axis defined by distance to edge
        # -> major axis defined by ellipse with same curvature at end of minor axis

        # ellipse parameters
        R = self.ri
        inside = r < R
        b_ = R - r[inside]
        a_ = np.sqrt(b_*R)  
        # ^-- see https://math.stackexchange.com/questions/4511168/how-to-find-the-radius-of-the-smallest-circle-such-that-the-inner-ellipse-is-tan
    
        # ellipse minor axis as vector
        b = np.zeros((b_.shape[0], 3))
        b[:, 0] = np.cos(theta[inside]) 
        b[:, 1] = np.sin(theta[inside]) 

        return a_, b_, b, inside

    def mask(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Get surface mask values. A value of 1 means the surface is defined here.

        :param x: x-coordinate array (numpy 1D or 2D array)
        :param y: y-coordinate array (numpy 1D or 2D array)
        :return: z-coordinate array (numpy 1D or 2D array, depending on shape of x and y)
        """
        x0, y0, z0 = self.pos
        r2 = (x - x0) ** 2 + (y - y0) ** 2
        return ((self.ri - self.N_EPS)**2 <= r2) & (r2 <= (self.r + self.N_EPS)**2)

    def random_positions(self, N: int) -> np.ndarray:
        """
        Get random 3D positions on the surface, uniformly distributed.

        :param N: number of positions
        :return: position array, shape (N, 3)
        """
        x, y = random.stratified_ring_sampling(self.ri, self.r, N)
        
        p = np.tile(self.pos, (N, 1))
        p[:, 0] += x
        p[:, 1] += y

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
