
from typing import Any  # Any type

import numpy as np  # matrix calculations

from ....property_checker import PropertyChecker as pc  # check types and values
from .rectangular_surface import RectangularSurface  # parent class




class SlitSurface(RectangularSurface):
    
    def __init__(self,
                 dim:               (list | np.ndarray),
                 dimi:              (list | np.ndarray),
                 **kwargs)\
            -> None:
        """
        Create a rectangular surface, perpendicular to the z-axis.

        :param dim: x and y side outer length as two element array/list
        :param dimi: x and y side slit lengths as two element array/list
        :param kwargs: additional keyword arguments for parent classes
        """
        super().__init__(dim, **kwargs)
        
        self._lock = False
        self._new_lock = False
        self.dimi = np.asarray_chkfinite(dimi, dtype=np.float64)
        self.lock()

    @property
    def info(self) -> str:
        """property string for UI information"""
        return super().info + ", dimi = [{self.dimi[0]:.5g} mm, {self.dimi[1]:.5g} mm]"

    def plotting_mesh(self, N: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get 2D plotting mesh. Note that the values are not gridded, the distance can be arbitrary.
        The only guarantee is that neighbouring array values are neighbouring values in 3D space.
        Parameter N has no effect for a slit surface and is only for compatibility to other classes.

        :param N: number of grid values in each dimension (int)
        :return: X, Y, Z coordinate array (all numpy 2D array)
        """
        y = np.array([self._extent[2], -self.dimi[1]/2, -self.dimi[1]/2+self.N_EPS, 
                      self.dimi[1]/2-self.N_EPS, self.dimi[1]/2, self._extent[3]])
        x = np.array([self._extent[0], -self.dimi[0]/2, -self.dimi[0]/2+self.N_EPS, 
                      self.dimi[0]/2-self.N_EPS, self.dimi[0]/2, self._extent[1]])

        # calculate rotated coordinates
        Y, X = np.meshgrid(y, x)
        x2, y2 = self._rotate_rc(X.flatten(), Y.flatten(), self._angle)
        X, Y = self.pos[0] + x2.reshape(X.shape), self.pos[1] + y2.reshape(Y.shape)

        # set inner values to nan to show the slit
        Z = np.full(Y.shape, np.float64(self.pos[2]))
        nm = np.zeros(Y.shape, dtype=bool)
        nm[2:4, 2:4] = True
        Z[nm] = np.nan

        return X, Y, Z
    
    def hurb_props(self, x: np.ndarray, y: np.ndarray)\
            -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculates the properties for Heisenberg Uncertainty Ray Bending.

        :param x: ray position at surface, x-coordinate
        :param y: ray position at surface, y-coordinate
        :return: distances axis 1, distances axis 2, left-right axis vector, ray mask for rays to bend
        """
        # rotate coordinates to rectangle coordinate system
        x_, y_ = self._rotate_rc(x - self.pos[0], y - self.pos[1], -self._angle)

        # see Edge diffraction in Monte Carlo ray tracing Edward R. Freniere, G. Groot Gregory, and Richard A. Hassler
        a_ = self.dimi[1] / 2 - np.abs(y_)
        b_ = self.dimi[0] / 2 - np.abs(x_)
        inside = (a_ > 0) & (b_ > 0)

        # side vector left-right
        b = np.zeros((b_.shape[0], 3))
        b[:, 0] = np.cos(self._angle) 
        b[:, 1] = np.sin(self._angle) 

        return a_, b_, b, inside

    def mask(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Get surface mask values. A value of 1 means the surface is defined here.

        :param x: x-coordinate array (numpy 1D or 2D array)
        :param y: y-coordinate array (numpy 1D or 2D array)
        :return: z-coordinate array (numpy 1D or 2D array, depending on shape of x and y)
        """
        # instead of rotating the mask we rotate the relative point positions towards an unrotated rectangle
        xr, yr = self._rotate_rc(x-self.pos[0], y-self.pos[1], -self._angle)
        xs, xe, ys, ye = -self.dimi[0]/2, self.dimi[0]/2, -self.dimi[1]/2, self.dimi[1]/2
        inside = (xs+self.N_EPS <= xr) & (xr <= xe-self.N_EPS) & (ys+self.N_EPS <= yr) & (yr <= ye-self.N_EPS)

        return super().mask(x, y) & (~inside)

    def __setattr__(self, key: str, val: Any) -> None:
        """
        assigns the value of an attribute
        :param key: attribute name
        :param val: value to assign
        """

        if key == "dimi":
            pc.check_type(key, val, np.ndarray)

            if val.ndim != 1 or val.shape[0] != 2:
                raise TypeError("dimi needs to have two elements.")

            if val[0] >= self.dim[0] or val[1] >= self.dim[1]:
                raise ValueError("Dimensions dimi must be smaller than dimension dim.")

            if val[0] <= 0 or val[1] <= 0:
                raise ValueError("Dimensions dimi need to be positive, but are {dim=}")

        super().__setattr__(key, val)
