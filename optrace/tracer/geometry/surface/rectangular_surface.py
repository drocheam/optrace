
from typing import Any  # Any type

import numpy as np  # calculations

from ... import misc  # calculations
from ...misc import PropertyChecker as pc  # check types and values
from .surface import Surface


class RectangularSurface(Surface):

    def __init__(self,
                 dim:               (list | np.ndarray),
                 **kwargs)\
            -> None:
        """
        Create a surface object.
        The z-coordinate in the pos array is irrelevant if the surface is used for a lens, since it wil be
        adapted inside the lens class

        :param dim:
        """
        self._lock = False
        
        super().__init__(1, **kwargs)
        
        self.dim = np.asarray_chkfinite(dim, dtype=np.float64)
        self.parax_roc = np.inf
        self.z_min = self.z_max = self.pos[2]
        
        self.lock()

    @property
    def info(self) -> str:
        """property string for UI information"""
        return f"{type(self).__name__}, pos = [{self.pos[0]:.5g} mm, {self.pos[1]:.5g} mm, "\
                f"{self.pos[2]:.5g} mm], dim = [{self.dim[0]:.5g} mm, {self.dim[1]:.5g} mm]"

    @property
    def extent(self) -> tuple[float, float, float, float, float, float]:
        """
        """
        return *(self.pos[:2].repeat(2) + self.dim.repeat(2)/2 * np.array([-1, 1, -1, 1])), \
               self.z_min, self.z_max

    def get_plotting_mesh(self, N: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get 2D plotting mesh. Note that the values are not gridded, the distance can be arbitrary.
        The only guarantee is that neighbouring array values are neighbouring values in 3D space.

        :param N: number of grid values in each dimension (int)
        :return: X, Y, Z coordinate array (all numpy 2D array)
        """
        if N < 10:
            raise ValueError("Expected at least N=10.")

        xs, xe, ys, ye, _, _ = self.extent

        X, Y = np.mgrid[xs:xe:N*1j, ys:ye:N*1j]
        Z = np.full_like(Y, self.pos[2], dtype=np.float64)

        return X, Y, Z

    def get_mask(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Get surface mask values. A value of 1 means the surface is defined here.

        :param x: x-coordinate array (numpy 1D or 2D array)
        :param y: y-coordinate array (numpy 1D or 2D array)
        :return: z-coordinate array (numpy 1D or 2D array, depending on shape of x and y)
        """
        xs, xe, ys, ye = self.extent[:4]
        inside = (xs-self.N_EPS <= x) & (x <= xe+self.N_EPS) & (ys-self.N_EPS <= y) & (y <= ye+self.N_EPS)
        return inside

    def get_edge(self, nc: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get surface values of the surface edge, assumes a circular edge.

        :param nc: number of points on edge (int)
        :return: X, Y, Z coordinate arrays (all numpy 2D array)
        """

        if nc < 20:
            raise ValueError("Expected at least nc=20")

        N4 = int(nc/4)
        dn = nc - 4*N4
        xs, xe, ys, ye = self.extent[:4]

        x = np.concatenate((np.linspace(xs, xe, N4),
                            np.full(N4, xe),
                            np.flip(np.linspace(xs, xe, N4)),
                            np.full(N4+dn, xs)))

        y = np.concatenate((np.full(N4, ys),
                            np.linspace(ys, ye, N4),
                            np.full(N4, ye),
                            np.flip(np.linspace(ys, ye, N4+dn))))

        return x, y, np.full_like(y, self.pos[2])

    def get_random_positions(self, N: int) -> np.ndarray:
        """

        :param N:
        :return:
        """
        p = np.zeros((N, 3), dtype=np.float64, order='F')
        p[:, 0], p[:, 1] = misc.uniform2(*self.extent[:4], N)
        p[:, 2] = self.pos[2]

        return p

    def __setattr__(self, key: str, val: Any) -> None:
        """
        assigns the value of an attribute
        :param key: attribute name
        :param val: value to assign
        """

        if key == "dim":
            pc.check_type(key, val, np.ndarray)

            if val.ndim != 1 or val.shape[0] != 2:
                raise TypeError("dim needs to have two elements.")

            if val[0] <= 0 or val[1] <= 0:
                raise ValueError("Dimensions dim need to be positive, but are {dim=}")

        super().__setattr__(key, val)
