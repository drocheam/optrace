
from typing import Any  # Any type

import numpy as np  # matrix calculations

from ... import misc  # calculations
from ...misc import PropertyChecker as pc  # check types and values
from .surface import Surface  # parent class


class RectangularSurface(Surface):
    
    rotational_symmetry: bool = False  #: has the surface rotational symmetry?

    def __init__(self,
                 dim:               (list | np.ndarray),
                 **kwargs)\
            -> None:
        """
        Create a rectangular surface, perpendicular to the z-axis.

        :param dim: x and y side length as two element array/list
        :param kwargs: additional keyword arguments for parent classes
        """
        self._lock = False
      
        # angle for rotation
        self._angle = 0

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
        Surface extent, values for a smallest box encompassing all of the surface

        :return: tuple of x0, x1, y0, y1, z0, z1
        """
        # side lengths get rotated       
        sx = np.abs(self.dim[0] * np.cos(self._angle)) + np.abs(self.dim[1] * np.sin(self._angle))
        sy = np.abs(self.dim[0] * np.sin(self._angle)) + np.abs(self.dim[1] * np.cos(self._angle))
        
        return self.pos[0]-sx/2, self.pos[0]+sx/2,\
               self.pos[1]-sy/2, self.pos[1]+sy/2,\
               self.z_min, self.z_max

    @property
    def _extent(self) -> tuple[float, float, float, float, float, float]:
        """extent relative to center"""
        return -self.dim[0]/2, self.dim[0]/2, -self.dim[1]/2, self.dim[1]/2, 0., 0.
   
    def rotate(self, angle: float) -> None:
        """
        rotate the surface around the z-axis

        :param angle: rotation angle in degrees
        """
        # create an copy with an incremented angle
        self._lock = False
        self._angle += np.deg2rad(angle)
        self.lock()

    def flip(self) -> None:
        """flip the surface around the x-axis"""
        self._lock = False
        self._angle *= -1
        self.lock()

    def get_plotting_mesh(self, N: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get 2D plotting mesh. Note that the values are not gridded, the distance can be arbitrary.
        The only guarantee is that neighbouring array values are neighbouring values in 3D space.

        :param N: number of grid values in each dimension (int)
        :return: X, Y, Z coordinate array (all numpy 2D array)
        """
        if N < 10:
            raise ValueError("Expected at least N=10.")

        # create a rotated grid
        xs, xe, ys, ye = self._extent[:4]
        X, Y = np.mgrid[xs:xe:N*1j, ys:ye:N*1j]
        x2, y2 = self._rotate_rc(X.flatten(), Y.flatten(), self._angle)
        X, Y = self.pos[0] + x2.reshape(X.shape), self.pos[1] + y2.reshape(Y.shape)

        Z = np.full_like(Y, self.pos[2], dtype=np.float64)

        return X, Y, Z

    def get_mask(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Get surface mask values. A value of 1 means the surface is defined here.

        :param x: x-coordinate array (numpy 1D or 2D array)
        :param y: y-coordinate array (numpy 1D or 2D array)
        :return: z-coordinate array (numpy 1D or 2D array, depending on shape of x and y)
        """
        # instead of rotating the mask we rotate the relative point positions towards an unrotated rectangle
        xr, yr = self._rotate_rc(x-self.pos[0], y-self.pos[1], -self._angle)
        xs, xe, ys, ye = self._extent[:4]

        inside = (xs-self.N_EPS <= xr) & (xr <= xe+self.N_EPS) & (ys-self.N_EPS <= yr) & (yr <= ye+self.N_EPS)
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
        xs, xe, ys, ye = self._extent[:4]

        x = np.concatenate((np.linspace(xs, xe, N4),
                            np.full(N4, xe),
                            np.flip(np.linspace(xs, xe, N4)),
                            np.full(N4+dn, xs)))

        y = np.concatenate((np.full(N4, ys),
                            np.linspace(ys, ye, N4),
                            np.full(N4, ye),
                            np.flip(np.linspace(ys, ye, N4+dn))))

        # rotate the edge points around the rectangle center
        x, y = self._rotate_rc(x, y, self._angle)

        return self.pos[0]+x, self.pos[1]+y, np.full_like(y, self.pos[2])

    def get_random_positions(self, N: int) -> np.ndarray:
        """
        Get random 3D positions on the surface, uniformly distributed

        :param N: number of positions
        :return: position array, shape (N, 3)
        """
        p = np.zeros((N, 3), dtype=np.float64, order='F')
        # grid for unrotated rectangle at (0, 0, 0)
        x, y = misc.uniform2(*self._extent[:4], N)

        # rotate and add offset (=position)
        p[:, 0], p[:, 1] = self._rotate_rc(x, y, self._angle)
        p[:, 0] += self.pos[0]
        p[:, 1] += self.pos[1]
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
