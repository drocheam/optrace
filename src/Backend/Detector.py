
"""
Detector class:
rectangular plane perpendicular to optical axis for creation of Detector Images
"""

import numpy as np

import numexpr as ne  # for speeding up the calculations
import copy  # for copy.deepcopy

from Backend.Surface import Surface  # for the Detector surface

# TODO reference point for the angle calculation

class Detector:

    def __init__(self,
                 Surface:   Surface,
                 pos:       (list | np.ndarray))\
            -> None:
        """
        Create a Detector object.

        :param Surface: the Detector surface
        :param pos: position in 3D space
        """
        # use a Surface copy, since we change its position in 3D space
        self.Surface = Surface.copy()

        if not self.Surface.hasHitFinding:
            raise RuntimeError(f"surface_type '{Surface.surface_type}' has no hit finding functionality.")

        self.moveTo(pos)

    def moveTo(self, pos: (list | np.ndarray)) -> None:
        """
        Moves the Detector in 3D space.

        :param pos: new 3D position of Detector center (list or numpy array)
        """
        self.Surface.moveTo(pos)

    def setSurface(self, surf: Surface) -> None:
        """
        Assign a new Surface to the Detector.

        :param surf: Surface to assign
        """
        pos = self.Surface.pos
        self.Surface = surf.copy()
        self.Surface.moveTo(pos)
    
    def copy(self) -> 'Detector':
        """
        Return a fully independent copy of the Detector object.

        :return: copy
        """
        return copy.deepcopy(self)

    @property
    def pos(self) -> np.ndarray:
        """ position of the Detector center """
        return self.Surface.pos
   
    @property
    def extent(self) -> tuple[float, float, float, float, float, float]:
        """ 3D extent of the Detector"""
        return self.Surface.getExtent()
    
    def getCylinderSurface(self, nc: int = 100, d: float = 0.1) \
            -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get a 3D surface representation of the Detector cylinder for plotting.

        :param nc: number of surface edge points (int)
        :param d: thickness for visualization (float)
        :return: tuple of coordinate arrays X, Y, Z (2D numpy arrays)
        """

        # get Surface edge. The edge is the same for both cylinder sides
        X1, Y1, Z1 = self.Surface.getEdge(nc)

        # create coordinates for cylinder front edge and back edge
        X = np.column_stack((X1, X1))
        Y = np.column_stack((Y1, Y1))
        Z = np.column_stack((Z1, Z1 + d))  # shift back edge by d

        return X, Y, Z


    def getAngleExtent(self) -> np.ndarray:
        """

        :return:
        """

        if self.Surface.surface_type not in ["Sphere", "Asphere"]:
            raise RuntimeError(f"No angle conversion defined for surface_type '{self.Surface.surface_type}'.")

        Surf = self.Surface
        zm = Surf.pos[2] + 2/Surf.rho
        ze = Surf.getEdge(nc=1)[2][0]

        theta = np.arctan(Surf.r/np.abs(ze-zm))*180/np.pi
        return np.array([-theta, theta, -theta, theta])


    def toAngleCoordinates(self, p: np.ndarray) -> np.ndarray:
        """

        :param p:
        :return:
        """

        if self.Surface.surface_type not in ["Sphere", "Asphere"]:
            raise RuntimeError(f"No angle conversion defined for surface_type '{self.Surface.surface_type}'.")

        Surf = self.Surface

        x, y, z = p[:, 0], p[:, 1], p[:, 2]
        x0, y0, z0 = Surf.pos

        zm = z0 + 2/Surf.rho
        r = ne.evaluate("sqrt((x-x0)**2  + (y-y0)**2)")

        pi = np.pi
        theta = ne.evaluate("arctan(r/abs(z-zm))*180/pi")
        phi = ne.evaluate("arctan2(y-y0, x-x0)")

        p_hit = p.copy()
        p_hit[:, 0] = ne.evaluate("theta*cos(phi)")
        p_hit[:, 1] = ne.evaluate("theta*sin(phi)")

        return p_hit
