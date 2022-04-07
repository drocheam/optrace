
"""
Detector class:
rectangular plane perpendicular to optical axis for creation of Detector Images
"""

import numpy as np

from optrace.Backend.Surface import *  # for the Detector surface
from optrace.Backend.SObject import *
import optrace.Backend.Misc as misc

# TODO reference point for the angle calculation

class Detector(SObject):

    def __init__(self,
                 Surface:   Surface,
                 pos:       (list | np.ndarray))\
            -> None:
        """
        Create a Detector object.

        :param Surface: the Detector surface
        :param pos: position in 3D space
        """
        super().__init__(Surface, pos)

        self.name = "Detector"
        self.short_name = "DET"
        
        if not self.Surface.hasHitFinding:
            raise RuntimeError(f"surface_type '{Surface.surface_type}' has no hit finding functionality.")

        self._new_lock = True

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
        r = misc.calc("sqrt((x-x0)**2  + (y-y0)**2)")

        theta = misc.calc("arctan(r/abs(z-zm))*180/pi")
        phi = misc.calc("arctan2(y-y0, x-x0)")

        p_hit = p.copy()
        p_hit[:, 0] = misc.calc("theta*cos(phi)")
        p_hit[:, 1] = misc.calc("theta*sin(phi)")

        return p_hit

