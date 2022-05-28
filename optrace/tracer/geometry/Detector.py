
"""
Detector class:
rectangular plane perpendicular to optical axis for creation of Detector Images
"""

import numpy as np

from optrace.tracer.geometry.Surface import *  # for the Detector surface
from optrace.tracer.geometry.SObject import *
import optrace.tracer.Misc as misc


class Detector(SObject):

    abbr = "DET"
    _allow_non_2D = False  # don't allow points or lines as surfaces

    def __init__(self,
                 Surface:   Surface,
                 pos:       (list | np.ndarray),
                 ar:        float=2.,
                 **kwargs)\
            -> None:
        """
        Create a Detector object.

        :param Surface: the Detector surface
        :param pos: position in 3D space
        :param ar: rho factor for AngleCoordinate Transformation
        """
        super().__init__(Surface, pos, **kwargs)

        self.ar = float(ar)
        
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
        zm = Surf.pos[2] + self.ar/Surf.rho
        ze = Surf.getEdge(nc=1)[2][0] # TODO what if no rotational symmetry?

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

        zm = z0 + self.ar/Surf.rho
        r = misc.calc("sqrt((x-x0)**2  + (y-y0)**2)")

        theta = misc.calc("arctan(r/abs(z-zm))*180/pi")
        phi = misc.calc("arctan2(y-y0, x-x0)")

        p_hit = p.copy()
        p_hit[:, 0] = misc.calc("theta*cos(phi)")
        p_hit[:, 1] = misc.calc("theta*sin(phi)")

        return p_hit

