
"""
Detector class:
rectangular plane perpendicular to optical axis for creation of Detector Images
"""

import numpy as np  # for ndarray type
import numexpr as ne  # faster calculations

from optrace.tracer.geometry.surface import Surface  # for the Detector surface
from optrace.tracer.geometry.s_object import SObject  # parent class
from optrace.tracer.misc import PropertyChecker as pc  # check types and values


class Detector(SObject):

    abbr = "DET"  # object abbreviation
    _allow_non_2D = False  # don't allow points or lines as surfaces

    def __init__(self,
                 surface:   Surface,
                 pos:       (list | np.ndarray),
                 ar:        float = 2.,
                 **kwargs)\
            -> None:
        """
        Create a Detector object.

        :param surface: the Detector surface
        :param pos: position in 3D space
        :param ar: rho factor for getAngleCoordinate() transformation
        """
        super().__init__(surface, pos, **kwargs)

        self.ar = ar
        self._new_lock = True  # no new properties after this

    def __setattr__(self, key, val):

        if key == "ar":
            pc.checkType(key, val, float | int)
            val = float(val)

        elif key == "Surface" and isinstance(val, Surface):
            if not val.has_hit_finding:
                raise RuntimeError(f"surface_type '{val.surface_type}' has no hit finding functionality.")

        super().__setattr__(key, val)

    def get_angle_extent(self) -> np.ndarray:
        """

        :return:
        """

        # Currently only works for predefined, rotational symmetric surfaces

        if self.surface.surface_type not in ["Sphere", "Asphere", "Circle"]:
            raise RuntimeError(f"No angle conversion defined for surface_type '{self.surface.surface_type}'.")

        Surf = self.surface
        zm = Surf.pos[2] + self.ar/Surf.rho
        ze = Surf.get_edge(nc=1)[2][0]  # only works for rotational symmetric surfaces

        theta = np.arctan(Surf.r/np.abs(ze-zm))*180/np.pi
        return np.array([-theta, theta, -theta, theta])

    def to_angle_coordinates(self, p: np.ndarray) -> np.ndarray:
        """

        :param p:
        :return:
        """

        if self.surface.surface_type not in ["Sphere", "Asphere"]:
            raise RuntimeError(f"No angle conversion defined for surface_type '{self.surface.surface_type}'.")

        Surf = self.surface

        x, y, z = p[:, 0], p[:, 1], p[:, 2]
        x0, y0, z0 = Surf.pos

        zm = z0 + self.ar/Surf.rho
        r = ne.evaluate("sqrt((x-x0)**2  + (y-y0)**2)")
    
        pi = np.pi
        theta = ne.evaluate("arctan(r/abs(z-zm))*180/pi")
        phi = ne.evaluate("arctan2(y-y0, x-x0)")

        p_hit = p.copy()
        p_hit[:, 0] = ne.evaluate("theta*cos(phi)")
        p_hit[:, 1] = ne.evaluate("theta*sin(phi)")

        return p_hit
