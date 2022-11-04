
"""
Surface class:
Provides the functionality for the creation of numerical or analytical surfaces.
The class contains methods for interpolation, surface masking and normal calculation
"""

import numpy as np  # calculations
import numexpr as ne  # faster calculations

from .conic_surface import ConicSurface


class SphericalSurface:
    pass


class SphericalSurface(ConicSurface):

    sphere_projection_methods: list[str] = ["Equidistant", "Equal-Area", "Stereographic"]
    """projection methods for mapping a sphere surface onto a plane"""

    def __init__(self,
                 r:                 float,
                 R:                 float,
                 **kwargs)\
            -> None:
        """
        Create a surface object.
        The z-coordinate in the pos array is irrelevant if the surface is used for a lens, since it wil be
        adapted inside the lens class

        :param R: curvature circle for surface_type="Conic" or "Sphere" (float)
        :param r: radial size for surface_type="Conic", "Sphere", "Circle" or "Ring" (float)
        """
        self._lock = False
        
        super().__init__(r, R, k=0, **kwargs)
        self.lock()

    def sphere_projection(self, p: np.ndarray, projection_method: str = "Equidistant") -> np.ndarray:
        """

        Sign convention: projection coordinates are positive, 
        when they point in the same direction as cartesian coordinates for the corresponding axis.
        E.g: cartesian coordinates x = +5, y = -2 relative to sphere center 
        would have some projection coordinate px > 0, py < 0

        :param p:
        :param projection_method:
        :return:
        """

        x, y, z = p[:, 0], p[:, 1], p[:, 2]
        x0, y0, z0 = self.pos
        R = self.R
        Rs = np.sign(R)
        pi = np.pi

        zm = z0 + R  # center z-position of sphere
        r = ne.evaluate("sqrt((x-x0)**2  + (y-y0)**2)")

        # different sign convention than for mathematical rotation angles
        # in our case the coordinates are positive, when they point in the same direction
        # as the values in the cartesian axes

        if projection_method == "Equidistant":
            # https://en.wikipedia.org/wiki/Azimuthal_equidistant_projection

            theta = ne.evaluate("arctan(r/(z-zm))")
            phi = ne.evaluate("arctan2(y-y0, x-x0)")
        
            p_hit = p.copy()
            p_hit[:, 0] = ne.evaluate("-Rs*theta*cos(phi)")
            p_hit[:, 1] = ne.evaluate("-Rs*theta*sin(phi)")

        elif projection_method == "Stereographic":

            # https://en.wikipedia.org/wiki/Stereographic_map_projection
            theta = ne.evaluate("pi/2 - arctan(r/(z-zm))")
            phi = ne.evaluate("arctan2(y-y0, x-x0)")
            r = ne.evaluate("2*tan(pi/4 - theta/2)")
            
            p_hit = p.copy()
            p_hit[:, 0] = -r*Rs*np.cos(phi)
            p_hit[:, 1] = -r*Rs*np.sin(phi)

        elif projection_method == "Equal-Area":

            # https://en.wikipedia.org/wiki/Lambert_azimuthal_equal-area_projection
            x_ = (x - x0) / np.abs(R)
            y_ = (y - y0) / np.abs(R)
            z_ = (z - zm) / R

            p_hit = p.copy()
            p_hit[:, 0] = ne.evaluate("sqrt(2/(1-z_))*x_")
            p_hit[:, 1] = ne.evaluate("sqrt(2/(1-z_))*y_")

        else:
            raise ValueError(f"Invalid projection_method {projection_method}, "
                             f"must be one of {self.sphere_projection_methods}.")

        return p_hit

    def reverse(self) -> SphericalSurface:

        return SphericalSurface(r=self.r, R=-self.R, desc=self.desc, long_desc=self.long_desc,
                                silent=self.silent, threading=self.threading)

