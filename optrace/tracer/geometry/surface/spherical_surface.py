
import numpy as np  # calculations

from .conic_surface import ConicSurface  # parent class


class SphericalSurface(ConicSurface):

    sphere_projection_methods: list[str] = ["Equidistant", "Orthographic", "Equal-Area", "Stereographic"]
    """projection methods for mapping a sphere surface onto a plane"""
    
    rotational_symmetry: bool = True  #: has the surface rotational symmetry?

    def __init__(self,
                 r:                 float,
                 R:                 float,
                 **kwargs)\
            -> None:
        """
        Create a spherical surface object.

        :param R: curvature circle for surface_type="Conic" or "Sphere" (float)
        :param r: radial size for surface_type="Conic", "Sphere", "Circle" or "Ring" (float)
        :param kwargs: additional keyword arguments for parent classes
        """
        self._lock = False
        
        super().__init__(r, R, k=0, **kwargs)
        self.lock()

    @property
    def info(self) -> str:
        """property string for UI information"""
        return super(ConicSurface, self).info + f", R = {self.R:.5g} mm"

    def sphere_projection(self, p: np.ndarray, projection_method: str = "Equidistant") -> np.ndarray:
        """

        Sign convention: projection coordinates are positive, 
        when they point in the same direction as cartesian coordinates for the corresponding axis.
        E.g: cartesian coordinates x = +5, y = -2 relative to sphere center 
        would have some projection coordinate px > 0, py < 0

        :param p: cartesian positions in 3D space
        :param projection_method: one of sphere_projection_methods defined in this class
        :return: projected coordinates, same shape as input
        """
        if projection_method == "Orthographic":
            return p.copy()

        x, y, z = p[:, 0], p[:, 1], p[:, 2]
        x0, y0, z0 = self.pos
        zm = z0 + self.R  # center z-position of sphere

        # different sign convention than for mathematical rotation angles
        # in our case the coordinates are positive, when they point in the same direction
        # as the values in the cartesian axes

        if projection_method == "Equidistant":
            # https://en.wikipedia.org/wiki/Azimuthal_equidistant_projection

            r = np.sqrt((x-x0)**2  + (y-y0)**2)
            theta = -np.sign(self.R)*np.arctan(r/(z-zm))
            phi = np.arctan2(y-y0, x-x0)
        
            p_hit = p.copy()
            p_hit[:, 0] = theta*np.cos(phi)
            p_hit[:, 1] = theta*np.sin(phi)

        elif projection_method == "Stereographic":

            # https://en.wikipedia.org/wiki/Stereographic_map_projection
            r = np.sqrt((x-x0)**2  + (y-y0)**2)
            theta = np.pi/2 - np.arctan(r/(z-zm))
            phi = np.arctan2(y-y0, x-x0)
            r = -2*np.sign(self.R)*np.tan(np.pi/4 - theta/2)
            
            p_hit = p.copy()
            p_hit[:, 0] = r*np.cos(phi)
            p_hit[:, 1] = r*np.sin(phi)

        elif projection_method == "Equal-Area":

            # https://en.wikipedia.org/wiki/Lambert_azimuthal_equal-area_projection
            x_ = (x - x0) / np.abs(self.R)
            y_ = (y - y0) / np.abs(self.R)
            z_ = (z - zm) / self.R

            p_hit = p.copy()
            p_hit[:, 0] = np.sqrt(2/(1-z_))*x_
            p_hit[:, 1] = np.sqrt(2/(1-z_))*y_

        else:
            raise ValueError(f"Invalid projection_method {projection_method}, "
                             f"must be one of {self.sphere_projection_methods}.")

        return p_hit

