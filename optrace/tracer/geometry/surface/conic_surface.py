
from typing import Any  # "Any" type

import numpy as np  # calculations
import numexpr as ne  # faster calculations

from ...misc import PropertyChecker as pc  # check types and values
from .surface import Surface  # parent class


class ConicSurface(Surface):
    
    rotational_symmetry: bool = True  #: has the surface rotational symmetry?

    def __init__(self,
                 r:            float,
                 R:            float,
                 k:            float,
                 **kwargs)\
            -> None:
        """
        Define a conic section surface, following the equation:
        z(r) = 1/R * r**2 / (1 + sqrt(1 - (k+1) * r**2 / R**2))

        :param R: curvature circle for surface_type="Conic" or "Sphere" (float)
        :param r: radial size for surface_type="Conic", "Sphere", "Circle" or "Ring" (float)
        :param k: conic constant for surface_type="Conic" (float)
        :param kwargs: additional keyword arguments for parent classes
        """
        self._lock = False

        super().__init__(r, **kwargs)

        self.R, self.k = R, k
        self.parax_roc = R
        
        # (k+1) * r**2 / R**2 = 1 defines the edge of the conic section
        # we can only calculate it inside this area
        # sections with negative k have no edge
        if (self.k + 1)*(self.r/self.R)**2 >= 1:
            raise ValueError("Surface radius r larger than radius of conic section.")

        # depending on the sign of R, z_max or z_min can be on the edge or center
        z0 = self.pos[2]
        self.z_max = 0  # set to some value so values() can be called
        z1 = z0 + self._values(np.array([r]), np.array([0]))[0]
        # note _values uses x, y coordinates relative to the surface center
        
        self.z_min, self.z_max = min(z0, z1), max(z0, z1)

        self.lock()

    @property
    def info(self) -> str:
        """property string for UI information"""
        return super().info + f", R = {self.R:.5g} mm, k = {self.k:.5g}"

    def _values(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Get surface values but in relative coordinate system to surface center.
        And without masking out values beyond the surface extent

        :param x: x-coordinate array (numpy 1D or 2D array)
        :param y: y-coordinate array (numpy 1D or 2D array)
        :return: z-coordinate array (numpy 1D or 2D array, depending on shape of x and y)
        """
        x0, y0, z0 = self.pos
        k, rho = self.k, 1/self.R
        r2 = ne.evaluate("x**2 + y**2")
        return ne.evaluate("rho * r2 /(1 + sqrt(1 - (k+1) * rho**2 *r2))")

    def normals(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Get normal vectors of the surface.

        :param x: x-coordinates (numpy 1D array)
        :param y: y-coordinates (numpy 1D array)
        :return: normal vector array of shape (x.shape[0], 3), components in second dimension (numpy 2D array)
        """
        n = np.tile(np.array([0., 0., 1.], dtype=np.float64), (x.shape[0], 1))

        # coordinates actually on surface
        m = self.mask(x, y)
        xm, ym = x[m], y[m]

        x0, y0, z0 = self.pos
        k, rho = self.k, 1/self.R
        r = ne.evaluate("sqrt((xm-x0)**2 + (ym-y0)**2)")
        phi = ne.evaluate("arctan2(ym-y0, xm-x0)")

        # the derivative of a conic section formula is
        #       m := dz/dr = r*rho / sqrt(1 - (k+1) * rho**2 *r**2)
        # the angle beta of the n_r-n_z normal triangle is
        #       beta = pi/2 + arctan(m)
        # the radial component n_r is
        #       n_r = cos(beta)
        # all this put together can be simplified to
        #       n_r = -rho * r / sqrt(1 - k * rho**2 * r**2)
        # since n = [n_r, n_z] is a unity vector
        #       n_z = +sqrt(1 - n_r**2)
        # this holds true since n_z is always positive in our raytracer
        n_r = ne.evaluate("-rho * r  / sqrt(1 - k* rho**2 * r**2 )")

        # n_r is rotated by phi to get n_x, n_y
        n[m, 0] = ne.evaluate("n_r*cos(phi)")
        n[m, 1] = ne.evaluate("n_r*sin(phi)")
        n[m, 2] = ne.evaluate("sqrt(1 - n_r**2)")

        return n

    def find_hit(self, p: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Find hit/intersections of rays with this surface.

        :param p: ray position array, shape (N, 3)
        :param s: unity ray direction vectors, shape (N, 3)
        :return: intersection position (shape (N, 3)), boolean array (shape N) declaring a hit,
                 indices of ill-conditioned rays
        """

        o = p - self.pos  # coordinates relative to surface center
        ox, oy, oz = o[:, 0], o[:, 1], o[:, 2]
        sx, sy, sz = s[:, 0], s[:, 1], s[:, 2]
        k, rho = self.k, 1/self.R

        with np.errstate(invalid='ignore'):  # suppresses nan warnings for now

            A = ne.evaluate("1 + k*sz**2")
            B = ne.evaluate("sx*ox + sy*oy + sz*(oz*(k+1) - 1/rho)")
            C = ne.evaluate("oy**2 + ox**2 + oz*(oz*(k+1) - 2/rho)")

            # if there are not hits, this term gets imaginary
            # we'll handle this case afterwards
            D = ne.evaluate("sqrt(B**2 - C*A)")

            # we get two possible ray parameters
            # nan values for A == 0, we'll handle them later
            t1 = ne.evaluate("(-B - D) / A")
            t2 = ne.evaluate("(-B + D) / A")

            # choose t that leads to z-position inside z-range of surface and 
            # to a larger z-position than starting point (since for us rays only propagate in +z direction)
            z1 = p[:, 2] + sz*t1
            z2 = p[:, 2] + sz*t2
            z = p[:, 2]
            z_min, z_max = self.z_min - self.N_EPS, self.z_max + self.N_EPS
            t = ne.evaluate("where((z_min <= z1) & (z1 <= z_max) & (z1 >= z) &"\
                            " ~((z_min <= z2) & (z2 <= z_max) & (z2 >= z) & (t2 < t1)), t1, t2)")
            # chose the smaller one of t1, t2 that produces a z-value inside the surface extent
            # and a hit point behind the starting point

            # calculate hit points and hit mask
            p_hit = p + s*t[:, np.newaxis]
            is_hit = self.mask(p_hit[:, 0], p_hit[:, 1])

            # case A == 0 and B != 0 => one intersection
            mask = (A == 0) & (B != 0)
            if np.any(mask):
                t = -C[mask]/(2*B[mask])
                p_hit[mask] = p[mask] + s[mask]*t[:, np.newaxis]
                is_hit[mask] = self.mask(p_hit[mask, 0], p_hit[mask, 1])

            # cases with no hit:
            # D imaginary, means no hit with whole surface function
            # (p_hit[:, 2] < z_min) | (p_hit[:, 2] > z_max): Hit with surface function,
            #               but for the wrong side or outside our surfaces definition region
            # (A == 0) | (B == 0) : Surface is a Line => not hit (
            # technically infinite hits for C == 0, but we'll ignore this)
            # in the simplest case: ray shooting straight at center of a parabola with R = inf,
            # which is so steep, it's a line
            # the ray follows the parabola line exactly => infinite solutions
            #   s = (0, 0, 1), o = (0, 0, oz), k = -1, R=0  => A = 0, B = 0, C = 0 => infinite solutions
            nh = ~is_hit | ~np.isfinite(D) | ((A == 0) & (B == 0)) | (p_hit[:, 2] < z_min) | (p_hit[:, 2] > z_max)

            if np.any(nh):
                # set intersection to plane z = z_max
                tnh = (self.z_max - p[nh, 2]) / s[nh, 2]
                p_hit[nh] = p[nh] + s[nh]*tnh[:, np.newaxis]
                is_hit[nh] = False

        # set hit to current position when ray starts after surface
        m = p[:, 2] > self.z_max
        p_hit[m] = p[m]
        is_hit[m] = False

        return p_hit, is_hit, np.array([])

    def flip(self) -> None:
        """flip the surface around the x-axis"""

        self._lock = False
        self.R *= -1
        self.parax_roc *= -1
        a = self.pos[2] - (self.z_max - self.pos[2])
        b = self.pos[2] + (self.pos[2] - self.z_min)
        self.z_min, self.z_max = a, b
        self.lock()

    def __setattr__(self, key: str, val: Any) -> None:
        """
        assigns the value of an attribute
        :param key: attribute name
        :param val: value to assign
        """
        if key in ["R", "k"]:
            pc.check_type(key, val, float | int)
            val = float(val)

            if key == "R" and (val == 0 or not np.isfinite(val)):
                raise ValueError("R needs to be non-zero and finite. Use planar surface types for planar surfaces.")

        super().__setattr__(key, val)
