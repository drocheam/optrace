
"""
Surface class:
Provides the functionality for the creation of numerical or analytical surfaces.
The class contains methods for interpolation, surface masking and normal calculation
"""

from typing import Any  # Any type

import numpy as np  # calculations
import numexpr as ne  # faster calculations
import scipy.interpolate  # biquadratic interpolation

from .. import misc  # calculations
from .surface_function import SurfaceFunction  # user surface functions
from ..base_class import BaseClass  # parent class
from ..misc import PropertyChecker as pc  # check types and values


# mode Data: receives a square matrix as data, but only values inside a circular area are used
# biquadratic interpolation is used to avoid glass planes, leading to concetrated ray regions
# only finite data in matrix allowed


class Surface(BaseClass):

    surface_types: list[str] = ["Conic", "Sphere", "Circle", "Rectangle", "Function", "Data", "Ring"]
    """possible surface types"""

    sphere_projection_methods: list[str] = ["Equidistant", "Equal-Area", "Stereographic"]
    """projection methods for mapping a sphere surface onto a plane"""
    
    N_EPS: float = 1e-10
    """numerical epsilon. Used for floating number comparisons. As well as adding small differences for plotting"""

    def __init__(self,
                 surface_type:      str,
                 r:                 float = 3.,
                 R:                 float = 10,
                 k:                 float = -0.444,
                 ri:                float = 0.1,
                 dim:               (list | np.ndarray) = None,
                 normal:            (list | np.ndarray) = None,
                 z_min:             float = None,
                 z_max:             float = None,
                 curvature_circle:  float = None,
                 data:              np.ndarray = None,
                 func:              SurfaceFunction = None,
                 **kwargs)\
            -> None:
        """
        Create a surface object.
        The z-coordinate in the pos array is irrelevant if the surface is used for a lens, since it wil be
        adapted inside the lens class

        :param surface_type: "Conic", "Sphere", "Circle", "Rectangle", "Function",  "Ring" or "Data" (string)
        :param R: curvature circle for surface_type="Conic" or "Sphere" (float)
        :param r: radial size for surface_type="Conic", "Sphere", "Circle" or "Ring" (float)
        :param k: conic constant for surface_type="Conic" (float)
        :param dim:
        :param normal:
        :param z_min:
        :param z_max:
        :param curvature_circle:
        :param data: uses copy of this array, grid z-coordinate array for surface_type="Data" (numpy 2D array)
        :param ri: radius of inner circle for surface_type="Ring" (float)
        :param func:
        """
        self._lock = False
        self.surface_type = surface_type

        if (z_max is not None or z_min is not None) and self.surface_type != "Function":
            raise ValueError("z_max and z_min can only be provided for surface_type='Function'")
        
        if curvature_circle is not None and self.surface_type not in ["Function", "Data"]:
            raise ValueError("curvature_circle can only be provided for surface_type='Function' and 'Data'")
        
        self.pos = np.asarray_chkfinite([0., 0., 0.], dtype=np.float64)

        dim = dim if dim is not None else [6, 6]
        self.dim = np.asarray_chkfinite(dim, dtype=np.float64)

        self.r, self.ri = r, ri
        self.R, self.k = R, k
        self.normal = normal
        self.func = func
        self._interp, self._offset = None, 0.
        self._curvature_circle = curvature_circle

        super().__init__(**kwargs)

        match surface_type:

            case ("Conic" | "Sphere"):

                if surface_type == "Sphere":
                    self.k = 0.

                # (k+1) * r**2 / R**2 = 1 defines the edge of the conic section
                # we can only calculate it inside this area
                # sections with negative k have no edge
                if (self.k + 1)*(self.r/self.R)**2 >= 1:
                    raise ValueError("Surface radius r larger than radius of conic section.")

                # depending on the sign of R, z_max or z_min can be on the edge or center
                z0 = self.pos[2]
                self.z_max = 0  # set to some value so get_values() can be called
                z1 = z0 + self._get_values(np.array([r]), np.array([0]))[0]  
                # note _get_values uses x, y coordinates relative to the surface center
                
                self.z_min, self.z_max = min(z0, z1), max(z0, z1)

            case "Data":

                Z = np.asarray_chkfinite(data, dtype=np.float64)

                ny, nx = Z.shape
                if nx != ny:
                    raise ValueError("Array 'data' needs to have a square shape.")

                if nx < 50:
                    raise ValueError("For a good surface representation 'data' should be at least 50x50")

                if nx < 200:
                    self.print("200x200 or larger is advised for a 'data' matrix.")

                xy = np.linspace(-self.r, self.r, nx)  # numeric r vector

                # remove offset at center
                if nx % 2:
                    Z -= np.array([Z[ny//2, nx//2], Z[ny//2+1, nx//2], Z[ny//2, nx//2+1], Z[ny//2+1, nx//2+1]]).mean()
                else:
                    Z -= Z[ny//2, nx//2]

                # we need spline order n=4 so curvature and curvature changes are smooth
                self._interp = scipy.interpolate.RectBivariateSpline(xy, xy, Z, kx=4, ky=4)
                self._offset = self._interp(0, 0, grid=False)  # remaining z_offset at center

                # self.z_min, self.z_max = np.nanmin(Z), np.nanmax(Z)  # provisional values
                self.z_min, self.z_max = self.__find_bounds()
              
                # biquadratic interpolation can lead to an increased z-value range
                # e.g. spheric surface, but the center is not defined by a data point
                X, Y = np.meshgrid(xy, xy)
                M = self.get_mask(X.ravel(), Y.ravel()).reshape(X.shape)
                z_range0 = np.max(Z[M]) - np.min(Z[M])
                z_range1 = self.z_max - self.z_min
                if np.abs(z_range0 - z_range1) > self.N_EPS:
                    z_change = (z_range1 - z_range0) / z_range0
                    add_warning = "WARNING: Deviations this high can be due to noise or abrupt changes in the data."\
                                  " DO NOT USE SUCH SURFACES HERE." if z_change > 0.05 else ""

                    self.print(f"Due to biquadratic interpolation the z_range of the surface {repr(self)}"
                               f" has increased from {z_range0} to {z_range1},"
                               f" a change of {z_change*100:.5g}%. {add_warning}")

            case "Function":
                self._offset = 0  # provisional offset
                self._offset = self._get_values(np.array([0]), np.array([0]))[0]
                self.z_min, self.z_max = self.__find_bounds()

                if z_max is not None and z_min is not None:
                    z_range_probed = self.z_max - self.z_min
                    z_range_provided = z_max - z_min
                    
                    if z_range_probed and z_range_provided + self.N_EPS < z_range_probed:
                        self.print(f"Provided a z-extent of {z_range_provided} for surface {repr(self)},"
                                   f"but measured range is at least {z_range_probed}, an increase of at "
                                   f"least {100*(z_range_probed - z_range_provided)/z_range_probed:.5g}."
                                   f" I will use the measured values for now.")
                    else:

                        range_factor = 1.2
                        if z_range_provided > range_factor*z_range_probed:
                            self.print(f"WARNING: Provided z-range is more than {(range_factor-1)*100:.5g}% "
                                       f"larger than measured z-range")

                        z_max_ = self.z_max + self._offset
                        z_min_ = self.z_min + self._offset

                        if z_max + self.N_EPS < z_max_:
                            self.print(f"WARNING: Provided z_max={z_max} lower than measured value of {z_max_}."
                                       f" Using the measured values for now")
                        
                        elif z_min - self.N_EPS > z_min_:
                            self.print(f"WARNING: Provided z_min={z_min} higher than measured value of {z_min_}."
                                       f" Using the measured values for now")
                        else:
                            self.z_min, self.z_max = z_min - self._offset, z_max - self._offset

                elif z_max is None and z_min is None:
                    self.print(f"Estimated z-bounds of surface {repr(self)}: [{self._offset+self.z_min}, "
                               f"{self._offset+self.z_max}], provide actual values to make it more exact.")
                
                else:
                    raise ValueError(f"z_max and z_min need to be both None or both need a value")

            case "Rectangle":
                self.z_min = self.z_max = self.pos[2]

            case "Circle":
                self.z_min = self.z_max = self.pos[2]

                if self.normal is None:
                    self.normal = [0, 0, 1]

                if not self.no_z_extent():
                    phi = np.arctan2(self.normal[1], self.normal[0])
                    R = self.r - self.N_EPS
                    val1 = self.pos[2] + self._get_values(np.array([R*np.cos(phi)]), np.array([R*np.sin(phi)]))[0]
                    val2 = self.pos[2] + self._get_values(np.array([-R*np.cos(phi)]), np.array([-R*np.sin(phi)]))[0]
                    self.z_min, self.z_max = min(val1, val2), max(val1, val2)

            case "Ring":
                self.z_min = self.z_max = self.pos[2]
                if ri >= r:
                    raise ValueError("ri needs to be smaller than r.")

            case _:  # case only for developing new types # pragma: no cover
                assert False, f"surface_type={self.surface_type} not handled."

        self.lock()

    def has_hit_finding(self) -> bool:
        """:return: if this surface has an analytic hit finding method implemented"""

        return self.surface_type in ["Circle", "Ring", "Rectangle", "Sphere", "Conic"] \
            or (self.surface_type == "Function" and self.func.hit_func is not None)

    def no_z_extent(self) -> bool:
        """:return: if the surface has no extent in z-direction"""
        if self.surface_type == "Circle":
            return np.all(self.normal == [0, 0, 1])
        return self.surface_type in ["Ring", "Rectangle"]

    def has_rotational_symmetry(self) -> bool:
        return self.curvature_circle is not None

    def __find_bounds(self) -> tuple[float, float]:
        """
        Estimate min and max z-value on Surface by sampling dozen values.

        :return: min and max z-value on Surface
        """
        # how to regularly sample a circle area, while sampling
        # as much different phi and r values as possible?
        # => sunflower sampling for surface area
        # see https://stackoverflow.com/a/44164075

        N = 10000
        ind = np.arange(0, N, dtype=np.float64) + 0.5

        r = np.sqrt(ind/N) * self.r
        phi = 2*np.pi * (1 + 5**0.5)/2 * ind

        vals = self._get_values(r*np.cos(phi), r*np.sin(phi))

        # mask out invalid values
        mask = self.get_mask(r * np.cos(phi) - self.pos[0], r * np.sin(phi) - self.pos[1])
        vals[~mask] = np.nan

        # in many cases the minimum and maximum are at the center or edge of the surface
        # => sample them additionally

        # values at surface edge
        phi2 = np.linspace(0, 2*np.pi, 1001)  # N is odd, since step size is 1/(N-1) * 2*pi,
        r2 = np.full_like(phi2, self.r, dtype=np.float64)
        vals2 = self._get_values(r2*np.cos(phi2), r2*np.sin(phi2))
        mask = self.get_mask(r2 * np.cos(phi2) - self.pos[0], r2 * np.sin(phi2) - self.pos[1])
        vals2[~mask] = np.nan

        # surface center
        vals3 = self._get_values(np.array([0.]), np.array([0.]))
        if not self.get_mask(np.array([self.pos[0]]), np.array([self.pos[1]])):
            vals3 = [np.nan]

        # add all surface values into one array
        vals = np.concatenate((vals, vals2, vals3))

        # find minimum and maximum value
        z_min = np.nanmin(vals)
        z_max = np.nanmax(vals)

        return z_min, z_max

    @property
    def curvature_circle(self):
        """surface center paraxial curvature"""

        if self.no_z_extent():
            return np.inf
        elif self.surface_type in ["Conic", "Sphere"]:
            return self.R
        elif self.surface_type in ["Function", "Data"] and self._curvature_circle is not None:
            return self._curvature_circle
        else:
            return None

    def move_to(self, pos: (list | np.ndarray)) -> None:
        """
        Moves the surface in 3D space.

        :param pos: 3D position to move to (list or numpy 1D array)
        """

        self._lock = False

        self.z_min += pos[2] - self.pos[2]
        self.z_max += pos[2] - self.pos[2]
        
        # update position
        self.pos = np.asarray_chkfinite(pos, dtype=np.float64)

        self.lock()

    @property
    def extent(self) -> tuple[float, float, float, float, float, float]:
        """
        """
        if self.surface_type == "Rectangle":
            return *(self.pos[:2].repeat(2) + self.dim.repeat(2)/2 * np.array([-1, 1, -1, 1])), \
                   self.z_min, self.z_max
        else:
            return *(self.r*np.array([-1, 1, -1, 1]) + self.pos[:2].repeat(2)), \
                   self.z_min, self.z_max

    @property
    def d(self) -> float:
        """surface thickness. Difference between highest and lowest point on surface"""
        return self.z_max - self.z_min

    @property
    def dn(self) -> float:
        """thickness between center z-position and lowest point on surface"""
        return self.pos[2] - self.z_min
    
    @property
    def dp(self) -> float:
        """thickness between highest point on surface and center z-position"""
        return self.z_max - self.pos[2]

    def get_values(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Get surface values. Absolute coordinates. Points outside the surface are set to z_max.

        :param x: x-coordinate array (numpy 1D or 2D array)
        :param y: y-coordinate array (numpy 1D or 2D array)
        :return: z-coordinate array (numpy 1D or 2D array, depending on shape of x and y)
        """
        z = np.full_like(x, self.z_max, dtype=np.float64)
        
        if self.no_z_extent():
            return z
    
        else:
            x0, y0, z0 = self.pos
            r2 = ne.evaluate("(x-x0)**2 + (y-y0)**2")
            inside = r2 <= (self.r + self.N_EPS)**2
            z[inside] = self.pos[2] + self._get_values(x[inside] - self.pos[0], y[inside] - self.pos[1])

            return z

    def _get_values(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Get surface values but in relative coordinate system to surface center.
        And without masking out values beyond the surface extent

        :param x: x-coordinate array (numpy 1D or 2D array)
        :param y: y-coordinate array (numpy 1D or 2D array)
        :return: z-coordinate array (numpy 1D or 2D array, depending on shape of x and y)
        """
        if self.no_z_extent():
            return np.zeros_like(x, dtype=np.float64)

        else:
            x0, y0, z0 = self.pos

            match self.surface_type:

                case ("Conic" | "Sphere"):
                    k, rho = self.k, 1/self.R
                    r2 = ne.evaluate("x**2 + y**2")
                    return ne.evaluate("rho * r2 /(1 + sqrt(1 - (k+1) * rho**2 *r2))")

                case "Data":
                    # uses quadratic interpolation for smooth surfaces and a constantly changing slope
                    return self._interp(x, y, grid=False) - self._offset

                case "Function":
                    # additional mask function is handled in get_values()
                    return self.func.get_values(x, y) - self._offset

                case "Circle" if not self.no_z_extent():
                    # slope in x and y direction from normal vector
                    mx = -self.normal[0]/self.normal[2]
                    my = -self.normal[1]/self.normal[2]
                    # no division by zero because we enforce normal[:, 2] > 0,
                    return x*mx + y*my

    def get_plotting_mesh(self, N: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get 2D plotting mesh. Note that the values are not gridded, the distance can be arbitrary.
        The only guarantee is that neighbouring array values are neighbouring values in 3D space.

        :param N: number of grid values in each dimension (int)
        :return: X, Y, Z coordinate array (all numpy 2D array)
        """
        if N < 10:
            raise ValueError("Expected at least N=10.")

        # return rectangle for rectangle type
        if self.surface_type == "Rectangle":
            xs, xe, ys, ye, _, _ = self.extent

            X, Y = np.mgrid[xs:xe:N*1j, ys:ye:N*1j]
            Z = np.full_like(Y, self.pos[2], dtype=np.float64)

            return X, Y, Z

        # get z-values on rectangular grid
        X, Y = np.mgrid[-self.r:self.r:N*1j,
                        -self.r:self.r:N*1j]
        z = self._get_values(X.ravel(), Y.ravel())

        # convert to polar coordinates
        R = np.sqrt(X**2 + Y**2)
        Phi = np.arctan2(Y, X)

        # masks for values outside of circular area
        r = self.r
        mask = R.ravel() >= r
        mask2 = R >= r

        # move values outside surface to the surface edge
        # this defines the edge with more points, making it more circular instead of step-like
        z[mask] = self._get_values(r * np.cos(Phi.ravel()[mask]), r * np.sin(Phi.ravel()[mask]))
        X[mask2] = r*np.cos(Phi[mask2])
        Y[mask2] = r*np.sin(Phi[mask2])

        # extra precautions to plot the inner ring
        # otherwise the inner circle or the ring could be too small to resolve, depending on the plotting resolution
        if self.surface_type == "Ring":

            # move points near inner edge towards the edge line
            # create two circles, one slightly outside the edge (mask5)
            # and one slightly below (mask4)

            # ring larger
            if self.ri < self.r/2:
                rr = self.r - self.ri  # diameter of ring area
                mask4 = R <= (self.ri + rr/3)
                mask5 = (R > (self.ri + rr/3)) & (R < (self.ri + 2/3*rr))
            # diameter of inner circle larger
            else:
                mask4 = (R < self.ri/2)
                mask5 = (R < self.ri) & (R >= self.ri/2)

            # move points onto the two circles
            # we need to shift by more than eps, since values with eps are still on the surface
            X[mask4] = (self.ri - 4*self.N_EPS) * np.cos(Phi[mask4])
            Y[mask4] = (self.ri - 4*self.N_EPS) * np.sin(Phi[mask4])
            X[mask5] = (self.ri + 4*self.N_EPS) * np.cos(Phi[mask5])
            Y[mask5] = (self.ri + 4*self.N_EPS) * np.sin(Phi[mask5])

        # plot nan values inside
        mask3 = self.get_mask(X.ravel()+self.pos[0], Y.ravel()+self.pos[1])  # get_mask has absolute coordinates
        z[~mask3] = np.nan

        # make 2D
        Z = z.reshape(X.shape)

        # return offset values
        return X+self.pos[0], Y+self.pos[1], Z+self.pos[2]

    def get_mask(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Get surface mask values. A value of 1 means the surface is defined here.

        :param x: x-coordinate array (numpy 1D or 2D array)
        :param y: y-coordinate array (numpy 1D or 2D array)
        :return: z-coordinate array (numpy 1D or 2D array, depending on shape of x and y)
        """
        if self.surface_type == "Rectangle":
            xs, xe, ys, ye = self.extent[:4]
            inside = (xs-self.N_EPS <= x) & (x <= xe+self.N_EPS) & (ys-self.N_EPS <= y) & (y <= ye+self.N_EPS)
            return inside

        else:
            # use r^2 instead of r, saves sqrt calculation for all points
            x0, y0, z0 = self.pos
            r2 = ne.evaluate("(x - x0) ** 2 + (y - y0) ** 2")

            if self.surface_type == "Ring":
                return ((self.ri - self.N_EPS)**2 <= r2) & (r2 <= (self.r + self.N_EPS)**2)
            else:
                mask = r2 <= (self.r + self.N_EPS)**2

                # additionally evaluate SurfaceFunction mask if defined
                if self.surface_type == "Function" and self.func.mask_func is not None:
                    mask = mask & self.func.get_mask(x - self.pos[0], y - self.pos[1])

                return mask

    def get_normals(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Get normal vectors of the surface.

        :param x: x-coordinates (numpy 1D array)
        :param y: y-coordinates (numpy 1D array)
        :return: normal vector array of shape (x.shape[0], 3), components in second dimension (numpy 2D array)
        """

        n = np.tile([0., 0., 1.], (x.shape[0], 1))

        if self.no_z_extent():
            return n

        # coordinates actually on surface
        m = self.get_mask(x, y)
        xm, ym = x[m], y[m]

        match self.surface_type:

            case "Circle":  # circle with normal
                n[m] = self.normal

            case "Function" if self.func.deriv_func is not None:
                nxn, nyn = self.func.get_derivative(xm - self.pos[0], ym - self.pos[1])
                n[m, 0] = -nxn
                n[m, 1] = -nyn
                n[m] = misc.normalize(n[m])

            case "Data":
                # the interpolation object provides partial derivatives
                n[m, 0] = -self._interp(xm - self.pos[0], ym - self.pos[1], dx=1, dy=0, grid=False)
                n[m, 1] = -self._interp(xm - self.pos[0], ym - self.pos[1], dx=0, dy=1, grid=False)
                n[m] = misc.normalize(n[m])

            case ("Sphere" | "Conic"):
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

            # get normals the numerical way
            case _:
                # approximate optimal step width for differentiation
                # for a second derivative of form (f(x+h) - f(x-h)) / (2h) the optimal h is
                # h* = (3*e*abs(f(x)/f'''(x))^(1/3)  with e being the machine precision
                # for us one general assumption can be abs(f(x)/f'''(x) = 50
                # another condition is that x+h* != x, meaning it is representable
                # so make sure x+h* is representable as float number for every point on the surface
                # in a latter step we will use relative coordinates from the center
                # so for eps_deriv we are scaling e with s_max, the maximal extent in x, y or z dimension
                # source h* formula: 
                # http://www.uio.no/studier/emner/matnat/math/MAT-INF1100/h08/kompendiet/diffint.pdf, p.241
                # the C++ boost library does it the same way:
                # https://github.com/boostorg/math/blob/1ce6dda2fb9b8d3cd2f54c76fe5a8cc3d0e430f9/include/boost/math/differentiation/finite_difference.hpp
                eps_f = np.finfo(np.float64).eps
                eps_deriv = (3*eps_f*50)**(1/3)
                ext = np.array(self.extent)
                eps_num = np.spacing(ext[1::2] - ext[::2])
                eps = max(eps_deriv, *eps_num)

                # uz is the surface change in x direction, vz in y-direction, the normal vector
                # of derivative vectors u = (2*ds, 0, uz) and v = (0, 2*ds, vz) is n = (-2*uz*ds, -2*ds*vz, 4*ds*ds)
                # this can be rescaled to n = (-uz, -vz, 2*ds)
                x_, y_ = xm - self.pos[0], ym - self.pos[1]  # use coordinates relative to center
                n[m, 0] = self._get_values(x_ - eps, y_) - self._get_values(x_ + eps, y_)  # -uz
                n[m, 1] = self._get_values(x_, y_ - eps) - self._get_values(x_, y_ + eps)  # -vz
                n[m, 2] = 2*eps

                # normalize vectors
                n[m] = misc.normalize(n[m])

        return n

    def get_edge(self, nc: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get surface values of the surface edge, assumes a circular edge.

        :param nc: number of points on edge (int)
        :return: X, Y, Z coordinate arrays (all numpy 2D array)
        """

        if nc < 20:
            raise ValueError("Expected at least nc=20")

        if self.surface_type == "Rectangle":
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

        else:
            theta = np.linspace(0, 2 * np.pi, nc)
            xd = self.r * np.cos(theta)
            yd = self.r * np.sin(theta)
            zd = self._get_values(xd, yd)

            return xd+self.pos[0], yd+self.pos[1], zd+self.pos[2]

    def get_random_positions(self, N: int) -> np.ndarray:
        """

        :param N:
        :return:
        """

        p = np.zeros((N, 3), dtype=np.float64, order='F')

        match self.surface_type:

            case ("Circle" | "Ring"):
                assert self.no_z_extent()  # don't implemented for a user defined self.normal

                rs = 0 if self.surface_type == "Circle" else self.ri  # choose inner radius
                # weight with square root to get equally distributed points
                r, theta = misc.uniform2(rs**2, self.r**2, 0, 2*np.pi, N)
                r = np.sqrt(r)

                x0, y0 = self.pos[0], self.pos[1]

                p[:, 0] = ne.evaluate("x0 + r*cos(theta)")
                p[:, 1] = ne.evaluate("y0 + r*sin(theta)")
                p[:, 2] = self.pos[2]

            case "Rectangle":
                p[:, 0], p[:, 1] = misc.uniform2(*self.extent[:4], N)
                p[:, 2] = self.pos[2]

            case _:
                assert False, f"surface_type={self.surface_type} not handled."

        return p

    def find_hit(self, p: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """

        :param p:
        :param s:
        :return:
        """
        match self.surface_type:

            case ("Circle" | "Ring" | "Rectangle"):

                if self.surface_type == "Circle" and not self.no_z_extent():
                    # intersection ray with plane
                    # see https://www.scratchapixel.com/lessons/3d-basic-rendering/
                    # minimal-ray-tracer-rendering-simple-shapes/ray-plane-and-ray-disk-intersection
                    normal = np.broadcast_to(self.normal, (p.shape[0], 3))
                    t = misc.rdot(self.pos - p, normal) / misc.rdot(s, normal)
                    p_hit = p + s*t[:, np.newaxis]

                    # rays don't hit -> intersect with xy plane at z=z_max
                    m = is_hit = self.get_mask(p_hit[:, 0], p_hit[:, 1])
                    tnm = (self.z_max - p[~m, 2])/s[~m, 2]
                    p_hit[~m] = p[~m] + s[~m]*tnm[:, np.newaxis]
                else:
                    # intersection with xy plane
                    t = (self.pos[2] - p[:, 2])/s[:, 2]
                    p_hit = p + s*t[:, np.newaxis]
                    is_hit = self.get_mask(p_hit[:, 0], p_hit[:, 1])

                return p_hit, is_hit

            case "Function" if self.func.hit_func is not None:
                # spatial 3D shift so we have coordinates relative to the surface center
                # _offset is removed, since it was introduced by Surface and is unknown to the SurfaceFunction object
                dp = self.pos - [0, 0, self._offset]

                p_hit0 = self.func.get_hits(p - dp, s)
                is_hit = self.func.get_mask(p_hit0[:, 0], p_hit0[:, 1])

                return p_hit0 + dp, is_hit  # transform to standard coordinates

            case ("Sphere" | "Conic"):

                o = p - self.pos  # coordinates relative to surface center
                ox, oy, oz = o[:, 0], o[:, 1], o[:, 2]
                sx, sy, sz = s[:, 0], s[:, 1], s[:, 2]
                k, rho = self.k, 1/self.R

                # we want to ignore numpy errors, since we divide by zero in some cases (e.g for A == 0)
                # these errors are handled afterwards, but checking and masking
                # all arrays beforehand would decrease performance
                old_err = np.seterr("ignore")

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
                z = p[:, 2]
                z_min, z_max = self.z_min, self.z_max
                t = ne.evaluate("where((z_min <= z1) & (z1 <= z_max) & (z1 >= z), t1, t2)")

                # calculate hit points and hit mask
                p_hit = p + s*t[:, np.newaxis]
                is_hit = self.get_mask(p_hit[:, 0], p_hit[:, 1])

                # case A == 0 and B != 0 => one intersection
                mask = (A == 0) & (B != 0)
                if np.any(mask):
                    t = -C[mask]/(2*B[mask])
                    p_hit[mask] = p[mask] + s[mask]*t[:, np.newaxis]
                    is_hit[mask] = self.get_mask(p_hit[mask, 0], p_hit[mask, 1])

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

                np.seterr(**old_err)  # restore numpy error settings

                return p_hit, is_hit

            case _:
                raise RuntimeError(f"Hit finding not defined for {self.surface_type=}.")

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

        if self.surface_type != "Sphere":
            raise RuntimeError(f"Sphere projections are for spheres only, not for surface_type='{self.surface_type}'")

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

    def __setattr__(self, key: str, val: Any) -> None:
        """
        assigns the value of an attribute
        :param key: attribute name
        :param val: value to assign
        """
        match key:

            case "surface_type":
                pc.check_type(key, val, str)
                pc.check_if_element(key, val, self.surface_types)

            case ("r" | "R" | "k" | "ri" | "z_max" | "z_min" | "curvature_circle"):
                pc.check_type(key, val, float | int)
                val = float(val)

                if key in ["r", "ri", "curvature_circle"]:
                    pc.check_above(key, val, 0)

                if key == "R" and (val == 0 or not np.isfinite(val)):
                    raise ValueError("R needs to be non-zero and finite. For a plane use surface_type=\"Circle\","
                                     " for a point use the Point class.")

            case "normal" if not val is None:
                if self.surface_type != "Circle":
                    raise RuntimeError("'normal' only supported for surface_type='Circle'")

                pc.check_type(key, val, list | np.ndarray)
                val2 = np.asarray_chkfinite(val, dtype=np.float64) / np.linalg.norm(val)  # normalize
               
                pc.check_above("normal[2]", val2[2], 0)
                
                super().__setattr__(key, val2)
                return
                
            case "func":
                pc.check_type(key, val, SurfaceFunction | None)

                # copy object
                if val is not None:
                    super().__setattr__(key, val.copy())
                    return

            case "dim":
                pc.check_type(key, val, np.ndarray)

                if val[0] <= 0 or val[1] <= 0:
                    raise ValueError("Dimensions dim need to be positive, but are {dim=}")

        super().__setattr__(key, val)
