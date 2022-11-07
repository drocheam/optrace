
"""
Surface class:
Provides the functionality for the creation of numerical or analytical surfaces.
The class contains methods for interpolation, surface masking and normal calculation
"""

from typing import Any  # Any type

import numpy as np  # calculations
import numexpr as ne  # faster calculations

from ... import misc  # calculations
from ...base_class import BaseClass  # parent class
from ...misc import PropertyChecker as pc  # check types and values

# NOTE for classes Surface, Point, Line, reverse returns a reversed version at [0, 0, 0]
# for higher classes like Element and Marker the objects itself maintains its position


class Surface:
    pass

class Surface(BaseClass):

    C_EPS: float = 1e-6
    """ calculation epsilon. In some numerical methods this is the desired solution precision """
    
    N_EPS: float = 1e-10
    """numerical epsilon. Used for floating number comparisons. As well as adding small differences for plotting"""

    def __init__(self,
                 r:                 float,
                 **kwargs)\
            -> None:
        """
        Create a surface object.
        The z-coordinate in the pos array is irrelevant if the surface is used for a lens, since it wil be
        adapted inside the lens class

        """
        self._lock = False

        self.pos = np.asarray_chkfinite([0., 0., 0.], dtype=np.float64)

        self.r = r
        self.parax_roc = None
        self.z_min, self.z_max = np.nan, np.nan

        super().__init__(**kwargs)

    def is_flat(self) -> bool:
        """:return: if the surface has no extent in z-direction"""
        return self.z_max == self.z_min

    def has_rotational_symmetry(self) -> bool:
        return self.parax_roc is not None

    @property
    def info(self) -> str:
        """property string for UI information"""
        return f"{type(self).__name__}, pos = [{self.pos[0]:.5g} mm, {self.pos[1]:.5g} mm, "\
               f"{self.pos[2]:.5g} mm], r = {self.r:.5g} mm"

    def _find_bounds(self) -> tuple[float, float]:
        """
        Estimate min and max z-value on Surface by sampling dozen values.

        :return: min and max z-value on Surface
        """
        # how to regularly sample a circle area, while sampling
        # as much different phi and r values as possible?
        # => sunflower sampling for surface area
        # see https://stackoverflow.com/a/44164075

        N = 50000
        ind = np.arange(0, N, dtype=np.float64)  # start at 0

        r = np.sqrt(ind/N) * self.r
        phi = 2*np.pi * (1 + 5**0.5)/2 * ind

        # get values and mask out invalid values
        vals = self._get_values(r*np.cos(phi), r*np.sin(phi))
        mask = self.get_mask(r * np.cos(phi) - self.pos[0], r * np.sin(phi) - self.pos[1])
        vals[~mask] = np.nan

        # in many cases the minimum and maximum are at the  edge of the surface
        # => sample it additionally

        # values at surface edge
        xv, yv, vals2 = self.get_edge(3001)
        vals2 -= self.pos[2]
        mask = self.get_mask(xv, yv)
        vals2[~mask] = np.nan

        # find minimum and maximum value
        z_min = min(np.nanmin(vals), np.nanmin(vals2))
        z_max = max(np.nanmax(vals), np.nanmax(vals2))

        return z_min, z_max

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
        
        if self.is_flat():
            return z
    
        else:
            inside = self.get_mask(x, y)
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
        assert self.is_flat(), "function not implemented for sub-class"
        return np.zeros_like(x, dtype=np.float64)

    def get_plotting_mesh(self, N: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get 2D plotting mesh. Note that the values are not gridded, the distance can be arbitrary.
        The only guarantee is that neighbouring array values are neighbouring values in 3D space.

        :param N: number of grid values in each dimension (int)
        :return: X, Y, Z coordinate array (all numpy 2D array)
        """
        if N < 10:
            raise ValueError("Expected at least N=10.")

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
        # use r^2 instead of r, saves sqrt calculation for all points
        x0, y0, z0 = self.pos
        r2 = ne.evaluate("(x - x0) ** 2 + (y - y0) ** 2")

        return r2 <= (self.r + self.N_EPS)**2

    def get_normals(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Get normal vectors of the surface.

        :param x: x-coordinates (numpy 1D array)
        :param y: y-coordinates (numpy 1D array)
        :return: normal vector array of shape (x.shape[0], 3), components in second dimension (numpy 2D array)
        """

        n = np.tile([0., 0., 1.], (x.shape[0], 1))

        if self.is_flat():
            return n

        # coordinates actually on surface
        m = self.get_mask(x, y)
        xm, ym = x[m], y[m]

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

        theta = np.linspace(0, 2 * np.pi, nc)
        xd = self.r * np.cos(theta)
        yd = self.r * np.sin(theta)
        zd = self._get_values(xd, yd)

        return xd+self.pos[0], yd+self.pos[1], zd+self.pos[2]

    def find_hit(self, p: np.ndarray, s: np.ndarray)\
            -> tuple[np.ndarray, np.ndarray]:
        """
        Find the position of hits on surface using the iterative regula falsi algorithm.

        :param p: position array (numpy 2D array, shape (N, 3))
        :param s: direction array (numpy 2D array, shape (N, 3))
        :param i:
        :param msg:
        :return: positions of hit (shape as p), bool numpy 1D array if ray hits lens
        """

        if self.is_flat():
            # intersection with xy plane
            t = (self.pos[2] - p[:, 2])/s[:, 2]
            p_hit = p + s*t[:, np.newaxis]
            is_hit = self.get_mask(p_hit[:, 0], p_hit[:, 1])

            return p_hit, is_hit

        else:
            # get search bounds
            t1, t2, f1, f2, p1, p2 = self._find_hit_bounds(p, s)

            # contraction factor (m = 0.5 : Illinois Algorithm)
            m = 0.5

            # assign non-finite and rays converged in first iteration
            w = np.ones(t1.shape, dtype=bool)  # bool array for hit search
            w[~np.isfinite(t1) | ~np.isfinite(t2)] = False  # exclude non-finite t1, t2
            w[(t2 - t1) < self.C_EPS] = False  # exclude rays that already converged
            c = ~w  # bool array for which ray has converged

            # arrays for estimated hit points
            p_hit = np.zeros_like(s, dtype=np.float64, order='F')
            p_hit[c] = p1[c]  # assign already converged rays

            it = 1  # number of iteration
            # do until all rays have converged
            while np.any(w):

                # secant root
                t1w, t2w, f1w, f2w = t1[w], t2[w], f1[w], f2[w]
                ts = ne.evaluate("t1w - f1w/(f2w-f1w)*(t2w-t1w)")

                # position of root on rays
                pl = p[w] + s[w]*ts[:, np.newaxis]

                # difference between ray and surface at root
                fts = pl[:, 2] - self.get_values(pl[:, 0], pl[:, 1])

                # sign of fts*f2 decides which case is handled for each ray
                prod = fts*f2[w]

                # case 1: fts, f2 different sign => change [t1, t2] interval to [t2, ts]
                mask = prod < 0
                wm = misc.part_mask(w, mask)
                t1[wm], t2[wm], f1[wm], f2[wm] = t2[wm], ts[mask], f2[wm], fts[mask]

                # case 2: fts, f2 same sign => change [t1, t2] interval to [t1, ts]
                mask = prod > 0
                wm = misc.part_mask(w, mask)
                t2[wm], f1[wm], f2[wm] = ts[mask], m*f1[wm], fts[mask]

                # case 3: fts or f2 is zero => ts and fts are the found solution
                mask = prod == 0
                wm = misc.part_mask(w, mask)
                t1[wm], t2[wm], f1[wm], f2[wm] = ts[mask], ts[mask], fts[mask], fts[mask]

                # masks for rays converged in this iteration
                cn = np.abs(t2[w]-t1[w]) < self.C_EPS/10
                wcn = misc.part_mask(w, cn)

                # assign found hits and update bool arrays
                p_hit[wcn] = pl[cn]
                c[wcn] = True
                w[wcn] = False

                # timeout
                if it == 100:  # how to check a timeout? # pragma: no cover
                    raise RuntimeError(f"Timeout after {it} iterations in hit finding.") 
                it += 1

            # check if hit is an actual hit. For this the surface needs to be defined at this x and y
            # as well as the actual surface value being near
            is_hit = self.get_mask(p_hit[:, 0], p_hit[:, 1])
            zs = self.get_values(p_hit[:, 0], p_hit[:, 1])
            is_hit = is_hit & (np.abs(zs - p_hit[:, 2]) < self.C_EPS)

        return p_hit, is_hit

    def _find_hit_bounds(self, p, s):
        """

        Surface hits can only occur inside the cylindrical extent of the surface.
        This is the smallest cylinder with rotation axis in z-direction that encompasses all of the surface.
        Find intersections between the ray projected in xy direction and surface projected onto a xy circle,
        but only the ray section starts or ends outside of it.

        If the intersections reduce the search range, while also starting above and ending behind the surface,
        we use these bound values from now on.

        Why do we do this?
        The area outside a valid surface extent is set to a constant value, where the numeric algorithm has problems
        converging and finding the correct regions. While ensuring the rays are only in xy regions, 
        where the surface is defined, we reduce these issues.
        Another thing is, that a hit with this infinite outside-plane is guaranteed for every ray,
        but in most cases there is also a hit with the surface.
        This procedure also ensures that the actual hit position is preferred.

        :param p:
        :param s:
        :param surface:
        :return:
        """

        # ray parameters for just above and below the surface
        t1 = (self.z_min - self.C_EPS / 10 - p[:, 2]) / s[:, 2]
        t2 = (self.z_max + self.C_EPS / 10 - p[:, 2]) / s[:, 2]

        # set to -eps since we can't move in -z anyway (t1 <0)
        t1[t1 < 0] = -self.C_EPS

        # check if surface is defined for t1 and t2
        p1 = p + s*t1[:, np.newaxis]
        p2 = p + s*t2[:, np.newaxis]
        mt1 = self.get_mask(p1[:, 0], p1[:, 1])
        mt2 = self.get_mask(p2[:, 0], p2[:, 1])

        # cost function values
        f1 = np.full_like(t1, np.nan)
        f2 = np.full_like(t1, np.nan)
        
        fb = ~(mt1 & mt2) & ~((s[:, 0] == 0) & (s[:, 1] == 0))  # need-to-fix-bounds bool array
        # but can't do this when there is only propagation in z direction

        if np.any(fb):

            # equations taken from
            # https://www.scratchapixel.com/lessons/3d-basic-rendering/
            # minimal-ray-tracer-rendering-simple-shapes/ray-sphere-intersection

            # the next part only works for a = D**2 != 0
            # which is s[:, 0] == 0 and s[:, 1] == 0
            # but we excluded this case above

            R = self.r - self.C_EPS  # slightly smaller so we are on the surrface without doubt
            C = self.pos[:2]
            O = p[fb, :2]
            D = s[fb, :2]
            OMC = O-C
            a = misc.rdot(D, D)
            b = 2*misc.rdot(D, OMC)
            c = misc.rdot(OMC, OMC) - R**2

            D2 = ne.evaluate("sqrt(b**2 - 4*a*c)")
            t1n = ne.evaluate("(-b - D2)/(2*a)")
            t2n = ne.evaluate("(-b + D2)/(2*a)")

            # coordinates for hit of x-y circle projection
            p1n = p[fb] + s[fb]*t1n[:, np.newaxis]
            p2n = p[fb] + s[fb]*t2n[:, np.newaxis]

            # cost function = difference between ray z-postion and self height at the same x, y
            diff1 = p1n[:, 2] - self.get_values(p1n[:, 0], p1n[:, 1])
            diff2 = p2n[:, 2] - self.get_values(p2n[:, 0], p2n[:, 1])

            # assign rays with better t1
            t1n_for_t1 = (diff1 < 0) & ((diff2 >= 0)) | ((diff2 < 0) & (diff1 > diff2))
            t1c = np.where(t1n_for_t1, t1n, t2n)
            p1c = np.where(t1n_for_t1[:, np.newaxis], p1n, p2n)
            diff1c = np.where(t1n_for_t1, diff1, diff2)
            make_t1 = (t1c > 0) & (t1c > t1[fb]) & (p1c[:, 2] < self.z_max) & (p1c[:, 2] > self.z_min)\
                      & np.isfinite(t1c) & (diff1c < 0)
            fbmt1 = misc.part_mask(fb, make_t1)
            t1[fbmt1] = t1c[make_t1]
            p1[fbmt1] = p1c[make_t1]
            f1[fbmt1] = diff1c[make_t1]
        
            # assign rays with better t2
            t2n_for_t2 = (diff2 > 0) & ((diff1 <= 0) | ((diff1 > 0) & (diff2 < diff1)))
            t2c = np.where(t2n_for_t2, t2n, t1n)
            p2c = np.where(t2n_for_t2[:, np.newaxis], p2n, p1n)
            diff2c = np.where(t2n_for_t2, diff2, diff1)
            make_t2 = (t2c > 0) & (t2c < t2[fb]) & (p2c[:, 2] < self.z_max) & (p2c[:, 2] > self.z_min)\
                       & np.isfinite(t2c) & (diff2c > 0)
            fbmt2 = misc.part_mask(fb, make_t2)
            t2[fbmt2] = t2c[make_t2]
            p2[fbmt2] = p2c[make_t2]
            f2[fbmt2] = diff2c[make_t2]

            assert ~np.any((t2c == t1c) & make_t1 & make_t2)

        # assign missing f1
        f1ci = np.isnan(f1)
        f1[f1ci] = p1[f1ci, 2] - self.get_values(p1[f1ci, 0], p1[f1ci, 1])

        # assign missing f2
        f2ci = np.isnan(f2)
        f2[f2ci] = p2[f2ci, 2] - self.get_values(p2[f2ci, 0], p2[f2ci, 1])
       
        # update mask arrays, since rays with fb changed their p1, p2
        mt1[fb] = self.get_mask(p1[fb, 0], p1[fb, 1])
        mt2[fb] = self.get_mask(p2[fb, 0], p2[fb, 1])

        return t1, t2, f1, f2, p1, p2
    
    def reverse(self) -> Surface:

        assert self.is_flat()  # reverse otherwise not implemented

        # reversed version has same properties, but pos at [0, 0, 0]
        S = self.copy()
        S.move_to([0, 0, 0])
        return S

    def __setattr__(self, key: str, val: Any) -> None:
        """
        assigns the value of an attribute
        :param key: attribute name
        :param val: value to assign
        """
        if key == "r":
            pc.check_type(key, val, float | int)
            val = float(val)
            pc.check_above(key, val, 0)
        
        elif key == "parax_roc" and val is not None:
            pc.check_type(key, val, float | int)
            val = float(val)

        super().__setattr__(key, val)
