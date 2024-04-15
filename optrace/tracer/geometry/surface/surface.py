
from typing import Any  # Any type

import numpy as np  # calculations
import numexpr as ne  # faster calculations

from ... import misc  # calculations
from ...base_class import BaseClass  # parent class
from ...misc import PropertyChecker as pc  # check types and values
from ....warnings import warning

# NOTE for classes Surface, Point, Line, reverse returns a reversed version at [0, 0, 0]
# for higher classes like Element and Marker the objects itself maintains its position


class Surface(BaseClass):

    C_EPS: float = 1e-6
    """ calculation epsilon. In some numerical methods this is the desired solution precision """
    
    N_EPS: float = 1e-10
    """numerical epsilon. Used for floating number comparisons. As well as adding small differences for plotting"""

    rotational_symmetry: bool = False
    """has the surface rotational symmetry? Needs to be overwritten by child classes"""


    def __init__(self,
                 r:                 float,
                 **kwargs)\
            -> None:
        """
        Create a surface object, parent class of all other surface types.

        :param r: surface radius
        :param kwargs: additional keyword arguments for parent classes
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

    @property
    def info(self) -> str:
        """property string for UI information"""
        return f"{type(self).__name__}, pos = [{self.pos[0]:.5g} mm, {self.pos[1]:.5g} mm, "\
               f"{self.pos[2]:.5g} mm], r = {self.r:.5g} mm"

    def _find_bounds(self) -> tuple[float, float]:
        """
        Estimate min and max z-value on Surface by sampling many values.

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
        rcos, rsin = ne.evaluate("r*cos(phi)"), ne.evaluate("r*sin(phi)")
        vals = self._values(rcos, rsin)
        mask = self.mask(rcos - self.pos[0], rsin - self.pos[1])
        vals[~mask] = np.nan

        # in many cases the minimum and maximum are at the  edge of the surface
        # => sample it additionally

        # values at surface edge
        xv, yv, vals2 = self.edge(3001)
        vals2 -= self.pos[2]
        mask = self.mask(xv, yv)
        vals2[~mask] = np.nan

        # find minimum and maximum value
        z_min = min(np.nanmin(vals), np.nanmin(vals2))
        z_max = max(np.nanmax(vals), np.nanmax(vals2))

        return z_min, z_max

    # TODO make private
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
        Surface extent, values for a smallest box encompassing all of the surface

        :return: tuple of x0, x1, y0, y1, z0, z1
        """
        return *(self.r*np.array([-1, 1, -1, 1]) + self.pos[:2].repeat(2)), \
               self.z_min, self.z_max

    @property
    def ds(self) -> float:
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

    def values(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Get surface values. Absolute coordinates. 
        Points outside the surface are intersected with the radially continued edge.

        :param x: x-coordinate array (numpy 1D or 2D array)
        :param y: y-coordinate array (numpy 1D or 2D array)
        :return: z-coordinate array (numpy 1D or 2D array, depending on shape of x and y)
        """
        z = np.full_like(x, self.z_max, dtype=np.float64)
        
        if self.is_flat():
            return z
    
        else:
            inside = self.mask(x, y)
            z[inside] = self.pos[2] + self._values(x[inside] - self.pos[0], y[inside] - self.pos[1])
            r = self.r - self.N_EPS

            # continue the edge value in radial direction
            if np.any(~inside):
                if not self.rotational_symmetry:
                    xni, x0, yni, y0 = x[~inside], self.pos[0], y[~inside], self.pos[1]
                    phi = ne.evaluate("arctan2(yni - y0, xni - x0)")
                    z[~inside] = self.pos[2] + self._values(r * np.cos(phi), r * np.sin(phi))
                else:
                    z[~inside] = self.pos[2] + self._values(np.array([r]), np.array([0.]))[0]

            return z

    def _values(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Get surface values but in relative coordinate system to surface center.
        And without masking out values beyond the surface extent

        :param x: x-coordinate array (numpy 1D or 2D array)
        :param y: y-coordinate array (numpy 1D or 2D array)
        :return: z-coordinate array (numpy 1D or 2D array, depending on shape of x and y)
        """
        assert self.is_flat(), "function not implemented for sub-class"
        return np.zeros_like(x, dtype=np.float64)

    def plotting_mesh(self, N: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get 2D plotting mesh. Note that the values are not gridded, the distance can be arbitrary.
        The only guarantee is that neighbouring array values are neighbouring values in 3D space.

        :param N: number of grid values in each dimension (int)
        :return: X, Y, Z coordinate array (all numpy 2D array)
        """
        if N < 10:
            raise ValueError("Expected at least N=10.")

        # rectangular grid
        Y, X = np.mgrid[-self.r:self.r:N*1j,
                        -self.r:self.r:N*1j]

        # convert to polar coordinates
        R = np.sqrt(X**2 + Y**2)
        Phi = np.arctan2(Y, X)

        # masks for values outside of circular area
        r = self.r
        mask = R.ravel() >= r
        mask2 = R >= r
        
        # values inside circular area
        z = np.zeros(mask.shape, dtype=np.float64)
        z[~mask] = self._values(X[~mask2].ravel(), Y[~mask2].ravel())

        # move values outside surface to the surface edge
        # this defines the edge with more points, making it more circular instead of step-like
        z[mask] = self._values(r * np.cos(Phi.ravel()[mask]), r * np.sin(Phi.ravel()[mask]))
        X[mask2] = r*np.cos(Phi[mask2])
        Y[mask2] = r*np.sin(Phi[mask2])

        # plot nan values inside
        mask3 = self.mask(X.ravel() + self.pos[0], Y.ravel() + self.pos[1])  # mask has absolute coordinates
        z[~mask3] = np.nan

        # make 2D
        Z = z.reshape(X.shape)

        # return offset values
        return X+self.pos[0], Y+self.pos[1], Z+self.pos[2]

    def mask(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
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

    def normals(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
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
        m = self.mask(x, y)
        xm, ym = x[m], y[m]

        # approximate optimal step width for differentiation  for a second derivative
        # of form (f(x+h) - f(x-h)) / (2h) the optimal h is
        # h* = (3*e*abs(f(x)/f'''(x))^(1/3)  with e being the machine precision
        # for us one general assumption can be e.g. abs(f(x)/f'''(x) = 50, additionally x+h needs to be representable
        # check the documentation for more info
        eps_f = np.finfo(np.float64).eps
        eps_deriv = (3*eps_f*50)**(1/3)
        ext = np.array(self.extent)
        eps_num = np.spacing(ext[1::2] - ext[::2])
        eps = max(eps_deriv, *eps_num)

        # uz is the surface change in x direction, vz in y-direction, the normal vector
        # of derivative vectors u = (2*ds, 0, uz) and v = (0, 2*ds, vz) is n = (-2*uz*ds, -2*ds*vz, 4*ds*ds)
        # this can be rescaled to n = (-uz, -vz, 2*ds)
        x_, y_ = xm - self.pos[0], ym - self.pos[1]  # use coordinates relative to center
        n[m, 0] = self._values(x_ - eps, y_) - self._values(x_ + eps, y_)  # -uz
        n[m, 1] = self._values(x_, y_ - eps) - self._values(x_, y_ + eps)  # -vz
        n[m, 2] = 2*eps

        # normalize vectors
        n[m] = misc.normalize(n[m])

        return n

    def edge(self, nc: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get surface values of the surface edge.

        :param nc: number of points on edge (int)
        :return: X, Y, Z coordinate arrays (all numpy 2D array)
        """

        if nc < 20:
            raise ValueError("Expected at least nc=20")

        # start at -135 degrees, which is consistent how the RectangularSurface plots the edge
        theta = np.linspace(-3/4*np.pi, 5/4 * np.pi, nc)
        xd = self.r * np.cos(theta)
        yd = self.r * np.sin(theta)
        zd = self._values(xd, yd)

        return xd + self.pos[0],\
            yd + self.pos[1],\
            zd + self.pos[2]

    def find_hit(self, p: np.ndarray, s: np.ndarray)\
            -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Find the position of hits on surface using the iterative regula falsi algorithm.

        :param p: position array (numpy 2D array, shape (N, 3))
        :param s: direction array (numpy 2D array, shape (N, 3))
        :return: positions of hit (shape as p), bool numpy 1D array if ray hits lens, indices of ill-conditioned rays
        """

        if self.is_flat():
            # intersection with xy plane
            t = (self.pos[2] - p[:, 2])/s[:, 2]
            p_hit = p + s*t[:, np.newaxis]
            is_hit = self.mask(p_hit[:, 0], p_hit[:, 1])

            self._find_hit_handle_abnormal(p, s, p_hit, is_hit)

            return p_hit, is_hit, np.array([])

        else:
            # ray parameters for just above and below the surface
            t1 = (self.z_min - self.C_EPS / 10 - p[:, 2]) / s[:, 2]
            t2 = (self.z_max + self.C_EPS / 10 - p[:, 2]) / s[:, 2]

            # set to -eps since we can't move in -z anyway (t1 <0)
            t1[t1 < 0] = -self.C_EPS

            # check if surface is defined for t1 and t2
            p1 = p + s*t1[:, np.newaxis]
            p2 = p + s*t2[:, np.newaxis]

            # cost function values
            f1 = p1[:, 2] - self.values(p1[:, 0], p1[:, 1])
            f2 = p2[:, 2] - self.values(p2[:, 0], p2[:, 1])

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

            # we need values with different signs to find the root
            ill = f1*f2 > 0

            it = 1  # number of iteration
            # do until all rays have converged
            while np.any(w):

                # secant root
                t1w, t2w, f1w, f2w = t1[w], t2[w], f1[w], f2[w]
                ts = ne.evaluate("t1w - f1w/(f2w-f1w)*(t2w-t1w)")

                # position of root on rays
                pl = p[w] + s[w]*ts[:, np.newaxis]

                # difference between ray and surface at root
                fts = pl[:, 2] - self.values(pl[:, 0], pl[:, 1])

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
                if it == 200:  # how to check a timeout? # pragma: no cover
                    raise TimeoutError(f"Timeout after {it} iterations in hit finding.") 
                it += 1

            # check if hit is an actual hit. For this the surface needs to be defined at this x and y
            # as well as the actual surface value being near
            is_hit = self.mask(p_hit[:, 0], p_hit[:, 1])

            # handle rays that start behind surface or inside its extent 
            self._find_hit_handle_abnormal(p, s, p_hit, is_hit)

        return p_hit, is_hit, ill

    def flip(self) -> None:
        """flip the surface around the x-axis"""
        assert self.is_flat()  # flip otherwise not implemented

    def rotate(self, angle: float) -> None:
        """
        rotate the surface around the z-axis
        :param angle: rotation angle in degrees
        """
        assert self.rotational_symmetry

    def _rotate_rc(self, x: np.ndarray, y: np.ndarray, alpha: float)\
            -> tuple[np.ndarray, np.ndarray]:
        """helper function rotating surface coordinates so the surface does not need to be rotated itself"""

        if alpha:
            return ne.evaluate("x*cos(alpha) - y*sin(alpha)"),\
                ne.evaluate("x*sin(alpha) + y*cos(alpha)")

        return x, y

    def _find_hit_handle_abnormal(self, 
                                  p:        np.ndarray, 
                                  s:        np.ndarray, 
                                  p_hit:    np.ndarray, 
                                  is_hit:   np.ndarray)\
            -> None:
        """
        this handles "abnormal" rays.
        Ray starts after z_max -> p_hit = p
        Ray starts after the surface but before z=z_max  -> intersect ray with plane z=z_max
        Hit is in negative z-direction -> intersect ray at plane z=z_max
        Other kind of deviation between hit z-value and surface z-value -> intersect with plane z=z_max
        All these rays are set as non-hitting by updating is_hit

        All normal rays are kept intact.

        :param p: ray position vectors
        :param s: ray unity direction vectors
        :param p_hit: intersection positions
        :param is_hit: boolean hit array
        """

        zs = self.values(p_hit[:, 0], p_hit[:, 1])  # surface values
        dev = np.abs(p_hit[:, 2] - zs) > self.C_EPS  # z-value deviates
        beh = p[:, 2] > self.z_max + self.N_EPS  # rays start after surface
        neg = p_hit[:, 2] < p[:, 2] - self.C_EPS  # hit is behind ray start
        bet = (neg | dev) & ~beh
        # ^-- rays don't hit, hit in negative direction or start after surface, but before highest surface value
       
        # "bet"-rays should intersect plane at z=z_max
        tnm = (self.z_max - p[bet, 2])/s[bet, 2]
        p_hit[bet] = p[bet] + s[bet]*tnm[:, np.newaxis]
        is_hit[bet] = False # ray starting behind the surface don't count as hit

        # "beh" rays should keep their coordinates
        p_hit[beh] = p[beh]
        is_hit[beh] = False

        brok = bet | beh
        if np.any(brok):
            brokc = np.count_nonzero(brok)
            # TODO handle differently (return indices and let it handle by main thread?)
            warning(f"Broken sequentiality. {brokc} rays start behind the current surface. "
                     "The simulation results for these rays are most likely wrong. Check the geometry.")

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
