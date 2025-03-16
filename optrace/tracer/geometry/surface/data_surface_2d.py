
import numpy as np  # calculations
import scipy.interpolate  # interpolation

from ... import misc  # calculations
from ...misc import PropertyChecker as pc  # check types and values
from .surface import Surface  # parent class
from ....warnings import warning


class DataSurface2D(Surface):
    
    rotational_symmetry: bool = False  #: has the surface rotational symmetry?

    _1D: bool = False
    """1D or 2D data surface, basically if we use an 1D profile as whole surface description"""

    def __init__(self,
                 r:                 float,
                 data:              np.ndarray,
                 parax_roc:         float = None,
                 **kwargs)\
            -> None:
        """
        Create a surface object, defined by a two dimensional data set.

        :param r: radial size of surface
        :param parax_roc: paraxial radius of curvature, optional
        :param data: uses copy of this array, grid z-coordinate array for surface_type="Data" (numpy 2D array)
        :param kwargs: additional keyword arguments for parent classes
        """
        self._lock = False

        super().__init__(r, **kwargs)

        # sign used for reversing the function
        self._sign = 1
    
        # rotation angle
        self._angle = 0

        self._interp, self._offset = None, 0.
        self.parax_roc = parax_roc

        pc.check_type("data", data, np.ndarray | list)
        Z = np.asarray_chkfinite(data, dtype=np.float64)

        surf_name = f"{type(self).__name__} {self.get_desc(hex(id(self)))}"

        # too few values exception
        nx = Z.shape[0]
        if nx < 50:
            raise ValueError("For a good surface representation 'data' should have at least 50 values per dimension")

        # too few values warning
        if nx < 200:
            warning(f"{surf_name}: At least 200 values per dimension are advised for a 'data' matrix, "
                    f"but got {nx} values for surface {self.get_desc(hex(id(self)))}.")

        if self._1D:
            
            if Z.ndim != 1:
                raise ValueError("data array needs to have exactly one dimension.")
            
            # remove offset at first value, actual offset at center will be removed later
            Z -= Z[0]
        
            # create r vector
            r0 = np.linspace(0, self.r, Z.shape[0], dtype=np.float64)

            # mirror values around r axis, we now have a symmetric profile including the lens center
            r2 = np.concatenate((-np.flip(r0[1:]), r0))
            z2 = np.concatenate((np.flip(Z[1:]), Z))

            # we need spline order n=4 so curvature and curvature changes are smooth
            self._interp = scipy.interpolate.InterpolatedUnivariateSpline(r2, z2, k=4)
            self._offset = self._call(0, 0)  # remaining z_offset at center
        
            # get max and min z values
            rn = np.linspace(0, self.r, 10000)
            zn = self._values(rn, np.zeros_like(rn))
            self.z_min, self.z_max = np.min(zn), np.max(zn)

            # range of input data, not to be confused with height of interpolated data
            z_range0 = np.max(Z) - np.min(Z)

        else:
            if Z.ndim != 2:
                raise ValueError("data array needs to have exactly two dimensions.")
            
            ny, nx = Z.shape
            if nx != ny:
                raise ValueError("Array 'data' needs to be of square shape.")
            
            # remove offset at center
            if nx % 2:
                Z -= np.array([Z[ny//2, nx//2], Z[ny//2+1, nx//2], Z[ny//2, nx//2+1], Z[ny//2+1, nx//2+1]]).mean()
            else:
                Z -= Z[ny//2, nx//2]
        
            xy = np.linspace(-self.r, self.r, nx)  # numeric r vector

            # we need spline order n=4 so curvature and curvature changes are smooth
            self._interp = scipy.interpolate.RectBivariateSpline(xy, xy, Z, kx=4, ky=4)
            self._offset = self._call(0, 0)  # remaining z_offset at center

            # self.z_min, self.z_max = np.nanmin(Z), np.nanmax(Z)  # provisional values
            self.z_min, self.z_max = self._find_bounds()
       
            # get range of input data, but only inside circular area (= enforced by masking)
            X, Y = np.meshgrid(xy, xy)
            M = self.mask(X.ravel(), Y.ravel()).reshape(X.shape)
            z_range0 = np.max(Z[M]) - np.min(Z[M])
      
        # interpolation can lead to an increased z-value range
        # e.g. spheric surface, but the center is not defined by a data point
        z_range1 = self.z_max - self.z_min
        if np.abs(z_range0 - z_range1) > self.N_EPS:
            z_change = (z_range1 - z_range0) / z_range0
            add_warning = "WARNING: Deviations this high can be due to noise or abrupt changes in the data."\
                          " DO NOT USE SUCH SURFACES HERE." if z_change > 0.05 else ""

            warning(f"{surf_name}: Due to biquadratic interpolation the z_range of the surface"
                    f" has increased from {z_range0:.9g} to {z_range1:.9g},"
                    f" a change of {z_change*100:.5g}%. {add_warning}")

        # lock properties
        self.lock()

    def _call(self, x: np.ndarray, y: np.ndarray, **kwargs):
        """internal function.
        get surface values relative to center, directly works on interpolation object without rotation or flipping
        """
        if self._1D:
            r = np.hypot(x, y)
            return self._interp(r, **kwargs)
        else:
            return self._interp(x, y, grid=False, **kwargs)

    def _values(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Get surface values but in relative coordinate system to surface center.
        And without masking out values beyond the surface extent

        :param x: x-coordinate array (numpy 1D or 2D array)
        :param y: y-coordinate array (numpy 1D or 2D array)
        :return: z-coordinate array (numpy 1D or 2D array, depending on shape of x and y)
        """
        # rotate if needed and surface has no rotational symmetry
        x_, y_ = self._rotate_rc(x, y, -self._angle) if not self.rotational_symmetry else (x, y)

        # uses quadratic interpolation for smooth surfaces and a constantly changing slope
        return self._sign*(self._call(x_, self._sign*y_) - self._offset)

    def normals(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Get normal vectors of the surface.

        :param x: x-coordinates (numpy 1D array)
        :param y: y-coordinates (numpy 1D array)
        :return: normal vector array of shape (x.shape[0], 3), components in second dimension (numpy 2D array)
        """

        n = np.tile([0., 0., 1.], (x.shape[0], 1))

        # coordinates actually on surface
        m = self.mask(x, y)
       
        # relative coordinates
        xm = x[m] - self.pos[0]
        ym = y[m] - self.pos[1]

        if not self._1D:
            # rotate surface
            x_, y_ = self._rotate_rc(xm, ym, -self._angle)
            
            # rotating [x, y, z] around [1, 0, 0] by pi gives us [x, -y, -z]
            # we need to negate this, so the vector points in +z direction
            # -> [-x, y, z]

            # the y value for the interpolation needs to be negated, since the surface is flipped around the x-axis
            nxn = self._call(x_, self._sign*y_, dx=1, dy=0)*self._sign
            nyn = self._call(x_, self._sign*y_, dx=0, dy=1)
            nxn, nyn = self._rotate_rc(nxn, nyn, self._angle)
        else:
            # get vector n = (n_r, n_z) and rotate n_r to get n = (nx, ny, nz)
            phi = np.arctan2(ym, xm)
            nr = self._sign*self._call(xm, ym, nu=1)
            nxn, nyn = nr*np.cos(phi), nr*np.sin(phi)

        # the interpolation object provides partial derivatives
        n[m, 0] = -nxn
        n[m, 1] = -nyn
        n[m] = misc.normalize(n[m])

        return n

    def flip(self) -> None:
        """flip the surface around the x-axis"""
        
        # unlock
        self._lock = False

        # invert sign
        self._sign *= -1

        # invert curvature circle if given
        self.parax_roc = self.parax_roc if self.parax_roc is None else -self.parax_roc

        # assign new values for z_min, z_max. Both are negated and switched
        a = self.pos[2] - (self.z_max - self.pos[2])
        b = self.pos[2] - (self.z_min - self.pos[2])
        self.z_min, self.z_max = a, b

        # lock
        self.lock()
    
    def rotate(self, angle: float) -> None:
        """
        rotate the surface around the z-axis

        :param angle: rotation angle in degrees
        """
        if not self.rotational_symmetry:
            self._lock = False
            self._angle += np.deg2rad(angle)  # add relative rotation
            self.lock()
