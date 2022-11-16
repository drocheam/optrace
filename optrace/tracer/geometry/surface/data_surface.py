
import numpy as np  # calculations
import scipy.interpolate  # biquadratic interpolation

from ... import misc  # calculations
from ...misc import PropertyChecker as pc  # check types and values
from .surface import Surface


# TODO DataSurface -> DataSurface2D
# TODO New DataSurface1D, r and values are provided as arrays, no need for them to be equally spaced
# spline interpolateion as in DataSurface. Add mirrored copy for r < 0 so we have a smooth curvature at r = 0


class DataSurface:
    pass


class DataSurface(Surface):
    
    rotational_symmetry: bool = False


    def __init__(self,
                 r:                 float,
                 data:              np.ndarray,
                 parax_roc:         float = None,
                 **kwargs)\
            -> None:
        """
        Create a surface object.
        The z-coordinate in the pos array is irrelevant if the surface is used for a lens, since it wil be
        adapted inside the lens class

        :param r: radial size for surface_type="Conic", "Sphere", "Circle" or "Ring" (float)
        :param parax_roc:
        :param data: uses copy of this array, grid z-coordinate array for surface_type="Data" (numpy 2D array)
        """
        self._lock = False

        super().__init__(r, **kwargs)

        # sign used for reversing the function
        self._sign = 1
    
        self._angle = 0

        self._interp, self._offset = None, 0.
        self.parax_roc = parax_roc

        pc.check_type("data", data, np.ndarray | list)
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
        self.z_min, self.z_max = self._find_bounds()
      
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

        self.lock()

    def _get_values(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Get surface values but in relative coordinate system to surface center.
        And without masking out values beyond the surface extent

        :param x: x-coordinate array (numpy 1D or 2D array)
        :param y: y-coordinate array (numpy 1D or 2D array)
        :return: z-coordinate array (numpy 1D or 2D array, depending on shape of x and y)
        """
        x_, y_ = self._rotate_rc(x, y, -self._angle)

        # uses quadratic interpolation for smooth surfaces and a constantly changing slope
        return self._sign*(self._interp(x_, self._sign*y_, grid=False) - self._offset)

    def get_normals(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Get normal vectors of the surface.

        :param x: x-coordinates (numpy 1D array)
        :param y: y-coordinates (numpy 1D array)
        :return: normal vector array of shape (x.shape[0], 3), components in second dimension (numpy 2D array)
        """

        n = np.tile([0., 0., 1.], (x.shape[0], 1))

        # coordinates actually on surface
        m = self.get_mask(x, y)
       
        # relative coordinates
        xm = x[m] - self.pos[0]
        ym = y[m] - self.pos[1]

        # rotate surface
        x_, y_ = self._rotate_rc(xm, ym, -self._angle)
        
        # rotating [x, y, z] around [1, 0, 0] by pi gives us [x, -y, -z]
        # we need to negate this, so the vector points in +z direction
        # -> [-x, y, z]

        # the y value for the interpolation needs to be negated, since the surface is flipped around the x-axis

        nxn = self._interp(x_, self._sign*y_, dx=1, dy=0, grid=False)*self._sign
        nyn = self._interp(x_, self._sign*y_, dx=0, dy=1, grid=False)
        nxn, nyn = self._rotate_rc(nxn, nyn, self._angle)

        # the interpolation object provides partial derivatives
        n[m, 0] = -nxn
        n[m, 1] = -nyn
        n[m] = misc.normalize(n[m])

        return n

    def flip(self) -> None:
        
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

        self._lock = False
        self._angle += np.deg2rad(angle)  # add relative rotation
        self.lock()
