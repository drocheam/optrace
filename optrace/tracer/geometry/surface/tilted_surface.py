
"""
Surface class:
Provides the functionality for the creation of numerical or analytical surfaces.
The class contains methods for interpolation, surface masking and normal calculation
"""

from typing import Any  # Any type

import numpy as np  # calculations

from ... import misc  # calculations
from ...misc import PropertyChecker as pc  # check types and values
from .surface import Surface


class TiltedSurface:
    pass



class TiltedSurface(Surface):
    
    rotational_symmetry: bool = False


    def __init__(self,
                 r:                 float,
                 normal:            (list | np.ndarray) = None,
                 normal_sph:        (list | np.ndarray) = None,
                 **kwargs)\
            -> None:
        """
        Create a surface object.
        The z-coordinate in the pos array is irrelevant if the surface is used for a lens, since it wil be
        adapted inside the lens class

        :param r: radial size for surface_type="Conic", "Sphere", "Circle" or "Ring" (float)
        :param normal:
        """
        self._lock = False

        super().__init__(r, **kwargs)

        self.r = r
        self.parax_roc = None
        self.z_min = self.z_max = self.pos[2]

        # assign normal
        if normal is not None:
            self.normal = normal
        elif normal_sph is not None:
            pc.check_type("normal_sph", normal_sph, list | np.ndarray)
            theta, phi = np.radians(normal_sph[0]), np.radians(normal_sph[1])
            self.normal = [np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)]
        else:
            raise RuntimeError("normal or normal_sph parameter needs to be specified.")

        phi = np.arctan2(self.normal[1], self.normal[0])
        R = self.r
        val1 = self.pos[2] + self._get_values(np.array([R*np.cos(phi)]), np.array([R*np.sin(phi)]))[0]
        val2 = self.pos[2] + self._get_values(np.array([-R*np.cos(phi)]), np.array([-R*np.sin(phi)]))[0]
        self.z_min, self.z_max = min(val1, val2), max(val1, val2)

        self.lock()

    @property
    def info(self) -> str:
        """property string for UI information"""
        return super().info + f", normal = [{self.normal[0]:.4f}, {self.normal[1]:.4f},"\
            f" {self.normal[2]:.4f}]"

    def _get_values(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Get surface values but in relative coordinate system to surface center.
        And without masking out values beyond the surface extent

        :param x: x-coordinate array (numpy 1D or 2D array)
        :param y: y-coordinate array (numpy 1D or 2D array)
        :return: z-coordinate array (numpy 1D or 2D array, depending on shape of x and y)
        """
        # slope in x and y direction from normal vector
        mx = -self.normal[0]/self.normal[2]
        my = -self.normal[1]/self.normal[2]
        # no division by zero because we enforce normal[:, 2] > 0,
        return x*mx + y*my

    def get_normals(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Get normal vectors of the surface.

        :param x: x-coordinates (numpy 1D array)
        :param y: y-coordinates (numpy 1D array)
        :return: normal vector array of shape (x.shape[0], 3), components in second dimension (numpy 2D array)
        """
        # coordinates actually on surface
        m = self.get_mask(x, y)
        n = np.tile([0., 0., 1.], (x.shape[0], 1))
        n[m] = self.normal

        return n

    def find_hit(self, p: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """

        :param p:
        :param s:
        :return:
        """
        # intersection ray with plane
        # see https://www.scratchapixel.com/lessons/3d-basic-rendering/
        # minimal-ray-tracer-rendering-simple-shapes/ray-plane-and-ray-disk-intersection
        normal = np.broadcast_to(self.normal, (p.shape[0], 3))
        t = misc.rdot(self.pos - p, normal) / misc.rdot(s, normal)
        p_hit = p + s*t[:, np.newaxis]
        is_hit = self.get_mask(p_hit[:, 0], p_hit[:, 1])  # rays not hitting

        # handle rays that start behind surface or inside its extent 
        self._find_hit_handle_abnormal(p, s, p_hit, is_hit)

        return p_hit, is_hit

    def flip(self) -> None:

        # rotating [x, y, z] around [1, 0, 0] by pi gives us [x, -y, -z]
        # we need to negate this, so the vector points in +z direction
        # -> [-x, y, z]
        self._lock = False
        self.normal.flags.writeable = True
        self.normal[0] *= -1
        self.lock()

    def rotate(self, angle: float) -> None:
        
        self._lock = False
        self.normal.flags.writeable = True
        self.normal[:2] = self._rotate_rc(self.normal[0], self.normal[1], np.deg2rad(angle))
        self.lock()

    def __setattr__(self, key: str, val: Any) -> None:
        """
        assigns the value of an attribute
        :param key: attribute name
        :param val: value to assign
        """
        if key == "normal" and val is not None:

            pc.check_type(key, val, list | np.ndarray)
            val2 = np.asarray_chkfinite(val, dtype=np.float64) / np.linalg.norm(val)  # normalize
           
            pc.check_above("normal[2]", val2[2], 0)
            super().__setattr__(key, val2)
        else:
            super().__setattr__(key, val)
