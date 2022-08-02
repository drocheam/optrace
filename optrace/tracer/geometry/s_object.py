

"""
Parent class for :obj:`Backend.Filter`, :obj:`Backend.RaySource`, :obj:`Backend.Lens` and :obj:`Backend.Detector`

A SObject has a FrontSurface and an optional BackSurface.

Meaning of 'Front' and 'Back'
 * Defined by the :obj:`Backend.Surface` z-position
 * :obj:`SObject.FrontSurface`: Surface with smaller z-position
 * :obj:`SObject.BackSurface`: Surface with larger z-position

Properties with FrontSurface only:
 * :obj:`SObject.pos` is :obj:`SObject.FrontSurface.pos`.
 * :obj:`SObject.extent` is :obj:`SObject.FrontSurface.extent`.
 * :obj:`SObject.Surface` and :obj:`SObject.setSurface`
        are aliases for :obj:`SObject.FrontSurface` and :obj:`SObject.setFrontSurface`.

Properties with FrontSurface + BackSurface:
 * (:obj:`SObject.d1` + :obj:`SObject.d2`) is the whole z-difference of the SObject in z-direction
 * :obj:`SObject.d1` defines the z-distance between z-pos of the SObject and the z-pos of :obj:`SObject.FrontSurface`
 * :obj:`SObject.d2` defines the z-distance between z-pos of the SObject and the z-pos of :obj:`SObject.BackSurface`
 * :obj:`SObject.pos` is (z-pos of FrontSurface + d1) or (z-pos of BackSurface - d2),
                    which by above definitions are the same.
 * :obj:`SObject.extent` is the extent of both surfaces,
                    each value is determined by checking which Surface has a larger extent in this dimension
"""

import numpy as np  # calculations

from optrace.tracer.geometry.surface import Surface  # for the SObject surface
from optrace.tracer.base_class import BaseClass  # parent class
from optrace.tracer.misc import PropertyChecker as pc  # check types and values


class SObject(BaseClass):

    abbr = "SO"  # abbreviation for objects of this class
    _allow_non_2D = True  # allow points or lines as surfaces

    def __init__(self, 
                 front_surface: Surface, 
                 pos:           (list | np.ndarray),
                 back_surface:  Surface = None,
                 d1:            float = None,
                 d2:            float = None,
                 **kwargs)\
            -> None:
        """

        :param FrontSurface:
        :param pos:
        :param BackSurface:
        :param d1:
        :param d2:
        :param kwargs:
        """
        self._geometry_lock = False

        self.FrontSurface = front_surface
        self.BackSurface = back_surface
        self.d1 = d1
        self.d2 = d2

        if self.has_back_surface():

            if d1 is None or d2 is None:
                raise ValueError("d1 and d2 need to be specfied for a SObject with BackSurface")

            if d1 < 0 or d2 < 0:
                raise ValueError("Thicknesses d1, d2 need to be non-negative.")
        
        self.move_to(pos)

        super().__init__(**kwargs)
        
        self._geometry_lock = True

    def has_back_surface(self) -> bool:
        """:return: if the SObject has a BackSurface"""
        return self.BackSurface is not None

    def set_surface(self, surf: Surface) -> None:
        """
        Assign a new Surface to the SObject.

        :param surf: Surface to assign
        """
        if self.has_back_surface():
            raise RuntimeError("Replacing of Surfaces only supported for objects with one surface")

        self._geometry_lock = False
        pos = self.FrontSurface.pos
        self.FrontSurface = surf.copy()
        self.FrontSurface.move_to(pos)
        self._geometry_lock = True

    def move_to(self, pos: (list | np.ndarray)) -> None:
        """
        Moves the SObject in 3D space.

        :param pos: new 3D position of SObject center (list or numpy array)
        """
        pc.checkType("pos", pos, list | np.ndarray)
        pos = np.array(pos, dtype=np.float64)

        if pos.shape[0] != 3:
            raise ValueError("pos needs to have 3 elements.")

        if not self.has_back_surface():
            self.FrontSurface.move_to(pos)
        else:
            self.FrontSurface.move_to(pos - [0, 0, self.d1])
            self.BackSurface.move_to(pos + [0, 0, self.d2])
    
    @property
    def surface(self):
        """alias for :obj:`SObject.FrontSurface` for a SObject without BackSurface"""
        return self.FrontSurface

    @property
    def pos(self) -> np.ndarray:
        """ position of the SObject center """
        return self.FrontSurface.pos + [0, 0, 0 if not self.has_back_surface() else self.d1]

    @property
    def extent(self) -> tuple[float, float, float, float, float, float]:
        """ 3D extent of the SObject"""
        if not self.has_back_surface():
            return self.FrontSurface.get_extent()
        else:
            front_ext = self.FrontSurface.get_extent()[:4]
            back_ext = self.BackSurface.get_extent()[:4]

            return min(front_ext[0], back_ext[0]),\
                max(front_ext[1], back_ext[1]),\
                min(front_ext[2], back_ext[2]),\
                max(front_ext[3], back_ext[3]),\
                self.FrontSurface.zmin,\
                self.BackSurface.zmax

    def get_desc(self) -> str:
        """"""
        if self.has_back_surface():
            fallback = f"{self.FrontSurface.surface_type} + {self.BackSurface.surface_type} "\
                       f"at [{self.pos[0]:.04g}, {self.pos[1]:.04g}, {self.pos[2]:.04g}]"
        else:
            fallback = f"{self.surface.surface_type} at [{self.pos[0]:.04g}, "\
                       f"{self.pos[1]:.04g}, {self.pos[2]:.04g}]"

        return super().get_desc(fallback)

    def get_cylinder_surface(self, nc: int = 100, d: float = 0.1) \
            -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get a 3D surface representation of the SObject cylinder for plotting.

        :param nc: number of surface edge points (int)
        :param d: thickness for visualization (float)
        :return: tuple of coordinate arrays X, Y, Z (2D numpy arrays)
        """

        X1, Y1, Z1 = self.FrontSurface.get_edge(nc)
        X2, Y2, Z2 = self.BackSurface.get_edge(nc) if self.has_back_surface() else (X1, Y1, Z1 + d)

        return np.column_stack((X1, X2)),\
            np.column_stack((Y1, Y2)),\
            np.column_stack((Z1, Z2))

    def __setattr__(self, key, val):

        # lock changing of geometry directly
        if "_geometry_lock" in self.__dict__ and self._geometry_lock:
            if key in ["d1", "d2", "FrontSurface", "Surface", "BackSurface"]:
                raise RuntimeError("Use Functions setSurface to reassign a new Surface or its thickness.")
            if key == "pos":
                raise RuntimeError("Use moveTo(pos) to move the Object")

        match key:

            case "FrontSurface":
                pc.checkType(key, val, Surface)
                if val is not None and not self._allow_non_2D and not val.is_2d():
                    raise RuntimeError(f"FrontSurface of a {self.__class__.__name__} object needs to be 2 dimensional.")

                super().__setattr__(key, val.copy())  # save internal copy
                return

            case "BackSurface":
                pc.checkType(key, val, Surface | None)
                if val is not None and not self._allow_non_2D and not val.is_2d():
                    raise RuntimeError(f"BackSurface of a {self.__class__.__name__} object needs to be 2 dimensional.")

                super().__setattr__(key, val.copy() if val is not None else None)  # save internal copy
                return
            
            case ("d1" | "d2"):
                pc.checkType(key, val, int | float | None)
                val = float(val) if val is not None else val

        super().__setattr__(key, val)
    
