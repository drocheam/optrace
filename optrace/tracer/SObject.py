

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
 * :obj:`SObject.Surface` and :obj:`SObject.setSurface` are aliases for :obj:`SObject.FrontSurface` and :obj:`SObject.setFrontSurface`.

Properties with FrontSurface + BackSurface:
 * (:obj:`SObject.d1` + :obj:`SObject.d2`) is the whole z-difference of the SObject in z-direction
 * :obj:`SObject.d1` defines the z-distance between z-pos of the SObject and the z-pos of :obj:`SObject.FrontSurface`
 * :obj:`SObject.d2` defines the z-distance between z-pos of the SObject and the z-pos of :obj:`SObject.BackSurface`
 * :obj:`SObject.pos` is (z-pos of FrontSurface + d1) or (z-pos of BackSurface - d2), which by above definitions are the same.
 * :obj:`SObject.extent` is the extent of both surfaces, each value is determined by checking which Surface has a larger extent in this dimension

"""


from optrace.tracer.Surface import *  # for the SObject surface

from typing import Callable  # for function type hints
import numpy as np
from optrace.tracer.BaseClass import *


class SObject(BaseClass):

    abbr = "SO" # abbreviation for objects of this class
    _allow_non_2D = True # allow points or lines as surfaces


    def __init__(self, 
                 FrontSurface:  Surface, 
                 pos:           (list | np.ndarray),
                 BackSurface:   Surface = None,
                 d1:            float = None,
                 d2:            float = None,
                 **kwargs)\
            -> None:
        """
        Create a SObject object..

        :param FrontSurface: Surface object
        :param pos: 3D position of SObject center (numpy array or list)
        """
        self._geometry_lock = False

        # use a Surface copy, since we change its position in 3D space
        self.FrontSurface = FrontSurface.copy()
        self.BackSurface = BackSurface.copy() if BackSurface is not None else None

        self.d1 = d1
        self.d2 = d2

        if self.hasBackSurface():

            if d1 is None or d2 is None:
                raise ValueError("d1 and d2 need to be specfied for a SObject with BackSurface")

            if d1 < 0 or d2 < 0:
                raise ValueError("Thicknesses de, d1, d2 need to be non-negative.")
        
        self.moveTo(pos)

        super().__init__(**kwargs)
        
        self._geometry_lock = True

    def hasBackSurface(self) -> bool:
        """:return: if the SObject has a BackSurface"""
        return self.BackSurface is not None

    def setSurface(self, surf: Surface) -> None:
        """alias for :obj:`SObject.setFrontSurface` for a SObject with no BackSurface"""
        self.setFrontSurface(surf)

    def setFrontSurface(self, surf: Surface) -> None:
        """
        Assign a new Surface to the SObject.

        :param surf: Surface to assign
        """
        self._geometry_lock = False
        pos = self.FrontSurface.pos
        self.FrontSurface = surf.copy()
        self.FrontSurface.moveTo(pos)
        self._geometry_lock = True

    def setBackSurface(self, surf: Surface) -> None:
        """
        Assign a new Surface to the SObject.

        :param surf: Surface to assign
        """
        self._geometry_lock = False
        pos = self.BackSurface.pos
        self.BackSurface = surf.copy()
        self.BackSurface.moveTo(pos)
        self._geometry_lock = True

    def moveTo(self, pos: (list | np.ndarray)) -> None:
        """
        Moves the SObject in 3D space.

        :param pos: new 3D position of SObject center (list or numpy array)
        """
        self._checkType("pos", pos, list | np.ndarray)
        pos = np.array(pos, dtype=np.float64)

        if not self.hasBackSurface():
            self.FrontSurface.moveTo(pos)
        else:
            self.FrontSurface.moveTo(pos - [0, 0, self.d1])
            self.BackSurface.moveTo(pos + [0, 0, self.d2])
    
    @property
    def Surface(self):
        """alias for :obj:`SObject.FrontSurface` for a SObject without BackSurface"""
        return self.FrontSurface

    @property
    def pos(self) -> np.ndarray:
        """ position of the SObject center """
        return self.FrontSurface.pos + [0, 0, 0 if not self.hasBackSurface() else self.d1]

    @property
    def extent(self) -> tuple[float, float, float, float, float, float]:
        """ 3D extent of the SObject"""
        if not self.hasBackSurface():
            return self.FrontSurface.getExtent()
        else:
            front_ext = self.FrontSurface.getExtent()[:4]
            back_ext = self.BackSurface.getExtent()[:4]

            return min(front_ext[0], back_ext[0]),\
                   max(front_ext[1], back_ext[1]),\
                   min(front_ext[2], back_ext[2]),\
                   max(front_ext[3], back_ext[3]),\
                   self.FrontSurface.minz,\
                   self.BackSurface.maxz

    def getDesc(self) -> str:
        """"""
        if self.hasBackSurface():
            fallback = f"{self.FrontSurface.surface_type} + {self.BackSurface.surface_type} "\
                       f"at [{self.pos[2]:.04g}, {self.pos[1]:.04g}, {self.pos[2]:.04g}]"
        else:
            fallback =  f"{self.Surface.surface_type} at [{self.pos[2]:.04g}, "\
                        f"{self.pos[1]:.04g}, {self.pos[2]:.04g}]"

        return super().getDesc(fallback)

    def getCylinderSurface(self, nc: int = 100, d: float = 0.1) \
            -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get a 3D surface representation of the SObject cylinder for plotting.

        :param nc: number of surface edge points (int)
        :param d: thickness for visualization (float)
        :return: tuple of coordinate arrays X, Y, Z (2D numpy arrays)
        """

        X1, Y1, Z1 = self.FrontSurface.getEdge(nc)
        X2, Y2, Z2 = self.BackSurface.getEdge(nc) if self.hasBackSurface() else (X1, Y1, Z1+d)

        return np.column_stack((X1, X2)),\
               np.column_stack((Y1, Y2)),\
               np.column_stack((Z1, Z2))

    def __setattr__(self, key, val):

        # lock changing of geometry directly
        if "_geometry_lock" in self.__dict__ and self._geometry_lock:
            if key in ["d1", "d2", "FrontSurface", "Surface", "BackSurface"]:
                raise RuntimeError("Use Functions setFrontSurface and setBackSurface to reassign a new Surface or its thickness.")
            if key == "pos":
                raise RuntimeError("Use moveTo(pos) to move the Object")

        match key:

            case "FrontSurface":
                self._checkType(key, val, Surface)
                if val is not None and not self._allow_non_2D and not val.is2D():
                    raise RuntimeError(f"FrontSurface of a {self.__class__.__name__} object needs to be 2 dimensional.")

            case "BackSurface":
                self._checkType(key, val, Surface | None)
                if val is not None and not self._allow_non_2D and not val.is2D():
                    raise RuntimeError(f"BackSurface of a {self.__class__.__name__} object needs to be 2 dimensional.")
            
            case ("d1" | "d2"):
                self._checkType(key, val, int | float | None)
                val = float(val) if val is not None else val

        super().__setattr__(key, val)
    
