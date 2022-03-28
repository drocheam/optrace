

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


from Backend.Surface import Surface  # for the SObject surface

from typing import Callable  # for function type hints
import copy  # for copy.deepcopy
import numpy as np


class SObject:

    def __init__(self, 
                 FrontSurface:       Surface, 
                 pos:           (list | np.ndarray),
                 BackSurface:   Surface=None,
                 d1:            float=None,
                 d2:            float=None)\
            -> None:
        """
        Create a SObject object..

        :param FrontSurface: Surface object
        :param pos: 3D position of SObject center (numpy array or list)
        """

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
        pos = self.FrontSurface.pos
        self.FrontSurface = surf.copy()
        self.FrontSurface.moveTo(pos)

    def setBackSurface(self, surf: Surface) -> None:
        """
        Assign a new Surface to the SObject.

        :param surf: Surface to assign
        """
        pos = self.BackSurface.pos
        self.BackSurface = surf.copy()
        self.BackSurface.moveTo(pos)

    def moveTo(self, pos: (list | np.ndarray)) -> None:
        """
        Moves the SObject in 3D space.

        :param pos: new 3D position of SObject center (list or numpy array)
        """

        pos = np.array(pos, dtype=np.float64)

        if not self.hasBackSurface():
            self.FrontSurface.moveTo(pos)
        else:
            self.FrontSurface.moveTo(pos - [0, 0, self.d1])
            self.BackSurface.moveTo(pos + [0, 0, self.d2])
    
    def copy(self) -> 'SObject':
        """
        Return a fully independent copy of the SObject.

        :return: copy
        """
        return copy.deepcopy(self)

    @property
    def Surface(self):
        """alias for :obj:`SObject.FrontSurface` for a SObject without BackSurface"""
        return self.FrontSurface

    @property
    def pos(self) -> np.ndarray:
        """ position of the SObject center """
        if not self.hasBackSurface():
            return self.FrontSurface.pos
        else:
            return self.FrontSurface.pos + [0, 0, self.d1]

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

    def getCylinderSurface(self, nc: int = 100, d: float = 0.1) \
            -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get a 3D surface representation of the SObject cylinder for plotting.

        :param nc: number of surface edge points (int)
        :param d: thickness for visualization (float)
        :return: tuple of coordinate arrays X, Y, Z (2D numpy arrays)
        """

        X1, Y1, Z1 = self.FrontSurface.getEdge(nc)

        if not self.hasBackSurface():
            X2, Y2, Z2 = X1, Y1, Z1 + d
        else:
            X2, Y2, Z2 = self.BackSurface.getEdge(nc)

        X = np.column_stack((X1, X2))
        Y = np.column_stack((Y1, Y2))
        Z = np.column_stack((Z1, Z2))

        return X, Y, Z
