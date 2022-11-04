

"""
Parent class for :obj:`optrace.tracer.geometry.filter.Filter`, :obj:`optrace.tracer.geometry.aperture.Aperture`,
:obj:`optrace.tracer.geometry.ray_source.RaySource`, :obj:`optrace.tracer.geometry.lens.Lens` and :obj:`optrace.tracer.geometry.detector.Detector`

A Element has a front and an optional back.

Meaning of 'front' and 'back'
 * Defined by the :obj:`surface` z-position
 * :obj:`Element.front`: Surface with smaller z-position
 * :obj:`Element.back`: Surface with larger z-position

Properties with FrontSurface only:
 * :obj:`Element.pos` is :obj:`Element.front.pos`.
 * :obj:`Element.extent` is :obj:`Element.front.extent`.
 * :obj:`Element.surface` is an alias for :obj:`Element.front`

Properties with FrontSurface + BackSurface:
 * d = (:obj:`Element.d1` + :obj:`Element.d2`) is the whole z-difference between the centers of both surfaces
 * :obj:`Element.d1` defines the z-distance between z-pos of the Element and the z-pos of :obj:`Element.front`
 * :obj:`Element.d2` defines the z-distance between z-pos of the Element and the z-pos of :obj:`Element.back`
 * :obj:`Element.pos` is (z-pos of front + d1) or (z-pos of back - d2),
                    which by above definitions are the same.
 * :obj:`Element.extent` is the extent of both surfaces,
                    each value is determined by checking which surface has a larger extent in this dimension
"""


from typing import Any  # Any type
import copy

import numpy as np  # calculations

from . import Surface, Point, Line

from ..base_class import BaseClass  # parent class
from ..misc import PropertyChecker as pc  # check types and values


class Element:
    pass

class Element(BaseClass):

    abbr: str = "EL"  #: abbreviation for objects of this class
    _allow_non_2D: bool = True  # allow points or lines as surfaces

    def __init__(self,
                 front: Surface | Line | Point,
                 pos:   (list | np.ndarray),
                 back:  Surface | Point | Line = None,
                 d1:    float = None,
                 d2:    float = None,
                 **kwargs)\
            -> None:
        """

        :param front:
        :param pos:
        :param back:
        :param d1:
        :param d2:
        :param kwargs: keyword arguments for :obj:`optrace.tracer.base_class.BaseClass`
        """
        self._geometry_lock = False

        self.front = front
        self.back = back
        self.d1 = d1
        self.d2 = d2

        if self.has_back():

            if d1 is None or d2 is None:
                raise ValueError("d1 and d2 need to be specified for a Element with a back surface")

            if d1 < 0 or d2 < 0:
                raise ValueError(f"Thicknesses d1, d2 need to be non-negative but are {d1=} and {d2=}.")

        self.move_to(pos)

        # double surface object -> check for collisions between back and front
        if self.has_back():
            collision, xc, yc = self.check_collision(self.front, self.back)
            if collision:
                raise RuntimeError(f"Collision of front and back surface of {self} at {xc=}, {yc=}.")

        super().__init__(**kwargs)
        self._geometry_lock = True

    def has_back(self) -> bool:
        """:return: if the Element has a BackSurface"""
        return self.back is not None

    def set_surface(self, surf: Surface) -> None:
        """
        Assign a new Surface to the Element.

        :param surf: Surface to assign
        """
        if self.has_back():
            raise RuntimeError("Replacing of Surfaces only supported for objects with one surface")

        self._geometry_lock = False
        pos = self.front.pos
        self.front = surf.copy()
        self.front.move_to(pos)
        self._geometry_lock = True

    def move_to(self, pos: (list | np.ndarray)) -> None:
        """
        Moves the Element in 3D space.

        :param pos: new 3D position of Element center (list or numpy array)
        """
        pc.check_type("pos", pos, list | np.ndarray)
        pos = np.asarray_chkfinite(pos, dtype=np.float64)

        if pos.shape[0] != 3:
            raise ValueError("pos needs to have 3 elements.")

        if not self.has_back():
            self.front.move_to(pos)
        else:
            self.front.move_to(pos - [0, 0, self.d1])
            self.back.move_to(pos + [0, 0, self.d2])

    @property
    def surface(self):
        """alias for :obj:`Element.FrontSurface` for a Element without BackSurface"""
        return self.front

    @property
    def pos(self) -> np.ndarray:
        """ position of the Element center """
        return self.front.pos + [0, 0, 0 if not self.has_back() else self.d1]

    @property
    def extent(self) -> tuple[float, float, float, float, float, float]:
        """ 3D extent of the Element"""
        if not self.has_back():
            return self.front.extent
        else:
            # choose extent that contains both surfaces
            ext = np.zeros(6, dtype=np.float64)
            exts = np.column_stack((self.front.extent, self.back.extent))
            ext[[0, 2, 4]] = np.min(exts, axis=1)[[0, 2, 4]]
            ext[[1, 3, 5]] = np.max(exts, axis=1)[[1, 3, 5]]
            return tuple(ext)

    def get_desc(self, fallback: str = None) -> str:
        """

        :param fallback: unused parameter
        :return:
        """
        stype1 = type(self.front).__name__

        if self.has_back():
            stype2 = type(self.back).__name__
            fallback = f"{stype1} + {stype2}, z = {self.pos[2]:.04g}"
        else:
            fallback = f"{stype1}, z = {self.pos[2]:.04g}"

        return super().get_desc(fallback)

    def get_cylinder_surface(self, nc: int = 100) \
            -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get a 3D surface representation of the Element cylinder for plotting.

        :param nc: number of surface edge points (int)
        :return: tuple of coordinate arrays X, Y, Z (2D numpy arrays)
        """
        X1, Y1, Z1 = self.front.get_edge(nc)
        X2, Y2, Z2 = self.back.get_edge(nc) if self.has_back() else (X1, Y1, Z1)

        return np.column_stack((X1, X2)), np.column_stack((Y1, Y2)), np.column_stack((Z1, Z2))

    def reverse(self) -> Element:

        if self.has_back():
            front = self.back.reverse()
            back = self.front.reverse()

            args = {}
            for key, val in self.__dict__.items():
                if not key.startswith("_") and not key in ["front", "back", "pos", "d1", "d2"]:
                    args[key] = copy.deepcopy(val)

            return type(self)(front, back, pos=self.pos, d1=self.d2, d2=self.d1, **args)
        else:
            el = self.copy()
            el.set_surface(self.front.reverse())
            return el

    @staticmethod
    def check_collision(front: Surface | Line | Point, back: Surface | Line | Point, res: int = 100)\
            -> tuple[bool, np.ndarray, np.ndarray]:
        """
        
        Check for collisions.
        A collision is defined as the front surface havin a higher z-value than the back surface,
        at a point where both surfaces are defined

        :param front:
        :param back:
        :param res:
        :return:
        """

        # we only compare when at least one object is a surface
        if not (isinstance(front, Surface) or isinstance(back, Surface)):
            raise TypeError(f"At least one object needs to be a Surface for collision detection")
        
        # check if point and surface hit. Basically if order of surface and point parameter is correct
        elif isinstance(front, Point) or isinstance(back, Point):
            rev, pt, surf = (False, front, back) if isinstance(front, Point) else (True, back, front)

            # check value at surface
            x, y = np.array([pt.pos[0]]), np.array([pt.pos[1]])
            z = surf.get_values(x, y)

            # check if hitting, surface needs to be defined at this point
            hit = (z < pt.pos[2]) if not rev else (z > pt.pos[2])
            hit = hit & surf.get_mask(x, y)
            where = np.where(hit)[0]
            return np.any(hit), x[where], y[where]

        # intersection of surface and line
        elif isinstance(front, Line) or isinstance(back, Line):
            rev, line, surf = (False, front, back) if isinstance(front, Line) else (True, back, front)

            # some line x, y values
            t = np.linspace(-line.r, line.r, 10*res)
            x = line.pos[0] + np.cos(line.angle)*t
            y = line.pos[1] + np.sin(line.angle)*t
            z = surf.get_values(x, y)

            # check if hitting and order correct
            hit = (z < line.pos[2]) if not rev else (z > line.pos[2])
            hit = hit & surf.get_mask(x, y)
            where = np.where(hit)[0]
            return np.any(hit), x[where], y[where]

        # extent of front and back
        xsf, xef, ysf, yef, zsf, zef = front.extent
        xsb, xeb, ysb, yeb, zsb, zeb = back.extent

        # no overlap of z extents -> no collision
        if zef < zsb:
            return False, np.array([]), np.array([])

        # get rectangular overlap area in xy-plane projection
        xs = max(xsf, xsb)
        xe = min(xef, xeb)
        ys = max(ysf, ysb)
        ye = min(yef, yeb)

        # no overlap in xy plane projection -> no collision
        if xs > xe or ys > ye:
            return False, np.array([]), np.array([])

        # grid for overlap area
        X, Y = np.mgrid[xs:xe:res*1j, ys:ye:res*1j]

        # sample surface mask
        x2, y2 = X.flatten(), Y.flatten()
        valid = front.get_mask(x2, y2) & back.get_mask(x2, y2)

        # sample valid surface values
        x2v, y2v = x2[valid], y2[valid]
        zfv = front.get_values(x2v, y2v)
        zbv = back.get_values(x2v, y2v)

        # check for collisions
        coll = zfv > zbv
        where = np.where(coll)[0]

        # return flag and collision samples
        return np.any(coll), x2v[where], y2v[where]

    def __setattr__(self, key: str, val: Any) -> None:
        """
        assigns the value of an attribute
        :param key: attribute name
        :param val: value to assign
        """
        # lock changing of geometry directly
        if "_geometry_lock" in self.__dict__ and self._geometry_lock:
            if key in ["d1", "d2", "front", "surface", "back"]:
                raise RuntimeError("Use Functions set_surface to reassign a new Surface or its thickness.")
            if key == "pos":
                raise RuntimeError("Use move_to(pos) to move the Object")

        match key:

            case "front":
                pc.check_type(key, val, (Surface | Point | Line | None) if self._allow_non_2D else Surface)
                super().__setattr__(key, val.copy())  # save internal copy
                return

            case "back":
                pc.check_type(key, val, (Surface | Point | Line | None) if self._allow_non_2D else (Surface | None))
                super().__setattr__(key, val.copy() if val is not None else None)  # save internal copy
                return

            case ("d1" | "d2"):
                pc.check_type(key, val, int | float | None)
                val = float(val) if val is not None else val

        super().__setattr__(key, val)
