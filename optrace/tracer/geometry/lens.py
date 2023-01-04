
from typing import Any  # Any type

import numpy as np  # calculations and ndarray type

from ..refraction_index import RefractionIndex  # material refraction index
from .surface import Surface  # Surface type
from .element import Element  # parent class
from ..misc import PropertyChecker as pc  # check types and values
from ..transfer_matrix_analysis import TMA  # paraxial analysis


class Lens(Element):

    abbr: str = "L"  #: object abbreviation
    _allow_non_2D: bool = False  # don't allow points or lines as surfaces
    
    is_ideal: bool = False

    def __init__(self,
                 front:   Surface,
                 back:    Surface,
                 n:       RefractionIndex,
                 pos:     (list | np.ndarray),
                 de:      float = 0,
                 d:       float = None,
                 d1:      float = None,
                 d2:      float = None,
                 n2:      RefractionIndex = None,
                 **kwargs)\
            -> None:
        """
        Creates a lens object using 2 surfaces and additional properties.
        Of the thickness parameters only one of d, de or (d1, d2) needs to be specified.

        A lens is an geometrical object with two refractive surfaces.
        A refractive index is specified for the material and one for the area behind the lens (optional)

        :param front: front surface (smaller z-position) (Surface object)
        :param back: back surface (higher z-position) (Surface object)
        :param de: thickness extension.
                    additional thickness between maximum height of front and minimal height of back, (float)
        :param d: thickness at the optical axis / lens center
        :param d1: thickness of front surface relative to surface center position (float)
        :param d2: thickness of back surface relative to surface center position (float)
        :param n: material refraction index (RefractionIndex object)
        :param n2: refraction index behind lens (positive z direction) (RefractionIndex object)
        :param pos: 3D position of lens center (list or numpy array)
        :param kwargs: additional keyword arguments for parent classes
        """

        self.n = n
        self.n2 = n2
        d1 = float(d1) if d1 is not None else d1
        d2 = float(d2) if d2 is not None else d2

        # only calculate if both are surfaces, otherwise there will be an exception anyway
        if isinstance(front, Surface) and isinstance(back, Surface):
            # calculate de from d and use de mode from now on
            if d is not None:
                de = d - front.dp - back.dn

                # de negative: z-extents of lens surfaces overlap,
                # possible in lenses with same curvature sign on both sides
                # (concave-convex or convex-concave)
                if de < 0:
                    # using the initial d10, d20 makes no sense here,
                    # just distribute d equally
                    d1 = d/2
                    d2 = d/2

            if de is not None and d1 is None and d2 is None:
                # de negative: z-extents of surfaces overlap,
                # possible in lenses with same curvature sign on both sides
                # (concave-convex or convex-concave)
                if de < 0:
                    # distribute de equally without taking d10, d20 into account
                    d1 = -de/2
                    d2 = -de/2
                else:
                    # distribute de equally
                    d1 = de / 2. + front.dp
                    d2 = de / 2. + back.dn

            elif d1 is None or d2 is None:
                raise ValueError("Both thicknesses d1, d2 need to be specified")

        super().__init__(front, pos, back, d1, d2, **kwargs)

        self._new_lock = True

    def tma(self, wl: float = 555., n0: RefractionIndex = None):
        """
        Matrix analysis object for the lens.
        Note that the lens does not know which medium n0 comes before it,
        if you want some value n0 != 1, you need to specify n0.
        The medium 2 after it is set by Lens.n2

        :param wl:
        :param n0:
        :return: transfer matrix analysis object"""
        return TMA([self], wl, n0)

    @property
    def d(self) -> float:
        """lens thickness at center"""
        return self.d1 + self.d2

    @property
    def de(self) -> float:
        """thickness extension"""
        return self.back.z_min - self.front.z_max

    def __setattr__(self, key: str, val: Any) -> None:
        """
        assigns the value of an attribute
        :param key: attribute name
        :param val: value to assign
        """
        if key == "n2":
            pc.check_type(key, val, RefractionIndex | None)

        if key == "n":
            pc.check_type(key, val, RefractionIndex)

        super().__setattr__(key, val)
