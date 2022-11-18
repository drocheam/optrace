
from typing import Any

import numpy as np  # calculations

from . import Filter, Aperture, Detector, Lens, RaySource, Surface, Marker, IdealLens, Element
from ..base_class import BaseClass

from ..transfer_matrix_analysis import TMA
from ..refraction_index import RefractionIndex
from ..misc import PropertyChecker as pc


"""
A Group contains multiple elements of types Lens, Marker, RaySource, Filter, Aperture, Detector.
This class provides the functionality for moving, rotating and flipping its whole geometry.
A well as naturally including, removing elements and clearing is whole content.
"""


class Group:
    pass


class Group(BaseClass):

    def __init__(self, elements: list[Element] = None, **kwargs) -> None:
        """
        Create a group by including elements from the 'elements' parameter.
        Without this parameter an empty group is created.

        :param elements: list of elements to add. Can be empty
        :param kwargs: additional keyword arguments for the BaseClass class
        """

        self.lenses = []  #: lenses in raytracing geometry
        self.apertures = []  #: apertures in raytracing geometry
        self.filters = []  #: filters in raytracing geometry
        self.detectors = []  #: detectors in raytracing geometry
        self.ray_sources = []  #: ray sources in raytracing geometry
        self.markers = []  #: markers in raytracing geometry

        super().__init__(**kwargs)

        if elements is not None:
            self.add(elements)

    @property
    def elements(self) -> list[Element]:
        """all included elements sorted in z-order"""
        return sorted([*self.lenses, *self.apertures, *self.filters, *self.ray_sources,
                      *self.detectors, *self.markers], key=lambda el: el.pos[2])
    
    @property
    def _elements(self) -> list[Element]:
        """all included elements unsorted"""
        return [*self.lenses, *self.apertures, *self.filters, *self.ray_sources,
                *self.detectors, *self.markers]

    @property
    def pos(self) -> np.ndarray:
        """position of first element (sorting in z-order)"""
        return self.elements[0].pos if len(self._elements) else [0, 0, 0]

    @property
    def tracing_surfaces(self) -> list[Surface]:
        """
        List of all tracing surfaces (lenses, apertures, filters).
        Sorted by center z-position.

        :return: sorted list of tracing surfaces
        """
        # add front and back surfaces of Lens, Filters, Apertures, but only if surfaces are not None
        surfs = []
        for el in self.elements:
            if isinstance(el, Lens | Filter | Aperture):
                surfs.append(el.front)
                if el.has_back() and not isinstance(el, IdealLens):
                    surfs.append(el.back)

        return surfs

    @property
    def extent(self) -> tuple[float, float, float, float, float, float]:
        """
        Extent of all elements. Equivalent to the smallest cuboid that encompasses all group objects inside.

        :return: tuple of the form (x0, x1, y0, y1, z0, z1)
        """
        els = self._elements

        # no elements in group
        if not len(els):
            return 0, 0, 0, 0, 0, 0

        ext = np.zeros((len(els), 6))
        for i, el in enumerate(els):
            ext[i] = np.array(el.extent)

        max_ext = np.max(ext, axis=0)
        min_ext = np.min(ext, axis=0)

        return min_ext[0], max_ext[1], min_ext[2], max_ext[3], min_ext[4], max_ext[5]

    def move_to(self, pos: list | np.ndarray) -> None:
        """
        Moves all elements such that pos is the new position of the first element.
        Relative distances between objects are mantained

        :param pos: new position (3 element list/array)
        """

        pc.check_type("pos", pos, list | np.ndarray)
        pos = np.asarray_chkfinite(pos, dtype=np.float64)

        if pos.shape[0] != 3:
            raise ValueError("pos needs to have exactly 3 elements.")

        pos0 = self.pos

        for el in self._elements:
            el.move_to(el.pos - (pos0 - pos))

    def tma(self, wl: float = 555., n0: RefractionIndex = None) -> TMA:
        """
        Creates an ray transfer matrix analysis object from the group geometry.
        Note that this is a snapshot and does not get updated.

        :param wl: wavelength for matrix analysis
        :param n0: ambient refraction index before first lens
        :return: ray transfer matrix analysis object
        """
        return TMA(self.lenses, wl=wl, n0=n0)

    def flip(self, y0: float = 0, z0: float = None) -> None:
        """
        Flip the group (= rotate by 180 deg) around an axis parallel to the x-axis and through the point (y0, z0).
        By default y0 = 0 and z0 is the center of the z-extent of the group

        :param y0: y-position of axis
        :param z0: z-position of axis
        """
        if not len(self._elements):
            return

        # elements, z-positions and media
        els = self.elements
        ns = [L.n2 for L in els if isinstance(L, Lens)]
        z0 = np.mean(self.extent[4:]) if z0 is None else z0

        for i, el in enumerate(els):
            el.flip()
            el.move_to([el.pos[0], y0 - (el.pos[1] - y0), z0 - (el.pos[2] - z0)])

        if len(ns):
            # with the lenses the order of ambient materials n2 is also reversed
            # n2 last element is lost, because it would be the medium before the group

            if ns[-1] is not None:
                self.print("WARNING: ambient index n2 of last lens is lost.")
            
            # reverse medium order, last medium is undefined
            ns.reverse()
            ns += [None]

            # assign
            for n2, L in zip(ns[1:], self.lenses):
                L.n2 = n2

    def rotate(self, angle: float, x0: float = 0, y0: float = 0) -> None:
        """
        Rotate the group around an axis at (x0, y0) with direction (0, 0, 1)
        The rotation angle is the angle in the xy-plane with counter-clockwise direction

        :param angle: rotation angle in degrees
        :param x0: x-position of axis
        :param y0: y-position of axis
         """
        if not len(self._elements):
            return

        # angle in radians
        ang = np.deg2rad(angle)

        # rotate every element around itself and additionally its center around the axis of the first element
        for el in self.elements:
            xr = el.pos[0] - x0
            yr = el.pos[1] - y0
            posr = [x0 + xr*np.cos(ang) - yr*np.sin(ang), y0 + xr*np.sin(ang) + yr*np.cos(ang), el.pos[2]]
            el.rotate(angle)
            el.move_to(posr)

    def add(self, el: Element | list[Element] | Group) -> None:
        """
        Add an element, list or group to the geometry.

        :param el: object to add
        """

        if not isinstance(el, list) and self.has(el):
            self.print("Element already included in geometry. Make a copy to include it another time.")
            return

        match el:
            case Aperture():                  self.apertures.append(el)
            case Filter():                    self.filters.append(el)
            case RaySource():                 self.ray_sources.append(el)
            case Detector():                  self.detectors.append(el)
            case Marker():                    self.markers.append(el)
            case Lens() | IdealLens():        self.lenses.append(el)

            case Group():
                for eli in el.elements:
                    self.add(eli)

            case list():
                for eli in el:
                    self.add(eli)
            case _:
                raise TypeError("Unsupported element type {type(el).__name__}.")

    def remove(self, el: Element | list[Element] | Group) -> bool:
        """
        Remove the element specified by its id from raytracing geometry.
        Returns True if element(s) have been found and removes, False otherwise.

        :param el:
        :return: if anything was found and removed
        """
        success = False

        if isinstance(el, list):
            for eli in el.copy():
                success = self.remove(eli) or success

        elif isinstance(el, Group):
            for eli in el._elements.copy():
                success = self.remove(eli) or success

        else:
            for list_ in [self.lenses, self.apertures, self.detectors,
                          self.filters, self.ray_sources, self.markers]:
                for lel in list_.copy():
                    if lel is el:
                        list_.remove(lel)
                        success = True

        return success

    def has(self, el: Element) -> bool:
        """
        checks if element is included in geometry.
        Don't use this function with lists or Groups.

        :param el:
        :return: if object is inside group geometry
        """
        return any(eli is el for eli in self._elements)

    def clear(self) -> None:
        """clear geometry, remove all objects from group"""
        for list_ in [self.lenses, self.apertures, self.filters, self.detectors,\
                      self.ray_sources, self.markers]:
            list_[:] = []

