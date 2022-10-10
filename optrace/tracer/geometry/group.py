import numpy as np  # calculations

from . import Filter, Aperture, Detector, Lens, RaySource, Surface, Marker
from ..base_class import BaseClass

from ..transfer_matrix_analysis import TMA
from ..refraction_index import RefractionIndex
from ..misc import PropertyChecker as pc



# forward declaration so we can use the Group type inside the Group class
class Group:
    pass


class Group(BaseClass):

    def __init__(self, elements = None, **kwargs):

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
    def elements(self):
        """all elements sorted in z-order"""
        return sorted([*self.lenses, *self.apertures, *self.filters, *self.ray_sources,
                      *self.detectors, *self.markers], key=lambda el: el.pos[2])
    
    @property
    def _elements(self):
        """all elements unsorted"""
        return [*self.lenses, *self.apertures, *self.filters, *self.ray_sources,
                      *self.detectors, *self.markers]
    @property
    def pos(self):
        """position of first element (after sorting in z-order)"""
        return self.elements[0].pos

    @property
    def extent(self):
        """extent of all elements"""
        els = self._elements

        ext = np.array((len(els), 6))
        for i, el in enumerate(els):
            ext[i] = el.extent

        max_ext = np.max(ext, axis=1)
        min_ext = np.min(ext, axis=1)

        return min_ext[0], max_ext[1], min_ext[2], max_ext[3], min_ext[4], max_ext[5]

    def move_to(self, pos):
        """move all elements such that pos is the position of the first element"""
        pc.check_type("pos", pos, list | np.ndarray)
        pos = np.asarray_chkfinite(pos, dtype=np.float64)

        if pos.shape[0] != 3:
            raise ValueError("pos needs to have 3 elements.")

        pos0 = self.pos

        for el in self._elements:
            el.move_to(el.pos - (pos0 - pos))

    def tma(self, wl: float = 555., n0: RefractionIndex = None) -> TMA:
        """

        :param wl:
        :return:
        """
        return TMA(self.lenses, wl=wl, n0=n0)

    def add(self, el: Lens | Aperture | Filter | RaySource | Detector | Marker | list | Group) -> None:
        """
        Add an element or a list of elements to the raytracer geometry.

        :param el: Element to add to raytracer
        """

        if not isinstance(el, list) and self.has(el):
            self.print("Element already included in geometry. Make a copy to include it another time.")
            return

        match el:
            case Aperture():    self.apertures.append(el)
            case Filter():      self.filters.append(el)
            case RaySource():   self.ray_sources.append(el)
            case Detector():    self.detectors.append(el)
            case Marker():      self.markers.append(el)
            case Lens():        self.lenses.append(el)

            case Group():
                for eli in el.elements:
                    self.add(eli)

            case list():
                for eli in el:
                    self.add(eli)
            case _:
                raise TypeError("Invalid element type.")

    def remove(self, el: Lens | Aperture | Filter | RaySource | Detector | Marker | list) -> bool:
        """
        Remove the element specified by its id from raytracing geometry.
        Returns True if element(s) have been found and removes, False otherwise.

        :param el:
        :return: if the element was found and removed
        """
        success = False

        if isinstance(el, list):
            for eli in el.copy():
                success = self.remove(eli) or success
        else:
            for list_ in [self.lenses, self.apertures, self.detectors,
                          self.filters, self.ray_sources, self.markers]:
                for lel in list_.copy():
                    if lel is el:
                        list_.remove(lel)
                        success = True

        return success

    def has(self, el) -> bool:
        """
        checks if element is included in geometry

        :param el:
        :return:
        """
        for eli in self._elements:
            if eli is el:
                return True

        return False

    def clear(self) -> None:
        """clear geometry"""
        self.lenses[:] = []
        self.apertures[:] = []
        self.filters[:] = []
        self.detectors[:] = []
        self.ray_sources[:] = []
        self.markers[:] = []
