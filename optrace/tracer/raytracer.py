
"""
raytracer class:
Provides the functionality for raytracing, autofocussing and rendering of source and detector images.
"""


from typing import Any  # Any type
from threading import Thread  # threading
import sys  # redirect progressbar to stdout
from enum import IntEnum  # integer enum

import numpy as np  # calculations
import numexpr as ne  # faster calculations and core count
import scipy.optimize  # numerical optimization methods
from progressbar import progressbar, ProgressBar  # fancy progressbars

# needed for raytracing geometry and functionality
from .geometry import Filter, Aperture, Detector, Lens, RaySource, Surface, Marker, RectangularSurface, IdealLens
from .geometry.element import Element
from .geometry.group import Group

from .spectrum import LightSpectrum
from .refraction_index import RefractionIndex
from .ray_storage import RayStorage
from .r_image import RImage
from .base_class import BaseClass

from . import misc  # calculations
from .misc import PropertyChecker as pc  # check types and values

from .transfer_matrix_analysis import TMA


# sub-threads for all actions should not throw! this would make exception handling difficult

class Raytracer(Group):

    N_EPS: float = 1e-11
    """ numerical epsilon. Used for floating number comparisions in some places or adding small differences """

    T_TH: float = 1e-4
    """ threshold for the transmission Filter
    values below this are handled as absorbed
    needed to avoid ghost rays, meaning rays that have a non-zero, but negligible power"""

    MAX_RAYS: int = 10000000
    """ maximum number of rays. Limited by RAM usage """

    ITER_RAYS_STEP: int = 1000000
    """ number of rays per iteration in :obj:`raytracer.iterative_render` """

    class INFOS(IntEnum):
        ABSORB_MISSING = 0
        TIR = 1
        OUTLINE_INTERSECTION = 2
        ONLY_HIT_FRONT = 3
        ONLY_HIT_BACK = 4
        T_BELOW_TTH = 5
        SEQ_BROKEN = 6
    """enum for info messages in raytracing"""

    autofocus_methods: list[str, str, str] = ['Position Variance', 'Airy Disc Weighting',
                                              'Irradiance Variance', 'Irradiance Maximum', 'Image Sharpness']
    """available autofocus methods"""

    def __init__(self,
                 outline:        (list | np.ndarray),
                 n0:              RefractionIndex = None,
                 absorb_missing:  bool = True,
                 no_pol:          bool = False,
                 **kwargs)\
            -> None:
        """
        Initialize the raytracer

        :param outline: outline of raytracer space [x1, x2, y1, y2, z1, z2] (numpy 1D array or list)
        :param n0: refraction index of the raytracer enviroment (RefractionIndex object)
        :param absorb_missing: if rays missing a lens are absorbed at the lens (bool)
        :param no_pol:
        :param silent:
        :param threading:
        """

        self.outline = outline  #: geometrical raytracer outline
        self.absorb_missing = absorb_missing  #: absorb rays missing lenses
        self.no_pol = no_pol  #: polarization calculation flag
        self.n0 = n0  #: ambient refraction index

        self.rays = RayStorage()  #: traced rays
        self._msgs = np.array([])
        self._ignore_geometry_error = False
        self.geometry_error = False
        self._force_threads = None

        super().__init__(**kwargs)
        self._new_lock = True

    def __setattr__(self, key: str, val: Any) -> None:
        """
        assigns the value of an attribute
        :param key: attribute name
        :param val: value to assign
        """
        match key:

            case "outline":
                pc.check_type(key, val, list | np.ndarray)

                o = np.asarray_chkfinite(val, dtype=np.float64)
                if o.shape[0] != 6 or o[0] >= o[1] or o[2] >= o[3] or o[4] >= o[5]:
                    raise ValueError("Outline needs to be specified as [x1, x2, y1, y2, z1, z2] "
                                     "with x2 > x1, y2 > y1, z2 > z1.")

                super().__setattr__(key, o)
                return

            case ("no_pol" | "absorb_missing"):
                pc.check_type(key, val, bool)

            case "n0":
                pc.check_type(key, val, RefractionIndex | None)
                if val is None:
                    val = RefractionIndex("Constant", n=1.)

        super().__setattr__(key, val)

    @property
    def extent(self):
        """equals the outline of the raytracer"""
        # overwrites Group.extent, which knows no outline
        return tuple(self.outline)

    @property
    def pos(self):
        """center position of front xy-outline plane"""
        # overwrites Group.pos, which knows no outline
        return np.mean(self.outline[:2]), np.mean(self.outline[2:4]), self.outline[4]
    
    def clear(self) -> None:
        """clear geometry and rays"""
        # overwrites Group.clear, since we also clear the ray storage here
        super().clear()
        self.rays.__init__()

    def tma(self, wl: float = 555.) -> TMA:
        """

        :param wl:
        :return:
        """
        # overwrites Group.tma since n0 is known
        return super().tma(wl, self.n0)

    def property_snapshot(self) -> dict:
        """

        :return:
        """
        return dict(Rays=self.rays.crepr(),
                    Ambient=[tuple(self.outline), self.n0.crepr()],
                    TraceSettings=[self.no_pol, self.absorb_missing],
                    Lenses=[D.crepr() for D in self.lenses],
                    Filters=[D.crepr() for D in self.filters],
                    Apertures=[D.crepr() for D in self.apertures],
                    RaySources=[D.crepr() for D in self.ray_sources],
                    Markers=[D.crepr() for D in self.markers],
                    Detectors=[D.crepr() for D in self.detectors])

    def compare_property_snapshot(self, h1: dict, h2: dict) -> dict:
        """

        :param h1:
        :param h2:
        :return:
        """

        diff = {key: h1[key] != h2[key] for key in h1.keys()}

        # Refraction Index Regions could have changed by Lens.n2, the ambient did therefore change
        diff["Ambient"] = diff["Ambient"] or diff["Lenses"]
        diff["Any"] = any(val for val in diff.values())

        return diff

    def _set_messages(self, msgs: list[np.ndarray]) -> None:
        """

        :param msgs:
        """
        # join messages from all threads
        self._msgs = np.zeros_like(msgs[0], dtype=int)
        for msg in msgs:
            self._msgs += msg

    def _show_messages(self, N) -> None:
        """

        :param N:
        """
        # print messages
        for type_ in range(self._msgs.shape[0]):
            for surf in range(self._msgs.shape[1]):
                if count := self._msgs[type_, surf]:
                    match type_:
                        case self.INFOS.TIR:
                            self.print(f"{count} rays ({100*count/N:.3g}% of all rays) "
                                       f"with total inner reflection at surface {surf}, treating as absorbed.")

                        case self.INFOS.ABSORB_MISSING:
                            self.print(f"{count} rays ({100*count/N:.3g}% of all rays) "
                                       f"missing surface {surf}, "
                                       "set to absorbed because of parameter absorb_missing=True.")

                        case self.INFOS.OUTLINE_INTERSECTION:
                            self.print(f"{count} rays ({100*count/N:.3g}% of all rays) "
                                       f"hitting outline, set to absorbed.")

                        case self.INFOS.T_BELOW_TTH:
                            self.print(f"{count} rays ({100*count/N:.3g}% of all rays) "
                                       f"with transmittivity at filter surface {surf} below threshold of "
                                       f"{self.T_TH*100:.3g}%, setting to absorbed.")

                        case self.INFOS.ONLY_HIT_FRONT:
                            self.print(f"{count} rays ({100*count/N:.3g}% of all rays) "
                                       f"hitting lens front but missing back, setting to absorbed.")

                        case self.INFOS.ONLY_HIT_BACK:
                            self.print(f"{count} rays ({100*count/N:.3g}% of all rays) "
                                       f"missing lens front but hitting back, setting to absorbed.")

                        case self.INFOS.SEQ_BROKEN:
                            self.print(f"Hit point behind last starting point at surface {surf} for "
                                       f"{count} rays. This can be a result of surface collisions "
                                       "or incorrect sequentiality.")

    def trace(self, N: int) -> None:
        """
        Execute raytracing, saves all rays in the internal RaySource object

        :param N: number of rays (int)
        """

        if (N := int(N)) < 1:
            raise ValueError(f"Ray number N needs to be at least 1, but is {N}.")

        if N > self.MAX_RAYS:
            raise ValueError(f"Ray number exceeds maximum of {self.MAX_RAYS}")

        # absorb_missing = False is only possible, if all ambient refraction indices are the same
        if not self.absorb_missing:
            for Lens_ in self.lenses:
                if Lens_.n2 is not None and self.n0 != Lens_.n2:
                    self.print("Outside refraction index defined for at least one lens, setting absorb_missing"
                               " simulation parameter to True")
                    self.absorb_missing = True
                    break

        elements = self.__make_tracing_element_list()
        self.__geometry_checks(elements)
        if self.geometry_error and not self._ignore_geometry_error:
            self.print("ABORTED TRACING")
            return

        # reserve space for all tracing surface intersections, +1 for invisible aperture at the outline z-end
        # and +1 for the ray starting points
        nt = len(self.tracing_surfaces) + 2

        cores = ne.detect_number_of_cores()
        N_threads = cores if N/cores >= 10000 and self.threading else 1
        N_threads = self._force_threads if self._force_threads is not None else N_threads  # overwrite if forced

        # will hold info messages from each thread
        msgs = [np.zeros((len(self.INFOS), nt), dtype=int) for i in range(N_threads)]

        # start a progressbar
        bar = ProgressBar(fd=sys.stdout, prefix="Raytracing: ", max_value=nt).start()\
            if not self.silent else None

        # create rays from RaySources
        self.rays.init(self.ray_sources, N, nt, self.no_pol)

        def sub_trace(N_threads: int, N_t: int) -> None:

            p, s, pols, weights, wavelengths, ns = self.rays.make_thread_rays(N_threads, N_t)
            if not self.silent and N_t == 0:
                bar.update(1)

            msg = msgs[N_t]
            i = 0  # surface counter
            n1 = self.n0  # ambient medium before all elements
            ns[:, 0] = n1(wavelengths)  # assign refractive index

            for element in elements:

                p[:, i+1], pols[:, i+1], weights[:, i+1] = p[:, i], pols[:, i], weights[:, i]
                hw = weights[:, i] > 0

                if isinstance(element, Lens):

                    hw_front = hw.copy()
                    p[hw, i+1], hit_front = self.__find_surface_hit(element.front, p[hw, i], s[hw], i, msg)
                    hwh = misc.part_mask(hw, hit_front)  # rays having power and hitting lens front

                    # index after lens
                    n2 = element.n2 or self.n0
                    n2_l = n2(wavelengths)
                    
                    if not element.is_ideal:
                   
                        n1_l = n1(wavelengths)
                        n_l = element.n(wavelengths)
                        
                        self.__refraction(element.front, p, s, weights, n1_l, n_l, pols, hwh, i, msg)

                        # treat rays that go outside outline
                        hwnh = misc.part_mask(hw, ~hit_front)  # rays having power and not hitting lens front
                        self.__outline_intersection(p, s, weights, hwnh, i, msg)

                        i += 1
                        if not self.silent and N_t == 0:
                            bar.update(i+1)
                        p[:, i+1], pols[:, i+1], weights[:, i+1] = p[:, i], pols[:, i], weights[:, i]
                        ns[:, i], ns[:, i+1] = n_l, n2_l

                        hw = weights[:, i] > 0
                        p[hw, i+1], hit_back = self.__find_surface_hit(element.back, p[hw, i], s[hw], i, msg)
                        hwb = misc.part_mask(hw, hit_back)  # rays having power and hitting lens back
                        self.__refraction(element.back, p, s, weights, n_l, n2_l, pols, hwb, i, msg)

                        # since we don't model the behaviour of the lens side cylinder, we need to absorb all rays passing
                        # through the cylinder
                        self.__absorb_cylinder_rays(p, weights, hw_front, hw, hit_front, hit_back, i, msg)

                    else:
                        hit_back = hit_front
                        self.__refraction_ideal_lens(element.front, element.D, p, s, pols, hwh, i, msg)
                        ns[:, i+1] = n2_l

                    # absorb rays missing lens, overwrite p to last ray starting point (=end of lens front surface)
                    if self.absorb_missing and not np.all(hit_back):
                        miss_mask = misc.part_mask(hw, ~hit_back)
                        miss_count = np.count_nonzero(miss_mask)
                        weights[miss_mask, i+1] = 0
                        if not element.is_ideal:
                            p[miss_mask, i+1] = p[miss_mask, i]
                        msg[self.INFOS.ABSORB_MISSING, i] += miss_count

                    # # treat rays that go outside outline
                    hwnb = misc.part_mask(hw, ~hit_back)  # rays having power and not hitting lens back
                    self.__outline_intersection(p, s, weights, hwnb, i, msg)

                    # index after lens is index before lens in the next iteration
                    n1 = n2

                elif isinstance(element, Filter | Aperture):  # pragma: no branch
                    p[hw, i+1], hit = self.__find_surface_hit(element.surface, p[hw, i], s[hw], i, msg)
                    hwh = misc.part_mask(hw, hit)  # rays having power and hitting filter

                    if isinstance(element, Filter):
                        self.__filter(element, weights, wavelengths, hwh, i, msg)
                    else:
                        weights[hwh, i+1] = 0

                    # treat rays that go outside outline
                    hwnh = misc.part_mask(hw, ~hit)  # rays having power and not hitting filter
                    self.__outline_intersection(p, s, weights, hwnh, i, msg)

                    ns[:, i+1] = ns[:, i]

                i += 1
                if not self.silent and N_t == 0:
                    bar.update(i+1)

        if N_threads > 1:
            thread_list = [Thread(target=sub_trace, args=(N_threads, N_t)) for N_t in np.arange(N_threads)]

            [thread.start() for thread in thread_list]
            [thread.join() for thread in thread_list]
        else:
            sub_trace(1, 0)

        # lock Storage
        self.rays.lock()

        # show info messages from tracing
        if not self.silent:
            bar.finish()

        self._set_messages(msgs)
        self._show_messages(N)

    def __make_tracing_element_list(self) -> list[Lens | Filter | Aperture]:
        """
        Creates a sorted element list from filters and lenses.

        :return: list of sorted elements
        """
        # add a invisible (the filter is not in self.filters) to the outline area at +z
        # it absorbs all light at this surface
        o = self.outline
        end_filter = Aperture(RectangularSurface(dim=[o[1] - o[0], o[3] - o[2]]),
                              pos=[(o[1] + o[0])/2, (o[2] + o[3])/2, o[5]])

        # add filters, apertures and lenses into one list,
        elements = [el for el in self.elements if isinstance(el, Lens | Filter | Aperture)]

        # add end filter
        return elements + [end_filter]

    def __geometry_checks(self, elements: list[Lens | Filter | Aperture]) -> None:
        """
        Checks geometry in raytracer for errors.
        Markers and Detectors can be outside the tracing outline, Filters, Apertures, Lenses and Sources not

        :param elements: element list from __makeElementList()
        """

        def is_inside(e: tuple | list) -> bool:
            o = self.outline + self.N_EPS*np.array([-1, 1, -1, 1, -1, 1])  # add eps because of finite float precision
            return o[0] <= e[0] and e[1] <= o[1] and o[2] <= e[2] and e[3] <= o[3] and o[4] <= e[4] and e[5] <= o[5]

        if not self.ray_sources:
            self.print("RaySource Missing.")
            self.geometry_error = True
            return

        for i, el in enumerate(elements):
            if not is_inside(el.extent):
                self.print(f"Element{i} {el} with extent {el.extent} outside outline {self.outline}.")
                self.geometry_error = True
                return

            if i + 1 < len(elements):
                coll, xc, yc = Element.check_collision(el.front, elements[i + 1].front)

                if not coll and el.has_back():
                    coll, xc, yc = Element.check_collision(el.back, elements[i + 1].front)

                if coll:
                    self.print(f"Collision between Element{i} and Element{i+1} at x={xc}, y={yc}.")
                    self.geometry_error = True
                    return

        for rs in self.ray_sources:

            if not is_inside(rs.extent):
                self.print(f"RaySource {rs} with extent {rs.extent} outside outline {self.outline}.")
                self.geometry_error = True
                return

            if rs.pos[2] >= elements[0].extent[4]:
                coll, xc, yc = Element.check_collision(rs.surface, elements[0].front)
                if coll:
                    self.print(f"Collision between Element{i} and Element{i+1} at x={xc}, y={yc}."
                                "RaySource needs to come before all lenses.")
                    self.geometry_error = True
                    return

        self.geometry_error = False

    def __absorb_cylinder_rays(self,
                               p:           np.ndarray,
                               weights:     np.ndarray,
                               hw_front:    np.ndarray,
                               hw_back:     np.ndarray,
                               hit_front:   np.ndarray,
                               hit_back:    np.ndarray,
                               i:           int,
                               msg:         np.ndarray)\
            -> None:
        """
        Checks if rays intersect with the lens side cylinder. Since the cylinder is not used for raytracing,
        we need to absorb these rays before they intersect with the cylinder.

        :param p: position array prior surface hit (numpy 2D array, shape (N, 3))
        :param weights: ray weights (numpy 1D array)
        :param i: surface number (int)
        :param hw_front: boolean array of rays handled at front surface
        :param hw_back: boolean array of rays handled at back surface
        :param hit_front: boolean array of rays hitting front surface
        :param hit_back: boolean array of rays hitting back surface
        """

        if np.all(hit_front) and np.all(hit_back):
            return

        hit_front_hw = misc.part_mask(hw_front, hit_front)
        hit_back_hw = misc.part_mask(hw_back, hit_back)

        miss_front_hw = misc.part_mask(hw_front, ~hit_front)
        miss_back_hw = misc.part_mask(hw_back, ~hit_back)

        abnormal_front = hit_front_hw & miss_back_hw
        abnormal_back = hit_back_hw & miss_front_hw
        ab_count_front = np.count_nonzero(abnormal_front)
        ab_count_back = np.count_nonzero(abnormal_back)

        if ab_count_front:
            msg[self.INFOS.ONLY_HIT_FRONT, i] += ab_count_front
            p[abnormal_front, i+1] = p[abnormal_front, i]
            weights[abnormal_front, i] = 0
            weights[abnormal_front, i+1] = 0

        if ab_count_back:
            msg[self.INFOS.ONLY_HIT_BACK, i + 1] += ab_count_back
            p[abnormal_back, i+1] = p[abnormal_back, i]
            weights[abnormal_back, i] = 0
            weights[abnormal_back, i+1] = 0

    def __outline_intersection(self,
                               p:           np.ndarray,
                               s:           np.ndarray,
                               w:           np.ndarray,
                               hw:          np.ndarray,
                               i:           int,
                               msg:         np.ndarray)\
            -> None:
        """
        Checks if the rays intersect with the outline, finds intersections points.
        Ray weights of intersecting rays are set to zero at that point.

        :param p: position array prior surface hit (numpy 2D array, shape (N, 3))
        :param s: direction array (numpy 2D array, shape (N, 3))
        :param w: ray weights (numpy 1D array)
        :param hw:
        :param i:
        """

        if not np.any(hw):
            return

        # check if examine points pe are inside outlines
        xs, xe, ys, ye, zs, ze = self.outline
        x, y, z = p[hw, i+1, 0], p[hw, i+1, 1], p[hw, i+1, 2]
        inside = ne.evaluate("(xs < x) & (x < xe) & (ys < y) & (y < ye) & (zs < z) & (z < ze)")

        # number of rays going outside
        n_out = np.count_nonzero(~inside)

        if n_out:

            hwi = misc.part_mask(hw, ~inside)

            OT = np.tile(self.outline, (n_out, 1))  # tile outline for every outside ray
            P = p[hwi, i].repeat(2).reshape(n_out, 6)  # repeat each column once
            S = s[hwi].repeat(2).reshape(n_out, 6)  # repeat each column once

            # calculate t Parameter for every outline coordinate and ray
            # replace zeros with nan for division
            nan = np.nan
            T_arr = ne.evaluate("(OT-P)/where(S != 0, S, nan)")

            # exclude negative t
            T_arr[T_arr <= 0] = np.nan

            # first intersection is the smallest positive t
            t = np.nanmin(T_arr, axis=1)

            # assign intersection positions and weights for outside rays
            p[hwi, i+1] = p[hwi, i] + s[hwi]*t[:, np.newaxis]
            w[hwi, i + 1] = 0

            coll_count = np.count_nonzero(hwi)
            msg[self.INFOS.OUTLINE_INTERSECTION, i] += coll_count

    def __refraction_ideal_lens(self,
                                surface:  Surface,
                                D:        float,
                                p:        np.ndarray,
                                s:        np.ndarray,
                                pols:     np.ndarray,
                                hwh:      np.ndarray,
                                i:        int,
                                msg:      np.ndarray)\
            -> None:
        """
        """
        
        # position relative to center
        x = p[hwh, i+1, 0] - surface.pos[0]
        y = p[hwh, i+1, 1] - surface.pos[1]

        s0 = s.copy()
        
        # decomposition in x and y component of tan theta
        # tan theta_x0 = sx / sz
        # tan theta_y0 = sy / sz
        s_z_inv = 1 / s[hwh, 2]
        tan_x_0 = s[hwh, 0] * s_z_inv
        tan_y_0 = s[hwh, 1] * s_z_inv

        # tan components after refraction
        # see https://www.sciencedirect.com/science/article/pii/S0030402615000364
        tan_x_1 = tan_x_0 - x*D/1000
        tan_y_1 = tan_y_0 - y*D/1000

        # pythogras: (tan theta1)^2 = (tan theta_x1)^2 + (tan theta_y1)^2
        # calculate theta_1, and phi1 = arctan2(theta_y1, theta_x1)
        # these components can be used as spherical coordinates, which are calculated back into cartesian
        # coordinates for the new vectors components of the direction s
        # sx = sin(theta_1)*cos(phi1),   sy = sin(theta1)*sin(phi1),   sz= cos(theta1)
        # actually, this can all can be simplified to the equations below
        frac = ne.evaluate("1/sqrt(tan_x_1**2 + tan_y_1**2 + 1)")
        s[hwh, 0] = tan_x_1 * frac
        s[hwh, 1] = tan_y_1 * frac
        s[hwh, 2] = frac

        # calculate polarization
        n = np.tile([0., 0., 1.], (x.shape[0], 1))
        ns = s0[hwh, 2]
        self.__rotate_polarization(ns, s0, s[hwh], n, pols, i, hwh)

    def __refraction(self,
                     surface:      Surface,
                     p:            np.ndarray,
                     s:            np.ndarray,
                     weights:      np.ndarray,
                     n1:           np.ndarray,
                     n2:           np.ndarray,
                     pols:         np.ndarray,
                     hwh:          np.ndarray,
                     i:            int,
                     msg:          np.ndarray)\
            -> None:
        """
        Calculate directions and weights after refraction. rays with total inner reflection are treated as absorbed.
        The weights are calculated using the Fresnel formulas, assuming 50% p and s polarized light.

        :param surface: Surface object
        :param p: position array (numpy 2D array, shape (N, 3))
        :param s: direction array (numpy 2D array, shape (N, 3))
        :param weights:
        :param n1: refraction indices prior surface (numpy 1D array)
        :param n2: refraction indices after surface (numpy 1D array)
        :param pols:
        :param hwh:
        :param i: number of surface
        """

        if not np.any(hwh):
            return

        n = surface.get_normals(p[hwh, i + 1, 0], p[hwh, i + 1, 1])

        n1_h = n1[hwh]
        n2_h = n2[hwh]
        s_h = s[hwh]

        # vectorial Snell's Law as in "Optik - Physikalisch-technische Grundlagen und Anwendungen,
        # Heinz Haferkorn, Auflage 4, Wiley-VCH, 2008", p.43
        # but with s_ = s', n1 = n, n2 = n', alpha = ε, beta = ε'

        ns = misc.rdot(n, s_h)  # equals cos(alpha)
        N = n1_h/n2_h  # ray wise refraction index quotient

        # rays with TIR mean an imaginary W, we'll handle this later
        W = ne.evaluate("sqrt(1 - N**2 * (1-ns**2))")
        N_m, ns_m, W_m = N[:, np.newaxis], ns[:, np.newaxis], W[:, np.newaxis]
        s_ = ne.evaluate("s_h*N_m - n*(N_m*ns_m - W_m)")

        # rotate polarization vectors and calculate amplitude components in s and p plane
        A_ts, A_tp = self.__rotate_polarization(ns, s, s_, n, pols, i, hwh)

        # reflection coefficients for non-magnetic (µ_r=1) and non-absorbing materials (κ=0)
        # according to the Fresnel equations
        # see https://de.wikipedia.org/wiki/Fresnelsche_Formeln#Spezialfall:_gleiche_magnetische_Permeabilit.C3.A4t
        #####
        cos_alpha, cos_beta = ns, W
        ts = ne.evaluate("2 * n1_h*cos_alpha / (n1_h*cos_alpha + n2_h*cos_beta)")
        tp = ne.evaluate("2 * n1_h*cos_alpha / (n2_h*cos_alpha + n1_h*cos_beta)")
        T = ne.evaluate("n2_h*cos_beta / (n1_h*cos_alpha) * ((A_ts*ts)**2 + (A_tp*tp)**2)")

        # handle rays with total internal reflection
        TIR = ~np.isfinite(W)
        if np.any(TIR):
            T[TIR] = 0
            TIR_count = np.count_nonzero(TIR)
            msg[self.INFOS.TIR, i] += TIR_count

        weights[hwh, i+1] = weights[hwh, i]*T
        s[hwh] = s_

    def __rotate_polarization(self, 
                              ns:       np.ndarray, 
                              s:        np.ndarray, 
                              s_:       np.ndarray, 
                              n:        np.ndarray, 
                              pols:     np.ndarray, 
                              i:        int, 
                              hwh:      np.ndarray)\
            -> tuple[np.ndarray, np.ndarray]:
        """
        rotate polarization vectors and calculate amplitude components in s and p plane
        """
           
        # no polarization -> don't set anything and return equal amplitude components
        if self.no_pol:
            return 1/np.sqrt(2), 1/np.sqrt(2)

        # calculate s polarization vector
        # ns==1 means surface normal is parallel to ray direction, exclude these rays for now
        mask = np.abs(ns) < 1-1e-9
        mask2 = misc.part_mask(hwh, mask)
        sm = s[mask2]

        # reduce slicing by storing separately
        polsm = pols[mask2, i]
        s_m = s_[mask]

        # s polarization vector
        ps = misc.cross(sm, n[mask])
        ps = misc.normalize(ps)
        pp = misc.cross(ps, sm)

        # init arrays
        # default for A_ts, A_tp are 1/sqrt(2)
        A_ts = np.full_like(ns, 1/np.sqrt(2), dtype=np.float32)
        A_tp = np.full_like(ns, 1/np.sqrt(2), dtype=np.float32)

        # A_ts is component of pol in ps
        A_ts[mask] = misc.rdot(ps, polsm)
        A_tp[mask] = misc.rdot(pp, polsm)

        # new polarization vector after refraction
        pp_ = misc.cross(ps, s_m)  # ps and s_m are unity vectors and perpendicular, so we need no normalization
        A_tsm, A_tpm = A_ts[mask, np.newaxis], A_tp[mask, np.newaxis]
        pols[mask2, i+1] = ne.evaluate("ps*A_tsm + pp_*A_tpm")

        # return amplitude components
        return A_ts, A_tp

    def __filter(self,
                 filter_:     Filter,
                 weights:     np.ndarray,
                 wl:          np.ndarray,
                 hwh:         np.ndarray,
                 i:           int,
                 msg:         np.ndarray)\
            -> None:
        """
        Get ray weights from positions on Filter and wavelengths.

        :param filter_: Filter object
        :param weights:
        :param wl: wavelength array (numpy 1D array)
        :param hwh:
        :param i:
        """

        if not np.any(hwh):
            return

        T = filter_(wl[hwh])

        # set transmittivity below a threshold to zero
        # useful when filter function is e.g. a gauss function
        # needed to avoid ghost rays, meaning rays that have a non-zero, but negligible power
        mask = (0 < T) & (T < self.T_TH)

        # only do absorbing due to T_TH if spectrum is not constant or rectangular
        if np.any(mask) and filter_.spectrum.spectrum_type != "Constant":
            T[mask] = 0
            m_count = np.count_nonzero(mask)
            msg[self.INFOS.T_BELOW_TTH, i] += m_count

        weights[hwh, i+1] = weights[hwh, i]*T

    def __find_surface_hit(self, surface: Surface, p: np.ndarray, s: np.ndarray, i: int, msg: np.ndarray)\
            -> tuple[np.ndarray, np.ndarray]:
        """
        Find the position of hits on surface using the iterative regula falsi algorithm.

        :param surface: Surface object
        :param p: position array (numpy 2D array, shape (N, 3))
        :param s: direction array (numpy 2D array, shape (N, 3))
        :param i:
        :param msg:
        :return: positions of hit (shape as p), bool numpy 1D array if ray hits lens
        """

        p_hit, is_hit = surface.find_hit(p, s)

        # rays breaking sequentiality
        if np.any(p_hit[:, 2] < p[:, 2]):
            msg[self.INFOS.SEQ_BROKEN, i] += np.count_nonzero(p_hit[:, 2] < p[:, 2])

        return p_hit, is_hit

    def _hit_detector(self,
                      info:              str,
                      detector_index:    int = 0,
                      source_index:      int = None,
                      extent:            list | np.ndarray = None,
                      projection_method: str = "Equidistant") \
            -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, str, ProgressBar]:
        """

        :param info:
        :param detector_index:
        :param source_index:
        :param extent:
        :param projection_method:
        :return:
        """
        if not self.detectors:
            raise RuntimeError("Detector Missing")

        if not self.rays.N:
            raise RuntimeError("No rays traced.")

        if source_index is not None and (source_index > len(self.ray_sources) - 1 or source_index < 0):
            raise ValueError("Invalid source_index.")

        if detector_index > len(self.detectors) - 1 or detector_index < 0:
            raise ValueError("Invalid detector_index.")

        bar = ProgressBar(fd=sys.stdout, prefix=f"{info}: ", max_value=4).start()\
            if not self.silent else None

        # range for selected rays in RayStorage
        Ns, Ne = self.rays.B_list[source_index:source_index + 2] if source_index is not None else (0, self.rays.N)

        def threaded(Ns, Ne, list_, list_i, bar):

            # current rays for loop iteration, this rays are used in hit finding for the next section
            rs = np.zeros(self.rays.N, dtype=bool)  # gets updated at every iteration
            rs[Ns:Ne] = True  # use rays of selected source

            # check if ray positions are behind, in front or inside detector extent
            bh_zmin = self.rays.p_list[Ns:Ne, :, 2] >= self.detectors[detector_index].extent[4]
            bh_zmax = self.rays.p_list[Ns:Ne, :, 2] >= self.detectors[detector_index].extent[5]
            
            # conclude if rays start after detector or don't reach that far
            no_start = np.all(bh_zmin & bh_zmax, axis=1)
            no_reach = np.all(~bh_zmin & ~bh_zmax, axis=1)
            ignore = no_start | no_reach

            # index for ray position before intersection
            rs2 = np.argmax(bh_zmin, axis=1) - 1
            rs2[rs2 < 0] = 0

            # exclude rays that start behind detector or end before it
            rs2z = misc.part_mask(rs, ignore)
            rs[rs2z] = False
            rs2 = rs2[~ignore]

            if bar is not None and not list_i:
                bar.update(1)

            p, s, _, w, wl, _, _ = self.rays.rays_by_mask(rs, rs2, ret=[1, 1, 0, 1, 1, 0, 0], normalize=True)

            if bar is not None and not list_i:
                bar.update(2)

            # init ph (position of hit) and is_hit bool array
            ph = np.zeros_like(p, dtype=np.float64, order='F')
            ish = np.zeros_like(wl, dtype=bool)
            rs3 = np.ones_like(wl, dtype=bool)

            while np.any(rs):

                rs2 += 1  # increment next-section-indices
                val = rs2 < self.rays.Nt  # valid-array, indices are below section count
                # these rays have no intersection with the detector, since they end at the last outline surface

                if not np.all(val):
                    rs = misc.part_mask(rs, val)  # only use valid rays
                    rsi = misc.part_mask(rs3, ~val)  # current rays that are not valid
                    rs3 = misc.part_mask(rs3, val)  # current rays that are not valid
                    w[rsi] = 0  # set invalid to weight of zero
                    p, s, rs2 = p[val], s[val], rs2[val]  # only use valid rays

                msg = np.zeros((len(self.INFOS), 2), dtype=int)  # find_surface_hit needs a message array
                ph[rs3], ish[rs3] = self.__find_surface_hit(self.detectors[detector_index].surface, p, s,
                                                            detector_index, msg)

                self._set_messages([msg])  # assign messages
                self._show_messages(s.shape[0])  # handle messages

                p2z = self.rays.p_list[rs, rs2, 2]
                rs3 = ph[rs3, 2] > p2z
                rs = misc.part_mask(rs, rs3)

                if np.any(rs):
                    rs2 = rs2[rs3]
                    p, s, _, w[rs3], _, _, _ = self.rays.rays_by_mask(rs, rs2, ret=[1, 1, 0, 1, 0, 0, 0])

            list_[list_i] = [ph, w, wl, ish]

        N_th = ne.detect_number_of_cores() if self.threading and Ne-Ns > 100000 else 1
        N_th = self._force_threads if self._force_threads is not None else N_th  # overwrite option for debugging

        threads = []
        list_ = [[]] * N_th

        if N_th > 1:
            for i in np.arange(N_th):
                # ray start and end index for each thread
                step = (Ne - Ns) // N_th
                Nsi = Ns + i*step
                Nei = Nsi + step if i != N_th - 1 else Ne

                thread = Thread(target=threaded, args=(Nsi, Nei, list_, i, bar))
                threads.append(thread)

            [thread.start() for thread in threads]
            [thread.join() for thread in threads]
        else:
            threaded(Ns, Ne, list_, 0, bar)

        # make arrays from thread data
        # it would be more efficient to have them assigned to their destination in the thread
        # however we only work on rays reaching that far to the detector, a number that is not known beforehand
        ph = np.vstack([list__[0] for list__ in list_])
        w = np.concatenate([list__[1] for list__ in list_])
        wl = np.concatenate([list__[2] for list__ in list_])
        ish = np.concatenate([list__[3] for list__ in list_])

        hitw = ish & (w > 0)
        ph, w, wl = ph[hitw], w[hitw], wl[hitw]
        if bar is not None:
            bar.update(3)

        if self.detectors[detector_index].surface.is_flat():
            projection = None

        else:
            ph = self.detectors[detector_index].surface.sphere_projection(ph, projection_method)
            projection = projection_method

        # define the extent
        if isinstance(extent, list | np.ndarray):
            # only use rays inside extent area
            inside = (extent[0] <= ph[:, 0]) & (ph[:, 0] <= extent[1]) \
                    & (extent[2] <= ph[:, 1]) & (ph[:, 1] <= extent[3])

            extent_out = np.asarray_chkfinite(extent.copy())
            ph, w, wl = ph[inside], w[inside], wl[inside]

        elif extent is None:
            extent_out = self.detectors[detector_index].pos[:2].repeat(2)
            if np.any(hitw):
                extent_out[[0, 2]] = np.min(ph[:, :2], axis=0)
                extent_out[[1, 3]] = np.max(ph[:, :2], axis=0)

        else:
            raise ValueError(f"Invalid extent '{extent}'.")

        detector = self.detectors[detector_index]
        pname = f": {detector.desc}" if detector.desc != "" else ""
        desc = f"{Detector.abbr}{detector_index}{pname} at z = {detector.pos[2]:.5g} mm"

        return ph, w, wl, extent_out, desc, projection, bar

    def detector_image(self,
                       N:                 int,
                       detector_index:    int = 0,
                       source_index:      int = None,
                       extent:            list | np.ndarray = None,
                       limit:             float = None,
                       projection_method: str = "Equidistant",
                       **kwargs) -> RImage:
        """

        :param N:
        :param detector_index:
        :param source_index:
        :param extent:
        :param projection_method:
        :param kwargs:
        :return:
        """
        N = int(N)
        if N <= 0:
            raise ValueError(f"Pixel number N needs to be a positive int, but is {N}")

        p, w, wl, extent_out, desc, projection, bar\
            = self._hit_detector("Detector Image", detector_index, source_index, extent, projection_method)

        # init image and extent, these are the default values when no rays hit the detector
        img = RImage(desc=desc, extent=extent_out, projection=projection,
                     threading=self.threading, silent=self.silent, limit=limit)
        img.render(N, p, w, wl, **kwargs)
        if bar is not None:
            bar.finish()

        return img

    def detector_spectrum(self,
                          detector_index:   int = 0,
                          source_index:     int = None,
                          extent:           list | np.ndarray = None,
                          **kwargs) -> LightSpectrum:
        """

        :param detector_index:
        :param source_index:
        :param extent:
        :param kwargs:
        :return:
        """
        p, w, wl, extent, desc, _, bar\
            = self._hit_detector("Detector Spectrum", detector_index, source_index, extent)

        spec = LightSpectrum.render(wl, w, desc=f"Spectrum of {desc}", **kwargs)
        if bar is not None:
            bar.finish()

        return spec

    def iterative_render(self,
                         N_rays:            int,
                         N_px_D:            int | list = 400,
                         N_px_S:            int | list = 400,
                         detector_index:    int | list = 0,
                         projection_method: str | list = "Equidistant",
                         pos:               int | list = None,
                         silent:            bool = False,
                         no_sources:        bool = False,
                         extent:            list | np.ndarray = None)\
            -> tuple[list[RImage], list[RImage]]:
        """
        Image render with N_rays rays.
        First return value is a list of rendered sources, the second a list of rendered detector images.

        If pos is not provided, a single detector image is rendered at the position of the detector specified by detector_index.
        >> RT.iterative_render(N_rays=10000, detector_index=1) 
        
        If pos is provided as coordinate, the detector is moved beforehand.
        >> RT.iterative_render(N_rays=10000, pos=[0, 1, 0], detector_index=1) 
        
        If pos is a list, len(pos) detector images are rendered. All other parameters are either automatically repeated len(pos) times or can be specified as list with the same length as pos
        >> RT.iterative_render(N_rays=10000, pos=[[0, 1, 0], [2, 2, 10]], detector_index=1, N_px_D=[128, 256]) 

        :param N_rays:
        :param N_px_D:
        :param N_px_S:
        :param detector_index:
        :param pos:
        :param projection_method:
        :param silent:
        :param no_sources: don't render sources
        :param extent:
        :return:
        """
        
        if not self.ray_sources:
            raise RuntimeError("Ray Sources Missing.")
        
        if (N_rays := int(N_rays)) <= 0:
            raise ValueError(f"Ray number N_rays needs to be a positive int, but is {N_rays}.")

        if not self.detectors and pos is not None:
            raise RuntimeError("No detectors in geometry.")

        # if there are detectors
        if self.detectors:

            # use current detector position if pos is empty
            if pos is None:
                if isinstance(detector_index, list):
                    raise ValueError("detector_index list needs to have the same length as pos list")
                pos = [self.detectors[detector_index].pos[2]]

            elif not isinstance(pos, list):
                pos = [pos]

            if not isinstance(N_px_D, list):
                N_px_D = [N_px_D] * len(pos)

            elif len(N_px_D) != len(pos):
                raise ValueError("list of N_px_D needs to have the same length as pos list")

            if not isinstance(detector_index, list):
                detector_index = [detector_index] * len(pos)
            
            elif len(detector_index) != len(pos):
                raise ValueError("detector_index list needs to have the same length as pos list")

            if not isinstance(projection_method, list):
                projection_method = [projection_method] * len(pos)

            elif len(projection_method) != len(pos):
                raise ValueError("projection_method list needs to have the same length as pos list")

            if not isinstance(extent, list) or isinstance(extent[0], int | float):
                extent = [extent] * len(pos)

            elif len(extent) != len(pos):
                raise ValueError("extent list needs to have the same length as pos list")
            
            for npxdi in N_px_D:
                if not isinstance(npxdi, int) or npxdi <= 0:
                    raise ValueError(f"Pixel number N_px_D needs to be a positive int, but is {npxdi}.")
        
            extentc = extent.copy()

        if not isinstance(N_px_S, list):
            N_px_S = [N_px_S] * len(self.ray_sources)

        elif len(N_px_S) != len(self.ray_sources):
            raise ValueError("N_px_S list needs to have the same length as ray_sources")

        for npxsi in N_px_S:
            if not isinstance(npxsi, int) or npxsi <= 0:
                raise ValueError(f"Pixel number N_px_S needs to be a positive int, but is {npxsi}.")

        rays_step = self.ITER_RAYS_STEP
        iterations = max(1, int(N_rays / rays_step))

        # turn off messages for raytracing iterations
        silent_old = self.silent
        self.silent = True

        # image list
        DIm_res = []
        SIm_res = []

        iter_ = range(iterations)
        iterator = progressbar(iter_, prefix="Rendering: ", fd=sys.stdout) if not silent\
            else iter_

        # for all render iterations
        for i in iterator:

            # add remaining rays in last step
            if i == iterations - 1:
                rays_step += int(N_rays - iterations*rays_step)  # additional rays for last iteration

            self.trace(N=rays_step)

            if self.detectors:
                # for all detector positions
                for j in np.arange(len(pos)):
                    pos_new = np.concatenate((self.detectors[detector_index[j]].pos[:2], [pos[j]]))
                    self.detectors[detector_index[j]].move_to(pos_new)
                    Imi = self.detector_image(N=N_px_D[j], detector_index=detector_index[j], 
                                              extent=extentc[j], _dont_rescale=True, 
                                              projection_method=projection_method[j])
                    Imi._img *= rays_step / N_rays

                    # append image to list in first iteration, after that just add image content
                    if i == 0:
                        DIm_res.append(Imi)
                        extentc[j] = Imi.extent
                    else:
                        DIm_res[j]._img += Imi._img

            if not no_sources:
                for j, _ in enumerate(self.ray_sources):
                    Imi = self.source_image(N=N_px_S[j], source_index=j, _dont_rescale=True)
                    Imi._img *= rays_step / N_rays

                    # append image to list in first iteration, after that just add image content
                    if i == 0:
                        SIm_res.append(Imi)
                    else:
                        SIm_res[j]._img += Imi._img

        # rescale images to update Im.Im, we only added Im._Im each
        # force rescaling even if Im has the same size as _Im, since only _Im holds the sum image of all iterations
        [SIm.rescale(N_px_S[i], _force=True) for i, SIm in enumerate(SIm_res)]
        [DIm.rescale(N_px_D[i], _force=True) for i, DIm in enumerate(DIm_res)]

        # revert silent to its state
        self.silent = silent_old

        return SIm_res, DIm_res

    def _hit_source(self, info: str, source_index: int = 0)\
            -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, ProgressBar]:
        """
        Rendered Image of RaySource. rays were already traced.

        :param N: number of image pixels in each dimension (int)
        :param source_index:
        :return: XYZIL Image (XYZ channels + Irradiance + Illuminance in third dimension) (numpy 3D array)
        """
        if not self.ray_sources:
            raise RuntimeError("Ray Sources Missing.")

        if not self.rays.N:
            raise RuntimeError("No rays traced.")

        if source_index > len(self.ray_sources) - 1 or source_index < 0:
            raise ValueError("Invalid source_index.")

        bar = ProgressBar(fd=sys.stdout, prefix=f"{info}: ", max_value=2).start()\
            if not self.silent else None

        extent = self.ray_sources[source_index].extent[:4]
        p, _, _, w, wl = self.rays.source_sections(source_index)
        if bar is not None:
            bar.update(1)

        rs = self.ray_sources[source_index]
        pname = f": {rs.desc}" if rs.desc != "" else ""
        desc = f"{RaySource.abbr}{source_index}{pname} at z = {rs.pos[2]:.5g} mm"

        return p, w, wl, extent, desc, bar

    def source_spectrum(self, source_index: int = 0, **kwargs) -> LightSpectrum:
        """

        :param source_index:
        :param kwargs:
        :return:
        """
        p, w, wl, extent, desc, bar = self._hit_source("Source Spectrum", source_index)

        spec = LightSpectrum.render(wl, w, desc=f"Spectrum of {desc}", **kwargs)
        if bar is not None:
            bar.finish()

        return spec

    def source_image(self, N: int, source_index: int = 0, limit: float=None, **kwargs) -> RImage:
        """
        Rendered Image of RaySource. rays were already traced.

        :param N: number of image pixels in each dimension (int)
        :param source_index:
        :return: XYZIL Image (XYZ channels + Irradiance + Illuminance in third dimension) (numpy 3D array)
        """

        if (N := int(N)) <= 0:
            raise ValueError(f"Pixel number N needs to be a positive int, but is {N}.")

        p, w, wl, extent, desc, bar = self._hit_source("Source Image", source_index)

        img = RImage(desc=desc, extent=extent, projection=None, limit=limit,
                    threading=self.threading, silent=self.silent)
        img.render(N, p, w, wl, **kwargs)
        if bar is not None:
            bar.finish()

        return img

    def __autofocus_cost_func(self,
                              z_pos:   float,
                              mode:    str,
                              pa:      np.ndarray,
                              sb:      np.ndarray,
                              w:       np.ndarray,
                              r0:      float = 1e-3,
                              ret_pos: bool = False)\
            -> float | tuple[float, tuple[float, float, float]]:
        """

        no checks if rays are still inside the outline!

        :param z_pos:
        :param mode:
        :param pa:
        :param sb:
        :param w:
        :param r0:
        :param ret_pos: if a second parameter contaning a 3D coordinate tuple should be returned
        :return:
        """
        # hit position at virtual xy-plane detector
        ph = pa + sb*z_pos
        x, y = ph[:, 0], ph[:, 1]

        if mode == "Airy Disc Weighting" or ret_pos:
            xm = np.average(x, weights=w)
            ym = np.average(y, weights=w)

        match mode:

            # not enough rays, return cost of 1 and invalid position
            case _ if w.shape[0] <= 1:
                cost = 1
                xm, ym, z_pos = np.nan, np.nan, np.nan

            case "Airy Disc Weighting":
                expr = ne.evaluate("w * exp(-0.5*((x - xm)**2 + (y-ym)**2)/ (0.42*r0)**2)")
                cost =  1 - np.sum(expr)/np.sum(w)

            case "Position Variance":
                var_x = np.cov(x, aweights=w)
                var_y = np.cov(y, aweights=w)
                cost = np.sqrt(var_x + var_y)

            case ("Irradiance Variance" | "Irradiance Maximum" | "Image Sharpness"):
                # adapt pixel number, so we have less noise for few rays,
                # but can see more image details and information for a larger number of rays
                # N rays are distributed on a square area,
                # scale default number by (1 + sqrt(N))
                # this equals 75 pixels per image side for a low amount of rays
                # and around 260 pixel for 4 million rays
                N_px = 75*int(1 + np.sqrt(w.shape[0])/800)
                N_px = N_px if N_px % 2 else N_px + 1  # enforce odd number
            
                # render power image
                Im, x_bin, y_bin = np.histogram2d(x, y, weights=w, bins=[N_px, N_px])

                if mode == "Image Sharpness":
                    Im = np.abs(np.fft.fft2(Im))
                    Im = np.fft.fftshift(Im)

                    X, Y = np.mgrid[-1:1:N_px*1j, -1:1:N_px*1j]
                    cost = 1/np.mean((X**2 + Y**2)*Im)
                else:
                    Im = Im[Im > 0]  # exclude empty pixels
                    Ap = (x_bin[1] - x_bin[0]) * (y_bin[1] - y_bin[0])  # pixel area

                    if mode == "Irradiance Variance":
                        cost = np.sqrt(Ap/np.std(Im))  # sqrt for better value range

                    else: # mode == "Irradiance Maximum":
                        cost = 1/np.sqrt(np.max(Im)/Ap)  # sqrt for better data range
                
        if not ret_pos:
            return cost

        return cost, (xm, ym, z_pos)

    def autofocus(self,
                  method:           str,
                  z_start:          float,
                  source_index:     int = None,
                  N:                int = 100000,
                  return_cost:      bool = False)\
            -> tuple[scipy.optimize.OptimizeResult, dict]:
        """
        Find the focal point using different methods. z_start defines the starting point,
        the search range is the region between lenses or the outline.
        The influence of filters and apertures is neglected. Outline intersections of rays are ignored.

        :param method:
        :param z_start: starting position z (float)
        :param source_index:
        :param N: maximum number of rays to evaluate for modes "Position Variance" and "Airy Disc Weighting"
        :param return_cost: False, if costly calculation of cost function array
                can be skipped in mode "Position Variance".
                In other modes it is generated on the way anyway
        :return:
        """

        if not (self.outline[4] <= z_start <= self.outline[5]):
            raise ValueError(f"Starting position z_start={z_start} outside raytracer"
                             " z-outline range {RT.outline[4:]}.")

        if method not in self.autofocus_methods:
            raise ValueError(f"Invalid method '{method}', should be one of {self.autofocus_methods}.")

        if N < 1:
            raise ValueError(f"N needs to be a positive value, but is {N}")

        if not self.rays.N:
            raise RuntimeError("No rays traced.")

        if source_index is not None and source_index < 0:
            raise ValueError(f"source_index needs to be >= 0, but is {source_index}")

        if (source_index is not None and source_index > len(self.rays.N_list)) or len(self.rays.N_list) == 0:
            raise ValueError(f"source_index={source_index} larger than number of simulated sources "
                             f"({len(self.rays.N_list)}. "
                             "Either the source was not added or the new geometry was not traced.")

        # get search bounds
        ################################################################################################################

        # sort list in z order
        lenses = sorted(self.lenses, key=lambda Element: Element.pos[2])

        # default bounds, left starts at end of all ray sources, right ends at outline
        b0 = self.N_EPS + np.max([rs.extent[5] for rs in self.ray_sources])
        b1 = self.outline[5] - self.N_EPS
        n_ambient = self.n0(550)

        # find region between lenses
        for i, lens in enumerate(lenses):
            if z_start < lens.pos[2]:
                b1 = lens.extent[4]
                n_ambient_ = lenses[i-1].n2 if i > 0 and lenses[i-1].n2 is not None else self.n0
                n_ambient = n_ambient_(550)
                break
            b0 = lens.extent[5]

        # create bounds
        bounds = [b0, b1]

        # show filter warning
        for filter_ in (self.filters + self.apertures):
            if bounds[0] <= filter_.pos[2] <= bounds[1]:
                self.print("WARNING: The influence of the filters/apertures in the autofocus range will be ignored.")

        # start progress bar
        ################################################################################################################

        Nt = 1024  # cost function sampling points, divisible by 2, 4, 8, 16, 32 threads
        N_th = ne.detect_number_of_cores() if self.threading else 1
        N_th = self._force_threads if self._force_threads is not None else N_th  # overwrite option for debugging
        steps = int(Nt/64)
        steps += 2  # progressbas steps
        bar = ProgressBar(fd=sys.stdout, prefix="Finding Focus: ", max_value=steps).start()\
            if not self.silent else None

        # get rays and properties
        ################################################################################################################
        # select source
        Ns, Ne = self.rays.B_list[source_index:source_index + 2] if source_index is not None else (0, self.rays.N)

        # make bool array
        rays_pos = np.zeros(self.rays.N, dtype=bool)
        pos = np.zeros(self.rays.N, dtype=int)
        rays_pos[Ns:Ne] = True
        # find section index for each ray for focussing search range
        z = bounds[0] + self.N_EPS
        pos[Ns:Ne] = np.argmax(z < self.rays.p_list[rays_pos, :, 2], axis=1) - 1
        
        if not self.silent:
            bar.update(1)

        # exclude already absorbed rays
        rays_pos[pos == -1] = False

        # make sure we have enough rays
        N_act = np.count_nonzero(rays_pos)
        N_use = min(N, N_act) if method in ["Position Variance", "Airy Disc Weighting"] else N_act
        if N_use < 1000:  # throw error when no rays are present
            self.print(f"WARNING: Less than 1000 rays for autofocus ({N_use}).")

        # no rays are used, return placeholder variables
        if N_use == 0:
            return scipy.optimize.OptimizeResult(),\
                dict(pos=[np.nan, np.nan, np.nan], bounds=bounds, z=np.full(Nt, np.nan), cost=np.full(Nt, np.nan), N=N_use)

        # select random rays from all valid
        rp = np.where(rays_pos)[0]
        rp2 = np.random.choice(rp, N_use, replace=False)

        # assign positions of random rays
        rays_pos = np.zeros(self.rays.N, dtype=bool)
        rays_pos[rp2] = True
        pos = pos[rp2]

        # get Ray parts
        p, s, _, weights, _, _, _ = self.rays.rays_by_mask(rays_pos, pos, ret=[1, 1, 0, 1, 0, 0, 0])

        if not self.silent:
            bar.update(2)

        # find focus
        ################################################################################################################

        # parameters for function hit position = pa + sb * z_pos
        pa = p - s/s[:, 2, np.newaxis]*p[:, 2, np.newaxis]
        sb = s/s[:, 2, np.newaxis]

        if method == "Position Variance":
            res = scipy.optimize.minimize_scalar(self.__autofocus_cost_func,
                                                 args=("Position Variance", pa, sb, weights),
                                                 options={'maxiter': 500, 'xatol':1e-6},
                                                 bounds=bounds, method='Bounded')

        if method == "Airy Disc Weighting":
            # calculate size of airy disc for method Airy Disc Weighting
            s_z = np.nanquantile(s[:, 2], 0.95)
            sin_alpha = np.sin(np.arccos(s_z))  # neglect outliers

            # lambda / Fnum, default to 3µm when sin_alpha = 0
            r0 = 550e-6 / (sin_alpha * n_ambient) if sin_alpha != 0 else 3e-3
        else:
            r0 = 3e-3

        if method != "Position Variance" or return_cost:
            # sample smaller region around minimum with proper method
            r = np.linspace(bounds[0], bounds[1], Nt)
            vals = np.zeros_like(r)

            def threaded(N_th, N_is, Nt, *afargs):
                Ns = N_is*int(Nt/N_th)
                Ne = (N_is+1)*int(Nt/N_th) if N_is != N_th-1 else Nt
                div = int(Nt / (steps - 2) / N_th)

                for i, Ni in enumerate(np.arange(Ns, Ne)):
                    if i % div == div-1 and not self.silent and not N_is:
                        bar.update(2+int(i/div + 1))
                    vals[Ni] = self.__autofocus_cost_func(r[Ni], *afargs)

            if N_th > 1:
                threads = [Thread(target=threaded, args=(N_th, N_is, Nt, method, pa, sb, weights, r0))
                           for N_is in range(N_th)]
                [th.start() for th in threads]
                [th.join() for th in threads]
            else:
                threaded(1, 0, Nt, method, pa, sb, weights, r0)
        else:
            r = None
            vals = None

        # find minimum for other methods, since they are susceptible to local minima
        if method != "Position Variance":
            # # start search at minimum of sampled data
            pos = np.argmin(vals)
            cost_func2 = lambda z, method: self.__autofocus_cost_func(z[0], method, pa, sb, weights, r0)
            res = scipy.optimize.minimize(cost_func2, r[pos], args=method, tol=None, callback=None,
                                          options={'maxiter': 300}, bounds=[bounds])
            res.x = res.x[0]

        if not self.silent:
            bar.finish()

        # print warning if result is near bounds
        ################################################################################################################

        rrl = (res.x - bounds[0]) < 10*(bounds[1] - bounds[0]) / Nt
        rrr = (bounds[1] - res.x) < 10*(bounds[1] - bounds[0]) / Nt
        if rrl or rrr:
            self.print("WARNING: Found minimum near search bounds, "
                       "this can mean the focus is outside of the search range.")

        pos = self.__autofocus_cost_func(res.x, method, pa, sb, weights, r0, ret_pos=True)[1]
        return res, dict(pos=pos, bounds=bounds, z=r, cost=vals, N=N_use)
