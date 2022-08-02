
"""
Raytracer class:
Provides the functionality for raytracing, autofocussing and rendering of source and detector images.
"""

import numpy as np  # calculations
import numexpr as ne  # faster calculations and core count
import scipy.optimize  # numerical optimization methods

from threading import Thread  # threading
from progressbar import progressbar, ProgressBar  # fancy progressbars
import sys  # redirect progressbar to stdout

# needed for raytracing geometry and functionality
from optrace.tracer.geometry import Filter, Aperture, Detector, Lens, RaySource, Surface

from optrace.tracer.spectrum import LightSpectrum
from optrace.tracer.refraction_index import RefractionIndex
from optrace.tracer.ray_storage import RayStorage
from optrace.tracer.r_image import RImage
from optrace.tracer.base_class import BaseClass

import optrace.tracer.misc as misc  # calculations
from optrace.tracer.misc import PropertyChecker as pc  # check types and values

from enum import IntEnum  # integer enum
import warnings  # print warnings


class Raytracer(BaseClass):

    EPS: float = 1e-12
    """ raytracer epsilon, used as precision in some places """

    T_TH: float = 1e-4
    """ threshold for the transmission Filter
    values below this are handled as absorbed
    needed to avoid ghost rays, meaning rays that have a non-zero, but negligible power"""

    MAX_RAYS: int = 10000000
    """ maximum number of rays. Limited by RAM usage """

    ITER_RAYS_STEP: int = 1000000
    """ number of rays per iteration in :obj:`Raytracer.iterativeDetectorImage` """

    # enum for info messages while tracing
    class _infos(IntEnum):
        absorb_missing = 0
        TIR = 1
        outline_intersection = 2
        only_hit_front = 3
        only_hit_back = 4
        T_below_TTH = 5

    autofocus_modes = ['Position Variance', 'Airy Disc Weighting', 'Irradiance Variance']
    
    def __init__(self,
                 outline:        (list | np.ndarray),
                 n0:             RefractionIndex = None,
                 absorb_missing:  bool = True,
                 no_pol:         bool = False,
                 **kwargs)\
            -> None:
        """
        Initialize the Raytracer

        :param outline: outline of raytracer space [x1, x2, y1, y2, z1, z2] (numpy 1D array or list)
        :param n0: refraction index of the raytracer enviroment (RefractionIndex object)
        :param absorb_missing: if rays missing a lens are absorbed at the lens (bool)
        :param no_pol:
        :param silent:
        :param threading:
        """

        self.outline = outline
        self.absorb_missing = absorb_missing
        self.no_pol = no_pol
        self.n0 = n0  # defaults to Air

        self.LensList = []
        self.ApertureList = []
        self.FilterList = []
        self.DetectorList = []
        self.RaySourceList = []
        self.Rays = RayStorage()
        self._msgs = np.array([])

        super().__init__(**kwargs)
        self._new_lock = True

    def __setattr__(self, key, val):

        match key:

            case "outline":
                pc.checkType(key, val, list | np.ndarray)

                o = np.array(val, dtype=np.float64)
                if o.shape[0] != 6 or o[0] >= o[1] or o[2] >= o[3] or o[4] >= o[5]:
                    raise ValueError("Outline needs to be specified as [x1, x2, y1, y2, z1, z2] "
                                     "with x2 > x1, y2 > y1, z2 > z1.")
                
                super().__setattr__(key, o)
                return 

            case ("no_pol" | "absorb_missing"):
                pc.checkType(key, val, bool)
            
            case "n0":
                pc.checkType(key, val, RefractionIndex | None)
                if val is None:
                    val = RefractionIndex("Constant", n=1.)

        super().__setattr__(key, val)

    def add(self, el: Lens | Aperture | Filter | RaySource | Detector | list) -> None:
        """
        Add an element or a list of elements to the Raytracer geometry.

        :param el: Element to add to Raytracer 
        """
        match el:
            case Lens():        self.LensList.append(el)
            case Aperture():    self.ApertureList.append(el)
            case Filter():      self.FilterList.append(el)
            case RaySource():   self.RaySourceList.append(el)
            case Detector():    self.DetectorList.append(el)

            case list():
                for eli in el:
                    self.add(eli)
            case _:             
                raise TypeError("Invalid element type.")

    def remove(self, ident: int | Lens | Aperture | Filter | RaySource | Detector) -> bool:
        """
        Remove the element specified by its id from raytracing geometry.
        Returns True if element(s) have been found and removes, False otherwise.

        :param ident: identifier of element from :obj:`Raytracer.add` or from :obj:`id`
        :return: if the element was found and removed
        """

        success = False
        ident_ = id(ident) if not isinstance(ident, int) else ident

        [(self.LensList.remove(El),      success := True) for El in self.LensList if id(El) == ident_]
        [(self.ApertureList.remove(El),  success := True) for El in self.ApertureList if id(El) == ident_]
        [(self.FilterList.remove(El),    success := True) for El in self.FilterList if id(El) == ident_]
        [(self.DetectorList.remove(El),  success := True) for El in self.DetectorList if id(El) == ident_]
        [(self.RaySourceList.remove(El), success := True) for El in self.RaySourceList if id(El) == ident_]

        return success

    def property_snapshot(self) -> dict:

        return dict(Rays=self.Rays.crepr(),
                    Ambient=[tuple(self.outline), self.n0.crepr()],
                    TraceSettings=[self.no_pol, self.EPS, self.T_TH, self.absorb_missing],
                    Lenses=[D.crepr() for D in self.LensList],
                    Filters=[D.crepr() for D in self.FilterList],
                    Apertures=[D.crepr() for D in self.ApertureList],
                    RaySources=[D.crepr() for D in self.RaySourceList],
                    Detectors=[D.crepr() for D in self.DetectorList])

    def compare_property_snapshot(self, h1: dict, h2: dict) -> dict:
       
        diff = dict()
        any_ = False

        for d in h1:
            diff[d] = h1[d] != h2[d]
            any_ = any_ or diff[d]

        # Refraction Index Regions could have changed by Lens.n2, the ambient did therefore change
        diff["Ambient"] = diff["Ambient"] or diff["Lenses"]
        diff["Any"] = any_

        return diff
  
    def _set_messages(self, msgs: list[np.ndarray]) -> None:

        # join messages from all threads
        msga = np.zeros_like(msgs[0], dtype=int)
        for msg in msgs:
            msga += msg
        self._msgs = msga

    def _show_messages(self, N) -> None:

        # print messages
        for type_ in range(self._msgs.shape[0]):
            for surf in range(self._msgs.shape[1]):
                if count := self._msgs[type_, surf]:
                    match type_:
                        case self._infos.TIR:
                            self.print(f"{count} rays ({100*count/N:.3g}% of all rays) "
                                       f"with total inner reflection at surface {surf}, treating as absorbed.")

                        case self._infos.absorb_missing:
                            self.print(f"{count} rays ({100*count/N:.3g}% of all rays) "
                                       f"missing surface {surf},"
                                       "set to absorbed because of parameter AbsorbMissing=True.")

                        case self._infos.outline_intersection:
                            self.print(f"{count} rays ({100*count/N:.3g}% of all rays) "
                                       f"hitting outline, set to absorbed.")

                        case self._infos.T_below_TTH:
                            self.print(f"{count} rays ({100*count/N:.3g}% of all rays) "
                                       f"with transmittivity at filter surface {surf} below threshold of "
                                       f"{self.T_TH*100:.3g}%, setting to absorbed.")

                        case self._infos.only_hit_front:
                            self.print(f"{count} rays ({100*count/N:.3g}% of all rays) "
                                       f"hitting lens front but missing back, setting to absorbed.")

                        case self._infos.only_hit_back:
                            self.print(f"{count} rays ({100*count/N:.3g}% of all rays) "
                                       f"missing lens front but hitting back, setting to absorbed.")

    def trace(self, N: int) -> None:
        """
        Execute raytracing, saves all Rays in the internal RaySource object

        :param N: number of rays (int)
        """

        if not self.RaySourceList:
            raise RuntimeError("RaySource Missing")

        if (N := int(N)) < 1000:
            raise ValueError(f"Ray number N needs to be at least 1000, but is {N}.")

        if N > self.MAX_RAYS:
            raise ValueError(f"Ray number exceeds maximum of {self.MAX_RAYS}")

        # AbsorbMissing = False is only possible, if all ambient refraction indices are the same
        if not self.absorb_missing:
            for Lens_ in self.LensList:
                if Lens_.n2 is not None and self.n0 != Lens_.n2:
                    warnings.warn("Outside refraction index defined for at least one lens, setting absorb_missing"
                                  " simulation parameter to True", RuntimeWarning)
                    self.absorb_missing = True
                    break

        Elements = self._make_element_list()
        self.__geometry_checks(Elements)

        # reserve space for all surface points, +1 for invisible aperture at the outline z-end
        # and +1 for the ray starting points
        nt = 2*len(self.LensList) + len(self.FilterList) + len(self.ApertureList) + 1 + 1
  
        cores = ne.detect_number_of_cores()
        N_threads = cores if N/cores >= 10000 and self.threading else 1

        # will hold info messages from each thread
        msgs = [np.zeros((len(self._infos), nt), dtype=int) for i in range(N_threads)]

        # start a progressbar
        bar = ProgressBar(fd=sys.stdout, prefix="Raytracing: ", max_value=nt, redirect_stderr=True).start()\
            if not self.silent else None

        # create Rays from RaySources
        self.Rays.init(self.RaySourceList, N, nt, no_pol=self.no_pol)
       
        def sub_trace(N_threads: int, N_t: int) -> None:

            p, s0, pols, weights, wavelengths = self.Rays.make_thread_rays(N_threads, N_t)
            if not self.silent and N_t == 0:
                bar.update(1)

            s = s0.copy()

            n0_l = self.n0(wavelengths) 
            msg = msgs[N_t]

            i = 0
            for Element in Elements:
    
                p[:, i+1], pols[:, i+1], weights[:, i+1] = p[:, i], pols[:, i], weights[:, i]

                # hw: has weight
                hw = weights[:, i] > 0

                if isinstance(Element, Lens):

                    # index inside lens
                    n1_l = Element.n(wavelengths) 

                    # choose ambient n or n specified in lens object for n after object
                    n2 = Element.n2 if Element.n2 is not None else self.n0
                    n2_l = n2(wavelengths) 

                    hw_front = hw.copy()
                    p[hw, i+1], hit_front = self.__find_surface_hit(Element.FrontSurface, p[hw, i], s[hw])
                    hwh = misc.part_mask(hw, hit_front)  # rays having power and hitting lens front
                    self.__refraction(Element.FrontSurface, p, s, weights, n0_l, n1_l, pols, hwh, i, msg)

                    # treat rays that go outside outlin                    
                    hwnh = misc.part_mask(hw, ~hit_front)  # rays having power and not hitting lens front
                    self.__outline_intersection(p, s, weights, hwnh, i, msg)

                    i += 1
                    if not self.silent and N_t == 0:
                        bar.update(i+1)
                    p[:, i+1], pols[:, i+1], weights[:, i+1] = p[:, i], pols[:, i], weights[:, i]

                    hw = weights[:, i] > 0
                    p[hw, i+1], hit_back = self.__find_surface_hit(Element.BackSurface, p[hw, i], s[hw])
                    hwb = misc.part_mask(hw, hit_back)  # rays having power and hitting lens back
                    self.__refraction(Element.BackSurface, p, s, weights, n1_l, n2_l, pols, hwb, i, msg)

                    # since we don't model the behaviour of the lens side cylinder, we need to absorb all rays passing
                    # through the cylinder
                    self.__absorb_cylinder_rays(p, weights, hw_front, hw, hit_front, hit_back, i, msg)

                    # absorb rays missing lens, overwrite p to last ray starting point (=end of lens front surface)
                    if self.absorb_missing and not np.all(hit_back):
                        miss_mask = misc.part_mask(hw, ~hit_back)
                        miss_count = np.count_nonzero(miss_mask)
                        weights[miss_mask, i+1] = 0
                        p[miss_mask, i+1] = p[miss_mask, i]
                        msg[self._infos.absorb_missing, i] += miss_count

                    # set n after object as next n before next object
                    n0_l = n2_l

                    # treat rays that go outside outline
                    hwnb = misc.part_mask(hw, ~hit_back)  # rays having power and not hitting lens back
                    self.__outline_intersection(p, s, weights, hwnb, i, msg)
                
                elif isinstance(Element, Filter | Aperture):
                    p[hw, i+1], hit = self.__find_surface_hit(Element.surface, p[hw, i], s[hw])
                    hwh = misc.part_mask(hw, hit)  # rays having power and hitting filter
                    
                    if isinstance(Element, Filter):
                        self.__filter(Element, weights, wavelengths, hwh, i, msg)
                    else:
                        weights[hwh, i+1] = 0

                    # treat rays that go outside outline
                    hwnh = misc.part_mask(hw, ~hit)  # rays having power and not hitting filter
                    self.__outline_intersection(p, s, weights, hwnh, i, msg)

                else:
                    raise RuntimeError(f"Invalid element type '{type(Element).__name__}' in raytracing")

                i += 1
                if not self.silent and N_t == 0:
                    bar.update(i+1)

        if N_threads > 1:
            thread_list = [Thread(target=sub_trace, args=(N_threads, N_t)) for N_t in np.arange(N_threads)]
            
            [thread.start() for thread in thread_list]
            [thread.join() for thread in thread_list]
        else:
            sub_trace(1, 0)

        self._set_messages(msgs)
        
        # lock Storage
        self.Rays.lock()

        # show info messages from tracing
        if not self.silent:
            bar.finish()
            self._show_messages(N)

    def _make_element_list(self) -> list[Lens | Filter | Aperture]:
        """
        Creates a sorted element list from filters and lenses.

        :return: list of sorted elements
        """

        # add a invisible (the filter is not in self.FilterList) to the outline area at +z
        # it absorbs all light at this surface
        o = self.outline
        EndFilter = Aperture(Surface("Rectangle", dim=[o[1] - o[0], o[3] - o[2]]),
                             pos=[(o[1] + o[0])/2, (o[2] + o[3])/2, o[5]])

        # add filters and lenses into one list,
        Elements = self.LensList + self.FilterList + self.ApertureList + [EndFilter]

        # sort list in z order
        Elements = sorted(Elements, key=lambda El: El.pos[2])

        return Elements

    def __geometry_checks(self, elements: list[Lens | Filter | Aperture]) -> None:
        """
        Checks geometry in raytracer for errors.

        :param elements: element list from __makeElementList()
        """

        def is_inside(e: tuple | list) -> bool:
            o = self.outline
            return o[0] <= e[0] and e[1] <= o[1] and o[2] <= e[2] and e[3] <= o[3] and o[4] <= e[4] and e[5] <= o[5]

        if not self.RaySourceList:
            raise RuntimeError("RaySource Missing.")

        for El in elements:
            if not is_inside(El.extent):
                raise RuntimeError(f"Element {El} of outside outline")

        for RS in self.RaySourceList:

            if not is_inside(RS.extent):
                raise RuntimeError(f"RaySource {RS} outside outline")

            if RS.pos[2] > elements[0].extent[4]:
                raise RuntimeError("The position of the RaySource needs to be in front of all objects.")
        
        for Det in self.DetectorList:
            Dx1, Dx2, Dy1, Dy2, _, _ = Det.extent
            Dz = Det.surface.pos[2]

            if not is_inside([Dx1, Dx2, Dy1, Dy2, Dz, Dz]):
                raise RuntimeError(f"Detector {Det} outside outline")

        # it's hard to check for surface collisions, for this we would need to find intersections of surfaces
        # instead we output a runtime error if the ray hits the surfaces at the wrong order while raytracing

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
            msg[self._infos.only_hit_front, i] += ab_count_front
            p[abnormal_front, i] = p[abnormal_front, i]
            weights[abnormal_front, i] = 0

        if ab_count_back:
            msg[self._infos.only_hit_back, i+1] += ab_count_back
            p[abnormal_back, i] = p[abnormal_back, i]
            weights[abnormal_back, i] = 0
    
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
            msg[self._infos.outline_intersection, i] += coll_count

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
        Calculate directions and weights after refraction. Rays with total inner reflection are treated as absorbed.
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

        # reflection coefficients for non-magnetic (µ_r=1) and non-absorbing materials (κ=0)
        # according to the Fresnel equations
        # see https://de.wikipedia.org/wiki/Fresnelsche_Formeln#Spezialfall:_gleiche_magnetische_Permeabilit.C3.A4t
        #####

        if not self.no_pol:
            # calculate s polarization vector
            # ns==1 means surface normal is parallel to ray direction, exclude these rays for now
            mask = np.abs(ns) < 1-1e-9
            mask2 = misc.part_mask(hwh, mask)

            # reduce slicing by storing separately
            polsm = pols[mask2, i]
            s_m = s_[mask]

            # s polarization vector
            ps = misc.cross(s_m, n[mask])
            ps = misc.normalize(ps)
            pp = misc.cross(ps, s_m)

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

        else:
            A_ts, A_tp = 1/np.sqrt(2), 1/np.sqrt(2)

        cos_alpha, cos_beta = ns, W
        ts = ne.evaluate("2 * n1_h*cos_alpha / (n1_h*cos_alpha + n2_h*cos_beta)")
        tp = ne.evaluate("2 * n1_h*cos_alpha / (n2_h*cos_alpha + n1_h*cos_beta)")
        T = ne.evaluate("n2_h*cos_beta / (n1_h*cos_alpha) * ((A_ts*ts)**2 + (A_tp*tp)**2)")
        
        # handle rays with total internal reflection
        TIR = ~np.isfinite(W)
        if np.any(TIR):
            T[TIR] = 0
            TIR_count = np.count_nonzero(TIR)
            msg[self._infos.TIR, i] += TIR_count

        weights[hwh, i+1] = weights[hwh, i]*T
        s[hwh] = s_

        if np.any(s_[:, 2] <= 0):
            raise RuntimeError(f"Non-positive ray z-direction after refraction on surface {i}.")

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
        if np.any(mask) and filter_.spectrum.spectrum_type not in ["Constant", "Rectangle"]:
            T[mask] = 0
            m_count = np.count_nonzero(mask)
            msg[self._infos.T_below_TTH, i] += m_count

        weights[hwh, i+1] = weights[hwh, i]*T

    def __find_surface_hit(self, surface: Surface, p: np.ndarray, s: np.ndarray)\
            -> tuple[np.ndarray, np.ndarray]:
        """
        Find the position of hits on surface using the iterative regula falsi algorithm.

        :param surface: Surface object
        :param p: position array (numpy 2D array, shape (N, 3))
        :param s: direction array (numpy 2D array, shape (N, 3))
        :return: positions of hit (shape as p), bool numpy 1D array if ray hits lens
        """

        # use surface's own hit finding for analytical surfaces
        if surface.has_hit_finding():
            p_hit, is_hit = surface.find_hit(p, s)

        else:
            # contraction factor (m = 0.5 : Illinois Algorithm)
            m = 0.5

            # ray parameters for just above and below the surface
            t1 = (surface.zmin - surface.eps/10 - p[:, 2])/s[:, 2]
            t2 = (surface.zmax + surface.eps/10 - p[:, 2])/s[:, 2]

            # ray coordinates at t1, t2
            p1 = p + s*t1[:, np.newaxis]
            p2 = p + s*t2[:, np.newaxis]

            # starting values for minimization function
            f1 = p1[:, 2] - surface.get_values(p1[:, 0], p1[:, 1])
            f2 = p2[:, 2] - surface.get_values(p2[:, 0], p2[:, 1])

            # assign non-finite and rays converged in first iteration
            w = np.ones(t1.shape, dtype=bool)  # bool array for hit search
            w[~np.isfinite(t1) | ~np.isfinite(t2)] = False  # exclude non-finite t1, t2
            w[(t2 - t1) < surface.eps] = False  # exclude rays that already converged
            c = ~w  # bool array for which ray has converged

            # arrays for estimated hit points
            p_hit = np.zeros_like(s, dtype=np.float64, order='F')
            p_hit[c] = p1[c]  # assign already converged rays

            it = 1  # number of iteration
            # do until all rays have converged
            while np.any(w):

                # secant root
                t1w, t2w, f1w, f2w = t1[w], t2[w], f1[w], f2[w]
                ts = ne.evaluate("t1w - f1w/(f2w-f1w)*(t2w-t1w)")

                # position of root on rays
                pl = p[w] + s[w]*ts[:, np.newaxis]

                # difference between ray and surface at root
                fts = pl[:, 2] - surface.get_values(pl[:, 0], pl[:, 1])

                # sign of fts*f2 decides which case is handled for each ray
                prod = fts*f2[w]

                # case 1: fts, f2 different sign => change [t1, t2] interval to [t2, ts]
                mask = prod < 0
                wm = misc.part_mask(w, mask)
                t1[wm], t2[wm], f1[wm], f2[wm] = t2[wm], ts[mask], f2[wm], fts[mask]

                # case 2: fts, f2 same sign => change [t1, t2] interval to [t1, ts]
                mask = prod > 0
                wm = misc.part_mask(w, mask)
                t2[wm], f1[wm], f2[wm] = ts[mask], m*f1[wm], fts[mask]

                # case 3: fts or f2 is zero => ts and fts are the found solution
                mask = prod == 0
                wm = misc.part_mask(w, mask)
                t1[wm], t2[wm], f1[wm], f2[wm] = ts[mask], ts[mask], fts[mask], fts[mask]

                # masks for rays converged in this iteration
                cn = t2[w]-t1[w] < surface.eps
                wcn = misc.part_mask(w, cn)

                # assign found hits and update bool arrays
                p_hit[wcn] = pl[cn]
                c[wcn] = True
                w[wcn] = False

                # timeout
                if it == 40:
                    raise RuntimeError(f"Non-convergence for {f1[w].shape[0]} rays in "
                                       f"surface hit finding after {it} iterations.")
                it += 1

            is_hit = surface.get_mask(p_hit[:, 0], p_hit[:, 1])

        if np.any(p_hit[:, 2] < p[:, 2]):
            raise RuntimeError("Hit point behind last starting point. This can be a result "
                               "of surface collisions or incorrect sequentiality.")

        return p_hit, is_hit

    def _hit_detector(self,
                      info:             str,
                      detector_index:   int = 0,
                      source_index:     int = None,
                      extent:           (list | np.ndarray | str) = "auto") \
            -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, str, ProgressBar]:
        """
        Rendered Detector Image. Rays are already traced.

        :param N: number of image pixels in each dimension (int)
        :param detector_index: index of Detector
        :param extent:
        :return: XYZIL Image (XYZ channels + Irradiance + Illuminance in third dimension) (numpy 3D array)
        """
        
        if not self.DetectorList:
            raise RuntimeError("Detector Missing")

        if not self.RaySourceList:
            raise RuntimeError("Raysource Missing")

        if source_index is not None and (source_index > len(self.RaySourceList) - 1 or source_index < 0):
            raise RuntimeError("Invalid RaySource index.")

        if detector_index > len(self.DetectorList) - 1 or detector_index < 0:
            raise RuntimeError("Invalid Detector index.")

        bar = ProgressBar(fd=sys.stdout, prefix=f"{info}: ", max_value=4, redirect_stdout=True).start()\
            if not self.silent else None

        # range for selected Rays in RayStorage
        Ns, Ne = self.Rays.B_list[source_index:source_index + 2] if source_index is not None else (0, self.Rays.N)

        # starting position of hit search
        z = self.DetectorList[detector_index].extent[4]

        # current rays for loop iteration, this rays are used in hit finding for the next section
        rs = np.zeros(self.Rays.N, dtype=bool)  # gets updated at every iteration
        rs[Ns:Ne] = True  # use rays of selected source

        # section index rs for each ray for section before z
        # rs2 == -1 can mean mask is true everywhere (ray starts after surface.zmin)
        # or false everywhere (rays don't reach surface),
        # so we need to check which case we're in. In the later case we don't need to calculate ray hits.
        mask = z <= self.Rays.p_list[Ns:Ne, :, 2]
        rs2 = np.argmax(mask, axis=1) - 1
        mask2 = np.all(mask, axis=1) & (rs2 < 0)
        rs2[mask2] = 0
        if bar is not None:
            bar.update(1)

        p, s, _, w, wl, _ = self.Rays.get_rays_by_mask(rs, rs2, ret=[1, 1, 0, 1, 1, 0])

        rs2z = misc.part_mask(rs, rs2 < 0)
        rs[rs2z] = False  # section index rs < 0 means ray does not reach that far
        p, s = p[rs2 >= 0], s[rs2 >= 0]
        w, wl = w[rs2 >= 0], wl[rs2 >= 0]
        rs2 = rs2[rs2 >= 0]
        # w[~rs] = 0  # not needed rs < 0 means ray is already absorbed
        if bar is not None:
            bar.update(2)

        # init ph (position of hit) and is_hit bool array
        ph = np.zeros_like(p, dtype=np.float64, order='F')
        ish = np.zeros_like(wl, dtype=bool)
        rs3 = np.ones_like(wl, dtype=bool)

        while np.any(rs):

            rs2 += 1  # increment next-section-indices
            val = rs2 < self.Rays.nt  # valid-array, indices are below section count
            # these rays have no intersection, since they end at the last outline surface

            if not np.all(val):
                rs = misc.part_mask(rs, val)  # only use valid rays
                rsi = misc.part_mask(rs3, ~val)  # current rays that are not valid
                w[rsi] = 0  # set invalid to weight of zero
                p, s, rs2, rs3 = p[val], s[val], rs2[val], rs3[val]  # only use valid rays

            ph[rs3], ish[rs3] = self.__find_surface_hit(self.DetectorList[detector_index].surface, p, s)

            p2z = self.Rays.p_list[rs, rs2, 2]
            rs3 = ph[rs3, 2] > p2z
            rs = misc.part_mask(rs, rs3)

            if np.any(rs):
                rs2 = rs2[rs3]
                p, s, _, w[rs3], _, _ = self.Rays.get_rays_by_mask(rs, rs2, ret=[1, 1, 0, 1, 0, 0])

        hitw = ish & (w > 0)
        ph, w, wl = ph[hitw], w[hitw], wl[hitw]
        if bar is not None:
            bar.update(3)

        if self.DetectorList[detector_index].surface.is_planar():
            extent_out = np.array(self.DetectorList[detector_index].extent[:4])
            coordinate_type = "Cartesian"

        else:
            if (stype := self.DetectorList[detector_index].surface.surface_type) not in ["Sphere", "Asphere"]:
                raise RuntimeError(f"Detector view not implemented for surface_type '{stype}'.")

            ph = self.DetectorList[detector_index].to_angle_coordinates(ph)
            extent_out = self.DetectorList[detector_index].get_angle_extent()
            coordinate_type = "Polar"

        # define the extent
        if isinstance(extent, list | np.ndarray):
            # only use rays inside extent area
            inside = (extent[0] <= ph[:, 0]) & (ph[:, 0] <= extent[1]) \
                    & (extent[2] <= ph[:, 1]) & (ph[:, 1] <= extent[3])

            extent_out = extent.copy()
            ph, w, wl = ph[inside], w[inside], wl[inside]

        elif extent == "auto":
            if np.any(hitw):
                # if auto mode and any rays hit the detector, adapt extent
                extent_out[[0, 2]] = np.min(ph[:, :2], axis=0)
                extent_out[[1, 3]] = np.max(ph[:, :2], axis=0)
                # to small dimension will be fixed by Image class

        elif extent == "whole":
            pass  # use already initialized extent

        else:
            raise ValueError(f"Invalid extent '{extent}'.")

        Detector_ = self.DetectorList[detector_index]
        pname = f": {Detector_.desc}" if Detector_.desc != "" else ""
        desc = f"{Detector.abbr}{detector_index}{pname} at z = {Detector_.pos[2]:.5g} mm"
        
        return ph, w, wl, extent_out, desc, coordinate_type, bar

    def detector_image(self,
                       N:               int,
                       detector_index:  int = 0,
                       source_index:    int = None,
                       extent:          (list | np.ndarray | str) = "auto",
                       **kwargs) -> RImage:
        """

        :param N:
        :param detector_index:
        :param source_index:
        :param extent:
        :param kwargs:
        :return:
        """
        N = int(N)
        if N <= 0:
            raise ValueError(f"Pixel number N needs to be a positive int, but is {N}")
        
        p, w, wl, extent_out, desc, coordinate_type, bar = self._hit_detector("Detector Image",
                                                                              detector_index, source_index, extent)

        # init image and extent, these are the default values when no rays hit the detector
        Im = RImage(desc=desc, extent=extent_out, coordinate_type=coordinate_type, 
                    threading=self.threading, silent=self.silent)
        Im.render(N, p, w, wl, **kwargs)
        if bar is not None:
            bar.finish()

        return Im

    def detector_spectrum(self,
                          detector_index:      int = 0,
                          source_index:     int = None,
                          extent:   (list | np.ndarray | str) = "auto",
                          **kwargs) -> LightSpectrum:
        """

        :param detector_index:
        :param source_index:
        :param extent:
        :param kwargs:
        :return:
        """
        p, w, wl, extent, desc, coordinate_type, bar = self._hit_detector("Detector Spectrum",
                                                                          detector_index, source_index)

        spec = LightSpectrum.render(wl, w, desc=f"Spectrum of {desc}", **kwargs)
        if bar is not None:
            bar.finish()

        return spec

    def iterative_render(self,
                         N_rays:            int,
                         N_px_D:            int | list = 400,
                         N_px_S:            int | list = 400,
                         detector_index:    int | list = 0,
                         pos:               list = None,
                         silent:            bool = False,
                         extent:            (str | list | np.ndarray) = "auto")\
            -> tuple[list[RImage], list[RImage]]:
        """

        :param N_rays:
        :param N_px_D:
        :param N_px_S:
        :param detector_index:
        :param pos:
        :param silent:
        :param extent:
        :return:
        """
        if (N_rays := int(N_rays)) <= 0:
            raise ValueError(f"Ray number N_rays needs to be a positive int, but is {N_rays}.")

        if (N_px_S := int(N_px_S)) <= 0:
            raise ValueError(f"Pixel number N_px_S needs to be a positive int, but is {N_px_S}.")
        
        if (N_px_D := int(N_px_D)) <= 0:
            raise ValueError(f"Pixel number N_px_D needs to be a positive int, but is {N_px_D}.")

        # if there are detectors
        if len(self.DetectorList):
            # use current detector position if pos is empty
            if pos is None:
                pos = [self.DetectorList[detector_index].pos[2]]

            if not isinstance(N_px_D, list):
                N_px_D = [N_px_D] * len(pos)
            
            if not isinstance(detector_index, list):
                detector_index = [detector_index] * len(pos)
            
            if not isinstance(extent, list) or isinstance(extent[0], int | float):
                extent = [extent] * len(pos)
            extentc = extent.copy()
            
        if not isinstance(N_px_S, list):
            N_px_S = [N_px_S] * len(self.RaySourceList)

        rays_step = self.ITER_RAYS_STEP
        iterations = int(N_rays / rays_step)
        diff = int(N_rays - iterations*rays_step)  # remaining rays for last iteration
        extra = diff > 0  # if there is a last iteration

        # turn off messages for raytracing iterations
        silent_old = self.silent
        self.silent = True

        # image list
        DIm_res = []
        SIm_res = []

        iter_ = range(iterations+extra)
        iterator = progressbar(iter_, prefix="Rendering: ", fd=sys.stdout, redirect_stdout=True) if not silent\
            else iter_

        # for all render iterations
        for i in iterator:

            # only true in extra step
            if i == iterations:
                rays_step = diff

            self.trace(N=rays_step)

            if len(self.DetectorList):
                # for all detector positions
                for j in np.arange(len(pos)):
                    pos_new = np.concatenate((self.DetectorList[detector_index[j]].pos[:2], [pos[j]]))
                    self.DetectorList[detector_index[j]].move_to(pos_new)
                    Imi = self.detector_image(N=N_px_D[j], detector_index=detector_index[j], extent=extentc[j])
                    Imi._Im *= rays_step/N_rays
                    
                    # append image to list in first iteration, after that just add image content
                    if i == 0:
                        DIm_res.append(Imi)
                        extentc[j] = Imi.extent
                    else:
                        DIm_res[j]._Im += Imi._Im

            for j, _ in enumerate(self.RaySourceList):
                Imi = self.source_image(N=N_px_S[j], source_index=j)
                Imi._Im *= rays_step/N_rays
                
                # append image to list in first iteration, after that just add image content
                if i == 0:
                    SIm_res.append(Imi)
                else:
                    SIm_res[j]._Im += Imi._Im

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
        Rendered Image of RaySource. Rays were already traced.

        :param N: number of image pixels in each dimension (int)
        :param source_index:
        :return: XYZIL Image (XYZ channels + Irradiance + Illuminance in third dimension) (numpy 3D array)
        """
        
        if not self.RaySourceList:
            raise RuntimeError("RaySource Missing")
        
        if source_index > len(self.RaySourceList) - 1 or source_index < 0:
            raise RuntimeError("Invalid RaySource index.")

        bar = ProgressBar(fd=sys.stdout, prefix=f"{info}: ", max_value=2, redirect_stdout=True).start()\
            if not self.silent else None

        extent = self.RaySourceList[source_index].extent[:4]
        p, _, _, w, wl = self.Rays.get_source_sections(source_index)
        if bar is not None:
            bar.update(1)

        RS = self.RaySourceList[source_index]
        pname = f": {RS.desc}" if RS.desc != "" else ""
        desc = f"{RaySource.abbr}{source_index}{pname} at z = {RS.pos[2]:.5g} mm"

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

    def source_image(self, N: int, source_index: int = 0, **kwargs) -> RImage:
        """
        Rendered Image of RaySource. Rays were already traced.

        :param N: number of image pixels in each dimension (int)
        :param source_index:
        :return: XYZIL Image (XYZ channels + Irradiance + Illuminance in third dimension) (numpy 3D array)
        """
        
        if (N := int(N)) <= 0:
            raise ValueError(f"Pixel number N needs to be a positive int, but is {N}.")
        
        p, w, wl, extent, desc, bar = self._hit_source("Source Image", source_index)

        Im = RImage(desc=desc, extent=extent, coordinate_type="Cartesian",
                    threading=self.threading, silent=self.silent)
        Im.render(N, p, w, wl, **kwargs)
        if bar is not None:
            bar.finish()

        return Im

    def __autofocus_cost_func(self,
                              z_pos:  float,
                              mode:   str,
                              pa:     np.ndarray,
                              sb:     np.ndarray,
                              w:      np.ndarray,
                              r0:     float = 1e-3)\
            -> float:
        """

        :param z_pos:
        :param mode:
        :param pa:
        :param sb:
        :param w:
        :param r0:
        :return:
        """
       
        ph = pa + sb*z_pos
        x, y = ph[:, 0], ph[:, 1]

        if mode == "Airy Disc Weighting":
            xm = np.average(x, weights=w)
            ym = np.average(y, weights=w)

            expr = ne.evaluate("w * exp(-0.5*((x - xm)**2 + (y-ym)**2)/ (0.42*r0)**2)")
            return 1 - np.sum(expr)/np.sum(w)
            
        elif mode == "Position Variance":
            var_x = np.cov(x, aweights=w)
            var_y = np.cov(y, aweights=w)

            # use pythagoras for overall variance
            return np.sqrt(var_x + var_y)

        elif mode == "Irradiance Variance":

            x0, x1, y0, y1, _, _ = self.outline
            inside = (x0 < x) & (x < x1) & (y0 < y) & (y < y1)
            x, y, w = x[inside], y[inside], w[inside]

            N_px = 151
            extent = np.min(x), np.max(x), np.min(y), np.max(y)

            # get hit pixel coordinate, subtract 1e-12 from N so all values are 0 <= xcor < N
            xcor = N_px*(1 - 1e-12) / (extent[1] - extent[0]) * (x - extent[0]) 
            ycor = N_px*(1 - 1e-12) / (extent[3] - extent[2]) * (y - extent[2])  

            # image can be 1D for standard deviation, speed things up a little
            ind = N_px*ycor.astype(int) + xcor.astype(int)

            Im = np.zeros(N_px**2, dtype=np.float64)
            np.add.at(Im, ind, w)

            return np.sqrt(1/np.std(Im[Im > 0]))  # sqrt for better value range

        else:
            raise ValueError(f"Invalid Autofocus Mode '{mode}'.")

    def autofocus(self,
                  method:           str,
                  z_start:          float,
                  source_index:     int = None,
                  N:                int = 75000,
                  return_cost:      bool = True)\
            -> tuple[scipy.optimize.OptimizeResult, list, np.ndarray | None, np.ndarray | None]:
        """
        Find the focal point using different methods. z_start defines the starting point, 
        the search range is the region between lenses or the outline.
        The influence of filters is neglected.

        :param source_index:
        :param z_start: starting position z (float)
        :param N: maximum number of rays to evaluate (int)
        :param method:
        :param return_cost: False, if costly calculation of cost function array
                can be skipped in mode "Position Variance".
                In other modes it is generated on the way anyway
        :return: position of focus (float)
        """

        if not (self.outline[4] <= z_start <= self.outline[5]):
            raise ValueError("Starting position z_start outside outline.")

        if method not in self.autofocus_modes:
            raise ValueError(f"Invalid method '{method}', should be one of {self.autofocus_modes}.")

        if N < 1000:
            raise ValueError("N should be at least 1000.")

        if source_index is not None and source_index < 0:
            raise ValueError("snum needs to be >= 0.")

        if (source_index is not None and source_index > len(self.Rays.N_list)) or len(self.Rays.N_list) == 0:
            raise ValueError("snum larger than number of simulated sources. "
                             "Either the source was not added or the geometry was not traced.")

        # get search bounds
        ################################################################################################################

        # sort list in z order
        Lenses = sorted(self.LensList, key=lambda Element: Element.pos[2])
        
        # no lenses or region lies before first lens
        if not Lenses or z_start < Lenses[0].pos[2]:
            n_ambient = self.n0(550)
            bounds = [self.outline[4], self.outline[5]] if not Lenses else [self.outline[4], Lenses[0].extent[4]]

            # start position needs to be behind all raysources
            for RS in self.RaySourceList:
                if RS.pos[2] > bounds[0]:
                    bounds[0] = RS.pos[2] + self.EPS

        # region lies behind last lens
        elif z_start > Lenses[-1].pos[2]:
            n_ambient = self.n0(550) if Lenses[-1].n2 is None else Lenses[-1].n2(550)
            bounds = [Lenses[-1].extent[5], self.outline[5]]

        # region lies between two lenses
        else:
            # find bounds between lenses
            for ind, Lens in enumerate(Lenses):
                if z_start < Lens.pos[2]:
                    bounds = [Lenses[ind-1].extent[5], Lens.extent[4]]
                    n_ambient = self.n0(550) if Lens.n2 is None else Lens.n2(550)
                    break

        # show filter warning
        for F in (self.FilterList + self.ApertureList):
            if bounds[0] <= F.pos[2] <= bounds[1]:
                warnings.warn("WARNING: The influence of the filters/apertures in the autofocus range will be ignored.")

        # start progress bar
        ################################################################################################################

        Nt = 1000
        N_th = ne.detect_number_of_cores()
        steps = np.ceil(Nt/N_th/10).astype(int) if self.threading else np.ceil(Nt/10).astype(int)
        steps += 2
        bar = ProgressBar(fd=sys.stdout, prefix="Finding Focus: ", max_value=steps, redirect_stdout=True).start()\
            if not self.silent else None

        # get rays and properties
        ################################################################################################################
        
        Ns, Ne = self.Rays.B_list[source_index:source_index + 2] if source_index is not None else (0, self.Rays.N)

        rays_pos = np.zeros(self.Rays.N, dtype=bool)
        pos = np.zeros(self.Rays.N, dtype=int)
        rays_pos[Ns:Ne] = True

        z = bounds[0] + self.EPS
        pos[Ns:Ne] = np.argmax(z < self.Rays.p_list[rays_pos, :, 2], axis=1) - 1
        if not self.silent:
            bar.update(1)

        rays_pos[pos == -1] = False
        
        N_is = min(N, np.count_nonzero(rays_pos))
        if N_is < 1000:  # throw error when no rays are present
            print(N_is, N, Ns, Ne, bounds)
            raise RuntimeError("Less than 1000 rays for autofocus.")

        # select random rays from all valid
        rp = np.where(rays_pos)[0]
        rp2 = np.random.choice(rp, N_is, replace=False)
    
        # assign positions of random rays
        rays_pos = np.zeros(self.Rays.N, dtype=bool)
        rays_pos[rp2] = True
        pos = pos[rp2]

        # get Ray parts
        p, s, _, weights, _, _ = self.Rays.get_rays_by_mask(rays_pos, pos, ret=[1, 1, 0, 1, 0, 0])

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

            # size of airy disc, default to 1µm when sin_alpha = 0
            r0 = 0.61 * 550e-6 / (sin_alpha * n_ambient) if sin_alpha != 0 else 1e-3 
        else:
            r0 = 1e-3

        if method != "Position Variance" or return_cost:
            # sample smaller region around minimum with proper method
            r = np.linspace(bounds[0], bounds[1], Nt)
            vals = np.zeros_like(r)

            def threaded(N_th, N_is, Nt, *afargs):
                Ns = N_is*int(Nt/N_th)
                Ne = (N_is+1)*int(Nt/N_th) if N_is != N_th-1 else Nt

                for i, Ni in enumerate(np.arange(Ns, Ne)):
                    if not i % 10 and not self.silent:
                        bar.update(2+int(i/10+1))
                    vals[Ni] = self.__autofocus_cost_func(r[Ni], *afargs)

            if self.threading:   
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
        if method in ["Airy Disc Weighting", "Irradiance Variance"]:
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
                       "this could mean the focus is outside the search range")

        return res, bounds, r, vals
