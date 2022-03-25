
# flag for toggling multithreading:
# useful, when the process should be kept in one thread (e.g. for debugging)
# or raytracer itself is run in multiple threads

"""
Raytracer class:
Provides the functionality for raytracing, autofocussing and rendering of source and detector images.
"""
import warnings
import numpy as np
import numexpr as ne

import time
import scipy.optimize

from threading import Thread
from typing import Callable

from Backend.Filter import Filter as Filter
from Backend.Detector import Detector as Detector
from Backend.Lens import Lens as Lens
from Backend.RaySource import RaySource as RaySource
from Backend.Surface import Surface as Surface
from Backend.RefractionIndex import RefractionIndex as RefractionIndex
from Backend.RaySourceList import RaySourceList as RaySourceList
from Backend.Image import Image as Image
from Backend.Misc import timer as timer

import Backend.Misc as misc


class Raytracer:

    EPS: float = 1e-12
    """ raytracer epsilon, used as precision in some places """

    T_TH: float = 1e-4
    """ threshold for the transmission Filter
    values below this are seen as absorbed
    needed to avoid ghost rays, meaning rays that have a non-zero, but negligible power"""

    MAX_RAYS: int = 10000000
    """ maximum number of rays. Limited by RAM usage """

    ITER_RAYS_STEP: int = 1000000
    """ number of rays per iteration in :obj:`Raytracer.iterativeDetectorImage` """

    def __init__(self,
                 outline:        (list | np.ndarray),
                 n0:             RefractionIndex = RefractionIndex("Constant", n=1),
                 AbsorbMissing:  bool = False,
                 no_pol:         bool = False,
                 silent:         bool = False,
                 multithreading: bool = True)\
            -> None:
        """
        Initialize the Raytracer

        :param outline: outline of raytracer space [x1, x2, y1, y2, z1, z2] (numpy 1D array or list)
        :param n0: refraction index of the raytracer enviroment (RefractionIndex object)
        :param AbsorbMissing: if rays missing a lens are absorbed at the lens (bool)
        :param no_pol:
        :param silent:
        :param multithreading:
        """

        self.outline            = np.array(outline, dtype=np.float64)
        self.AbsorbMissing      = AbsorbMissing
        self.no_pol             = no_pol
        self.n0                 = n0
        self.silent             = silent
        self.multithreading     = multithreading

        self.LensList = []
        self.FilterList = []
        self.DetectorList = []
        self.RaySources = RaySourceList() 

        # check outline
        if self.outline.shape[0] != 6 or \
           self.outline[0] >= self.outline[1] or \
           self.outline[2] >= self.outline[3] or \
           self.outline[4] >= self.outline[5]:
            raise ValueError("Outline needs to be specified as [x1, x2, y1, y2, z1, z2] "
                             "with x2 > x1, y2 > y1, z2 > z1.")

    def add(self, el: Lens | Filter | RaySource | Detector) -> int:
        """
        Add an element to the Raytracer geometry.

        :param el: Element to add to Raytracer 
        :return: identifier of object
        """
        if isinstance(el, Lens):
            self.LensList.append(el)

        elif isinstance(el, Filter):
            self.FilterList.append(el)

        elif isinstance(el, RaySource):
            self.RaySources.add(el)

        elif isinstance(el, Detector):
            self.DetectorList.append(el)

        else:
            raise TypeError("Invalid element type.")

        return id(el)
    
    def get(self, ident: int) -> (Lens | Filter | RaySource | Detector):
        """
        Get a reference to the element in the raytracing geometry specified by id.
        Returns None if element not found.

        :param ident:
        :return:
        """
        Elr = None
        [Elr := El for El in self.LensList if id(El) == ident]
        [Elr := El for El in self.FilterList if id(El) == ident]
        [Elr := El for El in self.DetectorList if id(El) == ident]
        [Elr := El for El in self.RaySources.List if id(El) == ident]

        return Elr

    def remove(self, ident: int) -> bool:
        """
        Remove the element specified by its id from raytracing geometry.
        Returns True if element(s) have been found and removes, False otherwise.

        :param ident: identifier of element from :obj:`Raytracer.add` or from :obj:`id`
        """
        success = False
        [(self.LensList.remove(El), success := True) for El in self.LensList if id(El) == ident]
        [(self.FilterList.remove(El), success := True) for El in self.FilterList if id(El) == ident]
        [(self.DetectorList.remove(El), success := True) for El in self.DetectorList if id(El) == ident]

        if not success:
            return self.RaySources.remove(ident)
        return success

    def trace(self, N: int) -> None:
        """
        Execute raytracing, saves all Rays in the internal RaySource object

        :param N: number of rays (int)
        """

        if not self.RaySources.hasSources():
            raise RuntimeError("RaySource Missing")

        if (N := int(N)) <= 0:
            raise ValueError(f"Ray number N needs to be a positive int, but is {N}.")

        if N > self.MAX_RAYS:
            raise ValueError(f"Ray number exceeds maximum of {self.MAX_RAYS}")

        if not self.silent:
            print("Raytracing...")

        # all rays have the same starting power.
        # The rays are distributed between the sources according to the ray source power
        # get source powers and overall power
        P_list = np.array([RS.power for RS in self.RaySources.List])
        P_all = np.sum(P_list)

        # calculate int ray number from power ratio
        N_list = (N*P_list/P_all).astype(int)
        dN = N-np.sum(N_list)  # difference to ray number

        # distribute difference dN randomly on all sources, with the power being the probability
        index_add = np.random.choice(N_list.shape[0], size=dN, p=P_list/P_all)
        np.add.at(N_list, index_add, np.ones(index_add.shape))

        if np.any(np.array(N_list) == 0):
            warnings.warn("There are RaySources that have no rays assigned. "\
                    "Change the power ratio or raise the overall ray number", RuntimeWarning)

        # AbsorbMissing = False is only possible, if all ambient refraction indices are the same
        if not self.AbsorbMissing:
            for Lens_ in self.LensList:
                if Lens_.n2 is not None and self.n0 != Lens_.n2:
                    warnings.warn("Outside refraction index defined for at least one lens, setting AbsorbMissing"\
                                  " simulation parameter to True", RuntimeWarning)
                    self.AbsorbMissing = True
                    break

        Elements = self.makeElementList()
        self.GeometryChecks(Elements)

        # reserve space for all surface points, +1 for invisible aperture at the outline z-end
        # and +1 for the ray starting points
        nt = 2*len(self.LensList) + len(self.FilterList) + 1 + 1
    
        cores = ne.detect_number_of_cores()
        N_threads = cores if N/cores >= 10000 and self.multithreading else 1

        self.RaySources.createRays(N_list, nt, threading=N_threads>1, no_pol=self.no_pol)

        def sub_trace(N_threads: int, N_t: int) -> None:

            p, s0, pols, weights, wavelengths = self.RaySources.getThreadRays(N_threads, N_t)
            s = s0.copy()

            n0_l = self.n0(wavelengths) 

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
                    p[hw, i+1], hit_front = self.findSurfaceHit(Element.front, p[hw, i], s[hw])
                    hwh, _ = misc.partMask(hw, hit_front)  # rays having power and hitting lens front
                    self.refraction(Element.front, p, s, weights, n0_l, n1_l, pols, hwh, i)

                    # treat rays that go outside outline
                    hwnh, _ = misc.partMask(hw, ~hit_front)  # rays having power and not hitting lens front
                    self.outlineIntersection(p, s, weights, hwnh, i)

                    i += 1
                    p[:, i+1], pols[:, i+1], weights[:, i+1] = p[:, i], pols[:, i], weights[:, i]

                    hw = weights[:, i] > 0
                    p[hw, i+1], hit_back = self.findSurfaceHit(Element.back, p[hw, i], s[hw])
                    hwb, _ = misc.partMask(hw, hit_back)  # rays having power and hitting lens back
                    self.refraction(Element.back, p, s, weights, n1_l, n2_l, pols, hwb, i)

                    # since we don't model the behaviour of the lens side cylinder, we need to absorb all rays passing
                    # through the cylinder
                    self.absorbCylinderRays(p, weights, hw_front, hw, hit_front, hit_back, i)

                    # absorb rays missing lens, overwrite p to last ray starting point (=end of lens front surface)
                    if self.AbsorbMissing and not np.all(hit_back):
                        miss_mask, _ = misc.partMask(hw, ~hit_back)
                        miss_count = np.count_nonzero(miss_mask)
                        weights[miss_mask, i+1] = 0
                        if not self.silent:
                            print(f"{miss_count} rays ({miss_count/self.RaySources.N:.3g}% of all rays) "\
                                  f"missing surface {i+1}, setting to absorbed because of parameter AbsorbMissing=True.")

                    # set n after object as next n before next object
                    n0_l = n2_l

                    # treat rays that go outside outline
                    hwnb, _ = misc.partMask(hw, ~hit_back)  # rays having power and not hitting lens back
                    self.outlineIntersection(p, s, weights, hwnb, i)
                
                elif isinstance(Element, Filter):
                    p[hw, i+1], hit = self.findSurfaceHit(Element.Surface, p[hw, i], s[hw])

                    hwh, _ = misc.partMask(hw, hit)  # rays having power and hitting filter
                    self.filter(Element, weights, wavelengths, hwh, i)

                    # treat rays that go outside outline
                    hwnh, _ = misc.partMask(hw, ~hit)  # rays having power and not hitting filter
                    self.outlineIntersection(p, s, weights, hwnh, i)

                else:
                    raise RuntimeError(f"Invalid element type '{type(Element).__name__}' in raytracing")

                i += 1

        if N_threads > 1:
            thread_list = [Thread(target=sub_trace, args=(N_threads, N_t)) for N_t in np.arange(N_threads)]
            
            [thread.start() for thread in thread_list]
            [thread.join()  for thread in thread_list]
        else:
            sub_trace(1, 0)

        if not self.silent:
            print("Raytracing Finished.\n")

    def makeElementList(self) -> list[Lens | Filter]:
        """
        Creates a sorted element list from filters and lenses.

        :return: list of sorted elements
        """

        # add a invisible (the filter is not in self.FilterList) to the outline area at +z
        # it absorbs all light at this surface
        EndFilter = Filter(Surface(surface_type="Rectangle", 
                                   dim=[self.outline[1]-self.outline[0], self.outline[3] - self.outline[2]]),
                           pos=[(self.outline[1]+self.outline[0])/2, (self.outline[2]+self.outline[3])/2, self.outline[5]])

        # add filters and lenses into one list,
        Elements = self.LensList + self.FilterList + [EndFilter]

        # sort list in z order
        Elements = sorted(Elements, key=lambda El: El.pos[2])

        return Elements

    def GeometryChecks(self, Elements: list[Lens | Filter]) -> None:
        """
        Checks geometry in raytracer for errors.

        :param Elements: element list from makeElementList()
        """

        def isinside(ext: list | np.ndarray) -> bool:
            return (self.outline[0] <= ext[0] and ext[1] <= self.outline[1] and
                    self.outline[2] <= ext[2] and ext[3] <= self.outline[3] and
                    self.outline[4] <= ext[4] and ext[5] <= self.outline[5])

        enum = 1
        z_max_old = self.outline[4]

        if not self.RaySources.hasSources():
            raise RuntimeError("RaySource Missing")

        # check if objects collide in z-direction and if all objects are inside outline
        for Element in Elements:

            z_min, z_max = Element.extent[4:]

            if not isinside(Element.extent):
                raise RuntimeError(f"Object {enum} (numbering in z-order) of type {Element} outside outline")

            if z_max_old > z_min:
                warnings.warn(f"Possible collision between objects {enum-1} and {enum} (numbering in z-order). "
                              f"However, only the maximum and minimum z-positions of the surfaces were compared.")

            z_max_old = z_max
            enum += 1


        for RS in self.RaySources.List:

            if not isinside(RS.extent):
                raise RuntimeError(f"RaySource {RS} outside outline")

            if RS.pos[2] > Elements[0].extent[4]:
                raise RuntimeError("The position of the RaySource needs to be in front of all objects.")
        
        for Det in self.DetectorList:
            Dx1, Dx2, Dy1, Dy2, _, _ = Det.extent
            Dz = Det.Surface.pos[2]

            if not isinside([Dx1, Dx2, Dy1, Dy2, Dz, Dz]):
                raise RuntimeError(f"Detector {Det} outside outline")

    # TODO: still working?
    def absorbCylinderRays(self,
                           p:           np.ndarray,
                           weights:     np.ndarray,
                           hw_front:    np.ndarray,
                           hw_back:     np.ndarray,
                           hit_front:   np.ndarray,
                           hit_back:    np.ndarray,
                           i:           int)\
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

        hit_front_hw, _  = misc.partMask(hw_front, hit_front)
        hit_back_hw, _   = misc.partMask(hw_back, hit_back)

        miss_front_hw, _ = misc.partMask(hw_front, ~hit_front)
        miss_back_hw, _  = misc.partMask(hw_back, ~hit_back)

        abnormal_front   = hit_front_hw & miss_back_hw
        abnormal_back    = hit_back_hw & miss_front_hw
        ab_count_front   = np.count_nonzero(abnormal_front)
        ab_count_back    = np.count_nonzero(abnormal_back)

        if ab_count_front:
            if not self.silent:
                print(f"{ab_count_front} rays ({100*ab_count_front/self.RaySources.N:.3g}% "\
                      f"of all rays) hitting lens front but missing back, setting to absorbed.")

            weights[abnormal_front, i] = 0
            p[abnormal_front, i] = p[abnormal_front, i-1]

        if ab_count_back:
            if not self.silent:
                print(f"{ab_count_back} rays ({100*ab_count_back/self.RaySources.N:.3g}% "\
                      f"of all rays) missing lens front but hitting back, setting to absorbed.")

            p[abnormal_back, i] = p[abnormal_back, i-1]
            weights[abnormal_back, i] = 0
    # @timer
    def outlineIntersection(self,
                            p:           np.ndarray,
                            s:           np.ndarray,
                            weights:     np.ndarray,
                            hw:          np.ndarray,
                            i:           int)\
            -> None:
        """
        Checks if the rays intersect with the outline, finds intersections points.
        Ray weights of intersecting rays are set to zero at that point.

        :param p: position array prior surface hit (numpy 2D array, shape (N, 3))
        :param s: direction array (numpy 2D array, shape (N, 3))
        :param weights: ray weights (numpy 1D array)
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

            hwi, _ = misc.partMask(hw, ~inside)

            OT = np.tile(self.outline, (n_out, 1)) # tile outline for every outside ray
            P = p[hwi, i].repeat(2).reshape(n_out, 6) # repeat each column once
            S = s[hwi].repeat(2).reshape(n_out, 6) # repeat each column once

            # calculate t Parameter for every outline coordinate and ray
            # replace zeros with nan for division
            nan = np.nan
            T_arr = ne.evaluate("(OT-P)/where(S != 0, S, nan)")

            # exclude negative t
            T_arr[T_arr <= 0] = np.nan

            # first intersection is smallest positive t
            t = np.nanmin(T_arr, axis=1)

            # assign intersection positions and weights for outside rays
            p[hwi, i+1] = p[hwi, i] + s[hwi]*t[:, np.newaxis]
            weights[hwi, i+1] = 0

            if not self.silent:
                coll_count = np.count_nonzero(hwi)
                print(f"{coll_count} rays ({100*coll_count/self.RaySources.N:.3g}% "\
                      f"of all rays) hitting outline, setting to absorbed.")

    def refraction(self,
                   surface:      Surface,
                   p:            np.ndarray,
                   s:            np.ndarray,
                   weights:      np.ndarray,
                   n1:           np.ndarray,
                   n2:           np.ndarray,
                   pols:         np.ndarray,
                   hwh:          np.ndarray,
                   i:            int)\
            -> None:
        """
        Calculate directions and weights after refraction. Rays with total inner r\eflection are treated as absorbed.
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

        n = surface.getNormals(p[hwh, i+1, 0], p[hwh, i+1, 1])

        n1_h = n1[hwh] 
        n2_h = n2[hwh] 
        s_h = s[hwh]

        # vectorial Snell's Law as in "Optik - Physikalisch-technische Grundlagen und Anwendungen,
        # Heinz Haferkorn, Auflage 4, Wiley-VCH, 2008", p.43
        # but with s_ = s', n1 = n, n2 = n', alpha = ε, beta = ε'

        ns = misc.rdot(n, s_h)  # equals cos(alpha)
        N = n1_h/n2_h  # ray wise refraction index quotient
       
        # rays with TIR mean an imaginary W, we'll handle this later
        W = ne.evaluate("sqrt(1 - N**2 * (1-ns**2))") # equals cos(beta), we'll need this later
        
        N_m, ns_m, W_m = N[:, np.newaxis], ns[:, np.newaxis], W[:, np.newaxis]
        s_ = ne.evaluate("s_h*N_m - n*(N_m*ns_m - W_m)")

        # reflection coefficients for non-magnetic (µ_r=1) and non-absorbing materials (κ=0)
        # according to the Fresnel equations
        # see https://de.wikipedia.org/wiki/Fresnelsche_Formeln#Spezialfall:_gleiche_magnetische_Permeabilit.C3.A4t
        #####

        if not self.no_pol:
            # calculate s polarization vector
            mask = ns != 1  # ns==1 means surface normal is parallel to ray direction, exclude these rays for now
            mask2, _ = misc.partMask(hwh, mask)

            # reduce slicing by storing separately
            polsm = pols[mask2, i]
            s_m = s_[mask]

            ps = misc.cross(n[mask], s_m)
            misc.normalize(ps)

            # calculate p polarization vector
            pp = misc.cross(ps, s[mask2])

            # init arrays
            # default for A_ts, A_tp are 1/sqrt(2)
            A_ts = np.full_like(ns, 1/np.sqrt(2), dtype=np.float32)
            A_tp = np.full_like(ns, 1/np.sqrt(2), dtype=np.float32)

            # amplitude components of ray polarization in s and p
            A_ts[mask] = misc.rdot(ps, polsm)
            A_tp[mask] = misc.rdot(pp, polsm)

            # new polarization vector after refraction
            pp_ = misc.cross(ps, s_m)
            pols_ = ps*A_ts[mask, np.newaxis] + pp_*A_tp[mask, np.newaxis]
        
            pols[mask2, i+1] = pols_

            # transmittance for s and p component
            cos_alpha, cos_beta = ns, W
            ts = ne.evaluate("2 * n1_h*cos_alpha / (n1_h*cos_alpha + n2_h*cos_beta)")
            tp = ne.evaluate("2 * n1_h*cos_alpha / (n2_h*cos_alpha + n1_h*cos_beta)")

            # overall transmittivity
            T = ne.evaluate("n2_h*cos_beta / (n1_h*cos_alpha) * ((A_ts*ts)**2 + (A_tp*tp)**2)")

        else:
            cos_alpha, cos_beta = ns, W
            T = ne.evaluate("2*n2_h*cos_beta*n1_h*cos_alpha * "\
                            "( 1 / (n1_h*cos_alpha + n2_h*cos_beta)**2 + 1/(n2_h*cos_alpha + n1_h*cos_beta)**2)")

        # handle rays with total internal reflection
        TIR = ~np.isfinite(W)
        if np.any(TIR):
            T[TIR] = 0

            if not self.silent:
                TIR_count = np.count_nonzero(TIR)
                print(f"{TIR_count} rays ({100*TIR_count/self.RaySources.N:.3g}% "\
                      f"of rays on surface) with total reflection at surface {i}, treating as absorbed.")

        weights[hwh, i+1] = weights[hwh, i]*T
        s[hwh] = s_

        if np.any(s[hwh, 2] <= 0):
            raise RuntimeError(f"Non-positive ray z-direction after refraction on surface {i}.")

    def filter(self,
               filter_:      Filter,
               weights:     np.ndarray,
               wl:          np.ndarray,
               hwh:         np.ndarray,
               i:           int)\
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
        mask = (T > 0) & (T < self.T_TH)

        if np.any(mask):
            T[mask] = 0
            
            if not self.silent:
                m_count = np.count_nonzero(mask)
                print(f"{m_count} rays ({100*m_count/self.RaySources.N:.3g}% of all rays) "\
                      f"with transmittivity at filter surface {i+1} below threshold of "\
                      f"{self.T_TH*100:.3g}%, setting to absorbed.")

        weights[hwh, i+1] = weights[hwh, i]*T

    def findSurfaceHit(self,
                       surface: Surface,
                       p:       np.ndarray,
                       s:       np.ndarray)\
            -> tuple[np.ndarray, np.ndarray]:
        """
        Find the position of hits on surface using the iterative regula falsi algorithm.

        :param surface: Surface object
        :param p: position array (numpy 2D array, shape (N, 3))
        :param s: direction array (numpy 2D array, shape (N, 3))
        :return: positions of hit (shape as p), bool numpy 1D array if ray hits lens
        """

        # use surface's own hit finding for analytical surfaces
        if surface.hasHitFinding():
            return surface.findHit(p, s)

        # contraction factor (m = 0.5 : Illinois Algorithm)
        m = 0.5

        # ray parameters for just above and below the surface
        t1 = (surface.minz - surface.eps/10 - p[:, 2])/s[:, 2]
        t2 = (surface.maxz + surface.eps/10 - p[:, 2])/s[:, 2]

        # ray coordinates at t1, t2
        p1 = p + s*t1[:, np.newaxis]
        p2 = p + s*t2[:, np.newaxis]

        # starting values for minimization function
        f1 = p1[:, 2] - surface.getValues(p1[:, 0], p1[:, 1])
        f2 = p2[:, 2] - surface.getValues(p2[:, 0], p2[:, 1])

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
            fts = pl[:, 2] - surface.getValues(pl[:, 0], pl[:, 1])

            # sign of fts*f2 decides which case is handled for each ray
            prod = fts*f2[w]

            # case 1: fts, f2 different sign => change [t1, t2] interval to [t2, ts]
            wm, mask = misc.partMask(w, prod < 0)
            t1[wm], t2[wm], f1[wm], f2[wm] = t2[wm], ts[mask], f2[wm], fts[mask]

            # case 2: fts, f2 same sign => change [t1, t2] interval to [t1, ts]
            wm, mask = misc.partMask(w, prod > 0)
            t2[wm], f1[wm], f2[wm] = ts[mask], m*f1[wm], fts[mask]

            # case 3: fts or f2 is zero => ts and fts are the found solution
            wm, mask = misc.partMask(w, prod == 0)
            t1[wm], t2[wm], f1[wm], f2[wm] = ts[mask], ts[mask], fts[mask], fts[mask]

            # masks for rays converged in this iteration
            wcn, cn = misc.partMask(w, t2[w]-t1[w] < surface.eps)

            # assign found hits and update bool arrays
            p_hit[wcn] = pl[cn]
            c[wcn] = True
            w[wcn] = False

            # timeout
            if it == 40:
                raise RuntimeError(f"Non-convergence for {f1[w].shape[0]} rays in "\
                                   f"surface hit finding after {it} iterations.")
            it += 1

        is_hit = surface.getMask(p_hit[:, 0], p_hit[:, 1])
        return p_hit, is_hit


    # TODO explanations
    def DetectorImage(self,
                      N:        int,
                      ind:      int = 0,
                      extent:   (list | np.ndarray | str) = "auto") \
            -> Image:
        """
        Rendered Detector Image. Rays are already traced.

        :param N: number of image pixels in each dimension (int)
        :param ind: index of detector
        :param extent:
        :return: XYZIL Image (XYZ channels + Irradiance + Illuminance in third dimension) (numpy 3D array)
        """

        N = int(N)
        if N <= 0:
            raise ValueError(f"Pixel number N needs to be a positive int, but is {N}")

        if not self.DetectorList:
            raise RuntimeError("Detector Missing")

        if not self.RaySources.hasSources():
            raise RuntimeError("Raysource Missing")

        # starting position of hit search
        z = self.DetectorList[ind].extent[4]

        # current rays for loop iteration, this rays are used in hit finding for the next section
        rs = np.ones(self.RaySources.N, dtype=bool) # gets updated at every iteration

        # section index rs for each ray for section before z
        # rs2 == -1 can mean mask is true everywhere (ray starts after surface.minz) or false everywhere (rays don't reach surface),
        # so we need to check which case we're in. In the later case we don't need to calculate ray hits.
        mask = z <= self.RaySources.p_list[:, :, 2]
        rs2 = np.argmax(mask, axis=1) - 1
        mask2 = np.all(mask, axis=1) & (rs2 < 0)
        rs2[mask2] = 0

        p, s, _, w, wl, _ = self.RaySources.getRaysByMask(rs, rs2, normalize=True, 
                                                            ret=[True, True, False, True, True, False])

        # init ph (position of hit) and is_hit bool array
        ph = np.zeros_like(p, dtype=np.float64)
        ish = np.zeros_like(rs, dtype=bool)

        rs[rs2 < 0] = False  # section index rs < 0 means ray does not reach that far
        rs2 = rs2[rs2 >= 0]
        # weights[~rs] = 0  # not needed rs < 0 means ray is already absorbed
        p, s = p[rs], s[rs]

        while np.any(rs):

            rs2 += 1  # increment next-section-indices
            inv = rs2 >= self.RaySources.nt  # invalid-array, indices are above section count
            # these rays have no intersection, since they end at the last outline surface

            if np.any(inv):
                rsi, _ = misc.partMask(rs, inv)  # current rays that are not valid 
                rs, val = misc.partMask(rs, ~inv)  # only use valid rays
                w[rsi] = 0  # set invalid to weight of zero
                p, s, rs2 = p[val], s[val], rs2[val] # only use valid rays

            ph[rs], ish[rs] = self.findSurfaceHit(self.DetectorList[ind].Surface, p, s)

            p2z = self.RaySources.p_list[rs, rs2, 2]
            rs, rsn = misc.partMask(rs, ph[rs, 2] > p2z)

            if np.any(rs):
                rs2 = rs2[rsn]
                p, s, _, w[rs], _, _ = self.RaySources.getRaysByMask(rs, rs2, normalize=True,
                                                                     ret=[True, True, False, True, False, False])

        hitw = ish & (w > 0)
        ph, w, wl = ph[hitw], w[hitw], wl[hitw]

        if self.DetectorList[ind].Surface.isPlanar():
            extent_out = np.array(self.DetectorList[ind].extent[:4])
            image_type = "Cartesian"

        else:
            if (stype := self.DetectorList[ind].Surface.surface_type) not in ["Sphere", "Asphere"]:
                raise RuntimeError(f"Detector view not implemented for surface_type '{stype}'.")

            ph = self.DetectorList[ind].toAngleCoordinates(ph)
            extent_out = self.DetectorList[ind].getAngleExtent()
            image_type = "Polar"

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

        # init image and extent, these are the default values when no rays hit the detector
        Im = Image(z=self.DetectorList[ind].pos[2], extent=extent_out, image_type=image_type)
        Im.makeImage(N, ph, w, wl, threading=self.multithreading)

        return Im


    def iterativeDetectorImage(self,
                            N_rays:     int,
                            N_px:       int,
                            ind:        int = 0,
                            pos:        list = [],
                            silent:     bool = False,
                            extent:     (str | list | np.ndarray) = "whole")\
            -> list[Image]:
        """
        Raytrace with N_rays and render Detector Image.

        :param N_rays: number of rays (int)
        :param N_px: number of image pixels in each dimension (int)
        :param ind:
        :param pos: list of z coordinates to render the Detector Images, current position is used if the list is empty
        :param extent: image extent, either "whole" or 4 element numpy 1D array or list
        :return: list of XYZIL Images (each numpy 3D array of XYZ channels + Irradiance + Illuminance)
        """
        if not isinstance(extent, list) and not isinstance(extent, np.ndarray) and extent == "auto":
            raise ValueError("Image extent can't be 'auto' for multiple image render.")

        if (N_rays := int(N_rays)) <= 0:
            raise ValueError(f"Ray number N_rays needs to be a positive int, but is {N_rays}.")

        if (N_px := int(N_px)) <= 0:
            raise ValueError(f"Pixel number N_px needs to be a positive int, but is {N_px}.")

        if not self.DetectorList:
            raise RuntimeError("Detector missing.")

        # use current detector position if pos is empty
        if not len(pos):
            pos = [self.DetectorList[ind].pos[2]]

        rays = 0
        rays_step = self.ITER_RAYS_STEP
        iterations = int(N_rays / rays_step)
        diff = int(N_rays - iterations*rays_step) # remaining rays for last iteration
        extra = diff > 0 # if there is a last iteration

        # turn off messages for raytracing iterations
        silent_old = self.silent
        self.silent = True

        # image list
        Im_res = []

        time_start = time.time()

        # for all render iterations
        for i in np.arange(iterations+extra):

            # only true in extra step
            if i == iterations:
                rays_step = diff

            self.trace(N=rays_step)

            # for all detector positions
            for j in np.arange(len(pos)):
                pos_new = np.concatenate((self.DetectorList[ind].pos[:2], [pos[j]]))
                self.DetectorList[ind].moveTo(pos_new)
                Imi = self.DetectorImage(N=N_px, ind=ind, extent=extent)
                
                # append image to list in first iteration, after that just add image content
                if i == 0:
                    Im_res.append(Imi)
                else:
                    Im_res[j].Im += rays_step/N_rays * Imi.Im

            # print render percentage and remaining time
            rays += rays_step
            time_remaining = (time.time() - time_start) * (N_rays - rays) / rays

            if not silent:
                print(f"Rendered {100*rays/N_rays:.1f}%, Estimated time remaining:", 
                        time.strftime('%H:%M:%S', time.gmtime(int(time_remaining))))

        if not silent:
            print("Finished Rendering.")
       
        # revert silent to its state
        self.silent = silent_old

        return Im_res

    def SourceImage(self,
                    N:      int,
                    sindex: int = 0) \
            -> Image:
        """
        Rendered Image of RaySource. Rays were already traced.

        :param N: number of image pixels in each dimension (int)
        :param sindex:
        :return: XYZIL Image (XYZ channels + Irradiance + Illuminance in third dimension) (numpy 3D array)
        """

        if not self.RaySources.hasSources():
            raise RuntimeError("RaySource Missing")

        if (N := int(N)) <= 0:
            raise ValueError(f"Pixel number N needs to be a positive int, but is {N}.")

        extent = self.RaySources.List[sindex].extent[:4]
        p, _, _, w, wavelengths = self.RaySources.getSourceRays(sindex)

        Im = Image(z=self.RaySources.List[sindex].pos[2], extent=extent, image_type="Cartesian", index=sindex+1)
        Im.makeImage(N, p, w, wavelengths)

        return Im

    # cost function for optimization
    def autofocus_cost_func(self,
                            z_pos:  float,
                            mode:   str,
                            p:      np.ndarray,
                            s:      np.ndarray,
                            w:      np.ndarray,
                            r0:     float = 1e-3)\
            -> float:
        """

        :param z_pos:
        :param mode:
        :param p:
        :param s:
        :param w:
        :param r0:
        :return:
        """
        t = (z_pos - p[:, 2]) / s[:, 2] 
        ph = p + s*t[:, np.newaxis]

        x, y = ph[:, 0], ph[:, 1]

        if mode == "Airy Disc Weighting":
            expr = ne.evaluate("w * exp(-0.5*(x**2 + y**2)/ (0.42*r0)**2)")
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

            N_px = 201
            extent = np.min(x), np.max(x), np.min(y), np.max(y)

            # get hit pixel coordinate, subtract 1e-12 from N so all values are 0 <= xcor < N
            xcor = (N_px - 1e-12) / (extent[1] - extent[0]) * (x - extent[0]) 
            ycor = (N_px - 1e-12) / (extent[3] - extent[2]) * (y - extent[2])  

            # image can be 1D for standard deviation, speed things up a little bit
            ind = N_px*ycor.astype(int) + xcor.astype(int)

            Im = np.zeros(N_px**2, dtype=np.float64)
            np.add.at(Im, ind, w)

            return 1/np.std(Im[Im > 0])

        else:
            raise ValueError(f"Invalid Autofocus Mode '{mode}'.")

    # TODO show information about search range?
    # TODO use source_index to focus for only one source
    def autofocus(self,
                  method,
                  z_start:      float,
                  N:            int = 100000,
                  ret_cost:     bool = True)\
            -> tuple[float, np.ndarray, np.ndarray]:
        """
        Find the focal point using different methods. z_start defines the starting point, 
        the search range is the region between lenses or the outline.
        The influence of filters is neglected.

        :param z_start: starting position z (float)
        :param N: maximum number of rays to evaluate (int)
        :param method:
        :param ret_cost: False, if costly calculation of cost function array can be skipped in mode "Position Variance". 
                In other modes it is generated on the way anyway
        :return: position of focus (float)
        """

        if not self.RaySources.hasSources():
            raise RuntimeError("RaySource Missing.")

        if not (self.outline[4] <= z_start <= self.outline[5]):
            raise ValueError("Starting position z_start outside outline.")

        # get search bounds
        ########################################################################################################

        # sort list in z order
        Lenses = sorted(self.LensList, key=lambda Element: Element.pos[2])
        
        # no lenses
        if not Lenses:
            n_ambient = self.n0(550)
            bounds = [self.outline[4], self.outline[5]]

        # region lies before the first lens
        elif z_start < Lenses[0].pos[2]:
            n_ambient = self.n0(550)
            bounds = [self.outline[4], Lenses[0].extent[4]]

            # start position needs to be behind all raysources
            for RS in self.RaySources.List:
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
        for F in self.FilterList:
            if bounds[0] <= F.pos[2] <= bounds[1]:
                warnings.warn("WARNING: The influence of the filters in the autofocus range will be ignored. ")

        # get rays and properties
        ########################################################################################################
        
        _, p, s, _, weights, _, _ = self.RaySources.getRaysAtZ(bounds[0]+self.EPS, normalize=True)

        # use only rays with weight
        hw = weights > 0
        p, s, weights = p[hw], s[hw], weights[hw]

        # select rays
        ########################################################################################################
        
        # get number of rays with weight
        N_is = p.shape[0]

        # use only selection
        if N_is > N:  # use subset if number is above N
            ch = np.random.choice(N_is, size=N, replace=False)
            p, s, weights = p[ch], s[ch], weights[ch]

        elif N_is == 0:  # throw error when no rays are present
            raise RuntimeError("No rays found for autofocus")

        else:
            pass # just use all rays


        # find focus
        ########################################################################################################
        
        # all methods start with Position Variance mode
        # find minimum for position variance
        res = scipy.optimize.minimize_scalar(self.autofocus_cost_func, 
                                             args=("Position Variance", p, s, weights),
                                             options={'maxiter': 500, 'xatol':1e-6},
                                             bounds=bounds, method='Bounded')

        db = (bounds[1] - bounds[0])/8  # use result of Position Variance mode
        rs = max(res.x-db, bounds[0])  # make sure it's inside bounds
        re = min(res.x+db, bounds[1])  # make sure it's inside bounds


        if method == "Airy Disc Weighting":
            # calculate size of airy disc for method Airy Disc Weighting 
            s_z = np.nanquantile(s[:, 2], 0.95)
            sin_alpha = np.sin(np.arccos(s_z))  # neglect outliers

            # size of airy disc, default to 1µm when sin_alpha = 0
            r0 = 0.61 * 550e-6 / (sin_alpha * n_ambient) if sin_alpha != 0 else 1e-3 
        else:
            r0 = 1e-3

        if method != "Position Variance" or ret_cost:
            # sample smaller region around minimum with proper method
            Nt = 150
            r = np.linspace(rs, re, Nt)
            vals = [self.autofocus_cost_func(rs, method, p, s, weights, r0=r0) for rs in r]
        else:
            r = None
            vals = None

        # find minimum for other methods, since they are susceptible to local minimums
        if method in ["Airy Disc Weighting", "Irradiance Variance"]:
            # # start search at minium of sampled data
            pos = np.argmin(vals)
            cost_func2 = lambda z, method: self.autofocus_cost_func(z[0], method, p, s, weights, r0=r0)
            res = scipy.optimize.minimize(cost_func2, r[pos], args=method, tol=None, callback=None,
                                          options={'maxiter': 300}, bounds=[bounds])

        # debug plots
        ########################################################################################################

        # return position of minimum
        return res.x, r, vals

