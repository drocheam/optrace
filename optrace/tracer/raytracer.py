
from typing import Any, assert_never
from threading import Thread  # threading
from enum import IntEnum  # integer enum

import numpy as np  # calculations
import scipy.optimize  # numerical optimization methods

# needed for raytracing geometry and functionality
from .geometry import Filter, Aperture, Detector, Lens, RaySource, Surface,\
                      RectangularSurface, Line, Point, SphericalSurface, Group, RingSurface, SlitSurface

from .spectrum import LightSpectrum
from .refraction_index import RefractionIndex
from .ray_storage import RayStorage
from .image.render_image import RenderImage

from ..global_options import global_options
from ..warnings import warning

from . import random
from ..progress_bar import ProgressBar
from . import misc  # calculations
from ..property_checker import PropertyChecker as pc  # check types and values


class Raytracer(Group):

    N_EPS: float = 1e-11
    """ numerical epsilon. Used for floating number comparisions in some places or adding small differences """

    T_TH: float = 1e-5
    """ threshold for the transmission Filter
    values below this are handled as absorbed
    needed to avoid ghost rays, meaning rays that have a non-zero, but negligible power"""

    HURB_FACTOR: float = 2**0.5
    """HURB uncertainty factor. Scales the tangent uncertainty expression. Original sources use 1, 
    but sqrt(2) matches the envelope better. See the documentation for details."""

    MAX_RAY_STORAGE_RAM: int = 8000000000
    """ Maximum available RAM for the stored rays """

    ITER_RAYS_STEP: int = 1000000
    """ number of rays per iteration in Raytracer.iterative_render()"""

    class INFOS(IntEnum):
        ABSORB_MISSING = 0
        TIR = 1
        T_BELOW_TTH = 2
        ILL_COND = 3
        OUTLINE_INTERSECTION = 4
        HURB_NEG_DIR = 5
    """enum for info messages in raytracing"""

    focus_search_methods: list[str, str, str] = ['RMS Spot Size', 'Irradiance Variance',
                                                 'Image Sharpness', 'Image Center Sharpness']
    """available focus_search methods"""

    def __init__(self,
                 outline:        (list | np.ndarray),
                 n0:              RefractionIndex = None,
                 no_pol:          bool = False,
                 use_hurb:        bool = False,
                 **kwargs)\
            -> None:
        """
        Initialize the raytracer

        :param outline: outline of raytracer space [x1, x2, y1, y2, z1, z2] (numpy 1D array or list)
        :param n0: refraction index of the raytracer environment (RefractionIndex object)
        :param no_pol: if polarization should be neglected to speed things up
        """

        self.outline = outline  #: geometrical raytracer outline
        self.no_pol = no_pol  #: polarization calculation flag
        self.use_hurb = use_hurb  #: flag for using Heisenberg uncertainty ray bending on apertures

        self.rays = RayStorage()  #: traced rays
        self._msgs = np.array([])
        self._ignore_geometry_error = False
        self.geometry_error = False  #: if geometry checks returned an error
        self._last_trace_snapshot = None
        self.fault_pos = np.array([])

        super().__init__(None, n0, **kwargs)
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

            case "no_pol":
                pc.check_type(key, val, bool)
            
            case "use_hurb":
                pc.check_type(key, val, bool)

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

    def property_snapshot(self) -> dict:
        """
        Creates a snapshot of properties of Elements and rays.
        Needed to detect changes.

        :return: dictionary of properties
        """
        return self.tracing_snapshot() |\
            dict(Markers=[D.crepr() for D in self.markers],
                 Volumes=[D.crepr() for D in self.volumes],
                 Detectors=[D.crepr() for D in self.detectors])

    def tracing_snapshot(self) -> dict:
        """
        Creates a snapshot of properties of rays and Elements relevant for tracing.
        Needed to detect changes since the last trace.

        :return: dictionary of properties
        """
        return dict(Rays=self.rays.crepr(),
                    Ambient=[tuple(self.outline), self.n0.crepr()],
                    TraceSettings=[self.no_pol, self.use_hurb, self.T_TH, self.HURB_FACTOR],
                    Lenses=[D.crepr() for D in self.lenses],
                    Filters=[D.crepr() for D in self.filters],
                    Apertures=[D.crepr() for D in self.apertures],
                    RaySources=[D.crepr() for D in self.ray_sources])

    def compare_property_snapshot(self, h1: dict, h2: dict) -> dict:
        """
        Compare two snapshots of property_snapshot and detect changes

        :param h1: snapshot 1
        :param h2: snapshot 2
        :return: dictionary of changes
        """

        diff = {key: h1[key] != h2[key] for key in h1.keys()}

        # Refraction Index Regions could have changed by Lens.n2, the ambient did therefore change
        diff["Ambient"] = diff["Ambient"] or diff["Lenses"]
        diff["Any"] = any(val for val in diff.values())

        return diff

    def check_if_rays_are_current(self) -> bool:
        """Check if anything tracing relevant changed since the last trace."""
        if self._last_trace_snapshot is None:
            return False

        now = self.tracing_snapshot()
        return not self.compare_property_snapshot(self._last_trace_snapshot, now)["Any"]

    def _set_messages(self, msgs: list[np.ndarray]) -> None:
        """
        Apply messages from threads

        :param msgs: list of message arrays
        """
        # join messages from all threads
        self._msgs = np.zeros_like(msgs[0], dtype=int)
        for msg in msgs:
            self._msgs += msg

    def _surface_names(self) -> list[str]:
        """Generate a list of surface names ordered by their z-position"""

        names = dict()

        for type_, els in zip(["Lens", "Aperture", "Filter"], [self.lenses, self.apertures, self.filters]):
            for i, el in enumerate(els):
                if not el.has_back():
                    names[f"surface of {type_} {el.abbr}{i}"] = el.pos[2]
                else:
                    names[f"front surface of {type_} {el.abbr}{i}"] = el.front.pos[2]
                    names[f"back surface of {type_} {el.abbr}{i}"] = el.back.pos[2]

        return ["RaySource"] + sorted(names, key=lambda k: names[k]) + ["Outline"]

    def _show_messages(self, N) -> None:
        """
        Show messages from tracing

        :param N: number of rays
        """
        surf_name = self._surface_names()

        # print messages
        for type_ in range(self._msgs.shape[0]):
            for surf in range(self._msgs.shape[1]):
                if count := self._msgs[type_, surf]:
                    match type_:
                        case self.INFOS.TIR:
                            warning(f"{count} rays ({100*count/N:.3g}% of all rays) "
                                    f"with total inner reflection at surface {surf} ({surf_name[surf]}),"
                                    " treating as absorbed.")

                        case self.INFOS.ABSORB_MISSING:
                            warning(f"{count} rays ({100*count/N:.3g}% of all rays) "
                                    f"missing lens surface {surf} ({surf_name[surf]}), set to absorbed")

                        case self.INFOS.T_BELOW_TTH:
                            warning(f"{count} rays ({100*count/N:.3g}% of all rays) "
                                    f"with transmittivity at filter surface {surf}  ({surf_name[surf]}) below threshold"
                                    f" of {self.T_TH*100:.3g}%, setting to absorbed.")
                        
                        case self.INFOS.ILL_COND:
                            warning(f"{count} rays ({100*count/N:.3g}% of all rays) are ill-conditioned for "
                                    f"numerical hit finding at surface {surf} ({surf_name[surf]}). "
                                    f"Where and whether they intersect might be wrong.")
                        
                        case self.INFOS.OUTLINE_INTERSECTION:
                            warning(f"{count} rays ({100*count/N:.3g}% of all rays) "
                                    f"hitting outline after surface {surf} ({surf_name[surf]}), set to absorbed.")
                        
                        case self.INFOS.HURB_NEG_DIR:
                            warning(f"{count} rays ({100*count/N:.3g}% of all rays) "
                                    f"have negative z-direction after ray bending at "
                                    f"surface {surf} ({surf_name[surf]}), set to absorbed.")
                        
                        case _:
                            assert_never(type_)

    def _pretrace_check(self, N: int) -> bool:
        """checks the geometry and N parameter. Returns if tracing is possible"""

        pc.check_type("N", N, int)

        if N < 1:
            raise ValueError(f"Ray number N needs to be at least 1, but is {N}.")

        # make element list and check geometry
        self.__geometry_checks()
        if self.geometry_error and not self._ignore_geometry_error:
            warning("ABORTED TRACING")
            return True

        return False

    def trace(self, N: int) -> None:
        """
        Execute raytracing for the current geometry

        Shows a warning on geometry errors, but does not throw an exception.
        This is useful in the TraceGUI, as the broken geometry can be displayed.

        :param N: number of rays (int)
        """
        if (dont_trace := self._pretrace_check(N)):
            return
        
        elements = self.__tracing_elements()

        # reserve space for all tracing surface intersections, +1 for invisible aperture at the outline z-end
        # and +1 for the ray starting points
        nt = len(self.tracing_surfaces) + 2
       
        if self.rays.storage_size(N, nt, self.rays.no_pol) > self.MAX_RAY_STORAGE_RAM:
            raise RuntimeError(f"More than {self.MAX_RAY_STORAGE_RAM*1e-9:.1f} GB RAM requested. Either decrease"
                                " the number of rays, surfaces or do an iterative render. If your system can handle"
                                " more RAM usage, increase the Raytracer.MAX_RAY_STORAGE_RAM parameter.")

        cores = misc.cpu_count()
        N_threads = max(1, min(cores, int(N/30000))) if global_options.multithreading else 1

        # will hold info messages from each thread
        msgs = [np.zeros((len(self.INFOS), nt), dtype=int) for i in range(N_threads)]

        # start a progress bar
        bar = ProgressBar("Raytracing: ", nt)

        # create rays from RaySources
        self.rays.init(self.ray_sources, N, nt, self.no_pol)

        def sub_trace(N_threads: int, N_t: int) -> None:

            p, s, pols, weights, wavelengths, ns = self.rays.thread_rays(N_threads, N_t)
            bar.update(N_t == 0)

            msg = msgs[N_t]
            i = 0  # surface counter
            n1 = self.n0  # ambient medium before all elements
            ns[:, 0] = n1(wavelengths)  # assign refractive index

            for en, element in enumerate(elements):

                p[:, i+1], pols[:, i+1], weights[:, i+1] = p[:, i], pols[:, i], weights[:, i]
                hw = weights[:, i] > 0

                if isinstance(element, Lens):

                    p[hw, i+1], hit_front, ill = element.front.find_hit(p[hw, i], s[hw])
                    hwh = misc.masked_assign(hw, hit_front)  # rays having power and hitting lens front
                    msg[self.INFOS.ILL_COND, i+1] += np.count_nonzero(ill)

                    # absorb rays not hitting surface
                    miss_mask = misc.masked_assign(hw, ~hit_front)
                    weights[miss_mask, i+1] = 0
                    msg[self.INFOS.ABSORB_MISSING, i+1] += np.count_nonzero(miss_mask)

                    # index after lens
                    n2 = element.n2 or self.n0
                    n2_l = n2(wavelengths)
                    
                    if not element.is_ideal:
                   
                        n1_l = n1(wavelengths)
                        n_l = element.n(wavelengths)
                        
                        self.__refraction(element.front, p, s, weights, n1_l, n_l, pols, hwh, i, msg)
                        
                        # treat rays that go outside outline
                        hwnh = misc.masked_assign(hw, ~hit_front)  # rays having power and not hitting lens front
                        self.__outline_intersection(p, s, weights, hwnh, i, msg)

                        i += 1
                        bar.update(N_t == 0)
                        p[:, i+1], pols[:, i+1], weights[:, i+1] = p[:, i], pols[:, i], weights[:, i]
                        ns[:, i], ns[:, i+1] = n_l, n2_l

                        hw = weights[:, i] > 0
                        p[hw, i+1], hit_back, ill = element.back.find_hit(p[hw, i], s[hw])
                        msg[self.INFOS.ILL_COND, i+1] += np.count_nonzero(ill)
                    
                        # absorb rays not hitting surface
                        miss_mask = misc.masked_assign(hw, ~hit_back)
                        weights[miss_mask, i+1] = 0
                        p[miss_mask, i+1] = p[miss_mask, i]  # rays should be absorbed at front surface of lens
                        msg[self.INFOS.ABSORB_MISSING, i+1] += np.count_nonzero(miss_mask)
                   
                        hwb = misc.masked_assign(hw, hit_back)  # rays having power and hitting lens back
                        self.__refraction(element.back, p, s, weights, n_l, n2_l, pols, hwb, i, msg)

                    else:
                        self.__refraction_ideal_lens(element.front, element.D, p, s, pols, hwh, i, msg)
                        ns[:, i+1] = n2_l
                        hit_back = hit_front
                    
                    # # treat rays that go outside outline
                    hwnb = misc.masked_assign(hw, ~hit_back)  # rays having power and not hitting lens back
                    self.__outline_intersection(p, s, weights, hwnb, i, msg)

                    # index after lens is index before lens in the next iteration
                    n1 = n2

                elif isinstance(element, Filter | Aperture):

                    p[hw, i+1], hit, ill = element.surface.find_hit(p[hw, i], s[hw])
                    msg[self.INFOS.ILL_COND, i+1] += np.count_nonzero(ill)
                    hwh = misc.masked_assign(hw, hit)  # rays having power and hitting filter
                    hwnh = misc.masked_assign(hw, ~hit)  # rays having power and not hitting filter

                    if isinstance(element, Filter):
                        self.__filter(element, weights, wavelengths, hwh, i, msg)
                    else:
                        weights[hwh, i+1] = 0

                        # use ray bending if activated, but ignore last surface that acts as absorber
                        if self.use_hurb and en != len(elements) - 1:
                            self.__hurb(p, s, pols, weights, wavelengths, ns, hwnh, i, element.surface, msg)

                    # treat rays that go outside outline
                    self.__outline_intersection(p, s, weights, hwnh, i, msg)
                    
                    ns[:, i+1] = ns[:, i]

                else:
                    assert_never(element)

                i += 1
                bar.update(N_t == 0)

        if N_threads > 1:
            thread_list = [Thread(target=sub_trace, args=(N_threads, N_t)) for N_t in np.arange(N_threads)]

            [thread.start() for thread in thread_list]
            [thread.join() for thread in thread_list]
        else:
            sub_trace(1, 0)

        # lock Storage
        self.rays.lock()

        bar.finish()
        self._set_messages(msgs)
        self._show_messages(N)
        
        # store settings for this trace
        self._last_trace_snapshot = self.tracing_snapshot()
                        

    def __hurb(self, 
               p:       np.ndarray, 
               s:       np.ndarray, 
               pols:    np.ndarray,
               weights: np.ndarray, 
               wl:      np.ndarray, 
               ns:      np.ndarray, 
               hwnh:    np.ndarray, 
               i:       int, 
               surf:    Surface, 
               msg:     np.ndarray)\
            -> None:
        """
        Approximate aperture diffraction asymptotically using HURB (Heisenberg Uncertainty Ray Bending),
        see Edge diffraction in Monte Carlo ray tracing Edward R. Freniere, G. Groot Gregory, and Richard A. Hassler

        :param p: position array
        :param s: direction array
        :param pols: ray polarizations
        :param weights: weight array
        :param wl: wavelength array
        :param ns: refractive index array
        :param hwnh: ray having an power and not hitting the surface
        :param i: surface number
        :param surf: aperture surface
        :param msg: message array
        """
        # get bending angles, axes and mask
        a_, b_, b, inside = surf.hurb_props(p[hwnh, i+1, 0], p[hwnh, i+1, 1])
        hwnhi = misc.masked_assign(hwnh, inside)  
        # ^-- rays having an power, not hitting the aperture, but inside of inner, ray bending aperture part
      
        # a is just b rotated by 90° in aperture plane
        a = b.copy()
        a[:, 0] = -b[:, 1]
        a[:, 1] = b[:, 0]

        # a tilted ray sees a projected aperture. The distance to the a-edge is smaller by cos(psi_a),
        # where psi_a = 90 - ang_a, with ang_a being the angle between the ray and the ellipsis axis a
        # scalar product s * a is cos(ang_a), so sin(ang_a) = sqrt(1 - cos^2(ang_a)) is cos(psi_a)
        cos_psi_a = np.sqrt(1 - misc.rdot(s[hwnhi], a)**2)
        cos_psi_b = np.sqrt(1 - misc.rdot(s[hwnhi], b)**2)

        # # std dev of direction Gaussian
        # # see Edge diffraction in Monte Carlo ray tracing Edward R. Freniere, G. Groot Gregory, and Richard A. Hassler
        # # additionally, the refractive index, cos-dependency from a above and a custom uncertainty factor were included
        k = 2*np.pi*ns[hwnhi, i] / (wl[hwnhi] * 1e-9)
        tan_sig_b = self.HURB_FACTOR / (2*b_*cos_psi_b*1e-3*k)
        tan_sig_a = self.HURB_FACTOR / (2*a_*cos_psi_a*1e-3*k)

        # # generated normally distributed tan(angles)
        tan_tha = np.random.normal(scale=tan_sig_a, size=a_.shape[0])
        tan_thb = np.random.normal(scale=tan_sig_b, size=b_.shape[0])

        # create sa, sb components
        # sa lies in plane containing s and a, but is orthogonal to s and sb
        # sb lies in plane containing s and b, but is orthogonal to s and sa
        sa = misc.cross(b, s[hwnhi])
        sa = misc.normalize(sa)
        sb = misc.cross(s[hwnhi], sa)
        
        # construct new direction from old one and sb, sa components
        sab = s[hwnhi] + sa*tan_tha[:, np.newaxis] + sb*tan_thb[:, np.newaxis]
        s0 = s.copy()
        s[hwnhi] = misc.normalize(sab)

        # absorb rays with negative direction after bending
        neg = s[:, 2] < 0
        weights[neg, i+1] = 0
        msg[self.INFOS.HURB_NEG_DIR, i+1] += np.count_nonzero(neg)
        
        # recalculate polarization
        n = np.broadcast_to([0., 0., 1.], (s0.shape[0], 3))
        self.__compute_polarization(s0, s[hwnhi], n, pols, i, hwnhi)

    def __tracing_elements(self) -> list[Lens | Filter | Aperture]:
        """
        Creates a sorted element list from filters, apertures and lenses.

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

    def __geometry_checks(self) -> None:
        """
        Checks geometry in raytracer for errors.
        Markers and Detectors can be outside the tracing outline, Filters, Apertures, Lenses and Sources not

        :param elements: element list from __make_element_list()
        """

        elements = self.__tracing_elements()

        def is_inside(e: tuple | list) -> bool:
            o = self.outline + self.N_EPS*np.array([-1, 1, -1, 1, -1, 1])  # add eps because of finite float precision
            return o[0] <= e[0] and e[1] <= o[1] and o[2] <= e[2] and e[3] <= o[3] and o[4] <= e[4] and e[5] <= o[5]

        if not self.ray_sources:
            warning("RaySource Missing.")
            self.geometry_error = True
            return

        coll = False
        for i, el in enumerate(elements):
            if not is_inside(el.extent):
                warning(f"Element{i} {el} with extent {el.extent} outside outline {self.outline}.")
                self.geometry_error = True
                return

            # collision of front and next front
            if i + 1 < len(elements):
                coll, xc, yc, zc = self.check_collision(el.front, elements[i + 1].front)
            
            # collision of front and back of element
            if not coll and el.has_back():
                coll, xc, yc, zc = self.check_collision(el.front, el.back)

            # collision of back and next front
            if not coll and el.has_back():
                coll, xc, yc, zc = self.check_collision(el.back, elements[i + 1].front)

            if self.use_hurb and i < len(elements) - 1 and isinstance(el, Aperture):
                if not (isinstance(el.front, RingSurface) or isinstance(el.front, SlitSurface)):
                    warning(f"Ray bending for surface type {type(el.front).__name__} not implemented.")
                    self.geometry_error = True
                    return

            if coll:
                break

        if not coll:
            for rs in self.ray_sources:

                if not is_inside(rs.extent):
                    warning(f"RaySource {rs} with extent {rs.extent} outside outline {self.outline}.")
                    self.geometry_error = True
                    return

                if rs.pos[2] >= elements[0].extent[4]:
                    coll, xc, yc, zc = self.check_collision(rs.surface, elements[0].front)

                if coll:
                    break

        if coll:
            warning(f"Detected collision between two Surfaces at {xc[0], yc[0], zc[0]}"
                    f" and at least {xc.shape[0]} other positions.")
            self.geometry_error = True
            self.fault_pos = np.column_stack((xc, yc, zc))
            return

        self.geometry_error = False

    @staticmethod
    def check_collision(front: Surface | Line | Point, back: Surface | Line | Point, res: int = 100)\
            -> tuple[bool, np.ndarray, np.ndarray, np.ndarray]:
        """
        Check for surface/point/line collisions.
        A collision is defined as the front surface havin a higher z-value than the back surface,
        at a point where both surfaces are defined

        :param front: first object (in regards to z-position)
        :param back: second object
        :param res: resolution measure, increase for better detection
        :return: bool value if there is a collision, collision x-value array,
            collision y-value array, collision z-value array
        """

        # we only compare when at least one object is a surface
        if not (isinstance(front, Surface) or isinstance(back, Surface)):
            raise TypeError(f"At least one object needs to be a Surface for collision detection")
        
        # check if point and surface hit. Basically if order of surface and point parameter is correct
        elif isinstance(front, Point) or isinstance(back, Point):
            rev, pt, surf = (False, front, back) if isinstance(front, Point) else (True, back, front)

            # check value at surface
            x, y = np.array([pt.pos[0]]), np.array([pt.pos[1]])
            z = surf.values(x, y)

            # check if hitting, surface needs to be defined at this point
            hit = (z < pt.pos[2]) if not rev else (z > pt.pos[2])
            hit = hit & surf.mask(x, y)
            where = np.where(hit)[0]
            return np.any(hit), x[where], y[where], z[where]

        # intersection of surface and line
        elif isinstance(front, Line) or isinstance(back, Line):
            rev, line, surf = (False, front, back) if isinstance(front, Line) else (True, back, front)

            # some line x, y values
            t = np.linspace(-line.r, line.r, 10*res)
            x = line.pos[0] + np.cos(line.angle)*t
            y = line.pos[1] + np.sin(line.angle)*t
            z = surf.values(x, y)

            # check if hitting and order correct
            hit = (z < line.pos[2]) if not rev else (z > line.pos[2])
            hit = hit & surf.mask(x, y)
            where = np.where(hit)[0]
            return np.any(hit), x[where], y[where], z[where]

        # extent of front and back
        xsf, xef, ysf, yef, zsf, zef = front.extent
        xsb, xeb, ysb, yeb, zsb, zeb = back.extent

        # no overlap of z extents -> no collision
        if zef < zsb:
            return False, np.array([]), np.array([]), np.array([])

        # get rectangular overlap area in xy-plane projection
        xs = max(xsf, xsb)
        xe = min(xef, xeb)
        ys = max(ysf, ysb)
        ye = min(yef, yeb)

        # no overlap in xy plane projection -> no collision
        if xs > xe or ys > ye:
            return False, np.array([]), np.array([]), np.array([])

        # grid for overlap area
        Y, X = np.mgrid[ys:ye:res*1j, xs:xe:res*1j]

        # sample surface mask
        x2, y2 = X.flatten(), Y.flatten()
        valid = front.mask(x2, y2) & back.mask(x2, y2)

        # sample valid surface values
        x2v, y2v = x2[valid], y2[valid]
        zfv = front.values(x2v, y2v)
        zbv = back.values(x2v, y2v)

        # check for collisions
        coll = zfv > zbv
        where = np.where(coll)[0]

        # return flag and collision samples
        return np.any(coll), x2v[where], y2v[where], zfv[where]

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
        :param hw: bool array for rays still having a weight
        :param i: surface/section number
        :param msg: message array
        """

        if not np.any(hw):
            return

        # check if examine points pe are inside outlines
        xs, xe, ys, ye, zs, ze = self.outline
        x, y, z = p[hw, i+1, 0], p[hw, i+1, 1], p[hw, i+1, 2]
        inside = (xs < x) & (x < xe) & (ys < y) & (y < ye) & (zs < z) & (z < ze)

        # number of rays going outside
        if (n_out := np.count_nonzero(~inside)):

            hwi = misc.masked_assign(hw, ~inside)

            OT = np.broadcast_to(self.outline, (n_out, 6)) # outline for every outside ray
            P = p[hwi, i].repeat(2).reshape(n_out, 6)  # repeat each column once
            S = s[hwi].repeat(2).reshape(n_out, 6)  # repeat each column once

            # calculate t Parameter for every outline coordinate and ray
            with np.errstate(divide="ignore"):
                T_arr = (OT-P) /  S

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
        Calculate polarization and direction for refraction at an ideal lens.

        :param surface: lens surface
        :param D: optical power of the ideal lens
        :param p: absolute position array
        :param s;  direction vector array
        :param pols: polarization array
        :param hwh: have-weight-and-hit-surface boolean array
        :param i: surface number
        :param msg: message array
        """
       
        # check the documentation for the math on this

        # copy, needed later
        s0 = s.copy()
        
        # position relative to lens center
        x = p[hwh, i+1, 0] - surface.pos[0]
        y = p[hwh, i+1, 1] - surface.pos[1]

        # helper variables
        f = 1000 / D
        fsz = f / s0[hwh, 2]

        # calculate and normalize new vector
        s[hwh, 0] = s0[hwh, 0] * fsz - x
        s[hwh, 1] = s0[hwh, 1] * fsz - y
        s[hwh, 2] = f
        s[hwh] = misc.normalize(s[hwh]) * np.sign(f)  # sign so it points always in +z

        # calculate polarization
        n = np.broadcast_to([0., 0., 1.], (x.shape[0], 3))
        self.__compute_polarization(s0, s[hwh], n, pols, i, hwh)

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
        :param weights: ray weights
        :param n1: refraction indices prior surface (numpy 1D array)
        :param n2: refraction indices after surface (numpy 1D array)
        :param pols: polarizations of the input ray
        :param hwh: boolean array for rays having a weight and hitting the lens
        :param i: number of the surface
        :param msg: message array
        """

        if not np.any(hwh):
            return

        n = surface.normals(p[hwh, i + 1, 0], p[hwh, i + 1, 1])

        n1_h = n1[hwh]
        n2_h = n2[hwh]
        s_h = s[hwh]

        # vectorial Snell's Law as in "Optik - Physikalisch-technische Grundlagen und Anwendungen,
        # Heinz Haferkorn, Auflage 4, Wiley-VCH, 2008", p.43
        # but with s_ = s', n1 = n, n2 = n', alpha = ε, beta = ε'

        ns = misc.rdot(n, s_h)  # equals cos(alpha)
        N = n1_h/n2_h  # ray wise refraction index quotient

        # rays with TIR mean an imaginary W, we'll handle this later
        with np.errstate(invalid="ignore"):
            W = np.sqrt(1 - N**2 * (1-ns**2))
            s_ = s_h*N[:, np.newaxis] - n*(N*ns - W)[:, np.newaxis]

        # rotate polarization vectors and calculate amplitude components in s and p plane
        A_ts, A_tp = self.__compute_polarization(s, s_, n, pols, i, hwh)

        # reflection coefficients for non-magnetic (µ_r=1) and non-absorbing materials (κ=0)
        # according to the Fresnel equations
        # see https://de.wikipedia.org/wiki/Fresnelsche_Formeln#Spezialfall:_gleiche_magnetische_Permeabilit.C3.A4t
        #####
        cos_alpha, cos_beta = ns, W
        n1_h_cos_alpha = n1_h*cos_alpha
        n2_h_cos_beta = n2_h*cos_beta
        ts = 2 * n1_h_cos_alpha / (n1_h_cos_alpha + n2_h_cos_beta)
        tp = 2 * n1_h_cos_alpha / (n2_h*cos_alpha + n1_h*cos_beta)
        T = n2_h_cos_beta / (n1_h_cos_alpha) * ((A_ts*ts)**2 + (A_tp*tp)**2)

        # handle rays with total internal reflection
        TIR = ~np.isfinite(W)
        if np.any(TIR):
            T[TIR] = 0
            TIR_count = np.count_nonzero(TIR)
            msg[self.INFOS.TIR, i] += TIR_count

        weights[hwh, i+1] = weights[hwh, i]*T
        s[hwh] = s_

    def __compute_polarization(self,
                              s:        np.ndarray,
                              s_:       np.ndarray,
                              n:        np.ndarray,
                              pols:     np.ndarray,
                              i:        int,
                              hwh:      np.ndarray)\
            -> tuple[np.ndarray | float, np.ndarray | float]:
        """
        compute new polarization vectors and calculate amplitude components in s and p plane

        :param s:  direction vectors before
        :param s_: direction vectors after (hwh mask alreay applied)
        :param n: surface normal
        :param pols: initial polarization
        :param i: surface number
        :param hwh: have-weight-and-hit-surface boolean array
        :return: amplitude components in s and p polarization direction
        """
        # no polarization -> don't set anything and return equal amplitude components
        if self.no_pol:
            return 1/np.sqrt(2), 1/np.sqrt(2)

        # exclude rays where direction did not change
        # these are the rays where p and s plane are not defined and the polarization stays the same
        mask = np.any(s[hwh] != s_, axis=1)
        mask2 = misc.masked_assign(hwh, mask)
        sm = s[mask2]

        # reduce slicing by storing separately
        polsm = pols[mask2, i]
        s_m = s_[mask]

        # ps is perpendicular to the plane of s and s_
        # pp is perpendicular to ps and s
        ps = misc.cross(s_m, sm)
        ps = misc.normalize(ps)
        pp = misc.cross(ps, sm)

        # init arrays
        # default for A_ts, A_tp are 1/sqrt(2)
        A_ts = np.full(s_.shape[0], 1/np.sqrt(2), dtype=np.float32)
        A_tp = np.full(s_.shape[0], 1/np.sqrt(2), dtype=np.float32)

        # A_ts is component of pol in ps
        A_ts[mask] = misc.rdot(ps, polsm)
        A_tp[mask] = misc.rdot(pp, polsm)

        # new polarization vector after refraction
        pp_ = misc.cross(ps, s_m)  # ps and s_m are unity vectors and perpendicular, so we need no normalization
        pols[mask2, i+1] = ps*A_ts[mask, np.newaxis] + pp_*A_tp[mask, np.newaxis]

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
        :param weights: ray weights
        :param wl: wavelength array (numpy 1D array)
        :param hwh: boolean array for rays having a weight and hitting the filter
        :param i: surface number
        :param msg: message array
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

    def _hit_detector(self,
                      info:              str,
                      detector_index:    int = 0,
                      source_index:      int = None,
                      extent:            list | np.ndarray = None,
                      projection_method: str = "Equidistant") \
            -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, ProgressBar, int]:
        """
        Internal function for calculating the detector intersection positions

        :param info: information about the detector
        :param detector_index: detector index
        :param source_index: ray source index, optional. Defaults to None, meaning all sources
        :param extent: extent to detect the intersections in as [x0, x1, y0, y1].
                Defaults to None, meaning the whole detector are is used
        :param projection_method: sphere projection method for a SphericalSurface detector
        :return: hit, weights, wavelengths, actual extent, actual projection_method, ProgressBar, 
            number of ill-conditioned rays
        """
        if not self.detectors:
            raise RuntimeError("Detector Missing")

        if not self.rays.N:
            raise RuntimeError("No rays traced.")

        if source_index is not None and (source_index > len(self.ray_sources) - 1 or source_index < 0):
            raise IndexError("Invalid source_index.")

        if detector_index > len(self.detectors) - 1 or detector_index < 0:
            raise IndexError("Invalid detector_index.")

        if not self.check_if_rays_are_current():
            raise RuntimeError("Tracing geometry/properties changed. Please retrace first.")

        bar = ProgressBar(f"{info}: ", 4)

        # range for selected rays in RayStorage
        Ns, Ne = self.rays.B_list[source_index:source_index + 2] if source_index is not None else (0, self.rays.N)

        dsurf = self.detectors[detector_index].surface

        def threaded(Ns, Ne, list_, list_i, bar):

            # current rays for loop iteration, this rays are used in hit finding for the next section
            rs = np.zeros(self.rays.N, dtype=bool)  # gets updated at every iteration
            rs[Ns:Ne] = True  # use rays of selected source

            # check if ray positions are behind, in front or inside detector extent
            bh_zmin = self.rays.p_list[Ns:Ne, :, 2] >= dsurf.extent[4]
            bh_zmax = self.rays.p_list[Ns:Ne, :, 2] >= dsurf.extent[5]
            
            # conclude if rays start after detector or don't reach that far
            no_start = np.all(bh_zmin & bh_zmax, axis=1)
            no_reach = np.all(~bh_zmin & ~bh_zmax, axis=1)
            ignore = no_start | no_reach

            # index for ray position before intersection
            rs2 = np.argmax(bh_zmin, axis=1) - 1
            rs2[rs2 < 0] = 0

            # mask for ill-conditioned rays
            ill_mask = np.zeros(rs2.shape[0], dtype=bool)

            # exclude rays that start behind detector or end before it
            rs2z = misc.masked_assign(rs, ignore)
            rs[rs2z] = False
            rs2 = rs2[~ignore]
            bar.update(not list_i)

            p, s, _, w, wl, _, _ = self.rays.rays_by_mask(rs, rs2, ret=[1, 1, 0, 1, 1, 0, 0], normalize=True)
            bar.update(not list_i)

            # init ph (position of hit) and is_hit bool array
            ph = np.zeros_like(p, dtype=np.float64, order='F')
            ish = np.zeros_like(wl, dtype=bool)
            rs3 = np.ones_like(wl, dtype=bool)

            # N: number of rays
            # Nt: number of rays in detector region and assigned to current thread
            # Ni: number of rays hitting the detector at the current surface in the current thread
            # Nn: number of rays in current thread that have not hit anything yet

            # rs: bool array with shape N, with true values being of number Nn
            # rs3: bool array with shape Nt, with true values being of number Nn
            # rs2: int array with shape Nn, holding the surface index for each ray
            # rs4: shape Nt, holds if the hit is valid

            while np.any(rs):

                rs2 += 1  # increment next-section-indices
                val = rs2 < self.rays.Nt  # valid-array, indices are below section count
                # these rays have no intersection with the detector, since they end at the last outline surface

                if not np.all(val):
                    rs = misc.masked_assign(rs, val)  # only use valid rays
                    rsi = misc.masked_assign(rs3, ~val)  # current rays that are not valid
                    rs3 = misc.masked_assign(rs3, val)  # current rays that are valid
                    w[rsi] = 0  # set invalid to weight of zero
                    p, s, rs2 = p[val], s[val], rs2[val]  # only use valid rays

                ph[rs3], ish[rs3], ill = dsurf.find_hit(p, s)  # find hit
                if np.any(ill):
                    ill_mask = ill_mask | ill  

                p2z = self.rays.p_list[rs, rs2, 2]
                rs4 = ph[rs3, 2] > p2z + dsurf.C_EPS  # hit point behind next surface intersection -> no valid hit
                rs3 = misc.masked_assign(rs3, rs4)
                rs = misc.masked_assign(rs, rs4)

                if np.any(rs):
                    rs2 = rs2[rs4]
                    p, s, _, w[rs3], _, _, _ = self.rays.rays_by_mask(rs, rs2, ret=[1, 1, 0, 1, 0, 0, 0])

            list_[list_i] = [ph, w, wl, ish, np.count_nonzero(ill_mask)]

        N_th = misc.cpu_count() if global_options.multithreading and Ne-Ns > 100000 else 1
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
        ill_count = sum([list__[4] for list__ in list_])

        hitw = ish & (w > 0)
        ph, w, wl = ph[hitw], w[hitw], wl[hitw]
        bar.update()

        if isinstance(dsurf, SphericalSurface) and projection_method is not None:
            ph = dsurf.sphere_projection(ph, projection_method)
            projection = projection_method
        else:
            projection = None

        # define the extent
        if isinstance(extent, list | np.ndarray):
            # only use rays inside extent area
            inside = (extent[0] <= ph[:, 0]) & (ph[:, 0] <= extent[1]) \
                    & (extent[2] <= ph[:, 1]) & (ph[:, 1] <= extent[3])

            extent_out = np.asarray_chkfinite(extent.copy(), dtype=np.float64)
            ph, w, wl = ph[inside], w[inside], wl[inside]

        elif extent is None:
            extent_out = self.detectors[detector_index].pos[:2].repeat(2)
            if np.any(hitw):
                extent_out[[0, 2]] = np.min(ph[:, :2], axis=0)
                extent_out[[1, 3]] = np.max(ph[:, :2], axis=0)

        else:
            raise ValueError(f"Invalid extent '{extent}'.")

        return ph, w, wl, extent_out, projection, bar, ill_count

    def detector_image(self,
                       detector_index:    int = 0,
                       source_index:      int = None,
                       extent:            list | np.ndarray = None,
                       limit:             float = None,
                       projection_method: str = "Equidistant",
                       **kwargs) -> RenderImage:
        """
        Render a detector image for a traced geometry

        :param detector_index: index/number of the detector
        :param source_index: index/number of the source. By default all sources are used.
        :param extent: rectangular extent [x0, x1, y0, y1] to detect to intersections in.
                By default the whole detector are is used.
        :param projection_method: sphere projection method for a SphericalSurface detector
        :param limit: resolution limit filter constant, see the documentation. Defaults to no filter.
        :param kwargs: keyword arguments for creating the RenderImage
        :return: rendered RenderImage
        """

        if limit is not None and extent is not None and not "_dont_filter" in kwargs:
            warning("Using the limit parameter in combination with a user defined extent"
                    " will produce an incorrect detector image, as the rays outside the extent"
                    " are not included in the convolution calculation.")

        p, w, wl, extent_out, projection, bar, ill_count\
                = self._hit_detector("Detector Image", detector_index, source_index, extent, projection_method)

        # create description
        detector = self.detectors[detector_index]
        pname = f": {detector.desc}" if detector.desc != "" else ""
        desc = f"{Detector.abbr}{detector_index}{pname} at z = {detector.pos[2]:.5g} mm"
        if source_index is not None:
            desc = f"Rays from RS{source_index} at " + desc
        
        # init image and extent, these are the default values when no rays hit the detector
        img = RenderImage(long_desc=desc, extent=extent_out, projection=projection)
        img.render(p, w, wl, limit=limit, **kwargs)
        bar.finish()

        if ill_count:
            warning(f"{ill_count} rays ({100*ill_count/self.rays.N:.3g}% of all rays) were ill-conditioned for "
                    f"numerical hit finding at detector {detector_index}. "
                     "Where and whether they intersect might be wrong.")

        return img

    def detector_spectrum(self,
                          detector_index:   int = 0,
                          source_index:     int = None,
                          extent:           list | np.ndarray = None,
                          **kwargs) -> LightSpectrum:
        """
        Render a detector spectrum for a traced geometry.

        :param detector_index: index/number of the detector
        :param source_index: index/number of the source. By default all sources are used.
        :param extent: rectangular extent [x0, x1, y0, y1] to detect to intersections in.
                By default the whole detector are is used.
        :param kwargs: optional keyword arguments for the created LightSpectrum
        :return: rendered LightSpectrum
        """
        p, w, wl, extent, _, bar, ill_count\
            = self._hit_detector("Detector Spectrum", detector_index, source_index, extent)
        
        # create description
        detector = self.detectors[detector_index]
        pname = f": {detector.desc}" if detector.desc != "" else ""
        desc = f"{Detector.abbr}{detector_index}{pname} at z = {detector.pos[2]:.5g} mm"
        desc = (f"Spectrum of RS{source_index} at " if source_index is not None else "Spectrum at ") + desc

        spec = LightSpectrum.render(wl, w, long_desc=desc, **kwargs)
        bar.finish()

        if ill_count:
            warning(f"{ill_count} rays ({100*ill_count/self.rays.N:.3g}% of all rays) were ill-conditioned for "
                    f"numerical hit finding at detector {detector_index}. "
                     "Where and whether they intersect might be wrong.")
        
        return spec

    def iterative_render(self,
                         N:                 int | float,
                         detector_index:    int | list = 0,
                         limit:             float | list = None,
                         projection_method: str | list = "Equidistant",
                         pos:               int | list = None,
                         extent:            list | np.ndarray = None)\
            -> list[RenderImage]:
        """
        Image render with N_rays rays.
        First returned value is a list of rendered sources, the second a list of rendered detector images.

        If pos is not provided,
        a single detector image is rendered at the position of the detector specified by detector_index.
        >> RT.iterative_render(N=10000, detector_index=1)
        
        If pos is provided as coordinate, the detector is moved beforehand.
        >> RT.iterative_render(N=10000, pos=[0, 1, 0], detector_index=1)
        
        If pos is a list, len(pos) detector images are rendered. All other parameters are either automatically
        repeated len(pos) times or can be specified as list with the same length as pos.
        Exemplary calls:
        >> RT.iterative_render(N=10000, pos=[[0, 1, 0], [2, 2, 10]], detector_index=1)
        >> RT.iterative_render(N=10000, pos=[[0, 1, 0], [2, 2, 10]], detector_index=[0, 1], \
        extent=[None, [-2, 2, -2, 2]])

        N_px_S can also be provided as list, note however, that when provided as list it needs
        to have the same length as the number of sources.

        This functions raises an exception if the geometry is incorrect.
        Note that no warnings for ill-conditioned rays are raised.

        :param N: number of rays
        :param detector_index: number/list of detector indices
        :param pos: 3D position(s) of the detector(s)
        :param limit: list/resolution limits for detector images
        :param projection_method: type/list of projection methods for SphericalSurface
        :param extent: list/value for the extent of the detector images
        :return: list of rendered detector images
        """
       
        if not self.ray_sources:
            raise RuntimeError("Ray Source(s) Missing.")
        
        if not self.detectors:
            raise RuntimeError("Detector(s) Missing.")
        
        if (N := int(N)) <= 0:
            raise ValueError(f"Ray number N_rays needs to be a positive int, but is {N}.")

        # use current detector position if pos is empty
        if pos is None:
            if isinstance(detector_index, list):
                raise ValueError("detector_index list needs to have the same length as pos list")
            pos = [self.detectors[detector_index].pos]

        elif isinstance(pos, list) and not isinstance(pos[0], list | np.ndarray):
            pos = [pos]

        if not isinstance(detector_index, list):
            detector_index = [detector_index] * len(pos)
        
        elif len(detector_index) != len(pos):
            raise ValueError("detector_index list needs to have the same length as pos list")
            
        if not isinstance(limit, list):
            limit = [limit] * len(pos)

        elif len(limit) != len(pos):
            raise ValueError("limit list needs to have the same length as pos list")

        if not isinstance(projection_method, list):
            projection_method = [projection_method] * len(pos)

        elif len(projection_method) != len(pos):
            raise ValueError("projection_method list needs to have the same length as pos list")
        
        if not isinstance(extent, list) or isinstance(extent[0], int | float):
            extent = [extent] * len(pos)

        elif len(extent) != len(pos):
            raise ValueError("extent list needs to have the same length as pos list")
        
        extentc = extent.copy()

        rays_step = self.ITER_RAYS_STEP
        iterations = max(1, int(N / rays_step))
        bar = ProgressBar("Rendering: ", iterations)

        # image list
        DIm_res = []

        # check geometry
        if (status := self._pretrace_check(rays_step)):
            raise RuntimeError("Geometry checks failed. Tracing aborted. Check the warnings.")

        # init cumulative warning messages
        nt = len(self.tracing_surfaces) + 2
        msgs_cum = np.zeros((len(self.INFOS), nt), dtype=int)

        # for all render iterations
        for i in range(iterations):

            # add remaining rays in last step
            if i == iterations - 1:
                rays_step += int(N - iterations*rays_step)  # additional rays for last iteration

            # turn off warnings and progress bar for the subtracing
            with global_options.no_warnings():
                with global_options.no_progress_bar():
                    self.trace(N=rays_step)
                    msgs_cum += self._msgs  # add to cumulative warnings

            # for all detector positions
            for j in np.arange(len(pos)):
               
                self.detectors[detector_index[j]].move_to(pos[j])
            
                with global_options.no_progress_bar():
                    Imi = self.detector_image(detector_index=detector_index[j],
                                              extent=extentc[j], limit=limit[j], _dont_filter=True,
                                              projection_method=projection_method[j])
               
                Imi._data *= rays_step / N

                # append image to list in first iteration, after that just add image content
                if i == 0:
                    DIm_res.append(Imi)
                    extentc[j] = Imi._extent0  # assign actual extent in case it was "auto"
                else:
                    DIm_res[j]._data += Imi._data

            global_options.show_progress_bar = True
            bar.update()
       
        # we didn't filter the images yet, as doing it with the finished image is faster (than doing it for each step)
        for i, DIm in enumerate(DIm_res):
            if limit[i] is not None:
                DIm._apply_rayleigh_filter()

        # show cumulative warnings
        bar.finish()
        self._msgs = msgs_cum
        self._show_messages(N)

        return DIm_res

    def _hit_source(self, info: str, source_index: int = 0)\
            -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, ProgressBar]:
        """
        Render a source image for a traced geometry.
        Internal function.

        :param N: number of image pixels in the smaller dimension
        :param source_index: source number, defaults to 0
        :return: positions, weights, wavelengths, actual extent, ProgressBar
        """
        if not self.ray_sources:
            raise RuntimeError("Ray Sources Missing.")

        if not self.rays.N:
            raise RuntimeError("No rays traced.")

        if source_index > len(self.ray_sources) - 1 or source_index < 0:
            raise IndexError("Invalid source_index.")
        
        if not self.check_if_rays_are_current():
            raise RuntimeError("Tracing geometry/properties changed. Please retrace first.")

        bar = ProgressBar(f"{info}: ", 2)

        extent = self.ray_sources[source_index].extent[:4]
        p, _, _, w, wl = self.rays.source_sections(source_index)
        bar.update()

        return p, w, wl, extent, bar

    def source_spectrum(self, source_index: int = 0, **kwargs) -> LightSpectrum:
        """
        Render a LightSpectrum for a source in a traced geometry.

        :param source_index: source number, default to 0
        :param kwargs: optional keyword arguments for the creation of the LightSpectrum
        :return: rendered LightSpectrum
        """
        p, w, wl, extent, bar = self._hit_source("Source Spectrum", source_index)

        # create description
        rs = self.ray_sources[source_index]
        pname = f": {rs.desc}" if rs.desc != "" else ""
        desc = f"Spectrum of {RaySource.abbr}{source_index}{pname} at z = {rs.pos[2]:.5g} mm"
        
        spec = LightSpectrum.render(wl, w, long_desc=desc, **kwargs)
        bar.finish()

        return spec

    def source_image(self, source_index: int = 0, limit: float = None, **kwargs) -> RenderImage:
        """
        Render a source image for a source in a traced geometry.

        :param source_index: source number, defaults to 0
        :param limit: resolution filter limit constant, see the documentation. Defaults to not filter
        :param kwargs: optional keyword arguments for creating the RenderImage
        :return: rendered RenderImage
        """
        p, w, wl, extent, bar = self._hit_source("Source Image", source_index)
        
        # create description
        rs = self.ray_sources[source_index]
        pname = f": {rs.desc}" if rs.desc != "" else ""
        desc = f"{RaySource.abbr}{source_index}{pname} at z = {rs.pos[2]:.5g} mm"

        img = RenderImage(long_desc=desc, extent=extent, projection=None)

        img.render(p, w, wl, limit=limit, **kwargs)
        bar.finish()

        return img

    def __focus_search_cost_function(self,
                                     z_pos:   float,
                                     mode:    str,
                                     pa:      np.ndarray,
                                     sb:      np.ndarray,
                                     w:       np.ndarray)\
            -> float | tuple[float, tuple[float, float, float]]:
        """
        Calculate the cost function value at this z position.
        There are no checks if rays are still inside the outline!

        :param z_pos: search starting position
        :param mode: focussing mode
        :param pa: auxiliary ray position
        :param sb: auxiliary ray direction
        :param w: ray weights
        :return: cost value
        """
        # hit position at virtual xy-plane detector
        ph = pa + sb*z_pos
        x, y = ph[:, 0], ph[:, 1]

        if mode == "RMS Spot Size":
            var_x = np.cov(x, aweights=w)
            var_y = np.cov(y, aweights=w)
            return np.sqrt(var_x + var_y)

        # adapt pixel number, so we have less noise for few rays,
        # but can see more image details and information for a larger number of rays
        # N rays are distributed on a square area,
        # scale default number by (1 + sqrt(N))
        N_px = 100*int(1 + np.sqrt(w.shape[0])/1000)
        N_px = N_px if N_px % 2 else N_px + 1  # enforce odd number
    
        # render power image
        Im, x_bin, y_bin = np.histogram2d(x, y, weights=w, bins=[N_px, N_px])

        if mode in ["Image Sharpness", "Image Center Sharpness"]:

            Y, X = np.mgrid[-1:1:N_px*1j, -1:1:N_px*1j]
            R2 = X**2 + Y**2
            R2[R2 > 1] = 1
            
            w2 = 1 - R2
            Im0 = Im*w2**2 if mode == "Image Center Sharpness" else Im
            
            if (Im0s := np.sum(Im0)):
                Im0 /= Im0s

            Im = np.abs(np.fft.fft2(Im0))
            Im = np.fft.fftshift(Im)

            rsm = np.mean(R2*Im)
            return 1/rsm**0.5 if rsm else 1

        else:  # Irradiance Variance
            Im = Im[Im > 0]  # exclude empty pixels
            Ap = (x_bin[1] - x_bin[0]) * (y_bin[1] - y_bin[0])  # pixel area
            Imstd = np.std(Im)
            return np.sqrt(Ap/Imstd) if Imstd else 1  # sqrt for better value range
              
    def __focus_rms_spot_direct_solution(self, pa: np.ndarray, sb: np.ndarray, w: np.ndarray, bounds: tuple)\
            -> scipy.optimize.OptimizeResult:
        """
        Direct RMS spot size solution from
        https://www.thepulsar.be/article/-devoptical-part-19--a-quick-focus-algorithm/
        my version is extended to include ray weights

        :param pa: auxiliary ray position
        :param sb: auxiliary ray direction
        :param w: ray weights
        :param bounds: search bounds
        :return: scipy optimize result
        """
        # mean position at search bounds
        pb0 = np.average(pa + sb*bounds[0], axis=0, weights=w)
        pb1 = np.average(pa + sb*bounds[1], axis=0, weights=w)

        # direction of mean position
        vx = pb1[0] - pb0[0]
        vy = pb1[1] - pb0[1]
        vz = bounds[1] - bounds[0]

        # relative ray starting position
        dx = pa[:, 0] - pb0[0]
        dy = pa[:, 1] - pb0[1]

        # relative ray directions
        dtx = sb[:, 0] - vx/vz
        dty = sb[:, 1] - vy/vz

        # calculate solution
        w2 = w**2
        dnorm = np.sum(w2*dtx**2 + w2*dty**2)
        d = -np.sum(dtx*dx*w2 + dty*dy*w2) / dnorm if dnorm else np.mean(bounds)
        d = np.clip(d, bounds[0], bounds[1])  # clip to search range

        # mock scipy optimize result
        res = scipy.optimize.OptimizeResult()
        res.x = d
        res.fun = self.__focus_search_cost_function(d, "RMS Spot Size", pa, sb, w)
        return res

    def focus_search(self,
                     method:           str,
                     z_start:          float,
                     source_index:     int = None,
                     return_cost:      bool = False)\
            -> tuple[scipy.optimize.OptimizeResult, dict]:
        """
        Find the focal point using different methods. z_start defines the starting point,
        the search range is the region between a lens, filter, aperture or the outline.
        Outline intersections of rays are ignored.

        :param method: focussing method from "focus_search_methods"
        :param z_start: starting position z (float)
        :param source_index: source number, defaults to None, so rays from all sources are used
        :param return_cost: If a cost function value array should be included in the output dictionary.
            Needed for plotting of the cost function. Deactivate to increase performance for some methods.
        :return: scipy optimize result and property dictionary
        """

        if not (self.outline[4] <= z_start <= self.outline[5]):
            raise ValueError(f"Starting position z_start={z_start} outside raytracer"
                             " z-outline range {RT.outline[4:]}.")

        if method not in self.focus_search_methods:
            raise ValueError(f"Invalid method '{method}', should be one of {self.focus_search_methods}.")

        if not self.rays.N:
            raise RuntimeError("No rays traced.")

        if source_index is not None and source_index < 0:
            raise IndexError(f"source_index needs to be >= 0, but is {source_index}")

        if (source_index is not None and source_index > len(self.rays.N_list)) or len(self.rays.N_list) == 0:
            raise IndexError(f"source_index={source_index} larger than number of simulated sources "
                             f"({len(self.rays.N_list)}. "
                             "Either the source was not added or the new geometry was not traced.")
        
        if not self.check_if_rays_are_current():
            raise RuntimeError("Tracing geometry/properties changed or last trace had errors. Please retrace first.")

        # get search bounds
        ################################################################################################################

        # default bounds, left starts at end of all ray sources, right ends at outline
        b0 = self.N_EPS + np.max([rs.extent[5] for rs in self.ray_sources])
        b1 = self.outline[5] - self.N_EPS

        # get surface position of Lens, Aperture or Filter around starting position
        for surf in self.tracing_surfaces:
            if surf.z_max > z_start:
                b1 = surf.z_min
                break
            b0 = surf.z_max

        # create bounds
        bounds = [b0, b1]

        # start progress bar
        ################################################################################################################

        Nt = 320  # cost function sampling points, good if divisible by 2, 4, 8, 16, 32 threads
        N_th = misc.cpu_count() if global_options.multithreading else 1
        steps = int(Nt/32) if return_cost or method in ["Image Sharpness", "Image Center Sharpness"] else 0
        steps += 3  # progressbars steps
        bar = ProgressBar("Finding Focus: ", steps)

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
        bar.update()

        # exclude already absorbed rays
        rays_pos[pos == -1] = False

        # make sure we have enough rays
        N_use = np.count_nonzero(rays_pos)
        if N_use < 1000:  # throw error when no rays are present
            warning(f"WARNING: Less than 1000 rays for focus_search ({N_use}).")

        # no rays are used, return placeholder variables
        if N_use <= 1:
            return scipy.optimize.OptimizeResult(),\
                dict(pos=[np.nan, np.nan, np.nan], bounds=bounds, z=np.full(Nt, np.nan),
                     cost=np.full(Nt, np.nan), N=N_use)

        # select valid positions and get ray parts
        rp = np.where(rays_pos)[0]
        pos = pos[rp]
        p, s, _, weights, _, _, _ = self.rays.rays_by_mask(rays_pos, pos, ret=[1, 1, 0, 1, 0, 0, 0])
        bar.update()

        # find focus
        ################################################################################################################

        # parameters for function hit position = pa + sb * z_pos
        pa = p - s/s[:, 2, np.newaxis]*p[:, 2, np.newaxis]
        sb = s/s[:, 2, np.newaxis]
            
        # sample cost function
        if return_cost or method in  ["Image Sharpness", "Image Center Sharpness"]:

            r = random.stratified_interval_sampling(bounds[0], bounds[1], Nt, shuffle=False)
            vals = np.zeros_like(r)

            def threaded(N_th, N_is, Nt, *afargs):
                Ns = N_is*int(Nt/N_th)
                Ne = (N_is+1)*int(Nt/N_th) if N_is != N_th-1 else Nt
                div = int(Nt / (steps - 3) / N_th)

                for i, Ni in enumerate(np.arange(Ns, Ne)):
                    vals[Ni] = self.__focus_search_cost_function(r[Ni], *afargs)
                    bar.update(i % div == div-1 and not N_is)

            if N_th > 1:
                threads = [Thread(target=threaded, args=(N_th, N_is, Nt, method, pa, sb, weights))
                           for N_is in range(N_th)]
                [th.start() for th in threads]
                [th.join() for th in threads]
            else:
                threaded(1, 0, Nt, method, pa, sb, weights)
       
        # method depending handling
        if method == "RMS Spot Size":
            res = self.__focus_rms_spot_direct_solution(pa, sb, weights, bounds)
        else:
            cost_func2 = lambda z, method: self.__focus_search_cost_function(z[0], method, pa, sb, weights)
            
            if method == "Irradiance Variance":
                res = scipy.optimize.minimize(cost_func2, np.mean(bounds), args=method, tol=None, callback=None,
                                              options={'maxiter': 100}, bounds=[bounds], method="Nelder-Mead")
                
            elif method in ["Image Sharpness", "Image Center Sharpness"]:
                pos = np.argmin(vals)
                res = scipy.optimize.minimize(cost_func2, r[pos], args=method, tol=None, callback=None,
                                              options={'maxiter': 30}, bounds=[bounds], method="COBYLA")
            else:
                assert_never(method)

            res.x = res.x[0]

        # print warning if result is near bounds
        ################################################################################################################
        
        bar.finish()

        rrl = (res.x - bounds[0]) < 10*(bounds[1] - bounds[0]) / Nt
        rrr = (bounds[1] - res.x) < 10*(bounds[1] - bounds[0]) / Nt
        if rrl or rrr:
            warning("Found minimum near search bounds, "
                    "this can mean the focus is outside of the search range.")

        # average ray position at focus
        pos = tuple(np.average(pa + sb*res.x, axis=0, weights=weights))

        if not return_cost:
            r = None
            vals = None

        return res, dict(pos=pos, bounds=bounds, z=r, cost=vals, N=N_use)

