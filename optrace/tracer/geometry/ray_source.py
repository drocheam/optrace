
# standard libs
from typing import Callable, Any  # Callable and Any type

# external libs
import numpy as np  # ndarray type and calculations
import numexpr as ne  # faster calculations

# geometries
from .element import Element  # parent class
from . import Surface, Line, Point  # source types
from .surface.rectangular_surface import RectangularSurface

# spectrum and color
from ..spectrum.light_spectrum import LightSpectrum  # spectrum of source
from .. import color  # for random_wavelengths_from_srgb() and _power_from_srgb()
from ..presets.light_spectrum import d65 as d65_spectrum  # default light spectrum

# misc
from ..misc import PropertyChecker as pc  # check types and values
from .. import misc  # calculations


class RaySource(Element):

    divergences: list[str] = ["None", "Lambertian", "Isotropic", "Function"]
    """Possible divergence types"""

    orientations: list[str] = ["Constant", "Converging", "Function"]
    """Possible orientation types"""

    polarizations: list[str] = ["Constant", "Uniform", "List", "Function", "x", "y", "xy"]
    """Possible polarization types"""

    abbr: str = "RS"  #: object abbreviation
    _allow_non_2D: bool = True  # allow points or lines as surfaces
    _max_image_px: float = 2e6  #: maximum number of pixels for images. Only here for performance reasons.

    def __init__(self,

                 # Surface Parameters
                 surface:           Surface | Line | Point,
                 pos:               (list | np.ndarray) = None,

                 # Divergence Parameters
                 divergence:        str = "None",
                 div_angle:         float = 0.5,
                 div_2d:            bool = False,
                 div_axis_angle:    float = 0,
                 div_func:          Callable[[np.ndarray], np.ndarray] = None,
                 div_args:          dict = {},

                 # Light Parameters
                 spectrum:          LightSpectrum = None,
                 power:             float = 1.,
                 image:             (str | np.ndarray) = None,

                 # Ray Orientation Parameters
                 s:                 (list | np.ndarray) = None,
                 s_sph:             (list | np.ndarray) = None,
                 orientation:       str = "Constant",
                 conv_pos:          list[float] | np.ndarray = None,
                 or_func:           Callable[[np.ndarray, np.ndarray], np.ndarray] = None,
                 or_args:           dict = {},

                 # Polarization Parameters
                 polarization:      str = "Uniform",
                 pol_angle:         float = 0.,
                 pol_angles:        list[float] = None,
                 pol_probs:         list[float] = None,
                 pol_func:          Callable[[np.ndarray], np.ndarray] = None,
                 pol_args:          dict = {},

                 **kwargs)\
            -> None:
        """
        Create a RaySource with a specific area, orientation, divergence, polarization, power and spectrum

        :param surface: emitting Surface object
        :param divergence: divergence type, see "divergences" list
        :param orientation: orientation type, see "orientations" list
        :param polarization: polarization type, see "polarizations" list
        :param spectrum: LightSpectrum of the RaySource
        :param pos: 3D position of RaySource center
        :param power: total power of the RaySource in W
        :param div_angle: cone opening angle in degrees
        :param div_2d: if divergence is inside of a circular arc instead a cone
        :param div_axis_angle: axis angle for 2D divergence with div_2d=True
        :param div_func: divergence function, must take angles in radians in range [0, div_angle]
                and return a probability
        :param div_args: additional keywords arguments for div_func in a dictionary
        :param conv_pos: convergence position for orientation='Converging', 3D position
        :param pol_angle: polarization angle as float, value in degrees 
        :param pol_angles: polarization angle list, values in degrees
        :param pol_probs: probabilities for the pol_angles
        :param pol_func: polarization function, must take an numpy array in range [0, 2*pi] and return a probability
        :param pol_args: dictionary of additional keyword arguments for pol_func
        :param s: 3D direction vector
        :param s_sph: 3D direction vector in spherical coordinates, specified as theta, phi, both in degrees
        :param or_func: orientation function,
            takes 1D array of x and y coordinates as input, returns (N, 3) numpy 2D array with orientations
        :param or_args: dictionary of additional keyword arguments for or_func
        :param image: image for modes source_type="BW_Image" or "RGB_Image",
                specified as path (string) or RGB array (numpy 3D array)
        :param kwargs: additional keyword arguments for parent classes
        """

        self._new_lock = False

        # geometry
        pos = pos if pos is not None else [0, 0, 0]
        super().__init__(surface, pos, **kwargs)

        # power and spectrum
        self.power = power
        self.spectrum = spectrum if spectrum is not None else d65_spectrum
        self.pIf = None
        self._mean_img_color = None
        self.image = image

        # polarization 
        self.polarization = polarization
        self.pol_angle = pol_angle
        self.pol_func = pol_func
        self.pol_angles = pol_angles
        self.pol_probs = pol_probs
        self.pol_args = pol_args

        # orientation and divergence
        self.divergence = divergence
        self.div_angle = div_angle
        self.orientation = orientation
        self.conv_pos = conv_pos if conv_pos is not None else [0, 0, 0]
        self.or_func = or_func
        self.or_args = or_args

        if s_sph is None:
            self.s = s if s is not None else [0, 0, 1]
        else:
            pc.check_type("s_sph", s_sph, list | np.ndarray)
            theta, phi = np.radians(s_sph[0]), np.radians(s_sph[1])
            self.s = [np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)]

        self.div_axis_angle = div_axis_angle
        self.div_func = div_func
        self.div_2d = div_2d
        self.div_args = div_args

        # lock assignment of new properties. New properties throw an error.
        self._new_lock = True

    def color(self, rendering_intent: str = "Ignore", clip=False) -> tuple[float, float, float]:
        """
        Get the mean color of the RaySource

        :param rendering_intent: rendering_intent for color calculation
        :param clip: if sRGB values are clipped
        :return: tuple of sRGB values with data range [0, 1]
        """
        if self.image is not None:
            return self._mean_img_color

        else:
            return self.spectrum.color(rendering_intent, clip)

    def create_rays(self, N: int, no_pol: bool = False, power: float = None)\
            -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate N rays according to the property of the RaySource

        :param N: number of rays
        :param no_pol: if polarization needs to be calculated
        :param power: power to use, when not specified internal object value is used
        :return: position array (numpy 2D array), direction array (numpy 2D array), polarization array (numpy 2D array)
                 weight array (numpy 1D array), wavelength array (numpy 1D array)
                 first dimension: ray number, second dimension: values
        """

        ## Generate ray weights
        ################################################################################################################
        power = power or self.power  # favour power parameter self.power
        weights = np.full(N, power/N, dtype=np.float32)

        ## Generate ray wavelengths
        ################################################################################################################

        if self.image is None:
            pc.check_type("RaySource.spectrum", self.spectrum, LightSpectrum)
            wavelengths = self.spectrum.random_wavelengths(N)

        ## Generate ray starting points
        ################################################################################################################

        if self.image is None:
            p = self.surface.random_positions(N)

        else:
            if not isinstance(self.surface, RectangularSurface):
                raise RuntimeError("Images can only be used with RectangularSurface")

            # special case image with only one pixel
            if self.image.shape[0] == 1 and self.image.shape[1] == 1:
                PY, PX = np.zeros(N, dtype=np.int32), np.zeros(N, dtype=np.int32)
            else:
                # get random pixel number, the pixel brightness is the probability
                P = misc.random_from_distribution(np.arange(self.pIf.shape[0]),
                                                  self.pIf, N, kind="discrete").astype(int)
                PY, PX = np.divmod(P, self.image.shape[1])  # pixel x, y position from pixel number

            # add random position inside pixel and calculate positions in 3D space
            rx, ry = misc.uniform2(0, 1, 0, 1, N)
            xs, xe, ys, ye = self.surface.extent[:4]
            Iy, Ix = self.image.shape[:2]

            p = np.zeros((N, 3), dtype=np.float64, order='F')
            ne.evaluate("(xe-xs)/Ix * (PX + rx) + xs", out=p[:, 0])
            ne.evaluate("(ye-ys)/Iy * (PY + ry) + ys", out=p[:, 1])
            p[:, 2] = self.pos[2]

            wavelengths = color.random_wavelengths_from_srgb(self.image[PY, PX])

        ## Generate orientations
        ################################################################################################################

        match self.orientation:

            case "Constant":
                s_or = np.tile(self.s, (N, 1))

            case "Converging":
                s_or = np.column_stack((self.conv_pos[0] - p[:, 0], 
                                        self.conv_pos[1] - p[:, 1], 
                                        self.conv_pos[2] - p[:, 2]))
                s_or = misc.normalize(s_or)

            case "Function":  # pragma: no branch
                pc.check_callable("RaySource.or_func", self.or_func)
                s_or = self.or_func(p[:, 0], p[:, 1], **self.or_args)

        ## Generate ray divergences relative to orientation
        ################################################################################################################

        # alpha: angle in plane perpendicular to base orientation s0
        # theta: angle between orientation s and base orientation s0

        # for 2D divergence theta has only two values, angle and angle + pi
        if self.div_2d:
            t = np.radians(self.div_axis_angle).repeat(2) + [0, np.pi]
            P = np.array([1., 1.])
            alpha = misc.random_from_distribution(t, P, N, kind="discrete")

        if self.divergence == "Function":
            pc.check_callable("RaySource.div_func", self.div_func)

        match self.divergence:

            case "None":
                s = s_or

            case "Lambertian" if not self.div_2d:
                # see https://doi.org/10.1080/10867651.1997.10487479
                # # see https://www.particleincell.com/2015/cosine-distribution/
                r, alpha = misc.ring_uniform(0, np.sin(np.radians(self.div_angle)), N, polar=True)
                theta = ne.evaluate("arcsin(r)")

            case "Lambertian" if self.div_2d:
                X0 = misc.uniform(0, np.sin(np.radians(self.div_angle)), N)
                theta = np.arcsin(X0)

            case "Isotropic" if not self.div_2d:
                # see https://doi.org/10.1080/10867651.1997.10487479
                # related https://mathworld.wolfram.com/SpherePointPicking.html
                r, alpha = misc.ring_uniform(0, np.sin(np.radians(self.div_angle)), N, polar=True)
                theta = ne.evaluate("arccos(1 - r**2)")
            
            case "Isotropic" if self.div_2d:
                theta = misc.uniform(0, np.radians(self.div_angle), N)
           
            case "Function" if not self.div_2d:
                div_sin = np.sin(np.radians(self.div_angle))
                r, alpha = misc.ring_uniform(0, div_sin, N, polar=True)
                x = np.linspace(0, np.radians(self.div_angle), 1000)
                # related https://www.scratchapixel.com/lessons/3d-basic-rendering/global-illumination-path-tracing/global-illumination-path-tracing-practical-implementation
                f = self.div_func(x, **self.div_args) * np.sin(x)
                X0 = r**2 / div_sin**2
                theta = misc.random_from_distribution(x, f, X0, kind="continuous")

            case "Function" if self.div_2d:  # pragma: no branch
                x = np.linspace(0, np.radians(self.div_angle), 1000)
                f = self.div_func(x, **self.div_args)
                theta = misc.random_from_distribution(x, f, N, kind="continuous")

        if self.divergence != "None":
            # vector perpendicular to s, created using  sy = [1, 0, 0] x s_or
            sy = np.zeros_like(s_or, dtype=np.float64, order='F')
            s_orx, s_ory, s_orz = s_or[:, 0], s_or[:, 1], s_or[:, 2]
            sy[:, 1] = ne.evaluate("-s_orz/sqrt(1-s_orx**2)")
            sy[:, 2] = ne.evaluate("s_ory/sqrt(1-s_orx**2)")

            # vector sx = s x sy
            sx = misc.cross(s_or, sy)

            # vector s has a component alpha in the sx-sy plane
            theta_, alpha_ = theta[:, np.newaxis], alpha[:, np.newaxis]
            s = ne.evaluate("cos(theta_)*s_or + sin(theta_)*(cos(alpha_)*sx + sin(alpha_)*sy)")

        if np.any(s[:, 2] <= 0):
            raise RuntimeError("All ray divergences s need to be in positive z-divergence")

        ## Assign ray polarization
        ################################################################################################################

        if no_pol:
            pols = np.full_like(p, np.nan, np.float64, order='F')

        else:
            match self.polarization:

                case "x":       
                    ang = 0.

                case "y":      
                    ang = np.pi/2

                case "xy":      
                    ang = misc.random_from_distribution(np.array([0, np.pi/2]), np.ones(2), N, kind="discrete")

                case "Constant":   
                    ang = np.radians(self.pol_angle)

                case "Uniform":
                    ang = misc.uniform(0, 2*np.pi, N)

                case "List":
                    pc.check_type("RaySource.pol_angles", self.pol_angles, np.ndarray | list)
                    if self.pol_probs is None:
                        self.pol_probs = np.ones_like(self.pol_angles)
                    ang = misc.random_from_distribution(self.pol_angles, self.pol_probs, N, kind="discrete")
                    ang = np.radians(ang)

                case "Function":  # pragma: no branch
                    pc.check_callable("RaySource.pol_func", self.pol_func)
                    x = np.linspace(0, 2*np.pi, 5000)
                    f = self.pol_func(x, **self.pol_args)
                    ang = misc.random_from_distribution(x, f, N, kind="continuous")
                    ang = np.radians(ang)

            # pol is rotated by an axis perpendicular to the plane of base divergence s = [0, 0, 1]
            # and the current divergence s_
            # let's call this axis ps. The resulting polarization pol_ is perpendicular to s_
            # and has the same component at ps
            ####
            # this is equivalent to a imaginary lens having focused pol-polarized parallel light.
            # The resulting pol_ for each ray is the same as ours.

            # ps = s_ x s = s_ x [0, 0, 1] = [s_1, -s_0, 0]/sqrt(s_0**2 + s_1**2)
            # pp = ps x s = ps x [0, 0, 1] = [-s_0, -s_1, 0]/sqrt(s_0**2 + s_1**2)
            # pp_ = ps x s_

            # A_ts = ps * pol
            # A_tp = pp * pol
            # pol_ = A_ts*ps + A_tp*pp_

            mask = s[:, 2] != 1
            pols = np.zeros_like(s, dtype=np.float64, order='F')

            sm = s[mask]
            ps = np.zeros_like(sm, dtype=np.float64, order='F')
            s_0, s_1, s_2 = sm[:, 0], sm[:, 1], sm[:, 2]

            # sqrt(s0**2 + s1**2) equals sqrt(1-s2**2) for a unity vector
            ne.evaluate("s_1/sqrt(1-s_2**2)", out=ps[:, 0])
            ne.evaluate("-s_0/sqrt(1-s_2**2)", out=ps[:, 1])

            pols[:, 0] = ne.evaluate("cos(ang)")
            pols[:, 1] = ne.evaluate("sin(ang)")

            ps0, ps1 = ps[:, 0], ps[:, 1]
            pol0m, pol1m = pols[mask, 0], pols[mask, 1]
            A_ts = ne.evaluate("ps0*pol0m + ps1*pol1m")[:, np.newaxis]
            A_tp = ne.evaluate("ps1*pol0m - ps0*pol1m")[:, np.newaxis]

            pp_ = misc.cross(ps, sm)
            pols[mask] = ne.evaluate("A_ts*ps + A_tp*pp_")

        ## return ray properties
        ################################################################################################################
        return p, s, pols, weights, wavelengths

    def __setattr__(self, key: str, val: Any) -> None:
        """
        assigns the value of an attribute
        :param key: attribute name
        :param val: value to assign
        """

        match key:

            case "divergence":
                pc.check_type(key, val, str)
                pc.check_if_element(key, val, self.divergences)

            case "orientation":
                pc.check_type(key, val, str)
                pc.check_if_element(key, val, self.orientations)

            case "polarization":
                pc.check_type(key, val, str)
                pc.check_if_element(key, val, self.polarizations)

            case ("pol_angle"):
                pc.check_type(key, val, int | float)
                val = float(val)

            case "div_axis_angle":
                pc.check_type(key, val, int | float)
                val = float(val)
            
            case ("power" | "div_angle"):
                pc.check_type(key, val, int | float)
                pc.check_above(key, val, 0)
                val = float(val)

            case "s":
                pc.check_type(key, val, list | np.ndarray)
                val2 = np.asarray_chkfinite(val, dtype=np.float64) / np.linalg.norm(val)  # normalize
                if val2.shape[0] != 3:
                    raise TypeError("s needs to have 3 dimensions")

                pc.check_above("s[2]", val2[2], 0)

                super().__setattr__(key, val2)
                return
            
            case "conv_pos":
                pc.check_type(key, val, list | np.ndarray)
                val2 = np.asarray_chkfinite(val, dtype=np.float64)
                if val2.shape[0] != 3:
                    raise TypeError("conv_pos needs to have 3 dimensions")
                
                super().__setattr__(key, val2)
                return

            case "div_2d":
                pc.check_type(key, val, bool)

            case "spectrum":
                pc.check_type(key, val, LightSpectrum)

            case ("or_func" | "div_func" | "pol_func"):
                pc.check_none_or_callable(key, val)

            case ("pol_angles" | "pol_probs") if val is not None:
                pc.check_type(key, val, list | np.ndarray)
                val2 = np.asarray_chkfinite(val, dtype=np.float64)
                super().__setattr__(key, val2)
                return

            case "image" if val is not None:

                pc.check_type(key, val, str | np.ndarray)

                if isinstance(val, str):
                    img = misc.load_image(val)
                else:
                    img = np.asarray_chkfinite(val, dtype=np.float64)

                if img.shape[0]*img.shape[1] > self._max_image_px:
                    raise RuntimeError("For performance reasons only images with less than 2 megapixels are allowed.")

                if img.ndim != 3 or img.shape[2] != 3 or not img.shape[0] or not img.shape[1]:
                    raise TypeError("Image array needs to be three dimensional with three values in third dimension,"
                                    f"but shape is {img.shape}.")

                if np.min(img) < 0 or np.max(img) > 1:
                    raise ValueError("Image values need to be inside range [0, 1]")

                # calculate pixel probability from relative power for each pixel
                If = color._power_from_srgb(img).ravel()
                Ifs = np.sum(If)

                if Ifs == 0:
                    raise ValueError("Image can not be completely black")

                self.pIf = 1/Ifs*If

                # calculate mean image color, needed for self.color
                # mean color needs to calculated in a linear colorspace, hence sRGBLinear
                sRGBL = color.srgb_to_srgb_linear(img)
                sRGBL_mean = np.mean(sRGBL, axis=(0, 1))
                sRGB_mean = color.srgb_linear_to_srgb(np.array([[[*sRGBL_mean]]]))
                self._mean_img_color = sRGB_mean[0, 0]

                super().__setattr__(key, img)
                return

            case "front":
                if not (isinstance(val, Point | Line) or (isinstance(val, Surface) and val.is_flat())):
                    raise ValueError("Currently only planar surfaces are supported for RaySources.")

        super().__setattr__(key, val)
