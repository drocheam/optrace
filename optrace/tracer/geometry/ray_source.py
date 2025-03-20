
# standard libs
from typing import Callable, Any  # Callable and Any type

# external libs
import numpy as np  # ndarray type and calculations

# geometries
from .element import Element  # parent class
from . import Surface, Line, Point  # source types
from .surface.rectangular_surface import RectangularSurface

# spectrum and color
from ..spectrum.light_spectrum import LightSpectrum  # spectrum of source
from .. import color  # for random_wavelengths_from_srgb() and _power_from_srgb()
from ..presets.light_spectrum import d65 as d65_spectrum  # default light spectrum

# misc
from ...property_checker import PropertyChecker as pc  # check types and values
from .. import misc  # calculations
from .. import random
from ..image.rgb_image import RGBImage
from ..image.linear_image import LinearImage



class RaySource(Element):

    divergences: list[str] = ["None", "Lambertian", "Isotropic", "Function"]
    """Possible divergence types"""

    orientations: list[str] = ["Constant", "Converging", "Function"]
    """Possible orientation types"""

    polarizations: list[str] = ["Constant", "Uniform", "List", "Function", "x", "y", "xy"]
    """Possible polarization types"""

    abbr: str = "RS"  #: object abbreviation
    _allow_non_2D: bool = True  # allow points or lines as surfaces
    _max_image_px: float = 2e6  #: maximum number of pixels for images. Only for performance reasons.


    def __init__(self,

                 # Surface Parameters
                 surface:           Surface | Line | Point | LinearImage | RGBImage,
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
        Create a RaySource with a specific area, orientation, divergence, polarization, power and spectrum.

        * When a 'surface' is provided as Surface (RectangularSurface, RingSurface, Point, Line, CircularSurface),
          there will be uniform emittance over the surface and the spectrum can be set by the 'spectrum' parameter.
        * When 'surface' is provided as LinearImage, the emittance follows the image intensity distribution,
          the spectrum can be set by the 'spectrum' parameter.
        * When 'surface' is provided as RGBImage,
          the spectrum and brightness for each pixel is generated from three primaries so it matches the pixel color.

        :param surface: emitting Surface, Point, Line, RGBImage or LinearImage object
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
        :param kwargs: additional keyword arguments for parent classes
        """

        self._new_lock = False

        # RGBImage -> color and distribution according to image
        if isinstance(surface, RGBImage):
            surface_ = RectangularSurface(dim=surface.s)
            self._image = surface
                
            # calculate pixel probability from relative power for each pixel
            If = color._power_from_srgb(self._image.data).ravel()
            Ifs = np.sum(If)
            self._pIf = 1/Ifs*If

            # calculate mean image color, needed for self.color
            # mean color needs to calculated in a linear colorspace, hence sRGBLinear
            sRGBL = color.srgb_to_srgb_linear(self._image.data)
            sRGBL_mean = np.mean(sRGBL, axis=(0, 1))
            sRGB_mean = color.srgb_linear_to_srgb(np.array([[[*sRGBL_mean]]]))
            self._mean_img_color = sRGB_mean[0, 0]
        
        # LinearImage -> distribution according to values, spectrum user-defined
        elif isinstance(surface, LinearImage):
            surface_ = RectangularSurface(dim=surface.s)
            self._image = surface
            self._mean_img_color = None
            
            If = surface.data.ravel()
            Ifs = np.sum(If)
            self._pIf = 1/Ifs*If
        
        else:
            surface_ = surface
            self._image = None
            self._pIf = None
            self._mean_img_color = None

        # geometry
        pos = pos if pos is not None else [0, 0, 0]
        super().__init__(surface_, pos, **kwargs)

        # power and spectrum
        self.power = power
        self.spectrum = spectrum if spectrum is not None else d65_spectrum

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
        Get the average color of the RaySource

        :param rendering_intent: rendering_intent for color calculation
        :param clip: if sRGB values are clipped
        :return: tuple of sRGB values with data range [0, 1]
        """
        if isinstance(self._image, RGBImage):
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

        # generate wavelengths from LightSpectrum, except for RGBImage
        if not isinstance(self._image, RGBImage):
            pc.check_type("RaySource.spectrum", self.spectrum, LightSpectrum)
            wavelengths = self.spectrum.random_wavelengths(N)

        ## Generate ray starting points
        ################################################################################################################

        if self._image is None:
            p = self.surface.random_positions(N)

        # RGBImage or LinearImage
        else:
            # special case image with only one pixel
            if self._image.shape[0] == 1 and self._image.shape[1] == 1:
                PY, PX = np.zeros(N, dtype=np.int32), np.zeros(N, dtype=np.int32)
            else:
                # get random pixel number, the pixel brightness is the probability
                P = random.inverse_transform_sampling(np.arange(self._pIf.shape[0]),
                                                      self._pIf, N, kind="discrete").astype(int)
                PY, PX = np.divmod(P, self._image.shape[1])  # pixel x, y position from pixel number

            # add random position inside pixel and calculate positions in 3D space
            rx, ry = random.stratified_rectangle_sampling(0, 1, 0, 1, N)
            xs, xe, ys, ye = self.surface.extent[:4]
            Iy, Ix = self._image.shape[:2]

            p = np.zeros((N, 3), dtype=np.float64, order='F')
            p[:, 0] = (xe-xs)/Ix * (PX + rx) + xs
            p[:, 1] = (ye-ys)/Iy * (PY + ry) + ys
            p[:, 2] = self.pos[2]

            if isinstance(self._image, RGBImage):
                wavelengths = color.random_wavelengths_from_srgb(self._image.data[PY, PX])
            # for LinearImage the wavelengths were already generated from the LightSpectrum

        ## Generate orientations
        ################################################################################################################

        match self.orientation:

            case "Constant":
                s_or = np.broadcast_to(self.s, (N, 3))

            case "Converging":
                s_or = misc.normalize(self.conv_pos - p)

            case "Function":  # pragma: no branch
                pc.check_callable("RaySource.or_func", self.or_func)
                s_or = self.or_func(p[:, 0], p[:, 1], **self.or_args)

        ## Generate ray divergences relative to orientation
        ################################################################################################################

        # alpha: angle in plane perpendicular to base orientation s0
        # theta: angle between orientation s and base orientation s0

        # NOTE creating a source with varying ray orientation but a specific divergence function
        # (mostly likely) leads to a different divergence function in the far field
        # e.g. a Lambertian radiator with spatially varying base orientation is not a lambertian radiator anymore

        # for 2D divergence theta has only two values, angle and angle + pi
        if self.div_2d:
            t = np.radians(self.div_axis_angle).repeat(2) + [0, np.pi]
            P = np.array([1., 1.], dtype=np.float64)
            alpha = random.inverse_transform_sampling(t, P, N, kind="discrete")

        if self.divergence == "Function":
            pc.check_callable("RaySource.div_func", self.div_func)

        match self.divergence:

            case "None":
                s = s_or

            case "Lambertian" if not self.div_2d:
                # see https://doi.org/10.1080/10867651.1997.10487479
                # # see https://www.particleincell.com/2015/cosine-distribution/
                r, alpha = random.stratified_ring_sampling(0, np.sin(np.radians(self.div_angle)), N, polar=True)
                theta = np.arcsin(r)

            case "Lambertian" if self.div_2d:
                X0 = random.stratified_interval_sampling(0, np.sin(np.radians(self.div_angle)), N)
                theta = np.arcsin(X0)

            case "Isotropic" if not self.div_2d:
                # see https://doi.org/10.1080/10867651.1997.10487479
                # related https://mathworld.wolfram.com/SpherePointPicking.html
                r, alpha = random.stratified_ring_sampling(0, np.sin(np.radians(self.div_angle)), N, polar=True)
                theta = np.arccos(1 - r**2)
            
            case "Isotropic" if self.div_2d:
                theta = random.stratified_interval_sampling(0, np.radians(self.div_angle), N)
           
            case "Function" if not self.div_2d:
                div_sin = np.sin(np.radians(self.div_angle))
                r, alpha = random.stratified_ring_sampling(0, div_sin, N, polar=True)
                x = np.linspace(0, np.radians(self.div_angle), 1000)
                # related https://www.scratchapixel.com/lessons/3d-basic-rendering/global-illumination-path-tracing/global-illumination-path-tracing-practical-implementation
                f = self.div_func(x, **self.div_args) * np.sin(x)
                X0 = r**2 / div_sin**2
                theta = random.inverse_transform_sampling(x, f, X0, kind="continuous")

            case "Function" if self.div_2d:  # pragma: no branch
                x = np.linspace(0, np.radians(self.div_angle), 1000)
                f = self.div_func(x, **self.div_args)
                theta = random.inverse_transform_sampling(x, f, N, kind="continuous")

        if self.divergence != "None":
            # vector perpendicular to s, created using  sy = [1, 0, 0] x s_or
            fa = 1 / np.sqrt(1 - s_or[:, 0]**2)
            sy = np.zeros_like(s_or, dtype=np.float64, order='F')
            sy[:, 1] = -s_or[:, 2] * fa
            sy[:, 2] = s_or[:, 1] * fa

            # vector sx = s x sy
            sx = misc.cross(s_or, sy)

            # vector s has a component alpha in the sx-sy plane
            theta_, alpha_ = theta[:, np.newaxis], alpha[:, np.newaxis]
            s = np.cos(theta_)*s_or + np.sin(theta_)*(np.cos(alpha_)*sx + np.sin(alpha_)*sy)

        if np.any(s[:, 2] <= 0):
            raise RuntimeError("All ray divergences s need to be in positive z-divergence")

        ## Assign ray polarization
        ################################################################################################################

        if no_pol:
            pols = np.full_like(p, np.nan, np.float16, order='F')

        else:
            match self.polarization:

                case "x":       
                    ang = 0.

                case "y":      
                    ang = np.pi/2

                case "xy":      
                    ang = random.inverse_transform_sampling(np.array([0, np.pi / 2]), np.ones(2), N, kind="discrete")

                case "Constant":   
                    ang = np.radians(self.pol_angle)

                case "Uniform":
                    ang = random.stratified_interval_sampling(0, 2 * np.pi, N)

                case "List":
                    pc.check_type("RaySource.pol_angles", self.pol_angles, np.ndarray | list)
                    if self.pol_probs is None:
                        self.pol_probs = np.ones_like(self.pol_angles)
                    ang = random.inverse_transform_sampling(self.pol_angles, self.pol_probs, N, kind="discrete")
                    ang = np.radians(ang)

                case "Function":  # pragma: no branch
                    pc.check_callable("RaySource.pol_func", self.pol_func)
                    x = np.linspace(0, 2*np.pi, 5000)
                    f = self.pol_func(x, **self.pol_args)
                    ang = random.inverse_transform_sampling(x, f, N, kind="continuous")
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
            sm = s[mask]
            pols = np.zeros_like(s, dtype=np.float64, order='F')

            # sqrt(s0**2 + s1**2) equals sqrt(1-s2**2) for a unity vector
            fa = 1 / np.sqrt(1 - sm[:, 2]**2)
            ps = np.zeros_like(sm, dtype=np.float64, order='F')
            ps[:, 0] = sm[:, 1] * fa
            ps[:, 1] = -sm[:, 0] * fa

            pols[:, 0] = np.cos(ang)
            pols[:, 1] = np.sin(ang)

            ps0, ps1 = ps[:, 0], ps[:, 1]
            pol0m, pol1m = pols[mask, 0], pols[mask, 1]
            A_ts = ps0*pol0m + ps1*pol1m
            A_tp = ps1*pol0m - ps0*pol1m

            pp_ = misc.cross(ps, sm)
            pols[mask] = ps*A_ts[:, np.newaxis] + pp_*A_tp[:, np.newaxis]

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

            case "_image" if val is not None:

                if val.shape[0]*val.shape[1] > self._max_image_px:
                    raise RuntimeError("For performance reasons only images with less than 2 megapixels are allowed.")
                
                if np.sum(val._data) <= 0:
                    raise ValueError("Image can not be completely black")

            case "front":
                if not (isinstance(val, Point | Line)\
                        or (isinstance(val, Surface) and val.is_flat()) or isinstance(val, RGBImage)):
                    raise ValueError("Currently only planar surfaces are supported for RaySources.")

        super().__setattr__(key, val)
