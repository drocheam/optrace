
"""
RaySource class:
This class generates the rays depending 
on the specified positional, directional, wavelength and brightness distributions.
The RaySource object also holds all rays and ray sections generated in raytracing from the Raytracer class
"""


# Why random sampling? Sampling the source, "sampling" the lens areas or aperture by the rays can lead to Nyquist Theorem violation. Also, this ensures that when you run it repeatedly, you get a different version of the image, and not the same one. E.g. with image compositions by several raytraces.

import numpy as np
import numexpr as ne

from typing import Callable

from PIL import Image as PILImage
from scipy.spatial.transform import Rotation

import Backend.Color as Color
from Backend.Surface import Surface as Surface
from Backend.SObject import SObject
from Backend.Misc import random_from_distribution
from Backend.Misc import timer as timer


# TODO check light_type=Function
# TODO remove BW_Image, ersetzen durch emittance_type = "Constant", "Image"
# TODO welche lines sind typisch?
# TODO Check: check image dimensions
class RaySource(SObject):

    def __init__(self,

                 # Surface Parameter
                 Surface:           Surface,
                
                 # Direction Parameters
                 direction_type:    str = "Parallel",
                 s:                 (list | np.ndarray) = [0., 0., 1.],
                 sr_angle:          float = 20.,

                 # Light Parameters
                 light_type:        str = "Monochromatic",
                 power:             float = 1.,
                 wl:                float = 550.,
                 lines:             (list | np.ndarray) = [486.13, 589.29, 656.27],
                 spec_func:         Callable[[np.ndarray], np.ndarray] = None,
                 T:                 float = 6504.,
                 Image:             (str | np.ndarray) = None,
                 
                 # Ray Orientation Parameters
                 orientation_type:  str = "Constant",
                 or_func:           Callable[[np.ndarray, np.ndarray], np.ndarray] = None,
                
                 # Polarization Parameters
                 polarization_type: str = "Random",
                 pol_ang:           float = 0,

                 # Misc
                 pos:               (list | np.ndarray) = [0., 0., 0.])\
            -> None:
        """
        Create a RaySource with a specific source_type, direction_type and light_type.

        :param Surface:
        :param direction_type: "Diverging" or "Parallel"
        :param orientation_type: "Constant" or "Function"
        :param light_type: "Monochromatic", "Blackbody", "Function", "Lines", "BW_Image", "RGB_Image" 
                            or one of "A", "C", "D50", "D55", "D65", "D75", "E", "F2", "F7", "F11".
        :param s: 3D direction vector
        :param sr_angle: cone opening angle in degree in mode direction_type="Diverging"
        :param pos: 3D position of RaySource center
        :param wl: light wavelength in nm in mode light_type="Monochromatic"
        :param spec_func: spectrum function for light_type="Function". Must take an argument in range [380, 780].
        :param lines: wavelengths of spectral lines
        :param power: total power of RaySource in W
        :param pol: polarisation angle as float. Specify 'x' or 'y' for x- or y-polarisation, 'xy' for both
                    and leave empty for unpolarized light
        :param T: blackbody color in K for light_type="Blackbody"
        :param or_func: orientation function for orientation_type="Function",
            takes 1D array of x and y coordinates as input, returns (N, 3) numpy 2D array with orientations
        :param Image: image for modes source_type="BW_Image" or "RGB_Image",
                specified as path (string) or RGB array (numpy 3D array)
        """

        super().__init__(Surface, pos)

        if direction_type not in ["Parallel", "Diverging"]:
            raise ValueError(f"Invalid direction_type '{direction_type}'.")

        if light_type not in ["Monochromatic", "Blackbody", "Function", "Lines", "RGB_Image",\
                              "A", "C", "D50", "D55", "D65", "D75", "E", "F2", "F7", "F11"]:
            raise ValueError(f"Invalid light_type '{light_type}'.")

        if orientation_type not in ["Constant", "Function"]:
            raise ValueError(f"Invalid orientation_type '{orientation_type}'.")

        if polarization_type not in ["x", "y", "xy", "Random", "Angle"]:
            raise ValueError(f"Invalid polarization_type '{polarization_type}'.")

        self.direction_type = direction_type
        self.orientation_type = orientation_type
        self.light_type = light_type
        self.polarization_type = polarization_type

        if not Surface.isPlanar():
            raise ValueError("Currently only planar surfaces are supported for RaySources.")

        self.sr_angle = float(sr_angle)
        self.pol_ang = float(pol_ang)
        self.or_func = or_func
        self.spec_func = spec_func
        self.power = float(power)
        self.T = float(T)
        self.wl = float(wl)

        # assign arrays, force conversion to np.array
        self.s = np.array(s, dtype=np.float64) / np.linalg.norm(s)  # normalize
        self.lines = np.array(lines, dtype=np.float32)

        if light_type == "RGB_Image":
            if isinstance(Image, str):
                self.Image = np.asarray(PILImage.open(Image).convert("RGB"), dtype=np.float64) / 2**8
            elif isinstance(Image, np.ndarray):
                self.Image = np.array(Image, dtype=np.float64)
            else:
                raise ValueError("Invalid image format")

            self.Image = np.flipud(self.Image)

        # check parameters
        if power <= 0:
            raise ValueError(f"Source power needs to be positive, but is {power}.")

        if sr_angle <= 0:
            raise ValueError(f"Cone angle sr_angle needs to be positive, but is {sr_angle}.")

        if T <= 0:
            raise ValueError(f"Blackbody temperature T needs to be positive, but is {T}.")

        if self.lines.shape[0] == 0:
            raise ValueError("'lines' can't be empty.")

        if (wlo := np.min(self.lines)) < 380 or (wlo := np.max(self.lines)) > 780:
            raise ValueError(f"'lines' need to be inside visible range [380nm, 780nm], but got a value of {wlo}nm.")

        if self.light_type == "Function" and spec_func is None:
            raise ValueError("light_type='Function', but spec_func not specified.")

        if self.orientation_type == "Function" and or_func is None:
            raise ValueError("orientation_type='Function', but or_func not specified.")


    def getColor(self) -> tuple[float, float, float]:
        """

        :return:
        """

        match self.light_type:

            case "RGB_Image":
                return np.mean(self.Image[:, :, 0]),\
                       np.mean(self.Image[:, :, 1]),\
                       np.mean(self.Image[:, :, 2])

            case ("A" | "C" | "D50" | "D55" | "D65" | "D75" | "E" | "F2" | "F7" | "F11"):
                wl = np.linspace(380, 780, 4000)
                spec = Color.Illuminant(wl, self.light_type)
                
            case "Blackbody":
                wl = np.linspace(380, 780, 4000)
                spec = Color.Blackbody(wl, T=self.T)
            
            case "Function":
                wl = np.linspace(380, 780, 4000)
                spec = self.spec_func(wl)

            case "Monochromatic":
                wl = np.array([self.wl])
                spec = np.array([1.])

            case "Lines":
                wl = np.array(self.lines)
                spec = np.ones_like(wl)

            case _:
                raise RuntimeError(f"light_type '{self.light_type}' not handled in getColor()")

        Xc = np.sum(spec * Color.Tristimulus(wl, "X"))
        Yc = np.sum(spec * Color.Tristimulus(wl, "Y"))
        Zc = np.sum(spec * Color.Tristimulus(wl, "Z"))

        XYZ = np.array([[[Xc, Yc, Zc]]])
        RGB = Color.XYZ_to_sRGB(XYZ)[0, 0]

        return tuple(RGB)

    def createRays(self, N: int, no_pol: bool = False)\
            -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate N rays and save them internally.

        :param N: number of rays (int)
        :param no_pol:
        :return: position array (numpy 2D array), direction array (numpy 2D array), weight array (numpy 1D array),
            first dimension: ray number, second dimension: values
        """

        ## Generate ray weights
        ################################################################################################################
        weights = np.full((N, ), self.power/N, dtype=np.float32)

        ## Generate ray wavelengths
        ################################################################################################################

        match self.light_type:

            case "Monochromatic":
                wavelengths = np.full((N,), self.wl, dtype=np.float32)

            case"Lines":
                wavelengths = np.random.choice(self.lines, N)

            case ("A" | "C" | "D50" | "D55" | "D65" | "D75" | "E" | "F2" | "F7" | "F11"):
                wl = np.linspace(380, 780, 4000)
                wavelengths = random_from_distribution(wl, Color.Illuminant(wl, self.light_type), N)

            case "Blackbody":
                wl = np.linspace(380, 780, 4000)
                wavelengths = random_from_distribution(wl, Color.Blackbody(wl, T=self.T), N)
            
            case "Function":
                if self.spec_func is None:
                    raise ValueError("spec_func not defined for light_type='Function'")

                wl = np.linspace(380, 780, 10000)
                wavelengths = random_from_distribution(wl, self.spec_func(wl), N)

            case "RGB_Image":
                pass  # will be handled later

            case _:
                raise RuntimeError(f"light_type '{self.light_type}' not handled.")

        ## Generate ray starting points
        ################################################################################################################

        if self.light_type != "RGB_Image":
            p = self.Surface.getRandomPositions(N)
        else:
            if self.Surface.surface_type != "Rectangle":
                raise RuntimeError("Images can only be used with surface_type='Rectangle'")

            RGB = Color.sRGB_to_sRGBLinear(self.Image)  # physical brightness is proportional to RGBLinear signal
            If = np.sum(RGB, axis=2).flatten()  # brightness is sum of RGBLinear components

            # get random pixel number, the pixel brightness is the probability
            P = np.random.choice(np.arange(If.shape[0]), N, p=If/np.sum(If))
            PY, PX = np.divmod(P, self.Image.shape[1])  # pixel x, y position from pixel number

            # add random position inside pixel and calculate positions in 3D space
            rx, ry = np.random.sample(N), np.random.sample(N)
            xs, xe, ys, ye = self.Surface.getExtent()[:4]
            Iy, Ix = self.Image.shape[:2]

            p = np.zeros((N, 3), dtype=np.float64, order='F')
            p[:, 2] = self.pos[2]

            ne.evaluate("(xe-xs)/Ix * (PX + rx) + xs", out=p[:, 0])
            ne.evaluate("(ye-ys)/Iy * (PY + ry) + ys", out=p[:, 1])

            wavelengths = Color.randomWavelengthFromRGB(self.Image[PY, PX])


        ## Generate orientations
        ################################################################################################################

        match self.orientation_type:

            case "Constant":
                s_or = np.tile(self.s, (N, 1))

            case "Function":
                s_or = self.or_func(p[:, 0], p[:, 1])

            case _:
                raise RuntimeError(f"orientation_type '{self.orientation_type}' not handled.")

        ## Generate ray directions relative to orientation
        ################################################################################################################

        match self.direction_type:

            case "Parallel":
                s = s_or  # all rays have the same direction

            # TODO sind die Strahlen gleichmäßig diverging?
            case "Diverging":
                # random direction inside cone, described with two angles
                alpha = self.sr_angle / 180 * np.pi * np.sqrt(np.random.sample(N))
                theta = np.random.uniform(0, 2 * np.pi, N)

                # create rotation vectors
                rv = np.zeros_like(s_or, dtype=np.float64, order='F')
                rv[:, 1] = alpha

                # rotate vectors in first cone dimension
                r1 = Rotation.from_rotvec(rv)
                s4 = r1.apply(s_or)

                # variable aliases
                s1, s2, s3 = s4[:, 0], s4[:, 1], s4[:, 2]
                n1, n2, n3 = s_or[:, 0], s_or[:, 1], s_or[:, 2]
                ca, sa, mca = np.cos(theta), np.sin(theta),  1 - np.cos(theta)

                # Rotation Matrix for a rotation around an arbitrary axis,
                # source: https://de.wikipedia.org/wiki/Drehmatrix#Drehmatrizen_des_Raumes_%E2%84%9D%C2%B3

                s = np.zeros_like(s_or, dtype=np.float64, order='F')

                # rotate in second cone dimension
                ne.evaluate("(n1**2*mca + ca)*s1    + (n1*n2*mca - n3*sa)*s2 + (n1*n3*mca + n2*sa)*s3", out=s[:, 0])
                ne.evaluate("(n2*n1*mca + n3*sa)*s1 + (n2**2*mca + ca)*s2    + (n2*n3*mca - n1*sa)*s3", out=s[:, 1])
                ne.evaluate("(n3*n1*mca - n2*sa)*s1 + (n3*n2*mca + n1*sa)*s2 + (n3**2*mca + ca)*s3",    out=s[:, 2])

            case _:
                raise RuntimeError(f"direction_type '{self.direction_type}' not handled.")

        if np.any(s[:, 2] <= 0):
            raise RuntimeError("All ray directions s need to be in positive z-direction")

        ## Assign ray polarization
        ################################################################################################################

        if no_pol:
            pols = np.full(p.shape, np.nan, np.float64, order='F')

        else:
            pols = np.zeros_like(p, np.float64, order='F')
            
            match self.polarization_type:

                case "x":
                    ang = np.zeros(N, dtype=np.float64)

                case "y":
                    ang = np.full(N, np.pi/2, dtype=np.float64)

                case "xy":
                    ang = np.random.choice([0, np.pi/2], N)

                case "Angle":
                    ang = np.full(N, self.pol_ang/360*2*np.pi, dtype=np.float64)

                case "Random":
                    ang = np.random.uniform(0, 2*np.pi, N)
        
                case _:
                    raise RuntimeError(f"polarization_type '{self.polarization_type}' not handled.")

            # sh is the unity polarization vector in xz-plane perpendicular to the ray direction
            # sh = 1/sqrt(sx^2 + sz^2)*[sz, 0, -sx] with ray direction s = [sx, sy, sz]
            # sv is perpendicular to s and sh
            # sv = 1/sqrt(sx^2 + sz^2)*[-sx*sy, sx^2 + sz^2, -sy*sz]
            # sx^2 + sy**2 is the same as 1-sy**2 since s is unity vector

            # polarization vector is pol = cos(a)*sh + sin(a)*sv
            sx, sy, sz = s[:, 0], s[:, 1], s[:, 2]
            ne.evaluate("(sz*cos(ang) + -sx*sy*sin(ang))/sqrt(1-sy**2)",  out=pols[:, 0])
            ne.evaluate("sin(ang)*sqrt(1-sy**2)",                         out=pols[:, 1])
            ne.evaluate("(-sx*cos(ang) + -sy*sz*sin(ang))/sqrt(1-sy**2)", out=pols[:, 2])

        ## return ray properties
        ################################################################################################################

        return p, s, pols, weights, wavelengths

