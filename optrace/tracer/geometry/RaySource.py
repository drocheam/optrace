
"""
RaySource class:
This class generates the rays depending 
on the specified positional, directional, wavelength and brightness distributions.
The RaySource object also holds all rays and ray sections generated in raytracing from the Raytracer class
"""

# Why random sampling? Sampling the source, "sampling" the lens areas or aperture by the rays can lead to Nyquist Theorem violation. Also, this ensures that when you run it repeatedly, you get a different version of the image, and not the same one. E.g. with image compositions by several raytraces.

import numpy as np
from typing import Callable

from optrace.tracer.geometry.Surface import * 
from optrace.tracer.geometry.SObject import *
from optrace.tracer.spectrum.LightSpectrum import *

from optrace.tracer.Misc import timer as timer
import optrace.tracer.Misc as misc
import optrace.tracer.Color as Color

# TODO remove BW_Image, ersetzen durch emittance_type = "Constant", "Image"
# TODO Check: check image dimensions
class RaySource(SObject):

    direction_types = ["Diverging", "Parallel"]
    orientation_types = ["Constant", "Function"]
    polarization_types = ["Angle", "Random", "x", "y", "xy"]

    abbr = "RS"  # object abbreviation
    _allow_non_2D = True  # allow points or lines as surfaces

    def __init__(self,

                 # Surface Parameters
                 Surface:           Surface,
                 pos:               (list | np.ndarray) = [0., 0., 0.],
                
                 # Direction Parameters
                 direction_type:    str = "Parallel",
                 s:                 (list | np.ndarray) = [0., 0., 1.],
                 sr_angle:          float = 20.,

                 # Light Parameters
                 spectrum:          LightSpectrum = None,
                 power:             float = 1.,
                 Image:             (str | np.ndarray) = None,
                 
                 # Ray Orientation Parameters
                 orientation_type:  str = "Constant",
                 or_func:           Callable[[np.ndarray, np.ndarray], np.ndarray] = None,
                
                 # Polarization Parameters
                 polarization_type: str = "Random",
                 pol_ang:           float = 0.,

                 **kwargs)\
            -> None:
        """
        Create a RaySource with a specific source_type, direction_type and light_type.

        :param Surface:
        :param direction_type: "Diverging" or "Parallel"
        :param orientation_type: "Constant" or "Function"
        :param s: 3D direction vector
        :param sr_angle: cone opening angle in degree in mode direction_type="Diverging"
        :param pos: 3D position of RaySource center
        :param power: total power of RaySource in W
        :param pol: polarisation angle as float. Specify 'x' or 'y' for x- or y-polarisation, 'xy' for both
                    and leave empty for unpolarized light
        :param or_func: orientation function for orientation_type="Function",
            takes 1D array of x and y coordinates as input, returns (N, 3) numpy 2D array with orientations
        :param Image: image for modes source_type="BW_Image" or "RGB_Image",
                specified as path (string) or RGB array (numpy 3D array)
        """
        self._new_lock = False

        self.pIf = None
        
        super().__init__(Surface, pos, **kwargs)

        self.direction_type = direction_type
        self.orientation_type = orientation_type
        self.polarization_type = polarization_type
        
        self.sr_angle = sr_angle
        self.pol_ang = pol_ang
        self.power = power
        self.spectrum = spectrum
        self.Image = Image
        self.or_func = or_func
        self.s = s

        # lock assignment of new properties. New properties throw an error.
        self._new_lock = True

    def getColor(self) -> tuple[float, float, float, float]:
        """

        :return:
        """
        if self.Image is not None:
            return np.mean(self.Image[:, :, 0]), np.mean(self.Image[:, :, 1]),\
                   np.mean(self.Image[:, :, 2]), 1.0
        else:
            return self.spectrum.getColor()
    
    def createRays(self, N: int, no_pol: bool = False, power: float=None)\
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
        power = self.power if power is None else power
        weights = np.full(N, power/N, dtype=np.float32)

        ## Generate ray wavelengths
        ################################################################################################################

        if self.Image is None:
            wavelengths = self.spectrum.randomWavelengths(N)

        ## Generate ray starting points
        ################################################################################################################

        if self.Image is None:
            p = self.Surface.getRandomPositions(N)

        else:
            if self.Surface.surface_type != "Rectangle":
                raise RuntimeError("Images can only be used with surface_type='Rectangle'")
            
            if self.Image is None:
                raise RuntimeError("Image parameter missing")

            # get random pixel number, the pixel brightness is the probability
            P = np.random.choice(self.pIf.shape[0], N, p=self.pIf)
            PY, PX = np.divmod(P, self.Image.shape[1])  # pixel x, y position from pixel number

            # add random position inside pixel and calculate positions in 3D space
            rx, ry = np.random.sample(N), np.random.sample(N)
            xs, xe, ys, ye = self.Surface.getExtent()[:4]
            Iy, Ix = self.Image.shape[:2]

            p = np.zeros((N, 3), dtype=np.float64, order='F')
            misc.calc("(xe-xs)/Ix * (PX + rx) + xs", out=p[:, 0])
            misc.calc("(ye-ys)/Iy * (PY + ry) + ys", out=p[:, 1])
            p[:, 2] = self.pos[2]

            wavelengths = Color.randomWavelengthFromRGB(self.Image[PY, PX])

        ## Generate orientations
        ################################################################################################################

        match self.orientation_type:

            case "Constant":
                s_or = np.tile(self.s, (N, 1))

            case "Function":
                if self.or_func is None:
                    raise RuntimeError("orientation_type='Function' but or_func is None")

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
                alpha = np.radians(self.sr_angle) * np.sqrt(np.random.sample(N))
                theta = np.random.uniform(0, 2*np.pi, N)

                # vector perpendicular to s, created using  sy = [1, 0, 0] x s_or
                sy = np.zeros_like(s_or, dtype=np.float64, order='F')
                s_orx, s_ory, s_orz = s_or[:, 0], s_or[:, 1], s_or[:, 2]
                sy[:, 1] = misc.calc("-s_orz/sqrt(1-s_orx**2)")
                sy[:, 2] = misc.calc("s_ory/sqrt(1-s_orx**2)")

                # vector sx = s x sy
                sx = misc.cross(s_or, sy)

                # vector s has a component alpha in the sx-sy plane
                theta_, alpha_ = theta[:, np.newaxis], alpha[:, np.newaxis]
                s = misc.calc("cos(alpha_)*s_or + sin(alpha_)*(cos(theta_)*sx + sin(theta_)*sy)")

            case _:
                raise RuntimeError(f"direction_type '{self.direction_type}' not handled.")

        if np.any(s[:, 2] <= 0):
            raise RuntimeError("All ray directions s need to be in positive z-direction")

        ## Assign ray polarization
        ################################################################################################################

        if no_pol:
            pols = np.full(p.shape, np.nan, np.float64, order='F')

        else:
            match self.polarization_type:
                case "x":       ang = 0.
                case "y":       ang = np.pi/2
                case "xy":      ang = np.random.choice([0, np.pi/2], N)
                case "Angle":   ang = np.radians(self.pol_ang)
                case "Random":  ang = np.random.uniform(0, 2*np.pi, N)
                case _:         raise RuntimeError(f"polarization_type '{self.polarization_type}' not handled.")

            # pol is rotated by an axis perpendicular to the plane of base direction s = [0, 0, 1] and the current direction s_
            # let's call this axis ps. The resulting polarization pol_ is perpendicular to s_ and has the same component at ps
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
            misc.calc("s_1/sqrt(1-s_2**2)", out=ps[:, 0])
            misc.calc("-s_0/sqrt(1-s_2**2)", out=ps[:, 1])

            pols[:, 0] = misc.calc("cos(ang)")
            pols[:, 1] = misc.calc("sin(ang)")

            ps0, ps1 = ps[:, 0], ps[:, 1]
            pol0m, pol1m = pols[mask, 0], pols[mask, 1]
            A_ts = misc.calc("ps0*pol0m + ps1*pol1m")[:, np.newaxis]
            A_tp = misc.calc("ps1*pol0m - ps0*pol1m")[:, np.newaxis]

            pp_ = misc.cross(ps, sm)
            pols[mask] = misc.calc("A_ts*ps + A_tp*pp_")

        ## return ray properties
        ################################################################################################################

        return p, s, pols, weights, wavelengths

    def __setattr__(self, key, val0):

        # work on copies of ndarray and list, so initial objects are not overwritten
        val = val0.copy() if isinstance(val0, list | np.ndarray) else val0

        match key:

            case "direction_type":
                self._checkType(key, val, str)
                self._checkIfIn(key, val, self.direction_types)

            case "orientation_type":
                self._checkType(key, val, str)
                self._checkIfIn(key, val, self.orientation_types)

            case "polarization_type":
                self._checkType(key, val, str)
                self._checkIfIn(key, val, self.polarization_types)

            case ("power" | "sr_angle" | "pol_ang"):
                self._checkType(key, val, int | float)
                self._checkNotBelow(key, val, 0)
                val = float(val)

            case "s":
                self._checkType(key, val, list | np.ndarray)
                val = np.array(val, dtype=np.float64) / np.linalg.norm(val)  # normalize

            case "spectrum":
                self._checkType(key, val, LightSpectrum | None)

            case "or_func":
                self._checkNoneOrCallable(key, val)

            case "Image" if val is not None:

                self._checkType(key, val, str | np.ndarray)

                val = misc.loadImage(val) if isinstance(val, str) else np.array(val, dtype=np.float64)

                if val.shape[0]*val.shape[1] > 2e6:
                    raise RuntimeError("For performance reasons only images with less than 2 megapixels are allowed.")

                val = np.flipud(val)

                # calculate pixel probability from relative power for each pixel
                If = Color._PowerFromSRGB(val).flatten()
                self.pIf = 1/np.sum(If)*If

            case "FrontSurface":
                self._checkType(key, val, Surface)

                if not val.isPlanar():
                    raise ValueError("Currently only planar surfaces are supported for RaySources.")

        super().__setattr__(key, val)

