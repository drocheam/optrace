from __future__ import annotations
from typing import Any  # Callable and Any type
from threading import Thread  # multithreading

import numpy as np  # calculations
import cv2
import scipy.constants  # for luminous efficacy
import scipy.special  # bessel functions

from ..base_class import BaseClass  # parent class
from ...property_checker import PropertyChecker as pc  # check types and values
from . import RGBImage, LinearImage

from .. import misc
from .. import color  # xyz_observers curves and sRGB conversions

from ...global_options import global_options

# from .image.base_image import BaseImage

# odd number of pixels per side, so we have a center pixel, which is useful in images with rotational symmetry
# otherwise a pixel at the center would fall in one of the neighboring pixel due to numeric errors
# 
# we render the image in the maximum resolution from the SIZES list
# it afterwards is rescaled to the desired size, which is the nearest of the SIZES array
# rescaling is done by summing neighboring pixels, the conversion is therefore is distinct and physically correct
# the side ratio of the image must be one of SIZES or 1/SIZES, so rescaling works for both dimensions
# for arbitrary side ratio this results in non-square pixels


class RenderImage(BaseClass):

    EPS: float = 1e-9
    """ Used for minimal extent """

    K: float = scipy.constants.physical_constants["luminous efficacy"][0]
    """Luminous Efficacy"""
    
    SIZES: list[int] = [1, 3, 5, 7, 9, 15, 21, 27, 35, 45, 63, 105, 135, 189, 315, 945]
    """valid image side lengths (smallest side)"""
    
    MAX_IMAGE_SIDE: int = SIZES[-1]
    """ maximum size of smaller image side in pixels"""
    
    MAX_IMAGE_RATIO: int = SIZES[2]
    """ maximum ratio of image side lengths. Images with ratios beyond will be corrected """

    image_modes: list[str] = ["sRGB (Absolute RI)", "sRGB (Perceptual RI)", "Outside sRGB Gamut", "Irradiance",
                              "Illuminance", "Lightness (CIELUV)", "Hue (CIELUV)", "Chroma (CIELUV)",
                              "Saturation (CIELUV)"]
    """possible display modes for the RenderImage"""


    def __init__(self,
                 extent:            (list | numpy.ndarray),
                 projection:        str = None,
                 **kwargs)\
            -> None:
        """
        Init an RenderImage object.
        This class is used to calculate
        and hold an Image consisting of the channels X, Y, Z, Illuminance and Irradiance.

        :param extent: image extent in the form [xs, xe, ys, ye]
        :param projection: string containing information about the sphere projection method, if any
        :param kwargs: additional keyword arguments for the BaseClass, like for instance desc and long_desc
        """
        self._new_lock = False

        self.extent: np.ndarray = extent
        """the image extent as [x0, x1, y0, y1] array"""

        self._extent0 = self.extent.copy()
        # initial extent before __fix_extent() changes

        self._data = None
        self._limit = None

        self.projection = projection

        super().__init__(**kwargs)

        self._new_lock = True

    def has_image(self) -> bool:
        """Check Image object contains an calculated image."""
        return self._data is not None

    def __check_for_image(self) -> None:
        """throw exception when image is missing"""
        if not self.has_image():
            raise RuntimeError("Image was not calculated.")

    @property
    def s(self) -> list[float, float]:
        """ geometric side lengths of image in direction (x, y)"""
        return [float(self.extent[1] - self.extent[0]), float(self.extent[3] - self.extent[2])]

    @property
    def shape(self) -> tuple[int, int]:
        """data image shape (y, x)"""
        self.__check_for_image()
        return self._data.shape

    @property
    def data(self) -> numpy.ndarray:
        """image data array"""
        return self._data.copy()

    @property
    def Apx(self) -> float:
        """ area per pixel """
        self.__check_for_image()
        return self.s[0] * self.s[1] / (self.shape[1] * self.shape[0])

    def power(self) -> float:
        """:return: calculated total image power"""
        self.__check_for_image()
        return float(np.sum(self._data[:, :, 3]))

    def luminous_power(self) -> float:
        """:return: calculated total image luminous power"""
        self.__check_for_image()
        return float(self.K * np.sum(self._data[:, :, 1]))

    @property
    def limit(self) -> float:
        """the resolution limit the RenderImage was rendered with"""
        return self._limit

    def get(self,
            mode:       str,
            N:          int = 315,
            L_th:       float = 0,
            chroma_scale:  float = None)\
            -> RGBImage | LinearImage:
        """
        Get a converted image with mode 'mode'. Must be one of RenderImage.image_modes.
        
        N describes the pixel count (of the smaller side) to which the image is rescaled to.
        N should be one of RenderImage.SIZES, but the nearest value is automatically selected.
        Rescaling is done by joining bins, so there is no interpolation.

        Depending on the image mode, the returned image is an RGBImage with three channels
        or a LinearImage with one channel.

        Parameters L_th and chroma_scale are only needed for mode='sRGB (Perceptual RI)',
        see function color.xyz_to_srgb_linear for more details.

        :param mode: one of RenderImage.image_modes
        :param N: pixel count of smaller side, nearest of RenderImage.SIZES is automatically selected
        :param L_th: lightness threshold for mode "sRGB (Perceptual RI)" 
        :param chroma_scale: chroma_scale option for mode "sRGB (Perceptual RI)"
        :return: RGBImage or LinearImage, depending on 'mode'
        """
        self.__check_for_image()

        N = int(N)  # enforce int
        if not 1 <= N <= self.MAX_IMAGE_SIDE:
            raise ValueError(f"N needs to be between 1 and {self.MAX_IMAGE_SIDE}")
        
        iargs = dict(extent=self.extent, projection=self.projection, desc=self.desc, 
                     long_desc=self.long_desc, quantity=mode, limit=self.limit)

        # get downscaling factor
        Ny, Nx, Nz = self._data.shape
        Na  = self.SIZES[np.argmin(np.abs(N - np.array(self.SIZES)))]
        fact = int(self.MAX_IMAGE_SIDE/Na)

        if fact != 1:
            # rescale by joining bins
            # we could just resize the relevant channel(s) instead of all of them,
            # but performance of cv2 seems to be optimized for four channels
            img = cv2.resize(self._data, [Nx // fact, Ny // fact], interpolation=cv2.INTER_AREA) #* fact**2
        else:
            img = self._data.copy()

        match mode:

            case "Irradiance":
                data = 1 / self.Apx * img[:, :, 3]
                return LinearImage(data, **iargs)

            case "Illuminance":
                # the Illuminance is just the unnormalized Y scaled by K = 683 lm/W and the inverse pixel area
                data = self.K / self.Apx * img[:, :, 1]
                return LinearImage(data, **iargs)

            case ("sRGB (Absolute RI)" |"sRGB (Perceptual RI)"):

                rendering_intent = "Absolute" if mode == "sRGB (Absolute RI)" else "Perceptual"
                data = color.xyz_to_srgb(img[:, :, :3], rendering_intent=rendering_intent, 
                                         L_th=L_th, chroma_scale=chroma_scale)

                return RGBImage(data, **iargs)

            case "Outside sRGB Gamut":
                # force conversion from bool to float64 so further algorithms work correctly
                data = np.array(color.outside_srgb_gamut(img[:, :, :3]), dtype=np.float64)
                return LinearImage(data, **iargs)

            case "Lightness (CIELUV)":
                data = color.xyz_to_luv(img[:, :, :3])[:, :, 0]
                return LinearImage(data, **iargs)

            case "Hue (CIELUV)":
                luv = color.xyz_to_luv(img[:, :, :3])
                data = color.luv_hue(luv)
                return LinearImage(data, **iargs)

            case "Chroma (CIELUV)":
                luv = color.xyz_to_luv(img[:, :, :3])
                data = color.luv_chroma(luv)
                return LinearImage(data, **iargs)

            case "Saturation (CIELUV)":
                luv = color.xyz_to_luv(img[:, :, :3])
                data = color.luv_saturation(luv)
                return LinearImage(data, **iargs)

            case _ :
                raise ValueError(f"Invalid display_mode {mode}, should be one of {self.image_modes}.")

    def __fix_extent(self) -> None:
        """
        Point images are given a valid 2D extent.
        Line images or images with a large side-to-side ratio are adapted.
        The image is extented to fit its convolved version (if RenderImage._limit is not None)
        """

        sx, sy = self.s  # use copies since extent changes along the way
        MR = self.MAX_IMAGE_RATIO  # alias for more readability
        self.extent = self._extent0.copy()

        # point image => make minimal dimensions
        if sx < 2*self.EPS and sy < 2*self.EPS:
            self.extent += self.EPS * np.array([-1, 1, -1, 1])

        # x side too small, expand
        elif not sx or sy/sx > MR:
            xm = (self._extent0[0] + self._extent0[1])/2  # center x position
            self.extent[0] = xm - sy/MR/2
            self.extent[1] = xm + sy/MR/2

        # y side too small, expand
        elif not sy or sx/sy > MR:
            ym = (self._extent0[2] + self._extent0[3])/2  # center y position
            self.extent[2] = ym - sx/MR/2
            self.extent[3] = ym + sx/MR/2

        # when filtering is active, add edges
        if self._limit is not None:
            # third zero would be at 2.655.. not 2.7
            self.extent += np.array([-1.0, 1.0, -1.0, 1.0]) * 2.7 * self._limit/1000.0

        
    def _apply_rayleigh_filter(self):
        """applies the rayleigh filter"""

        if self._limit is not None and self.projection is not None:
            raise RuntimeError("Resolution limit filter is not applicable for a projected image.")
            
        # limit size in pixels
        px = self._limit / 1000.0 / (self.s[0] / self._data.shape[1])
        py = self._limit / 1000.0 / (self.s[1] / self._data.shape[0])

        # larger of two sizes
        # third zero would be at 2.655.. not 2.7
        ps = int(np.ceil(2.7*max(px, py)))
        ps = ps+1 if ps % 2 else ps  # enforce even number, so we can use odd number for psf image

        # psf coordinates
        Y, X = np.mgrid[-ps:ps:(2*ps+1)*1j, -ps:ps:(2*ps+1)*1j]
        R = np.sqrt((X/px)**2 + (Y/py)**2) * 3.8317
        Rnz = R[R!=0]
        
        # psf
        psf = np.ones((2*ps+1, 2*ps+1), dtype=np.float64)
        psf[R != 0] = (2*scipy.special.j1(Rnz) / Rnz) ** 2
        psf[R > 10.1735] = 0  # only use up to third zero at 10.1735
        psf /= np.sum(psf)

        if global_options.multithreading:
            # for each of the XYZW channels:
            def threaded(ind, in_, psf):
                in_[:, :, ind] = scipy.signal.fftconvolve(in_[:, :, ind], psf, mode="same")

            threads = [Thread(target=threaded, args=(i, self._data, psf)) for i in range(self._data.shape[2])]
            [th.start() for th in threads]
            [th.join() for th in threads]

        else:
            self._data = scipy.signal.fftconvolve(self._data, psf[:, :, np.newaxis], mode="same", axes=(0, 1))

        # remove negative values that can arise by fft
        self._data[self._data < 0] = 0

    def save(self, path: str) -> None:
        """
        Save the RenderImage as .npz archive.
        Files are overridden. Throws IOError if the file could not been saved.

        :param path: path to save to
        """
        limit = self._limit if self._limit is not None else np.nan
        sdict = dict(_data=self._data, extent=self.extent, limit=limit,
                     desc=self.desc, long_desc=self.long_desc, proj=str(self.projection))

        # add file type and save
        path_ = path if path[-4:] == ".npz" else path + ".npz"
        np.savez_compressed(path_, **sdict)

    @staticmethod
    def load(path: str) -> RenderImage:
        """
        Load a saved RenderImage (.npz) from disk.

        :param path: path of the RenderImage archive
        :return: a saved image object from numpy archive to a image object
        """
        io = np.load(path)

        im = RenderImage(io["extent"], long_desc=io["long_desc"][()], desc=io["desc"][()], projection=io["proj"][()])
        im._limit = io["limit"][()] if not np.isnan(io["limit"]) else None
        im.projection = None if im.projection == "None" else im.projection  # None has been saved as string
        im._data = io["_data"]

        return im

    def __setattr__(self, key: str, val: Any) -> None:
        """
        assigns the value of an attribute
        :param key: attribute name
        :param val: value to assign
        """
        match key:
            
            case "extent":
                pc.check_type(key, val, list | tuple | np.ndarray)
                val2 = np.asarray_chkfinite(val, dtype=np.float64)

                if val2.shape[0] != 4:
                    raise ValueError("Extent needs to have 4 elements.")

                if val2[0] > val[1] or val2[2] > val2[3]:
                    raise ValueError("Extent needs to be an array with [x0, x1, y0, y1] with x0 < x1 and y0 < y1.")

                super().__setattr__(key, val2)
                return

            case "projection":
                pc.check_type(key, val, str | None)
            
            case "_limit" if val is not None:
                pc.check_type(key, val, float | int)
                pc.check_above(key, val, 0)
                val = float(val)

        super().__setattr__(key, val)


    def render(self,
               p:               np.ndarray = None,
               w:               np.ndarray = None,
               wl:              np.ndarray = None,
               limit:           float = None,
               _dont_filter:    bool = False)\
            -> None:
        """
        Creates an pixel image from ray positions on the detector.

        :param p: ray position matrix, xyz components in columns, (numpy 2D array)
        :param w: ray weight array (numpy 1D array)
        :param wl: ray wavelength array (numpy 1D array)
        :param limit: rayleigh limit, used to approximate wave optics using a airy disc kernel
        """
        self._limit = limit

        # fix point and line images as well as ones with a too large side ratio
        self.__fix_extent()

        # calculate image size. Smaller side is MAX_IMAGE_SIDE, larger MAX_IMAGE_SIDE*[1, 3, 5, ..., MAX_IMAGE_RATIO]
        # the resulting pixel size is not square. And the user doesn't exactly get his desired resolution
        Nrs = self.MAX_IMAGE_SIDE
        nf = lambda a: min(self.MAX_IMAGE_RATIO, 1 + 2*int(a/2))  
        # ^-- calculates nearest factor for a from [1, 3, 5, ..] below MAX_IMAGE_RATIO
        Nx = Nrs if self.s[0] <= self.s[1] else Nrs*nf(self.s[0]/self.s[1])
        Ny = Nrs if self.s[0] > self.s[1] else Nrs*nf(self.s[1]/self.s[0]) 

        # init image
        # x in first, y in second since np.histogram2d needs it that way
        self._data = np.zeros((Nx, Ny, 4), dtype=np.float64)

        if p is not None and p.shape[0]:

            # threading function
            def func(img, ind):
               
                tri = [color.x_observer, color.y_observer, color.z_observer]
                w_ = tri[ind](wl) * w if ind < 3 else w

                img[:, :, ind], _, _ = np.histogram2d(p[:, 0], p[:, 1], weights=w_, bins=[Nx, Ny], 
                                                      range=self.extent.reshape((2, 2)))

            # multithreading
            if global_options.multithreading and misc.cpu_count() > 3: 

                threads = [Thread(target=func, args=(self._data, i)) for i in np.arange(4)]
                [thread.start() for thread in threads]
                [thread.join() for thread in threads]

            # no multithreading
            else:
                [func(self._data, i) for i in np.arange(4)]

        # transpose since histogram2d returns x in first dimension, y in second
        self._data = np.transpose(self._data, (1, 0, 2))

        if not _dont_filter and self._limit is not None:
            self._apply_rayleigh_filter()  # apply resolution filter
