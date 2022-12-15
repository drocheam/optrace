from typing import Any  # Callable and Any type
from threading import Thread  # multithreading

import numpy as np  # calculations
from PIL import Image  # saving as png
import scipy.interpolate
import scipy.constants  # for luminous efficacy
import scipy.ndimage

from . import color  # xyz_observers curves and sRGB conversions
from . import misc  # interpolation and calculation methods

from .base_class import BaseClass  # parent class
from .misc import PropertyChecker as pc  # check types and values


# odd number of pixels per side, so we have a center pixel, which is useful in images with rotational symmetry
# otherwise a pixel at the center would fall in one of the neighboring pixel due to numeric errors
# 
# we render the image in the maximum resolution from the SIZES list
# it afterwards is rescaled to the desired size, which is the nearest of the SIZES array
# rescaling is done by summing neighboring pixels, the conversion is therefore is distinct and physically correct
# the side ratio of the image must be one of SIZES or 1/SIZES, so rescaling works for both dimensions
# for arbitrary side ratio this results in non-square pixels


class RImage(BaseClass):

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

    display_modes: list[str] = ["sRGB (Absolute RI)", "sRGB (Perceptual RI)", "Outside sRGB Gamut", "Irradiance",
                                "Illuminance", "Lightness (CIELUV)", "Hue (CIELUV)", "Chroma (CIELUV)",
                                "Saturation (CIELUV)"]
    """possible display modes for the RImage"""


    def __init__(self,
                 extent:            (list | np.ndarray),
                 projection:        str = None,
                 limit:             float = None,
                 **kwargs)\
            -> None:
        """
        Init an Image object.
        This class is used to calculate
        and hold an Image consisting of the channels X, Y, Z, Illuminance and Irradiance.
        The class also includes information like extent, z-position of image,
        an image plotting type and an index for tagging.

        :param extent: image extent in the form [xs, xe, ys, ye]
        :param limit: rayleigh limit, used to approximate wave optics using a gaussian blur filter
        """
        self._new_lock = False

        self.extent: np.ndarray = extent
        """the image extent as [x0, x1, y0, y1] array"""

        self.img = None
        self._img = None

        self.projection = projection
        self.limit = limit

        super().__init__(**kwargs)
        self._new_lock = True

    def has_image(self) -> bool:
        """Check Image objects contains an calculated image."""
        return self.img is not None

    def __check_for_image(self) -> None:
        """throw exception when image is missing"""
        if not self.has_image():
            raise RuntimeError("Image was not calculated.")

    @property
    def _sigma(self) -> float:
        # an airy disc with roots at +-0.5*d can be approximated by a gauss with sigma = 0.175*d
        # limit (=d) needs to be converted from um to mm first
        return 0.175*self.limit/1000 if self.limit is not None else self.limit

    @property
    def sx(self) -> float:
        """ geometric size of image in x direction """
        return self.extent[1] - self.extent[0]

    @property
    def sy(self) -> float:
        """ geometric size of image in y direction """
        return self.extent[3] - self.extent[2]

    @property
    def Nx(self) -> int:
        """ number of image pixels in x direction """
        self.__check_for_image()
        return self.img.shape[1]

    @property
    def Ny(self) -> int:
        """number of image pixels in y direction """
        self.__check_for_image()
        return self.img.shape[0]

    @property
    def N(self) -> int:
        """returns smaller side pixel length of Nx, Ny"""
        return min(self.Nx, self.Ny)

    @property
    def Apx(self) -> float:
        """ area per pixel """
        self.__check_for_image()
        return self.sx * self.sy / (self.Nx * self.Ny)

    def cut(self,
            mode:   str,
            x:      float = None,
            y:      float = None,
            log:    bool = False,
            imc:    np.ndarray = None)\
            \
            -> tuple[np.ndarray, list[np.ndarray]]:
        """

        
        :param mode:
        :param x:
        :param y:
        :param log:
        :param imc:
        :return: bin edge array, list of image cuts
        """
        if (x is not None and not self.extent[0] - self.EPS <= x <= self.extent[1] + self.EPS)\
           or (y is not None and not self.extent[2] - self.EPS <= y <= self.extent[3] + self.EPS):
            raise ValueError("Position outside image.")
        
        # bin edges for histogram
        bins_x = np.linspace(self.extent[0], self.extent[1], self.Nx+1)
        bins_y = np.linspace(self.extent[2], self.extent[3], self.Ny+1)

        # sampling points with positions in the center of the bins
        xp = bins_x[:-1] + (bins_x[1]-bins_x[0])/2
        yp = bins_y[:-1] + (bins_y[1]-bins_y[0])/2
        
        # get image
        img = self.get_by_display_mode(mode, log) if imc is None else imc

        # assign bins and sampling positions
        sp, xs, ys = (bins_y, np.full(self.Ny, x), yp) if x is not None else (bins_x, xp, np.full(self.Nx, y))

        # nearest neighbor interpolation, so we use actual pixel values
        iml = [img] if img.ndim == 2 else [img[:, :, 0], img[:, :, 1], img[:, :, 2]]
        ims = [scipy.interpolate.RegularGridInterpolator((xp, yp), imi.T, method="nearest", 
                                                         bounds_error=False, fill_value=None)((xs, ys))\
               for imi in iml]

        return sp, ims

    def get_by_display_mode(self, mode: str, log: bool = False) -> np.ndarray:
        """
        Modes only include displayable modes from self.modes, use dedicated functions for Luv and XYZ

        :param mode:
        :param log:
        :return:
        """

        self.__check_for_image()

        match mode:

            case "Irradiance":
                return 1 / self.Apx * self.img[:, :, 3]

            case "Illuminance":
                # the Illuminance is just the unnormalized Y scaled by K = 683 lm/W and the inverse pixel area
                return self.K / self.Apx * self.img[:, :, 1]

            case "sRGB (Absolute RI)":
                # return
                rgb = self.get_rgb(log=log, rendering_intent="Absolute")

                return rgb

            case "sRGB (Perceptual RI)":
                return self.get_rgb(log=log, rendering_intent="Perceptual")

            case "Outside sRGB Gamut":
                # force conversion from bool to int so further algorithms work correctly
                return np.array(color.outside_srgb_gamut(self.get_xyz()), dtype=int)

            case "Lightness (CIELUV)":
                return self.get_luv()[:, :, 0]

            case "Hue (CIELUV)":
                Luv = self.get_luv()
                return color.luv_hue(Luv)

            case "Chroma (CIELUV)":
                Luv = self.get_luv()
                return color.luv_chroma(Luv)

            case "Saturation (CIELUV)":
                Luv = self.get_luv()
                return color.luv_saturation(Luv)

            case _:
                raise ValueError(f"Invalid display_mode {mode}, should be one of {self.display_modes}.")

    def get_power(self) -> float:
        """:return: calculated total image power"""
        self.__check_for_image()
        return np.sum(self.img[:, :, 3])

    def get_luminous_power(self) -> float:
        """:return: calculated total image luminous power"""
        self.__check_for_image()
        return self.K * np.sum(self.img[:, :, 1])

    def get_xyz(self) -> np.ndarray:
        """:return: XYZ image (np.ndarray with shape (Ny, Nx, 3))"""
        self.__check_for_image()
        return self.img[:, :, :3]

    def get_luv(self) -> np.ndarray:
        """:return: CIELUV image"""
        xyz = self.get_xyz()
        return color.xyz_to_luv(xyz)

    def get_rgb(self, log: bool = False, rendering_intent: str = "Absolute") -> np.ndarray:
        """
        Get sRGB image

        :param log: if brightness should be logarithmically scaled
        :param rendering_intent:
        :return: sRGB image (np.ndarray with shape (Ny, Nx, 3))
        """
        img = color.xyz_to_srgb_linear(self.get_xyz(), rendering_intent=rendering_intent)

        # addition, multiplication etc. only work correctly in the linear color space
        # otherwise we would change the color ratios, but we only want the brightness to change
        if log and np.any(img > 0):
            rgbs = np.sum(img, axis=2)  # assume RGB channel sum as brightness
            wmax = np.max(rgbs)  # maximum brightness
            wmin = np.min(rgbs[rgbs > 0])  # minimum nonzero brightness
            maxrgb = np.max(img, axis=2)  # highest rgb value for each pixel

            # normalize pixel so highest channel value is 1, then rescale logarithmically.
            # Highest value is 1, lowest 0. Exclude all zero channels (maxrgb = 0) for calculation
            fact = 1 / maxrgb[maxrgb > 0] * (1 - np.log(rgbs[maxrgb > 0] / wmax)/np.log(wmin / wmax))
            img[maxrgb > 0] *= fact[:, np.newaxis]

        img = np.clip(img, 0, 1)
        img = color.srgb_linear_to_srgb(img)
        return img

    def __fix_extent(self) -> None:
        """
        Fix image extent. Point images are given a valid 2D extent.
        Line images or images with a large side-to-side ratio are adapted.
        """

        sx, sy = self.sx, self.sy  # use copies since extent changes along the way
        MR = self.MAX_IMAGE_RATIO  # alias for more readability

        # point image => make minimal dimensions
        if sx < 2*self.EPS and sy < 2*self.EPS:
            self.extent += self.EPS * np.array([-1, 1, -1, 1])

        # x side too small, expand
        elif not sx or sy/sx > MR:
            xm = (self.extent[0] + self.extent[1])/2  # center x position
            self.extent[0] = xm - sy/MR/2
            self.extent[1] = xm + sy/MR/2

        # y side too small, expand
        elif not sy or sx/sy > MR:
            ym = (self.extent[2] + self.extent[3])/2  # center y position
            self.extent[2] = ym - sx/MR/2
            self.extent[3] = ym + sx/MR/2

        # when filtering is active, add 5 sigma to the edges
        if self._sigma is not None:
            self.extent += np.array([-1, 1, -1, 1]) * 5*self._sigma

    def rescale(self, N: int) -> None:
        """

        :param N:
        :return:
        """
        N = int(N)  # enforce int
        if N < 1:
            raise ValueError("N needs to be an integer >= 1.")

        Ny, Nx, Nz = self._img.shape

        # get downscaling factor
        Na  = self.SIZES[np.argmin(np.abs(N - np.array(self.SIZES)))]
        Nm = self.MAX_IMAGE_SIDE
        fact = int(self.MAX_IMAGE_SIDE/Na)

        # for each of the XYZW channels:
        def threaded(ind, in_, out):

            # filter, if sigma provided
            if self._sigma is not None:
                # sigma size in pixel for each dimension
                fcx = self._sigma / (self.sx/Nx)
                fcy = self._sigma / (self.sy/Ny)
              
                # with this, with gaussian filtering the power sum in the image stays the same even after filtering
                in_2 = scipy.ndimage.gaussian_filter(in_[:, :, ind], sigma=[fcy, fcx], mode="constant", cval=0)
            else:
                in_2 = in_[:, :, ind]

            # this code basically sums up all pixel values that go into a new pixel value
            # this is done by only two sums and reshaping

            # example:
            # in_ = [[A0, A1, B0, B1], [A2, A3, B2, B3], [C0, C1, D0, D1], [ C2, C3, D2, D3]]
            # where each letter should be joined into one pixel => [[A, B], [C, D]] (Nx=4, Ny=4, fact=2)

            # reshape and sum such that all horizontal pixels that are joined together are in each line
            # => [[A0, A1], [B0, B1], [A2, A3], [B2, B3], [C0, C1], [D0, D1], [C2, C3], [D2, D3]]
            B2 = in_2.reshape((Ny*Nx//fact, fact))

            # sum over rows,
            # => [[A0+A1], [B0+B1], [A2+A3], [B2+B3], [C0+C1], [D0+D1], [C2+C3], [D2+D3]]
            B3 = B2.sum(axis=1)

            # reshape such that same pixel letters are in rows
            # [[A0+A1, A2+A3], [B0+B1, B2+B3], [C0+C1, C2+C3], [D0+D1, D2+D3]]
            B4 = B3.reshape((Nx//fact, Ny), order='F').reshape((Nx*Ny//fact**2, fact))

            # sum over rows
            # [[A0+A1+A2+A3], [B0+B1+B2+B3], [C0+C1+C2+C3], [D0+D1+D2+D3]]
            B5 = B4.sum(axis=1)

            # reshape to correct shape
            # out = [[A0+A1+A2+A3, B0+B1+B2+B3], [C0+C1+C2+C3, D0+D1+D2+D3]]
            out[:, :, ind] = B5.reshape((Ny//fact, Nx//fact), order='F')


        if fact <= 1 and self._sigma is None:
            self.img = self._img

        else:

            self.img = np.zeros((Ny // fact, Nx // fact, Nz))

            if self.threading:
                threads = [Thread(target=threaded, args=(i, self._img, self.img)) for i in range(Nz)]
                [th.start() for th in threads]
                [th.join() for th in threads]
            else:
                for i in range(Nz):
                    threaded(i, self._img, self.img)

    def refilter(self) -> None:
        """reapplies image filtering with the current limit setting"""
        self.rescale(self.N)

    def save(self,
             path:       str,
             save_32bit: bool = True,
             overwrite:  bool = False)\
            -> str:
        """
        Save the RImage as .npz archive.

        :param path: path to save to
        :param save_32bit: save image data in 32bit instead 64bit.
            Looses information in some darker regions of the image
        :param overwrite: if file can be overwritten. If no, it is saved in a fallback path
        :return: path of saved file
        """
        # save in float32 to save some space
        _img = np.array(self._img, dtype=np.float32) if save_32bit else self._img
        limit = self.limit if self.limit is not None else np.nan

        sdict = dict(_img=_img, extent=self.extent, N=min(*self.img.shape[:2]), limit=limit,
                     desc=self.desc, long_desc=self.long_desc, proj=str(self.projection))

        def sfunc(path_: str):
            np.savez_compressed(path_, **sdict)

        return misc.save_with_fallback(path, sfunc, "RImage", ".npz", overwrite, self.silent)

    def export_png(self,
                   path:         str,
                   mode:         str,
                   log:          bool = False,
                   flip:         bool = False,
                   overwrite:    bool = False)\
            -> str:
        """
        Export the RImage in a given display mode as png.
        The image is rescaled (and therefore interpolated) so we have square pixels before the export.
        Note that the side ratio slightly changes, since only integer side lengths are valid for the output,
        but the RImage side lengths can be of arbirtrary precision.

        :param path: path to save to
        :param mode: display mode for getByDisplayMode()
        :param log: logarithmic image (bool)
        :param flip: rotate image by 180 degrees
        :param overwrite: file if it exists, otherwise saved in a fallback path
        :return: path of saved file
        """
        im = self.get_by_display_mode(mode, log)
        if flip:
            im = np.fliplr(np.flipud(im))

        # approximate size we need to rescale to to make square pixels
        sh = im.shape[:2]
        if self.sx > self.sy:
            siz = (int(sh[0]*self.sx/self.sy), sh[0])  # specified as width, height
        else:
            siz = (sh[1], int(sh[1]*self.sy/self.sx))
        
        maxi = np.max(im)
        im = im/maxi*255 if maxi else im
        imp = Image.fromarray(im.astype(np.uint8))
        imp = imp.resize(siz)  # rescale so pixels are square

        def sfunc(path_: str):
            imp.save(path_)

        return misc.save_with_fallback(path, sfunc, "Image", ".png", overwrite, self.silent)

    @staticmethod
    def load(path: str) -> 'RImage':
        """:return: a saved image object from numpy archive to a image object"""

        # load npz archive
        io = np.load(path)

        im = RImage(io["extent"], long_desc=io["long_desc"][()], desc=io["desc"][()], projection=io["proj"][()])

        im.limit = io["limit"][()] if not np.isnan(io["limit"]) else None

        im._img = np.array(io["_img"], dtype=np.float64)
        im.projection = None if im.projection == "None" else im.projection  # None has been saved as string

        # create Im from _Im
        N = int(io["N"][()])
        im.rescale(N)

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
           
            case "limit" if val is not None:
                pc.check_type(key, val, float | int)
                pc.check_above(key, val, 0)
                val = float(val)

            case "projection":
                pc.check_type(key, val, str | None)

        super().__setattr__(key, val)

    def render(self,
               N:             int = MAX_IMAGE_SIDE,
               p:             np.ndarray = None,
               w:             np.ndarray = None,
               wl:            np.array = None,
               keep_extent:   bool = False,
               _dont_rescale: bool = False)\
            \
            -> None:
        """
        Creates an pixel image from ray positions on the detector.
        The image is saved in self.Im

        :param N: number of image pixels in smallest dimension (int)
        :param p: ray position matrix, xyz components in columns, (numpy 1D array)
        :param w: ray weight array (numpy 1D array)
        :param wl: ray wavelength array (numpy 1D array)
        :param keep_extent: True if :obj:`Image.__fixExtent` shouldn't be called before image calculation
        :param _dont_rescale:
        """

        # fix point and line images as well as ones with a too large side ratio
        if not keep_extent:
            self.__fix_extent()

        if N > self.MAX_IMAGE_SIDE or N < 1:
            raise ValueError(f"N needs to be between 1 and {self.MAX_IMAGE_SIDE}")

        # calculate image size. Smaller side is MAX_IMAGE_SIDE, larger MAX_IMAGE_SIDE*[1, 3, 5, ..., MAX_IMAGE_RATIO]
        # the resulting pixel size is not square. And the user doesn't exactly get his desired resolution
        Nrs = self.MAX_IMAGE_SIDE
        nf = lambda a: min(self.MAX_IMAGE_RATIO, 1 + 2*int(a/2))  
        # ^-- calculates nearest factor for a from [1, 3, 5, ..] below MAX_IMAGE_RATIO
        Nx = Nrs if self.sx <= self.sy else Nrs*nf(self.sx/self.sy)
        Ny = Nrs if self.sx > self.sy else Nrs*nf(self.sy/self.sx) 

        # init image
        # x in first, y in second since np.histogram2d needs it that way
        self._img = np.zeros((Nx, Ny, 4), dtype=np.float64)

        if p is not None and p.shape[0]:

            # threading function
            def func(img, ind):
               
                tri = [color.x_observer, color.y_observer, color.z_observer]
                w_ = tri[ind](wl) * w if ind < 3 else w

                img[:, :, ind], _, _ = np.histogram2d(p[:, 0], p[:, 1], weights=w_, bins=[Nx, Ny], 
                                                      range=self.extent.reshape((2,2)))

            # multithreading
            if self.threading: 

                threads = [Thread(target=func, args=(self._img, i)) for i in np.arange(4)]
                [thread.start() for thread in threads]
                [thread.join() for thread in threads]

            # no multithreading
            else:
                [func(self._img, i) for i in np.arange(4)]

        # transpose since histogram2d returns x in first dimension, y in second
        self._img = np.transpose(self._img, (1, 0, 2))

        if not _dont_rescale:
            self.rescale(N)  # create rescaled Image self.img
