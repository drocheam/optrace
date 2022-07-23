
import numpy as np  # calculations
from datetime import datetime  # date for fallback file naming
from pathlib import Path  # path handling for image saving

import optrace.tracer.Color as Color  # Tristimulus curves and sRGB conversions
import optrace.tracer.Misc as misc  # interpolation and calculation methods
from threading import Thread  # multithreading

from optrace.tracer.BaseClass import BaseClass # parent class

from PIL import Image  # saving as png
from typing import Callable  # Callable type


# Rendered Image class
class RImage(BaseClass):

    EPS: float = 1e-9
    """ Used for minimal extent """

    MAX_IMAGE_RATIO: float = 5.
    """ maximum ratio of image side lengths. Images with ratios beyond will be corrected """

    MAX_IMAGE_SIDE: int = 1024
    """ maximum size of smaller image side in pixels. Needs to be a power of 2 """
    
    display_modes = ["sRGB (Absolute RI)", "sRGB (Perceptual RI)", "Outside sRGB Gamut", "Irradiance", "Illuminance",\
                     "Lightness (CIELUV)", "Hue (CIELUV)", "Chroma (CIELUV)", "Saturation (CIELUV)"]
    """possible display modes for the RImage"""

    coordinate_types = ["Cartesian", "Polar"]
    """possible coordinate types for the RImage"""

    
    def __init__(self, 
                 extent:            (list | np.ndarray), 
                 coordinate_type:   str = "Cartesian", 
                 offset:            float = 0.,
                 **kwargs)\
            -> None:
        """
        Init an Image object.
        This class is used to calculate and hold an Image consisting of the channels X, Y, Z, Illuminance and Irradiance.
        The class also includes information like extent, z-position of image,
        an image plotting type and an index for tagging.

        :param extent: image extent in the form [xs, xe, ys, ye]
        :param coordinate_type: "Cartesian" or "Polar"
        """
        self._new_lock = False
        self.extent = extent
        self.coordinate_type = coordinate_type

        self.offset = offset
        """additional subjective ambient white for the rendered image.
        0 equals no additional white, 1 equals only additional white.
        An offset can increase visibility of dark tones. See RImage class for more details"""

        self.Im = None
        self._Im = None

        super().__init__(**kwargs)
        self._new_lock = True

    def hasImage(self) -> bool:
        """Check Image objects contains an calculated image."""
        return self.Im is not None

    def __checkForImage(self) -> None:
        """throw exception when image is missing"""
        if not self.hasImage():
            raise RuntimeError("Image was not calculated.")

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
        self.__checkForImage()
        return self.Im.shape[1]

    @property
    def Ny(self) -> int:
        """ number of image pixels in y direction """
        self.__checkForImage()
        return self.Im.shape[0]

    @property
    def Apx(self) -> float:
        """ area per pixel """
        self.__checkForImage()
        return self.sx * self.sy / (self.Nx * self.Ny)

    def cut(self, 
            mode:   str, 
            x:      float = None,
            y:      float = None, 
            log:    bool = False)\
            \
            -> tuple[np.ndarray, list[np.ndarray]]:
        """

        :param mode:
        :param x:
        :param y:
        :param log:
        :return:
        """
        xp = np.linspace(self.extent[0], self.extent[1], self.Nx)
        yp = np.linspace(self.extent[2], self.extent[3], self.Ny)
        Im = self.getByDisplayMode(mode, log)

        if (x is not None and not (self.extent[0] <= x <= self.extent[1]))\
           or (y is not None and not (self.extent[2] <= y <= self.extent[3])):
            raise RuntimeError("Position outside image.")

        if x is not None:
            sp, xs, ys = xp, np.full(self.Nx, x), yp
        else:
            sp, xs, ys = yp, xp, np.full(self.Ny, y)

        Iml = [Im] if Im.ndim == 2 else [Im[:, :, 0], Im[:, :, 1], Im[:, :, 2]]
        Ims = [misc.interp2d(xp, yp, Imi, xs, ys) for Imi in Iml]

        return sp, Ims

    def getByDisplayMode(self, mode: str, log: bool=False) -> np.ndarray:
        """Modes only include displayable modes from self.modes, use dedicated functions for Luv and XYZ"""

        self.__checkForImage()

        match mode:

            case "Irradiance":             
                return 1 / self.Apx * self.Im[:, :, 3]

            case "Illuminance":     
                # the Illuminance is just the unnormalized Y scaled by 683 lm/W and the inverse pixel area
                return 683 / self.Apx * self.Im[:, :, 1]

            case "sRGB (Absolute RI)":      
                return self.getRGB(log=log, RI="Absolute")

            case "sRGB (Perceptual RI)":    
                return self.getRGB(log=log, RI="Perceptual")

            case "Outside sRGB Gamut":
                # force conversion from bool to int so further algorithms work correctly
                return np.array(Color.outside_sRGB(self.getXYZ()), dtype=int)

            case "Lightness (CIELUV)":      
                return self.getLuv()[:, :, 0]

            case "Hue (CIELUV)":
                Luv = self.getLuv()
                return Color.getLuvHue(Luv)

            case "Chroma (CIELUV)":
                Luv = self.getLuv()
                return Color.getLuvChroma(Luv)

            case "Saturation (CIELUV)":     
                Luv = self.getLuv()
                return Color.getLuvSaturation(Luv)

            case _:                         
                raise ValueError(f"Invalid display_mode {mode}, should be one of {self.display_modes}.")

    def getPower(self) -> float:
        """Calculate total image power"""
        self.__checkForImage()
        return np.sum(self.Im[:, :, 3])

    def getLuminousPower(self) -> float:
        """Calculate Total image luminous power"""
        self.__checkForImage()
        return 683*np.sum(self.Im[:, :, 1])

    def getXYZ(self) -> np.ndarray:
        """Get XYZ image (np.ndarray with shape (Ny, Nx, 3))"""
        self.__checkForImage()

        if self.offset == 0: 
            return self.Im[:, :, :3]
        else:
            wp = np.array(Color._WP_D65_XYZ)
            loffset = ((100*self.offset+16)/116)**3 # relative Y to L (from Luv) conversion, see Color.XYZ_to_Luv
            Ymax = np.max(self.Im[:, :, 1])
            return wp*loffset*Ymax + (1-loffset) * self.Im[:, :, :3]

    def getLuv(self) -> np.ndarray:
        """CIELAB image"""
        XYZ = self.getXYZ()
        return Color.XYZ_to_Luv(XYZ)

    def getRGB(self, log: bool = False, RI: str = "Absolute") -> np.ndarray:
        """
        Get sRGB image

        :param log: if brightness should be logarithmically scaled
        :return: sRGB image (np.ndarray with shape (Ny, Nx, 3))
        """
        Im = Color.XYZ_to_sRGBLinear(self.getXYZ(), RI=RI)

        # addition, multiplication etc. only work correctly in the linear color space
        # otherwise we would change the color ratios, but we only want the brightness to change
        if log:
            RGBs = np.sum(Im, axis=2)  # assume RGB channel sum as brightness
            wmax = np.max(RGBs)  # maximum brightness
            wmin = np.min(RGBs[RGBs > 0])  # minimum nonzero brightness
            maxrgb = np.max(Im, axis=2)  # highest rgb value for each pixel

            # normalize pixel so highest channel value is 1, then rescale logarithmically.
            # Highest value is 1, lowest 0. Exclude all zero channels (maxrgb = 0) for calculation
            fact = 1 / maxrgb[maxrgb > 0] * (1 - np.log(RGBs[maxrgb > 0] / wmax)/np.log(wmin / wmax))
            Im[maxrgb > 0] *= fact[:, np.newaxis]

        Im = Color.sRGBLinear_to_sRGB(Im, clip=True)

        return Im

    def __fixExtent(self) -> None:
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
        elif sy/sx > MR:
            xm = (self.extent[0] + self.extent[1])/2  # center x position
            self.extent[0] = xm - sy/MR/2
            self.extent[1] = xm + sy/MR/2

        # y side too small, expand
        elif sx/sy > MR:
            ym = (self.extent[2] + self.extent[3])/2  # center y position
            self.extent[2] = ym - sx/MR/2
            self.extent[3] = ym + sx/MR/2

    def rescale(self, N: int, threading: bool = True, _force=False) -> None:
       
        if not isinstance(N, int) or N < 1:
            raise ValueError("N needs to be an integer >= 1.")

        Ny, Nx, Nz = self._Im.shape

        # get downscaling factor
        Nm = min(Nx, Ny)
        fact = Nm // 2**round(np.log2(N))

        # the image gets only rescaled to nearest 2**x resolution

        # for each of the XYZW channels:
        def threaded(ind, in_, out):
            # this code basically sums up all pixel values that go into a new pixel value
            # this is done by only two sums and reshaping

            # example:
            # in_ = [[A0, A1, B0, B1], [A2, A3, B2, B3], [C0, C1, D0, D1], [ C2, C3, D2, D3]]
            # where each letter should be joined into one pixel => [[A, B], [C, D]] (Nx=4, Ny=4, fact=2)

            # reshape and sum such that all horizontal pixels that are joined together are in each line
            # => [[A0, A1], [B0, B1], [A2, A3], [B2, B3], [C0, C1], [D0, D1], [C2, C3], [D2, D3]]
            B2 = in_[:, :, ind].reshape((Ny*Nx//fact, fact))

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

        if fact <= 1:
            self.Im = self._Im

        # only rescale if target image size is different from current Image (but not if force is active)
        elif self.Im is None or _force or min(*self.Im.shape[:2]) != Nm//fact:
            self.Im = np.zeros((Ny//fact, Nx//fact, Nz))

            if self.threading:
                threads = [Thread(target=threaded, args=(i, self._Im, self.Im)) for i in range(Nz)]
                [th.start() for th in threads]
                [th.join() for th in threads]
            else:
                for i in range(Nz):
                    threaded(i, self._Im, self.Im)

    def save(self, 
             path:       str, 
             save32bit:  bool = True, 
             overwrite:  bool = False)\
            -> str:
        """
        Save the RImage as .npz archive.

        :param path: path to save to
        :param save32bit: save image data in 32bit instead 64bit. Looses information in some darker regions of the image
        :param overwrite: if file can be overwritten. If no, it is saved in a fallback path
        :return: path of saved file
        """
        # save in float32 to save some space
        _Im = np.array(self._Im, dtype=np.float32) if save32bit else self._Im

        sdict = dict(_Im=_Im, extent=self.extent, N=min(*self.Im.shape[:2]), 
                     desc=self.desc, long_desc=self.long_desc, type_=self.coordinate_type)
       
        def sfunc(path: str):
            np.savez_compressed(path, **sdict)

        return self.__saveWithFallback(path, sfunc, "RImage", ".npz", overwrite)

    def exportPNG(self, 
                  path:         str, 
                  mode:         str, 
                  log:          bool = False, 
                  flip:         bool = False, 
                  overwrite:    bool = False)\
            -> str:
        """
        Export the RImage in a given display mode as png.

        :param path: path to save to
        :param mode: display mode for getByDisplayMode()
        :param log: logarithmic image (bool)
        :param flip: rotate image by 180 degrees
        :param overwrite file if it exists, otherwise saved in a fallback path
        :return: path of saved file
        """
        im = self.getByDisplayMode(mode, log)
        if flip:
            im = np.fliplr(np.flipud(Imd))

        imp = Image.fromarray((im/np.max(im)*255).astype(np.uint8))

        def sfunc(path: str):
            imp.save(path)

        return self.__saveWithFallback(path, sfunc, "Image", ".png", overwrite)

    def __saveWithFallback(self, 
                           path:        str, 
                           sfunc:       Callable, 
                           fname:       str, 
                           ending:      str, 
                           overwrite:   bool = False)\
            -> str:
        """saving with fallback path if the file exists and overwrite=False"""
        
        # called when invalid path or file exists but overwrite=False
        def fallback():
            # create a valid path and filename
            wd = Path.cwd()
            filename = f"{fname}_" + datetime.now().strftime('%Y-%m-%d_%H:%M:%S.%f%z') + ending
            path = str(wd / filename)

            # resave
            if not self.silent:
                print(f"Failed saving {fname}, resaving as \"{path}\"")
            sfunc(path)

            return path
       
        # append file ending if the path provided has none
        if path[-len(ending):] != ending:
            path += ending

        # check if file already exists
        exists = Path(path).exists()

        if overwrite or not exists:
            try:
                sfunc(path)
                if not self.silent:
                    print(f"Saved {fname} as \"{path}\"")
                return path
            except:
                return fallback()
        else:
            return fallback()

    @staticmethod
    def load(path: str) -> 'RImage':
        """load a saved image object numpy archive to a image object"""

        # load npz archive
        Io = np.load(path)

        Im = RImage(Io["extent"], long_desc=Io["long_desc"][()], desc=Io["desc"][()], coordinate_type=Io["type_"][()])
        Im._Im = np.array(Io["_Im"], dtype=np.float64)

        # create Im from _Im
        N = int(Io["N"][()])
        Im.rescale(N)

        return Im

    def __setattr__(self, key, val):
       
        match key:
          
            case "extent":
                self._checkType(key, val, list | tuple | np.ndarray)
                val2 = np.array(val, dtype=np.float64)

                if val2.shape[0] != 4:
                    raise ValueError("Extent needs to have 4 elements.")

                if val2[0] >= val[1] or val2[2] >= val2[3]:
                    raise ValueError("Extent needs to be an array with [x0, x1, y0, y1] with x0 < x1 and y0 < y1.")

                super().__setattr__(key, val2)
                return

            case "coordinate_type":
                self._checkType(key, val, str)
                self._checkIfIn(key, val, self.coordinate_types)

            case "offset":
                self._checkType(key, val, int | float)
                self._checkNotBelow(key, val, 0)
                self._checkNotAbove(key, val, 1)
        
        super().__setattr__(key, val)
    
    def render(self, 
               N:            int = MAX_IMAGE_SIDE, 
               p:            np.ndarray = None,
               w:            np.ndarray = None,
               wl:           np.array = None,
               keep_extent:  bool = False)\
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
        """

        # fix point and line images as well as ones with a too large side ratio
        if not keep_extent:
            self.__fixExtent()
        
        sx, sy = self.sx, self.sy
       
        if N > self.MAX_IMAGE_SIDE:
            raise ValueError("N needs to be equal or below 1024")

        # set smaller side to 1024 pixels, other one can be 1024* 2^x with x integer
        # upper bound is the side ratio MAX_IMAGE_RATIO
        # the resulting pixel size is not square. And the user doesn't exactly get his desired resolution
        Nrs = self.MAX_IMAGE_SIDE
        Nx = Nrs if sx <= sy else Nrs * 2**int(np.log2(sx / sy))
        Ny = Nrs if sx > sy  else Nrs * 2**int(np.log2(sy / sx))

        # init image
        self._Im = np.zeros((Ny, Nx, 4), dtype=np.float64)

        if p is not None and p.shape[0]:
            # get hit pixel coordinate, subtract 1e-12 from N so all values are 0 <= cor < N
            px, py, xs, ys = p[:, 0], p[:, 1], self.extent[0], self.extent[2]
            xcor = misc.calc("Nx / sx * (px - xs)")
            ycor = misc.calc("Ny / sy * (py - ys)")
        
            # handle case where coordinates land at exactly the edge (">=" for float errors)
            xcor[xcor >= self._Im.shape[1]] -= 1
            ycor[ycor >= self._Im.shape[0]] -= 1

            # calculate XYZP values, with P being the power
            XYZW = np.ones((wl.shape[0], 4), dtype=np.float64, order='F')
            XYZW[:, 0] = Color.Tristimulus(wl, "X")
            XYZW[:, 1] = Color.Tristimulus(wl, "Y")
            XYZW[:, 2] = Color.Tristimulus(wl, "Z")
            XYZW *= w[:, np.newaxis]

            # render image for positions and XYZW array
            np.add.at(self._Im, (ycor.astype(int), xcor.astype(int)), XYZW)

        self._Im = np.flipud(self._Im) # flip so [0, 0] is in the lower left
        self.rescale(N) # create rescaled Image Im

