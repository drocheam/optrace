
import numpy as np

import optrace.Backend.Color as Color  # for Tristimulus curves and sRGB conversions
from optrace.Backend.Misc import timer as timer  # for benchmarking
import optrace.Backend.Misc as misc
from threading import Thread  # for multithreading

# TODO functions for saving and loading the Image object from disc

# TODO settattr

# TODO save image internally in a higher dimension (e.g. at least 1000x1000), which is a integer multiple
# -> easy and explicit reducing operation possible
# use pil_image = Image.fromarray(np_array); p2 = pil_image.reduce(10); n2 = np.array(p2)
# see /bak/reduceTest.py

# Image.at() function, makes x or y cuts

class Image:

    EPS: float = 1e-9
    """ Used for minimal extent """

    MAX_IMAGE_RATIO: float = 5.
    """ maximum ratio of image side lengths. Images with ratios beyond will be corrected """

    def __init__(self, 
                 extent:        (list | np.ndarray),
                 z:             float = None,
                 index:         int = None,
                 image_type:    str = "Cartesian")\
            -> None:
        """
        Init an Image object.
        This class is used to calculate and hold an Image consisting of the channels X, Y, Z, Illuminance and Irradiance.
        The class also includes information like extent, z-position of image, an image plotting type and an index for tagging.

        :param extent: image extent in the form [xs, xe, ys, ye]
        :param z: position of image.Only used as tag.
        :param index: index of image for tagging
        :param image_type: "Cartesian" or "Polar"
        """
        self.extent = np.array(extent, dtype=np.float64)
        self.index = index
        self.image_type = image_type
        self.z = float(z) if z is not None else z
        self.Im = None

        # check for valid kind
        if self.image_type not in ["Cartesian", "Polar"]:
            raise ValueError(f"Invalid image_type '{image_type}'.")

    def hasImage(self) -> bool:
        """
        Check Image objects contains an calculated image.

        :return: True if contains Image, False otherwise
        """
        return self.Im is not None

    def getPower(self) -> float:
        """
        Calculate total image power

        :return: Total image power
        """
        if not self.hasImage():
            raise RuntimeError("Image was not calculated.")

        return np.sum(self.Im[:, :, 3])*self.Apx

    def getLuminousPower(self) -> float:
        """
        Calculate total image luminous power

        :return: Total image luminous power
        """
        if not self.hasImage():
            raise RuntimeError("Image was not calculated.")

        return np.sum(self.Im[:, :, 4])*self.Apx

    def getIrradiance(self) -> np.ndarray:
        """
        Get irradiance image

        :return: irradiance image (2D np.ndarray)
        """
        if not self.hasImage():
            raise RuntimeError("Image was not calculated.")

        # return flipped so index (0, 0) is in the bottom left
        return np.flipud(self.Im[:, :, 3])

    def getIlluminance(self) -> np.ndarray:
        """
        Get illuminance image

        :return: illuminance image (2D np.ndarray)
        """
        if not self.hasImage():
            raise RuntimeError("Image was not calculated.")

        # return flipped so index (0, 0) is in the bottom left
        return np.flipud(self.Im[:, :, 4])

    def getXYZ(self) -> np.ndarray:
        """
        Get XYZ image

        :return: XYZ image (np.ndarray with shape (Ny, Nx, 3))
        """
        if not self.hasImage():
            raise RuntimeError("Image was not calculated.")

        # return flipped so index (0, 0) is in the bottom left
        return np.flipud(self.Im[:, :, :3])

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
        if not self.hasImage():
            raise RuntimeError("Image was not calculated.")

        return self.Im.shape[1]

    @property
    def Ny(self) -> int:
        """ number of image pixels in y direction """
        if not self.hasImage():
            raise RuntimeError("Image was not calculated.")

        return self.Im.shape[0]

    @property
    def Apx(self) -> float:
        """ area per pixel """
        if not self.hasImage():
            raise RuntimeError("Image was not calculated.")

        # pixel are from pixel number and iamge extent
        return  self.sx * self.sy / (self.Nx * self.Ny)

    
    def getRGB(self, log: bool = False) -> np.ndarray:
        """
        Get sRGB image

        :param log: if brightness should be logarithmically scaled
        :return: sRGB image (np.ndarray with shape (Ny, Nx, 3))
        """
        Im = Color.XYZ_to_sRGBLinear(self.getXYZ())

        # addition, multiplication etc. only work correctly in the linear color space
        # otherwise we would change the color ratios, but we only want the brightness to change
        if log:
            RGBs = np.sum(Im[:, :, :3], axis=2)  # assume RGB channel sum as brightness
            wmax = np.max(RGBs)  # maximum brightness
            wmin = np.min(RGBs[RGBs > 0])  # minimum nonzero brightness
            maxrgb = np.max(Im[:, :, :3], axis=2)  # highest rgb value for each pixel

            # normalize pixel so highest channel value is 1, then rescale logarithmically.
            # Highest value is 1, lowest 0. Exclude all zero channels (maxrgb = 0) for calculation
            fact = 1 / maxrgb[maxrgb > 0] * (1 - np.log(RGBs[maxrgb > 0] / wmax)/np.log(wmin / wmax))
            Im[maxrgb > 0, :3] *= fact[:, np.newaxis]

        Im = Color.sRGBLinear_to_sRGB(Im, normalize=True)

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


    def makeImage(self, 
                  N:            int, 
                  p:            np.ndarray = None,
                  w:            np.ndarray = None,
                  wl:           np.array = None,
                  keep_extent:  bool = False,
                  threading:    bool = True)\
            -> None:
        """
        Creates an pixel image from ray positions on the detector.
        The image is saved in self.Im

        :param N: number of image pixels in each dimension (int)
        :param p: ray position matrix, xyz components in columns, (numpy 1D array)
        :param w: ray weight array (numpy 1D array)
        :param wl: ray wavelength array (numpy 1D array)
        :param keep_extent: True if :obj:`Image.__fixExtent` shouldn't be called before image calculation
        :param threading: True if multithreading should be enabled.
        """

        # fix point and line images as well as ones with a too large side ratio
        if not keep_extent:
            self.__fixExtent()
        
        # N is provided for smaller side of image, get the other one by scaling
        # note that the resulting pixel size is not square, since we are limited to int values
        Nx = N if self.sx <= self.sy else int(N * self.sx / self.sy)
        Ny = N if self.sx > self.sy  else int(N * self.sy / self.sx)

        # init image
        self.Im = np.zeros((Ny, Nx, 5), dtype=np.float64)

        # return if there are no rays
        if p is None or not p.shape[0]:
            return

        # init parameter for multithreading
        N_rays = wl.shape[0]
        N_threads = misc.getCoreCount() if threading and N_rays > 1e5 else 1
        N_step = int(N_rays/N_threads)

        # multithreading function
        def makeImages(i: int, Ims: list[np.ndarray]) -> None:

            # indices for thread rays
            Ns = i*N_step
            Ne = (i+1)*N_step if i != N_threads-1 else N_rays

            # get hit pixel coordinate, subtract 1e-12 from N so all values are 0 <= cor < N
            xcor = (Nx - 1e-12) / self.sx * (p[Ns:Ne, 0] - self.extent[0])
            ycor = (Ny - 1e-12) / self.sy * (p[Ns:Ne, 1] - self.extent[2])

            # calculate XYZW values, with W being the power
            wlt = wl[Ns:Ne]
            XYZW = np.column_stack((Color.Tristimulus(wlt, "X"),
                                    Color.Tristimulus(wlt, "Y"),
                                    Color.Tristimulus(wlt, "Z"),
                                    np.ones_like(wlt, dtype=np.float64)))  *  w[Ns:Ne, np.newaxis]

            # render image fro positions and XYZW array
            np.add.at(Ims[i][:, :, :4], (ycor.astype(int), xcor.astype(int)), XYZW)

        # multithreading mode
        if N_threads > 1:

            # create empty images for every thread
            Ims = [self.Im.copy() for N_t in np.arange(N_threads)]
        
            # create threads
            thread_list = [Thread(target=makeImages, args=[N_t, Ims]) for N_t in np.arange(N_threads)]
            
            # start and join threads
            [thread.start() for thread in thread_list]
            [thread.join()  for thread in thread_list]

            # add all sub-images together
            for Imi in Ims:
                self.Im += Imi

        # main thread mode
        else:
            makeImages(0, [self.Im])

        # conversion from Y to luminous flux
        self.Im[:, :, 4] = 683 * self.Im[:, :, 1]

        # normalize colors to maximum color value
        self.Im[:, :, :3] *= 1 / np.max(self.Im[:, :, :3])

        # scale radiant and luminous flux by pixel area to get irradiance and illuminance
        self.Im[:, :, 3:] *= 1 / self.Apx


