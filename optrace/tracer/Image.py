
import numpy as np

import optrace.tracer.Color as Color  # for Tristimulus curves and sRGB conversions
from optrace.tracer.Misc import timer as timer  # for benchmarking
import optrace.tracer.Misc as misc
from threading import Thread  # for multithreading

# TODO
# Image.at() function, makes x or y cuts
# quantile clip parameter

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
        self._new_lock = False
        self.extent = extent
        self.index = index
        self.image_type = image_type
        self.z = z
        self.Im = None
        self._Im = None
        self._new_lock = True

    def hasImage(self) -> bool:
        """
        Check Image objects contains an calculated image.

        :return: True if contains Image, False otherwise
        """
        return self.Im is not None

    def getPower(self) -> float:
        """Calculate total image power"""
        if not self.hasImage():
            raise RuntimeError("Image was not calculated.")

        return np.sum(self.Im[:, :, 3])

    def getLuminousPower(self) -> float:
        """Calculate Total image luminous power"""
        if not self.hasImage():
            raise RuntimeError("Image was not calculated.")

        return 683*np.sum(self.Im[:, :, 1])

    def getIrradiance(self) -> np.ndarray:
        """Get irradiance image (2D np.ndarray)"""
        if not self.hasImage():
            raise RuntimeError("Image was not calculated.")

        # return flipped so index (0, 0) is in the bottom left
        return np.flipud(1/self.Apx*self.Im[:, :, 3])

    def getIlluminance(self) -> np.ndarray:
        """Get illuminance image (2D np.ndarray)"""
        if not self.hasImage():
            raise RuntimeError("Image was not calculated.")

        # return flipped so index (0, 0) is in the bottom left
        # the Illuminance is just the unnormalized Y scaled by 683 lm/W and the inverse pixel area
        return np.flipud(683/self.Apx*self.Im[:, :, 1])

    def getXYZ(self) -> np.ndarray:
        """Get XYZ image (np.ndarray with shape (Ny, Nx, 3))"""
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

    def getRGB(self, log: bool = False, RI: str="Absolute") -> np.ndarray:
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

        Im = Color.sRGBLinear_to_sRGB(Im, normalize=True)

        return Im

    def getLuvLightness(self) -> np.ndarray:
        """Lightness in CIELAB"""
        return self.getLuv()[:, :, 0]

    def getLuvSaturation(self) -> np.ndarray:
        """Saturation in CIELAB"""       
        Luv = self.getLuv()
        return Color.getLuvSaturation(Luv)

    def getLuv(self) -> np.ndarray:
        """CIELAB image"""
        XYZ = self.getXYZ()
        return Color.XYZ_to_Luv(XYZ)

    def getLuvChroma(self) -> np.ndarray:
        """Chroma in CIELUV"""        
        Luv = self.getLuv()
        return Color.getLuvChroma(Luv)

    def getLuvHue(self) -> np.ndarray:
        """Hue in CIELUV"""        
        Luv = self.getLuv()
        return Color.getLuvHue(Luv)
    
    def getOutsidesRGB(self) -> np.ndarray:
        """Lightness in CIELAB"""
        return Color.outside_sRGB(self.getXYZ())

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

    # @timer
    def rescale(self, N, threading: bool=True):

        Ny, Nx, Nz = self._Im.shape
        fact = np.floor(min(Nx, Ny) / N).astype(int)

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

        if fact != 1:
            self.Im = np.zeros((Ny//fact, Nx//fact, Nz))

            if threading:
                threads = [Thread(target=threaded, args=(i, self._Im, self.Im)) for i in range(Nz)]
                [th.start() for th in threads]
                [th.join() for th in threads]
            else:
                for i in range(Nz):
                    threaded(i, self._Im, self.Im)
        else:
            self.Im = self._Im

    # TODO save in current folder if path is invalid
    def save(self, path):
        """save the image object in a numpy archive"""
        # save in float32 to save some space
        Im = np.array(self._Im, dtype=np.float32)

        np.savez_compressed(path, Im=Im, extent=self.extent, N=min(self.Im.shape[0], self.Im.shape[1]), 
                            z=self.z, index=self.index, type_=self.image_type)

    @staticmethod
    def load(path):
        """load a saved image object numpy archive to a image object"""

        # load npz archive
        Io = np.load(path)

        Im = Image(Io["extent"], z=Io["z"], index=Io["index"], image_type=Io["type_"])

        Im._Im = np.array(Io["Im"], dtype=np.float64)
        Im.rescale(Io["N"]) # also creates Im from _Im

        return Im

    def __setattr__(self, key, val):
        
        if key == "extent":

            if not isinstance(val, list | tuple | np.ndarray):
                raise TypeError(f"{key} needs to be of type list, tuple or np.ndarray")

            val = np.array(val, dtype=np.float64)

        if key == "index" and not isinstance(val, int | None):
            raise TypeError(f"{key} needs to be of type int")

        if key == "z" and val is not None:
            if not isinstance(val, int | float ):
                raise TypeError(f"{key} needs to be of type int or float.")
            val = float(val)

        if key == "image_type" and val not in ["Cartesian", "Polar"]:
            raise ValueError(f"Invalid image_type '{val}'.")
        
        if "_new_lock" in self.__dict__ and self._new_lock:
            if key not in self.__dict__:
                raise AttributeError(f"Invalid property '{key}'")
        
        self.__dict__[key] = val

    # TODO using threading is slower?
    # @timer
    def makeImage(self, 
                  N:            int, 
                  p:            np.ndarray = None,
                  w:            np.ndarray = None,
                  wl:           np.array = None,
                  keep_extent:  bool = False,
                  max_res:      bool = False,
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
        
        sx, sy = self.extent[1] - self.extent[0], self.extent[3] - self.extent[2]
       
        if N > 1000:
            raise ValueError("N needs to below 1000")

        # int factor that upscales N such that it is at least 1000
        # N is provided for smaller side of image, get the other one by scaling
        # note that the resulting pixel size is not square, since we are limited to int values
        Nrs = int(N*np.ceil(1000/N)) if not max_res else N
        Nx = Nrs if sx <= sy else Nrs*int(sx / sy)
        Ny = Nrs if sx > sy  else Nrs*int(sy / sx)

        Apx = sx*sy/Nx/Ny

        # init image
        self._Im = np.zeros((Ny, Nx, 4), dtype=np.float64)

        # return if there are no rays
        if p is None or not p.shape[0]:
            self.rescale(N, threading)
            return

        # init parameter for multithreading
        N_rays = wl.shape[0]

        # threading = False
        # maximum 4 threads, otherwise there are to many images in RAM
        N_threads = 1 # min(4, misc.getCoreCount()) if threading and N_rays > 1e5 else 1
        N_step = int(N_rays/N_threads)
    
        # multithreading function
        def makeImages(i: int, Ims: list[np.ndarray]) -> None:

            # indices for thread rays
            Ns = i*N_step
            Ne = (i+1)*N_step if i != N_threads-1 else N_rays

            # get hit pixel coordinate, subtract 1e-12 from N so all values are 0 <= cor < N
            px, py, xs, ys = p[Ns:Ne, 0], p[Ns:Ne, 1], self.extent[0], self.extent[2]
            xcor = misc.calc("(Nx - 1e-12) / sx * (px - xs)")
            ycor = misc.calc("(Ny - 1e-12) / sy * (py - ys)")

            # calculate XYZP values, with P being the power
            wlt = wl[Ns:Ne]
            XYZW = np.ones((wlt.shape[0], 4), dtype=np.float64, order='F')
            XYZW[:, 0] = Color.Tristimulus(wlt, "X")
            XYZW[:, 1] = Color.Tristimulus(wlt, "Y")
            XYZW[:, 2] = Color.Tristimulus(wlt, "Z")
            XYZW *= w[Ns:Ne, np.newaxis]

            # render image fro positions and XYZW array
            np.add.at(Ims[i], (ycor.astype(int), xcor.astype(int)), XYZW)

        # multithreading mode
        if N_threads > 1:

            # create empty images for every thread
            Ims = [self._Im.copy() for N_t in np.arange(N_threads)]
        
            # create threads
            thread_list = [Thread(target=makeImages, args=[N_t, Ims]) for N_t in np.arange(N_threads)]
            
            # start and join threads
            [thread.start() for thread in thread_list]
            [thread.join()  for thread in thread_list]

            # add all sub-images together
            for Imi in Ims:
                self._Im += Imi

        # main thread mode
        else:
            makeImages(0, [self._Im])

        self.rescale(N, threading)

