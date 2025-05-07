import sys
from threading import Thread

import numpy as np
import scipy.signal
import cv2

from . import misc
from . import color
from ..progress_bar import ProgressBar
from ..property_checker import PropertyChecker as pc
from .image import RGBImage, GrayscaleImage, RenderImage
from .. import global_options
from ..warnings import warning


class ConvShapes:
        
    iN:  np.ndarray  # image pixel counts
    ip:  np.ndarray  # image pixel size
    is_: np.ndarray  # image side lengths after m scaling

    ipad: np.ndarray  # image padding count
    i2N:  np.ndarray  # padded image pixel count
    i3N:  np.ndarray  # double padded image pixel count
    i4N:  np.ndarray  # result image pixel count
    i4s:  np.ndarray  # result image side lengths
    i4e:  np.ndarray  # result image extent

    pN:  np.ndarray  # psf pixel counts
    pp:  np.ndarray  # psf pixel sizes
    ps_: np.ndarray  # psf side lengths

    ppad: np.ndarray  # psf padding pixel count
    p2N:  np.ndarray  # pixel count of rescaled psf
    p3N:  np.ndarray  # pixel count of padded and rescaled psf

class ConvFlags:

    custom_padding: bool
    make_grayscale: bool
    three_psf:      bool
    psf_color:      bool
    img_color:      bool
    keep_size:      bool



def convolve(img:                RGBImage | GrayscaleImage,
             psf:                GrayscaleImage | RenderImage | list[RenderImage],
             m:                  float = 1,
             keep_size:          bool = False,
             padding_mode:       str = "constant",
             padding_value:      list | float = None,
             cargs:              dict = {})\
        -> RGBImage | GrayscaleImage:
    """
    Convolve an image with a point spread function.

    Different convolution cases:
    1. Grayscale image and PSF: Img and PSF are GrayscaleImage -> result is also GrayscaleImage
    2. Image grayscale or spectrally homogeneous, PSF has color information: Img GrayscaleImage, PSF RenderImage
    -> result is RGBImage
    3. Image has color information, PSF is color independent: Image RGBImage, PSF GrayscaleImage
    -> result is RGBImage
    4. Image has color information, PSF has color information: Image RGBImage, PSF list of R, G, B RenderImage rendered
    for the sRGB primary R, G, B preset spectra with the correct power ratios
    -> result is RGBImage
    
    The system magnification factor m scales the image before convolution.
    abs(m) > 1 means enlargement, abs(m) < 1 size reduction,
    m < 0 image flipping, m > 0 an upright image.

    See the documentation for more details.

    As the convolution requires values outside the initial image, a suitable padding method must be set.
    By default, padding_mode='constant' and padding_value are zeros.
    Padding modes from numpy.pad are supported. padding_value is only needed for the 'constant' mode.
    It must have three elements if img is an RGBImage, and one otherwise.

    If a result with the same side lengths and pixel count as the input is desired,
    parameter `keep_size` must be set to True so the image is cropped back to its original shape.

    Provide cargs=dict(normalize=False) so the output image intensities are not normalized automatically.
    Useful for images without bright regions.

    In the internals of this function the convolution is done in linear sRGB values, 
    while also using values outside the sRGB gamut.
    To remove this out-of-gamut colors in the resulting image, color transformations are performed.
    Parameter cargs overwrites parameters for color.xyz_to_srgb. By default these parameters are:
    dict(rendering_intent="Absolute", normalize=True, clip=True, L_th=0, chroma_scale=None).
    For instance, specify cargs=dict(normalize=False) to turn off normalization, but all other parameters unchanged

    :param img: initial image as either RGBImage or GrayscaleImage
    :param psf: point spread function, GrayscaleImage, RenderImage or three element list of R, G, B RenderImage
    :param m: magnification factor
    :param keep_size: if output image should be cropped back to size of the input image
    :param padding_mode: padding mode (from numpy.pad) for image padding before convolution.
    :param padding_value: padding value for padding_mode='constant'. 
     Three elements if img is a RGBImage, single element otherwise. Defaults to zeros.
    :param cargs: overwrite for parameters for color.xyz_to_srgb. By default these parameters are:
     dict(rendering_intent="Absolute", normalize=True, clip=True, L_th=0, chroma_scale=None).
     For instance, specify cargs=dict(normalize=False) to turn off normalization, but all other parameters unchanged
    :return: GrayscaleImage or RGBImage, see above 
    """
    # check types (image and psf are checked separately)
    pc.check_type("m", m, int | float)
    pc.check_type("cargs", cargs, dict)
    pc.check_above("abs(m)", abs(m), 0)
    pc.check_type("keep_size", keep_size, bool)
    
    # important flags
    flags = ConvFlags()
    flags.keep_size = keep_size
    flags.img_color = isinstance(img, RGBImage)
    flags.make_linear = isinstance(psf, GrayscaleImage) and isinstance(img, GrayscaleImage)
    flags.three_psf = isinstance(psf, list) and len(psf) == 3
    flags.psf_color = isinstance(psf, RenderImage) or flags.three_psf
    flags.threading = global_options.multithreading and misc.cpu_count() > 2 and not flags.make_linear
    # don't use multithreading when not set or when image has a single channel only anyway
    
    # create progress bar
    bar = ProgressBar("Convolving: ", 5)

    # Load image
    img_lin, pval_lin = _check_and_load_image(img, padding_mode, padding_value, flags)
    bar.update()

    # load PSF
    psf_lins, psfs = _check_and_load_psf(psf, flags)
    
    # check and calculate geometries
    s = _check_and_calculate_sizes(img, psfs, flags, m)

    # pad and flip image
    img_lin = _flip_and_pad_image(img_lin, s, m, flags, padding_mode, pval_lin)
    bar.update()
   
    # rescale and pad psf
    psf_lins2 = _rescale_and_pad_psf(psf_lins, s)
    bar.update()

    # Convolution
    img2 = _convolve_each(img_lin, psf_lins2, s, flags)
    bar.update()

    # Output Conversion
    img3 = _slice_and_convert_output_image(img2, cargs, s, flags)
    bar.finish()

    if flags.make_linear:
        return GrayscaleImage(img3, extent=s.i4e)

    return RGBImage(img3, extent=s.i4e)


def _check_and_load_image(img:              GrayscaleImage | RGBImage,
                          padding_mode:     str,
                          padding_value:    float | list[float],
                          flags:            ConvFlags)\
        -> tuple[np.ndarray, np.ndarray]:
    """
    Loads the image, checks its type and for a correct padding_value.
    Returns a img data array and padding_value in linear sRGB for a RGBImage img.
    Setting the flags.custom_padding flag.
    """
    pc.check_type("img", img, RGBImage | GrayscaleImage)
    
    # check padding_values
    if isinstance(img, RGBImage):
        pc.check_type("padding_value", padding_value, list | np.ndarray | None)
        
        pval = np.asarray(padding_value, dtype=np.float64) if padding_value is not None else np.array([0., 0., 0.])
        
        if pval.ndim != 1 or pval.shape[0] != 3:
            raise ValueError(f"padding_value must be a 3 element array/list, but has shape {pval.shape}")

        if np.any(pval < 0):
            raise ValueError(f"value in 'padding_value' needs to be non-negative.")

        pval_lin = color.srgb_to_srgb_linear(pval[np.newaxis, np.newaxis, :])[0, 0]
        img_lin = color.srgb_to_srgb_linear(img.data)

    else:
        pc.check_type("padding_value", padding_value, int | float | None)
        pval = float(padding_value) if padding_value is not None else 0.
        pc.check_not_below("padding_value", pval, 0)
        pval = np.array([pval, pval, pval], dtype=np.float64)

        pval_lin = color.srgb_to_srgb_linear(pval[np.newaxis, np.newaxis, :])[0, 0]
        img_lin = color.srgb_to_srgb_linear(img.data)

        if not flags.make_linear:
            img_lin = np.broadcast_to(img_lin[:, :, np.newaxis], [*img_lin.shape[:2], 3])
        else:
            pval_lin = pval_lin[0]

    flags.custom_padding = not (padding_mode == "constant" and np.sum(pval_lin) == 0)
    return img_lin, pval_lin


def _check_and_load_psf(psf: GrayscaleImage | RenderImage | list[RenderImage], flags: ConvFlags)\
        -> tuple[list[np.ndarray], list[GrayscaleImage | RenderImage]]:
    """
    Load the PSFs.
    Checks their type and correctness regarding extent and projection type.
    Returns a list of PSFs converted to a linear sRGB array, as well as a list of PSF objects.
    In the case of a single GrayscaleImage or RenderImage PSF a single element list is returned.
    """
    if flags.psf_color:
        psfs = [psf] if not flags.three_psf else psf
        pextent = psfs[0].extent

        if flags.img_color and not flags.three_psf:
            raise TypeError("A list of a R, G, B RenderImage PSF is required for convolving"
                            " a colored image with a colored PSF.")
        
        if not flags.img_color and flags.three_psf:
            raise TypeError("A single colored RenderImage is sufficient for a grayscale image.")

        psf_lins = []
        for i, psfi in enumerate(psfs):
            pc.check_type(f"psf[{i}]", psfi, RenderImage)

            if not np.all(pextent == psfi.extent):
                raise ValueError("All PSF sizes need to be the same. Render the detector image with"
                                 " the same manual extent option.")

            psf_lins.append(color.xyz_to_srgb_linear(psfi.data[:, :, :3], rendering_intent="Ignore", normalize=False))

    else:
        pc.check_type("psf", psf, GrayscaleImage)
        psfs = [psf]
        psf_lin = color.srgb_to_srgb_linear(psf.data)
        
        # normalize PSF
        if (psum := np.sum(psf_lin)):
            psf_lin /= psum

        if flags.make_linear:
            psf_lins = [psf_lin]
        else:
            psf_lins = [np.broadcast_to(psf_lin[:, :, np.newaxis], [*psf.shape[:2], 3])]

    return psf_lins, psfs


def _check_and_calculate_sizes(img:     RGBImage | GrayscaleImage, 
                               psfs:    list[RenderImage | GrayscaleImage], 
                               flags:   ConvFlags, 
                               m:       float)\
        -> ConvShapes:
    """Calculate image and psf shapes, sizes, extents for initial, processed and result images."""
    # grid coordinates:
    # image with side coordinates -5, 0, 5 has n = three pixels,
    # but a side length of s=10, as we measure from the center of the pixels
    # (the pixel edges would be at -7.5, -2.5, 2.5, 7.5)
    # the pixel size p is therefore s/(n-1)
    # from a known pixel size the length s is then (n-1)*p

    s = ConvShapes()

    # image and psf properties
    s.iN = np.array(np.flip(img.shape[:2]))  # image pixel count
    s.pN = np.array(np.flip(psfs[0].shape[:2]))  # psf pixel counts
    s.is_ = np.array(img.s) * abs(m)  # image side lengths scaled by magnification magnitude
    s.ps_ = np.array(psfs[0].s)  # psf side lengths
    s.ip = s.is_ / (s.iN - 1)  # image pixel sizes
    s.pp = s.ps_ / (s.pN - 1)  # psf pixel sizes

    if s.ps_[0] > 2 * s.is_[0] or s.ps_[1] > 2 * s.is_[1]:
        raise ValueError(f"m-scaled image size [{s.is_[0]:.5g}, {s.is_[1]:.5g}] is more than two times "
                         f"smaller than PSF size [{s.ps_[0]:.5g}, {s.ps_[1]:.5g}].")

    if s.pN[0] * s.pN[1] > 4e6:
        raise ValueError("PSF needs to be smaller than 4MP")

    if s.iN[0] * s.iN[1] > 4e6:
        raise ValueError("Image needs to be smaller than 4MP")

    if s.pp[0] > s.ip[0] or s.pp[1] > s.ip[1]:
        warning(f"PSF pixel sizes [{s.pp[0]:.5g}, {s.pp[1]:.5g}] larger than image pixel sizes"
                f" [{s.ip[0]:.5g}, {s.ip[1]:.5g}], generally you want a PSF in a higher resolution")

    if s.pN[0] < 50 or s.pN[1] < 50:
        raise ValueError(f"PSF too small with shape {psfs[0].shape}, "
                         "needs to have at least 50 values in each dimension.")

    if s.iN[0] < 50 or s.iN[1] < 50:
        raise ValueError(f"Image too small with shape {img.shape}, needs to have at least 50 values in each dimension.")

    if s.iN[0] * s.iN[1] < 2e4:
        warning(f"Low resolution image.")

    if s.pN[0] * s.pN[1] < 2e4:
        warning(f"Low resolution PSF.")

    if not (0.2 < s.pp[0] / s.pp[1] < 5):
        warning(f"Pixels of PSF are strongly non-square with side lengths [{s.pp[0]}mm, {s.pp[1]}mm]")

    if not (0.2 < s.ip[0] / s.ip[1] < 5):
        warning(f"Pixels of image are strongly non-square with side lengths [{s.ip[0]}mm, {s.ip[1]}mm]")

    # new psf sizes
    sc = s.pp / s.ip  # psf rescale factors
    s.ppad = np.array([4, 4], dtype=np.int32)
    s.p2N = np.where(s.pN * sc < 1, 1, np.round(s.pN * sc).astype(int))  # scale and round to new size
    s.p3N = s.p2N + 2 * s.ppad  # psf size after scaling and padding

    # new image sizes
    # we will convolve with the "full" mode of fftconvolve, which automatically pads the image
    # however, zero values are assumed beyond the image
    # with custom padding we need additional padding, so we don't have the descend towards black at the edges
    # the image will be sliced back later
    s.ipad = s.p3N if flags.custom_padding else np.array([0, 0], dtype=np.int32)  # additional padding size
    s.i2N = s.iN + 2 * s.ipad  # size after padding for padding mode
    s.i3N = s.i2N + s.p3N - 1  # size after padding for convolution

    # side length of output image after padding removal
    s.i4N = s.iN if flags.keep_size else s.iN + s.p3N - 1
    s.i4s = (s.i4N - 1) * s.ip

    # new image center is the sum of psf and old image center
    # (can be shown easily with convolution and shifting theorem of the fourier transform)
    extent = img.extent + psfs[0].extent
    xm = (extent[0] + extent[1]) / 2
    ym = (extent[2] + extent[3]) / 2
    s.i4e = [xm - s.i4s[0] / 2, xm + s.i4s[0] / 2, ym - s.i4s[1] / 2, ym + s.i4s[1] / 2]

    return s


def _flip_and_pad_image(img_lin:         np.ndarray,
                        s:               ConvShapes,
                        m:               float,
                        flags:           ConvFlags,
                        padding_mode:    str,
                        pval_lin:        float | np.ndarray)\
        -> np.ndarray:
    """
    Pads the image with the selected padding_mode and padding_value.
    Flips it according to the sign of m.
    """
    if flags.custom_padding:

        if padding_mode == "constant" and img_lin.ndim == 3:
            imgp = np.tile(pval_lin, (s.iN[1] + 2 * s.ipad[1], s.iN[0] + 2 * s.ipad[0], 1))
            imgp[s.ipad[1]:-s.ipad[1], s.ipad[0]:-s.ipad[0]] = img_lin
        else:
            pad_size = ((s.ipad[1], s.ipad[1]), (s.ipad[0], s.ipad[0]), (0, 0))
            shape = pad_size[:2] if img_lin.ndim == 2 else pad_size
            kwargs = dict(constant_values=pval_lin) if padding_mode == "constant" else {}
            imgp = np.pad(img_lin, shape, mode=padding_mode, **kwargs)
    else:
        imgp = img_lin

    # flip image
    if m < 0:
        imgp = np.fliplr(np.flipud(imgp))

    return imgp


def _rescale_and_pad_psf(psf_lins: list[np.ndarray], s: ConvShapes) -> list[np.ndarray]:
    """Rescale and pad PSFs to allow for a direct convolution."""
    
    psf2s = []
    for psf_lin in psf_lins:
        # INTER_AREA downscaling averages pixel according to their area
        # produces no aliasing effects and leaves the power sum (and also channel ratios) unchanged
        # rescaling by factors pny/pny2, and pnx/pnx2 is needed to leave the psf sum the same
        psf2 = cv2.resize(psf_lin, s.p2N, interpolation=cv2.INTER_AREA) * s.pN[0] * s.pN[1] / s.p2N[0] / s.p2N[1]

        # pad psf
        shape_xy = ((s.ppad[1], s.ppad[1]), (s.ppad[0], s.ppad[0]), (0, 0))
        shape = shape_xy[:2] if psf2.ndim == 2 else shape_xy
        psf2 = np.pad(psf2, shape, mode="constant", constant_values=0)

        psf2s.append(psf2)

    return psf2s


def _convolve_each(img_lin: list[np.ndarray], psf_lins: list[np.ndarray], s: ConvShapes, flags: ConvFlags)\
        -> np.ndarray:
    """Convolve image and PSFs."""

    # init output image
    out_shape = (s.i3N[1], s.i3N[0], 3) if not flags.make_linear else (s.i3N[1], s.i3N[0])
    img2 = np.zeros(out_shape, dtype=np.float64)
    img2_tmp = img2.copy()

    def threaded(img, psf, img_out, i):
        img_out[:, :, i] = scipy.signal.fftconvolve(img, psf, mode="full")

    if flags.three_psf:
            
        for i, psf_lin in enumerate(psf_lins):
        
            if not flags.threading:
                img2 += scipy.signal.fftconvolve(img_lin[:, :, i, np.newaxis], psf_lin, mode="full", axes=(0, 1))

            else:
                thread_list = []
                for j in range(3):
                    thread_list.append(Thread(target=threaded, args=(img_lin[:, :, i], psf_lin[:, :, j], img2_tmp, j)))
            
                [thread.start() for thread in thread_list]
                [thread.join() for thread in thread_list]
                img2 += img2_tmp

    else:
        if flags.make_linear:
            img2 = scipy.signal.fftconvolve(img_lin, psf_lins[0], mode="full")

        elif not flags.threading:
            img2 = scipy.signal.fftconvolve(img_lin, psf_lins[0], mode="full", axes=(0, 1))

        else:
            thread_list = []
            for i in range(3):
                thread_list.append(Thread(target=threaded, args=(img_lin[:, :, i], psf_lins[0][:, :, i], img2, i)))
            
            [thread.start() for thread in thread_list]
            [thread.join() for thread in thread_list]

    return img2


def _slice_and_convert_output_image(img2: np.ndarray, cargs: dict, s: ConvShapes, flags: ConvFlags) -> np.ndarray:
    """Apply slicing and color conversions back to sRGB color space."""

    # remove additional padding for custom padding modes
    if flags.custom_padding:
        img2 = img2[s.ipad[1]:-s.ipad[1], s.ipad[0]:-s.ipad[0]]

    # slice back to initial size, should be the same as fftconvolve with mode="same", but with our custom padding
    if flags.keep_size:
        i2sl = (s.i3N - s.i2N) // 2
        img2 = img2[i2sl[1]:i2sl[1] + s.iN[1], i2sl[0]:i2sl[0] + s.iN[0]]
 
    if flags.make_linear:
        # normalize if not set otherwise by user and if maximum exists
        if ("normalize" not in cargs or cargs["normalize"]) and (imax := np.max(img2)):
            img2 /= imax
        img2 = np.clip(img2, 0, 1)
        return color.srgb_linear_to_srgb(img2)
    
    # map color into sRGB gamut by converting from sRGB linear to XYZ to sRGB
    img2 = color.srgb_linear_to_xyz(img2)
    cargs0 = dict(rendering_intent="Absolute", normalize=True, clip=True, L_th=0, chroma_scale=None)
    img2 = color.xyz_to_srgb(img2, **(cargs0 | cargs))

    return img2
