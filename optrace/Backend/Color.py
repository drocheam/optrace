"""
Color conversion and processing functions

"""

import numpy as np
import colorio
import optrace.Backend.Misc as misc
from typing import Callable

WL_MIN: float = 380.
"""lower bound of wavelength range in nm
the wavelength range. Needs to be inside [380, 780] for the Tristimulus and Illuminant functions to work
Note that shrinking that range may lead to color deviations"""

WL_MAX: float = 780.
"""upper bound of wavelength range in nm"""

def wavelengths(N: int) -> np.ndarray:
    """
    Get wavelength range array with equal spacing and N points.
    :param N:
    :return:
    """
    return np.linspace(WL_MIN, WL_MAX, N)

def Blackbody(wl: np.ndarray, T: (int | float) = 6504) -> np.ndarray:
    """
    Get planck blackbody curve

    :param wl: wavelength vector in nm (numpy 1D array)
    :param T: temperature in Kelvin (float)
    :return: blackbody values (numpy 1D array)
    """

    # blackbody curve
    c = 299792458
    h = 6.62607015e-34
    k_B = 1.380649e-23

    spec = misc.calc("2 * h * c ** 2 / (wl*1e-9) ** 5 / (exp(h * c / (wl * 1e-9 * k_B * T)) - 1)")
    return spec


def Gauss(x: np.ndarray, mu: float, sig: float) -> np.ndarray:
    """
    normalized Gauss Function
    :param x:
    :param mu:
    :param sig:
    :return:
    """
    return 1/(sig*np.sqrt(2*np.pi)) * np.exp(-0.5 * (x - mu) ** 2 / sig ** 2)

def Tristimulus(wl: np.ndarray, name: str) -> np.ndarray:
    """
    Get tristimulus CIE 1931 2° observer data at specified wavelengths.
    Uses :obj:`colorio.observers` for curve data with additional interpolation.

    :param wl: wavelength vector
    :param name: One of "X", "Y", "Z".
    :return:    

    >>> Tristimulus(np.array([500, 600]), "X")
    array([0.0049, 1.0622])

    >>> Tristimulus(np.array([500, 600]), "Y")
    array([0.323, 0.631])
    """

    choices = ["X", "Y", "Z"]

    if name not in choices:
        raise ValueError("Invalid Tristimulus Type")

    ind = choices.index(name)

    observer = colorio.observers.cie_1931_2()
    return misc.interp1d(observer.lmbda_nm, observer.data[ind], wl)


def Illuminant(wl: np.ndarray, name: str) -> np.ndarray:
    """
    Get Illuminant data at specified wavelengths.
    Uses :obj:`colorio.illuminants` for curve data with additional interpolation.

    :param wl: wavelength vector
    :param name: One of "A", "C", "D50", "D55", "D65", "D75", "E", "F2", "F7", "F11".
    :return:    

    >>> Illuminant(np.array([500, 600]), "D50")
    array([95.7237, 97.6878])

    >>> Illuminant(np.array([500, 600]), "D65")
    array([109.3545,  90.0062])
    """

    if name not in ["A", "C", "D50", "D55", "D65", "D75", "E", "F2", "F7", "F11"]:
        raise ValueError("Invalid Illuminant Type")

    illu = eval(f"colorio.illuminants.{name.lower()}()")
    return misc.interp1d(illu.lmbda_nm, illu.data, wl)
    

def sRGB_to_sRGBLinear(RGB_in: np.ndarray) -> np.ndarray:
    """
    Conversion from sRGB to linear RGB values

    :param RGB_in: RGB values (numpy 1D, 2D or 3D array)
    :return: linear RGB values, array with same shape as input
    """

    RGB = RGB_in.copy()

    # remove gamma correction (sRGB -> RGBLinear)
    # Source: https://en.wikipedia.org/wiki/SRGB#Specification_of_the_transformation
    a = 0.055
    below = RGB <= 0.04045
    RGB[below] *= 1 / 12.92
    RGB[~below] = ((RGB[~below] + a) / (1 + a)) ** 2.4

    return RGB


def sRGBLinear_to_XYZ(RGBL_in: np.ndarray) -> np.ndarray:
    """
    Conversion from linear RGB values to XYZ

    :param RGBL_in: linear RGB image (numpy 3D array, RGB channels in third dimension)
    :return: XYZ image (numpy 3D array)
    """

    # # transform to XYZ
    # source for matrix: http://brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
    # note that the matrix differs in different sources
    trans = np.array([[0.4124564, 0.3575761, 0.1804375],
                      [0.2126729, 0.7151522, 0.0721750],
                      [0.0193339, 0.1191920, 0.9503041]])

    RGBL = RGBL_in.flatten().reshape((RGBL_in.shape[0] * RGBL_in.shape[1], 3))
    XYZ = (trans @ RGBL.T).T

    return XYZ.reshape((RGBL_in.shape[0], RGBL_in.shape[1], 3))


def sRGB_to_XYZ(RGB: np.ndarray) -> np.ndarray:
    """
    Conversion from sRGB to XYZ

    :param RGB: sRGB image (numpy 3D array, RGB channels in third dimension)
    :return: XYZ image (numpy 3D array)
    """

    # sRGB -> XYZ is just sRGB -> RGBLinear -> XYZ
    RGBL = sRGB_to_sRGBLinear(RGB)
    XYZ = sRGBLinear_to_XYZ(RGBL)

    return XYZ


def XYZ_to_sRGBLinear(XYZ_in: np.ndarray, normalize: bool = True) -> np.ndarray:
    """
    Conversion XYZ to linear RGB values.

    :param XYZ_in: XYZ image (numpy 3D array, XYZ channels in third dimension)
    :param normalize: if image is normalized to highest value before conversion (bool)
    :return: linear RGB image (numpy 3D array)
    """

    XYZ = XYZ_in.copy()

    # normalize to highest value
    if normalize:
        nmax = np.nanmax(XYZ)
        if nmax > 0:
            XYZ /= nmax

    # conversion XYZ -> RGB_linear
    # source for matrix: http://brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
    # note that the matrix differs in different sources
    trans = np.array([[3.2404542, -1.5371385, -0.4985314],
                      [-0.9692660, 1.8760108, 0.0415560],
                      [0.0556434, -0.2040259, 1.0572252]])

    XYZ_f = XYZ.flatten().reshape((XYZ.shape[0] * XYZ.shape[1], 3))

    RGBL = (trans @ XYZ_f.T).T

    return RGBL.reshape((XYZ.shape[0], XYZ.shape[1], 3))


def sRGBLinear_to_sRGB(RGBL: np.ndarray, normalize: bool = True) -> np.ndarray:
    """
    Conversion linear RGB to sRGB

    :param RGBL: linear RGB values (numpy 1D, 2D or 3D array)
    :param normalize: if RGB values are normalized before conversion (bool)
    :return: sRGB image (same shape as input)
    """
  
    RGB = RGBL.copy()

    # normalized RGBL signal
    if normalize:
        nmax = np.nanmax(RGB)
        if nmax > 0:
            RGB /= nmax

    # clip values
    RGB[RGB < 0] = 0
    RGB[RGB > 1] = 1
   
    # gamma correction. RGB -> sRGB
    # Source: https://en.wikipedia.org/wiki/SRGB#Specification_of_the_transformation
    a = 0.055
    below = RGB <= 0.0031308
    RGB[below] *= 12.92
    RGB[~below] = (1 + a) * RGB[~below] ** (1 / 2.4) - a

    return RGB


def XYZ_to_sRGB(XYZ: np.ndarray, normalize: bool = True) -> np.ndarray:
    """
    Conversion XYZ to sRGB

    :param XYZ: XYZ image (numpy 3D array, XYZ channels in third dimension)
    :param normalize: if values are normalized before conversion (bool)
    :return: sRGB image (numpy 3D array)
    """
    # XYZ -> sRGB is just XYZ -> RGBLinear -> sRGB
    RGBL = XYZ_to_sRGBLinear(XYZ, normalize=False)
    RGB = sRGBLinear_to_sRGB(RGBL, normalize)
    return RGB


def randomWavelengthFromRGB(RGB: np.ndarray) -> np.ndarray:
    """
    Choose random wavelengths from RGB colors.

    :param RGB: RGB values (numpy 2D array, RGB channels in second dimension)
    :return: random wavelengths for every color (numpy 1D array)

    """
    # calculation for arbitrary primaries using matrix inverse:
    # see Microchip AN1562 High Resolution RGB LED Color Mixing Application Note
    # https://ww1.microchip.com/downloads/en/Appnotes/00001562B.pdf
    #####
    # but in our case we use the RGB primaries and the RGBLinear values

    # use wavelengths in nm, otherwise the numeric range would be too large
    wl = wavelengths(3000)

    # spectra that lie at the sRGB primaries position in the xy-CIE Diagram
    r =  88.4033043 * (Gauss(wl, 660.255528, 35.6986569) + 0.0665761658 * Gauss(wl, 552.077348, 150.000000))
    g =  83.4999030 *  Gauss(wl, 539.131090, 33.3116417)
    b = 118.345477  *  Gauss(wl, 415.035902, 47.2130145)

    # the curves above have the same xy coordinates as the sRGB primaries, but the brightness Y is still different
    # the following scale factors are p_Y/integral(spec * y)  with spec being r, g or b; y being the y stimulus curve
    # and p_Y being the Y brightness value of the sRGB primary ([0.2126729, 0.7151522, 0.0721750] for red, green, blue)
    # The resulting factors here are normalized by the value received for the g curve
    r *= 1.24573718
    # g *= 1
    b *= 1.12354883

    # since our primaries have the same xyY values as the sRGB primaries, we can use the RGB intensities for mixing.
    # For the RGB intensities we convert the gamma corrected sRGB values to RGBLinear
    RGBL = sRGB_to_sRGBLinear(RGB)

    # we assign only one wavelength per ray, so we need to select one of the r, g, b curves and then a wavelength
    # underneath it. To select r, g or b we need to scale the mixing ratios Rr, Rg, Rb by the overall probability
    # (= integral of the r, g or b curve)
    RGBL[:, 0] *= 1.38950586
    # RGBL[:, 1] *= 1
    RGBL[:, 2] *= 1.22823756

    # in this part we select the r, g or b spectrum depending on the mixing ratios Rr, Rg, Rb
    rgb_sum = np.cumsum(RGBL, axis=-1)
    rgb_sum /= rgb_sum[:, -1, np.newaxis]

    # chose x, y or z depending on in which range rgb_choice fell
    rgb_choice = np.random.sample(RGB.shape[0])
    make_r = rgb_choice <= rgb_sum[:, 0]
    make_g = (rgb_choice > rgb_sum[:, 0]) & (rgb_choice <= rgb_sum[:, 1])
    make_b = (rgb_choice > rgb_sum[:, 1]) & (rgb_choice <= rgb_sum[:, 2])

    wl_out = np.zeros(RGB.shape[0], dtype=np.float32)
    wl_out[make_r] = misc.random_from_distribution(wl, r, np.count_nonzero(make_r))
    wl_out[make_g] = misc.random_from_distribution(wl, g, np.count_nonzero(make_g))
    wl_out[make_b] = misc.random_from_distribution(wl, b, np.count_nonzero(make_b))

    return wl_out


def spectralCM(N: int, wl0: float = WL_MIN, wl1: float = WL_MAX) -> np.ndarray:
    """
    Get a spectral colormap with N steps

    :param N: number of steps (int)
    :return: sRGBA array (numpy 2D array, shape (N, 4))

    >>> spectralCM(5)
    array([[ 97.32786529,   0.        ,  97.32786529,  97.32786529],
           [  0.        , 213.31047127, 255.        , 255.        ],
           [255.        , 255.        ,   0.        , 255.        ],
           [255.        ,   0.        ,   0.        , 255.        ],
           [ 97.32786529,   0.        ,   0.        ,  97.32786529]])
    """
    # approximate RGB values for spectral colors (only for visualization)
    # rewritten python version of http://www.physics.sfasu.edu/astro/color/spectra.html

    wl = np.linspace(wl0, wl1, N, dtype=np.float32)  # wavelength vector
    RGB = np.zeros((N, 4), dtype=np.float64, order='F')  # RGB matrix
    RGB[:, 3] = 1  # set alpha to 100%

    #### set spectra
    ###########
    
    sec = (wl >= 380) & (wl <= 440)
    RGB[sec, 0] = -(wl[sec]-440)/(440-380)
    RGB[sec, 2] = 1

    sec = (wl >= 440) & (wl <= 490)
    RGB[sec, 1] = (wl[sec] - 440) / (490 - 440)
    RGB[sec, 2] = 1

    sec = (wl >= 490) & (wl <= 510)
    RGB[sec, 1] = 1
    RGB[sec, 2] = -(wl[sec] - 510) / (510 - 490)

    sec = (wl >= 510) & (wl <= 580)
    RGB[sec, 0] = (wl[sec] - 510) / (580 - 510)
    RGB[sec, 1] = 1.

    sec = (wl >= 580) & (wl <= 645)
    RGB[sec, 0] = 1.
    RGB[sec, 1] = -(wl[sec] - 645) / (645 - 580)

    sec = (wl >= 645) & (wl <= 780)
    RGB[sec, 0] = 1.

    #### intensity fall-off at spectrum edges
    ###########
    SSS = np.ones_like(wl, dtype=np.float64)

    sec = wl > 700
    SSS[sec] = 0.3 + 0.7 * (780 - wl[sec]) / (780 - 700)

    sec = wl < 420
    SSS[sec] = 0.3 + 0.7 * (wl[sec] - 380) / (420 - 380)

    # Gamma correction and rescaling
    ##########
    GAMMA = 0.8
    RGB = 255 * (RGB * SSS[:, np.newaxis]) ** GAMMA

    return RGB



