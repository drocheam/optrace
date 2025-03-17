
import numpy as np  # calculations
import scipy.constants  # for c, h, k_B

from ...global_options import global_options as go

# default wavelength bounds
# defines the range wavelengths can exist in
_WL_MIN0 = 380.
_WL_MAX0 = 780.


def wavelengths(N: int) -> np.ndarray:
    """
    Get a wavelength range array with equal spacing and N points.
    The first and last wavelength are specified by global_options.wavelength_range

    :param N: number of values
    :return: wavelength vector in nm, 1D numpy array
    """
    return np.linspace(*go.wavelength_range, N)


def blackbody(wl: np.ndarray, T: float = 6504.) -> np.ndarray:
    """
    Get the spectral radiance of a planck blackbody curve. Unit is W/(sr m³).

    :param wl: wavelength vector in nm (numpy 1D array)
    :param T: blackbody temperature in Kelvin (float)
    :return: blackbody curve values (numpy 1D array)
    """

    # physical constants
    c, h, k_B = scipy.constants.c, scipy.constants.h, scipy.constants.k

    # wavelength in meters
    wlm = 1e-9*wl

    # blackbody equation
    # https://en.wikipedia.org/wiki/Planck%27s_law
    return 2 * h * c ** 2 / wlm**5 / (np.exp(h * c / (wlm * k_B * T)) - 1)

def normalized_blackbody(wl: np.ndarray, T: float = 6504.) -> np.ndarray:
    """
    Get the spectral power density of a planck blackbody curve.
    The values are normalized so that the highest value in the visible region,
    defined by global_options.wavelength_range is equal to 1.

    :param wl: wavelength vector in nm (numpy 1D array)
    :param T: blackbody temperature in Kelvin (float)
    :return: blackbody curve values (numpy 1D array)
    """

    l_w = 2897.771955 * 1e3 / T
    p_w, p_l, p_r = blackbody(np.array([l_w, *go.wavelength_range]), T)
    p_max = p_w if go.wavelength_range[0] <= l_w <= go.wavelength_range[1] else max(p_l, p_r)

    return blackbody(wl, T) / p_max

def has_color(rgb: np.ndarray, th: float = 1e-6) -> bool:
    """
    Check if color information are present in an RGB image.

    Calculated the channelwise standard deviation for each pixel.
    If it is above the threshold anywhere, the function returns true.

    :param rgb: RGB array shape (Ny, Nx, 3)
    :param th: threshold
    :return: boolean value if color information are found
    """
    return np.any(np.std(rgb, axis=2) > th)
