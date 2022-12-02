
import numpy as np  # calculations
import scipy.constants  # for c, h, k_B



# default wavelength bounds
_WL_MIN0 = 380.
_WL_MAX0 = 780.

WL_BOUNDS: list[float, float] = [_WL_MIN0, _WL_MAX0]
"""wavelength range"""


def wavelengths(N: int) -> np.ndarray:
    """
    Get wavelength range array with equal spacing and N points.

    :param N: number of values
    :return: wavelength vector in nm, 1D numpy array

    >>> wavelengths(5)
    array([ 380., 480., 580., 680., 780.])
    """
    return np.linspace(WL_BOUNDS[0], WL_BOUNDS[1], N)


def blackbody(wl: np.ndarray, T: float = 6504.) -> np.ndarray:
    """
    Get spectral radiance of a planck blackbody curve. Unit is W/(sr m³).

    :param wl: wavelength vector in nm (numpy 1D array)
    :param T: blackbody temperature in Kelvin (float)
    :return: blackbody curve values (numpy 1D array)

    >>> blackbody(np.array([380., 500., 600.]), T=5500)
    array([ 1.54073437e+13, 2.04744373e+13, 1.98272922e+13])
    """

    # physical constants
    c, h, k_B = scipy.constants.c, scipy.constants.h, scipy.constants.k

    # wavelength in meters
    wlm = 1e-9*wl

    # blackbody equation
    # https://en.wikipedia.org/wiki/Planck%27s_law
    return 2 * h * c ** 2 / wlm**5 / (np.exp(h * c / (wlm * k_B * T)) - 1)
