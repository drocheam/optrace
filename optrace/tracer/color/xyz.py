
import numpy as np  # matrix calculations

from .observers import x_observer, y_observer, z_observer  # spectrum to tristimulus


WP_D65_XYZ: list[float, float, float] = [0.95047, 1.00000, 1.08883]
"""whitepoint D65 in XYZ, see https://en.wikipedia.org/wiki/Illuminant_D65"""

WP_D65_XY: list[float, float] = [0.31272, 0.32903]
"""whitepoint D65 xy chromaticity coordinates, see CIE Colorimetry, 3. Edition, 2004, table 11.3"""


def xyz_to_xyY(xyz: np.ndarray) -> np.ndarray:
    """
    Convert XYZ to xyY coordinates.
    Black gets mapped to the whitepoint and Y=0

    :param xyz: input values, 2D image with channels in third dimension
    :return: converted values with same shape as input
    """
    s = np.sum(xyz, axis=2)
    mask = s > 0

    xyY = xyz.copy()

    xyY[mask, :2] /= s[mask, np.newaxis]
    xyY[~mask, :2] = WP_D65_XY  # set blacks to whitepoint chromacity

    xyY[:, :, 2] = xyz[:, :, 1]  # copy Y

    return xyY


def xyY_to_xyz(xyy: np.ndarray) -> np.ndarray:
    """
    Convert xyY to XYZ colorspace values.

    :param xyy: xyY values, 2D image with channels in third dimension
    :return: converted values, same shape as input
    """
    xyz = xyy.copy()

    # calculate z from x and y
    xyz[:, :, 2] = 1 - xyy[:, :, 0] - xyy[:, :, 1]
    
    # scale chromacity coordinates by ratio Y/y
    m = xyy[:, :, 1] != 0
    xyz[m] *= (xyy[m, 2] / xyy[m, 1]) [:, np.newaxis]

    return xyz


def xyz_from_spectrum(wl, spec, method="sum") -> np.ndarray:
    """
    Calculate the tristimulus values XYZ from a spectral distribution.

    :param wl: 1D wavelength vector
    :param spec: 1D spectral value vector, same shape as wl parameter
    :param method: "sum" or "trapz", method for numerical integration
    :return: numpy array of 3 elements
    """
    integrate = np.sum if method  == "sum" else np.trapz

    xyz = np.array([integrate(spec * x_observer(wl)),
                    integrate(spec * y_observer(wl)),
                    integrate(spec * z_observer(wl))])
    return xyz

