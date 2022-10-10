
import numpy as np

from .observers import x_tristimulus, y_tristimulus, z_tristimulus


WP_D65_XYZ: list[float, float, float] = [0.95047, 1.00000, 1.08883]
"""whitepoint D65 in XYZ, see https://en.wikipedia.org/wiki/Illuminant_D65"""

WP_D65_XY: list[float, float] = [0.31272, 0.32903]
"""whitepoint D65 xy chromaticity coordinates, see CIE Colorimetry, 3. Edition, 2004, table 11.3"""


def xyz_to_xyY(xyz: np.ndarray) -> np.ndarray:
    """
    Convert XYZ to xyY coordinates.
    Black gets mapped to the whitepoint and Y=0

    :param xyz:
    :return:
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

    :param xyy: xyY values
    :return:
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

    :param wl:
    :param spec:
    :param method:
    :return:
    """
    integrate = np.sum if method  == "sum" else np.trapz

    xyz = np.array([integrate(spec * x_tristimulus(wl)),
                    integrate(spec * y_tristimulus(wl)),
                    integrate(spec * z_tristimulus(wl))])
    return xyz
