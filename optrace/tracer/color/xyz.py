
import numpy as np  # matrix calculations
import scipy.interpolate  # curve interpolation
import scipy.integrate

from .observers import x_observer, y_observer, z_observer  # spectrum to tristimulus
from .tools import wavelengths


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
    xyY[~mask, :2] = WP_D65_XY  # set blacks to whitepoint chromaticity

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
    
    # scale chromaticity coordinates by ratio Y/y
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
    integrate = np.sum if method  == "sum" else scipy.integrate.trapezoid

    xyz = np.array([integrate(spec * x_observer(wl)),
                    integrate(spec * y_observer(wl)),
                    integrate(spec * z_observer(wl))])
    return xyz


def _chrom_angle(XYZ_s: np.ndarray, res: int = 10000):
    """
    Calculate the angle of a XYZ color coordinate relative to the whitepoint inside the CIE 1931 chromaticity diagram.
    Also returns an interpolation object for a function mapping such an angle to a spectral wavelength

    :param res: spectral locus sampling resolution
    :param XYZ_s: XYZ color coordinate (3 element array)
    :return: angle inside the chromaticity diagram, interpolation object
    """
    # whitepoint
    xw, yw = WP_D65_XY

    # wavelengths and corresponding angles inside the diagram, rotation around the whitepoint
    wl = wavelengths(res)
    XYZ = np.dstack((x_observer(wl), y_observer(wl), z_observer(wl)))
    xyY = xyz_to_xyY(XYZ)
    phi = np.arctan2(xyY[0, :, 1] - yw, xyY[0, :, 0] - xw)
    
    # a phi data range [-pi/2, 3/2*pi] leads to a injective function

    # enforce phi range [-pi/2, 3/2*pi]
    phi[(phi < 0) & (phi < -np.pi/2)] += 2*np.pi

    # create the interpolation object, outside angles are set to nan
    interp = scipy.interpolate.interp1d(phi, wl, bounds_error=False)

    # angle for the given color
    x_s, y_s, _ = xyz_to_xyY(np.array([[XYZ_s]])).ravel()
    phi_s = np.arctan2(y_s-yw, x_s-xw)

    # enforce phi range [-pi/2, 3/2*pi]
    if phi_s < 0 and phi_s < -np.pi/2:
        phi_s += 2*np.pi

    return phi_s, interp


def dominant_wavelength(XYZ_s: np.ndarray, res: int = 10000) -> float:
    """
    Calculate the dominant wavelength for a given XYZ color coordinate and the D65 whitepoint.

    :param XYZ_s: 3 element array with the XYZ values
    :param res: spectral locus sampling resolution
    :return: dominant wavelength in nm
    """
    phi_s, interp = _chrom_angle(XYZ_s, res)
    return interp(phi_s)[()]


def complementary_wavelength(XYZ_s: np.ndarray, res: int = 10000) -> float:
    """
    Calculate the complementary wavelength for a given XYZ color coordinate and the D65 whitepoint.

    :param XYZ_s: 3 element array with the XYZ values
    :param res: spectral locus sampling resolution
    :return: complementary wavelength in nm
    """
    phi_s, interp = _chrom_angle(XYZ_s, res)

    # complementary angle
    phi_c = phi_s - np.pi

    # still enforce phi range [-pi/2, 3/2*pi]
    if phi_c < 0 and phi_c < -np.pi/2:
        phi_c += 2*np.pi

    return interp(phi_c)[()]

