
import numexpr as ne
import numpy as np

from .. import misc
from .xyz import WP_D65_XYZ


WP_D65_LUV: list[float, float, float] = [100, 0.19783982, 0.4683363]
"""whitepoint D65 in Lu'v', calculate from whitepoint XYZ coordinates"""

WP_D65_UV: list[float, float] = WP_D65_LUV[1:]
"""whitepoint D65 in u'v', calculated from XYZ coordinates"""

# whitepoints and primary coordinates in u'v'
SRGB_R_UV: list[float, float] = [0.4507042254, 0.5228873239]  #: sRGB red primary in u'v' coordinates
SRGB_G_UV: list[float, float] = [0.125, 0.5625]  #: sRGB green primary in u'v' coordinates
SRGB_B_UV: list[float, float] = [0.1754385965, 0.1578947368]  #: sRGBgblue primary  in u'v' coordinates


def xyz_to_luv(xyz: np.ndarray, normalize: bool = True) -> np.ndarray:
    """
    Convert XYZ values to CIELUV colorspace.

    :param xyz: XYZ image (2D with channels in third dimension)
    :param normalize: if lightness is normalized
    :return: CIELUV image, same shape as input
    """

    _, un, vn = WP_D65_LUV

    # exclude Y = 0 otherwise divisions by zero occur
    mask = xyz[:, :, 1] > 0  # Y > 0 for 3D array
    X, Y, Z = xyz[mask, 0], xyz[mask, 1], xyz[mask, 2]

    if not X.shape[0]:
        return np.zeros_like(xyz)

    if normalize:
        Yn = np.nanmax(Y)
    else:
        Yn = WP_D65_XYZ[1]

    # we only need to normalize t using Yn (see definition of t below), since u and v consist of XYZ ratios

    # conversion using
    # http://www.brucelindbloom.com/Eqn_XYZ_to_Luv.html
    # with the "actual CIE standard" constants

    Luv = np.zeros_like(xyz)

    k = 903.3
    e = 0.008856
    t = 1/Yn * Y

    mask2 = t > e  # t > e for L > 0
    mask3 = misc.part_mask(mask, mask2)  # Y > 0 and t > e
    mask4 = misc.part_mask(mask, ~mask2)  # Y > 0 and t <= e

    tm, tn = t[mask2], t[~mask2]
    Luv[mask3, 0] = ne.evaluate("116*tm**(1/3) - 16")
    Luv[mask4, 0] = k * tn

    D = ne.evaluate("1/(X + 15*Y + 3*Z)")
    u = 4*X*D
    v = 9*Y*D

    L13 = 13*Luv[mask, 0]
    Luv[mask, 1] = L13*(u-un)
    Luv[mask, 2] = L13*(v-vn)

    return Luv


def luv_to_xyz(luv: np.ndarray) -> np.ndarray:
    """
    Convert CIELUV back to XYZ.

    :param luv: CIELUV image (2D, with channels in third dimension)
    :return: XYZ image, same shape as input
    """

    # calculations are a rewritten from of
    # http://www.brucelindbloom.com/Eqn_Luv_to_XYZ.html
    # with the "actual CIE standard" constants

    _, un, vn = WP_D65_LUV

    # exclude L == 0, otherwise divisions by zero
    mask = luv[:, :, 0] > 0
    L_, u_, v_ = luv[mask, 0], luv[mask, 1], luv[mask, 2]

    XYZ = np.zeros_like(luv)

    k = 903.3
    e = 0.008856
    mask2 = L_ > k*e
    mask3 = misc.part_mask(mask, mask2)
    mask4 = misc.part_mask(mask, ~mask2)

    Lm, Ln = L_[mask2], L_[~mask2]
    XYZ[mask3, 1] = ne.evaluate("((Lm+16)/116)**3")
    XYZ[mask4, 1] = 1/k * Ln

    Y = XYZ[mask, 1]
    L13 = 13 * luv[mask, 0]

    XYZ[mask, 0] = X = ne.evaluate("9/4*Y * (u_ + L13*un) / (v_ + L13*vn)")
    XYZ[mask, 2] = ne.evaluate("3*Y * (L13/(v_ + L13*vn) - 5/3) - X/3")

    return XYZ


def luv_to_u_v_l(luv: np.ndarray) -> np.ndarray:
    """
    Convert to CIELUV to chromaticity color space u'v'L

    :param luv: CIELUV image, 2D image with channels in third dimension
    :return: u'v'L image, same shape as input
    """
    _, un, vn = WP_D65_LUV
    mi = luv[:, :, 0] > 0

    u_v_L = np.zeros_like(luv)
    u_v_L[:, :, 0] = un
    u_v_L[:, :, 1] = vn
    u_v_L[:, :, 2] = luv[:, :, 0]

    Lm, um, vm = luv[mi, 0], luv[mi, 1], luv[mi, 2]
    u_v_L[mi, 0] += ne.evaluate("1/13 * um / Lm")
    u_v_L[mi, 1] += ne.evaluate("1/13 * vm / Lm")

    return u_v_L


def luv_saturation(luv: np.ndarray) -> np.ndarray:
    """
    Get Chroma from CIELUV values. Calculation using https://en.wikipedia.org/wiki/Colorfulness#Saturation

    :param luv: Luv Image, np.ndarray with shape (Ny, Nx, 3)
    :return: Saturation Image, np.ndarray with shape (Ny, Nx)
    """
    C = luv_chroma(luv)

    Sat = np.zeros_like(C)
    mask = luv[:, :, 0] > 0
    Sat[mask] = C[mask] / luv[mask, 0]

    return Sat


def luv_chroma(luv: np.ndarray) -> np.ndarray:
    """
    Get Chroma from CIELUV values. Calculation using https://en.wikipedia.org/wiki/Colorfulness#Chroma

    :param luv: Luv Image, np.ndarray with shape (Ny, Nx, 3)
    :return: Chroma Image, np.ndarray with shape (Ny, Nx)
    """
    u, v = luv[:, :, 1], luv[:, :, 2]
    return ne.evaluate("sqrt(u**2 + v**2)")


def luv_hue(luv: np.ndarray) -> np.ndarray:
    """
    Get Hue from CIELUV values. Calculation using https://en.wikipedia.org/wiki/Colorfulness#Chroma

    :param luv: Luv Image, np.ndarray with shape (Ny, Nx, 3)
    :return: Hue Image, np.ndarray with shape (Ny, Nx)
    """
    pi = np.pi
    u, v = luv[:, :, 1], luv[:, :, 2]
    hue = ne.evaluate("arctan2(v, u)/pi*180")
    hue[hue < 0] += 360
    return hue
