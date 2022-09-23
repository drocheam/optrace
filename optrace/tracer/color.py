"""
Color conversion and processing functions

"""

import pathlib

import numpy as np  # calculations
import numexpr as ne  # faster calculations
import scipy.constants  # for c, h, k_B

from . import misc  # calculations


# DO NOT CHANGE THIS WAVELENGTH RANGE IF YOU WANT COLORS TO WORK CORRECTLY

WL_MIN: float = 380.
"""lower bound of wavelength range in nm."""

WL_MAX: float = 780.
"""upper bound of wavelength range in nm"""

SRGB_RENDERING_INTENTS: list[str, str, str] = ["Ignore", "Absolute", "Perceptual"]
"""Rendering intents for XYZ to sRGB conversion"""

# Whitepoints in XYZ and Luv
WP_D65_XYZ: list[float, float, float] = [0.95047, 1.00000, 1.08883]
"""whitepoint D65 in XYZ, https://en.wikipedia.org/wiki/Illuminant_D65"""

WP_D65_LUV: list[float, float, float] = [100, 0.19783982, 0.4683363]
"""whitepoint D65 in Lu'v'"""

# whitepoints and primary coordinates in xy
SRGB_R_XY: list[float, float] = [0.64, 0.33]  #: sRGB red primary in xy coordinates
SRGB_G_XY: list[float, float] = [0.30, 0.60]  #: sRGB green primary in xy coordinates
SRGB_B_XY: list[float, float] = [0.15, 0.06]  #: sRGB blue primary in xy coordinates
SRGB_W_XY: list[float, float] = [0.31271, 0.32902]
"""whitepoint D65 xy coordinates, https://en.wikipedia.org/wiki/Illuminant_D65"""

# whitepoints and primary coordinates in u'v'
SRGB_R_UV: list[float, float] = [0.4507042254, 0.5228873239]  #: sRGB red primary in u'v' coordinates
SRGB_G_UV: list[float, float] = [0.125, 0.5625]  #: sRGB green primary in u'v' coordinates
SRGB_B_UV: list[float, float] = [0.1754385965, 0.1578947368]  #: sRGB blue primary  in u'v' coordinates
SRGB_W_UV: list[float, float] = WP_D65_LUV[1:]  #: D65 whitepoint in u'v' coordinates

# load illuminants
_ill_names = ["wl", "A", "C", "D50", "D55", "D65", "D75", "FL2", "FL7", "FL11",
              "LED-B1", "LED-B2", "LED-B3", "LED-B4", "LED-B5"]
_ill_path = pathlib.Path(__file__).resolve().parent.parent / "ressources" / "illuminants.csv"
_illuminants = np.genfromtxt(_ill_path, skip_header=1, delimiter=",", dtype=np.float64)

# load observers
_obs_names = ["wl", "x", "y", "z"]
_obs_path = pathlib.Path(__file__).resolve().parent.parent / "ressources" / "observers.csv"
_observers = np.genfromtxt(_obs_path, skip_header=1, delimiter=",", dtype=np.float64)

# Sources A, C, E, D50, D55, D65, D75, F2, F7, F11:
# CIE Colorimetry, 3. Edition, 2004

# Sources tristimulus values:
# https://cie.co.at/datatable/cie-1931-colour-matching-functions-2-degree-observer

# Sources LED Series:
# eigentlich CIE Colorimetry, 4th Edition, 2018, Werte aber von
# https://github.com/colour-science/colour/blob/develop/colour/colorimetry/datasets/illuminants/sds.py


def wavelengths(N: int) -> np.ndarray:
    """
    Get wavelength range array with equal spacing and N points.

    :param N: number of values
    :return: wavelength vector in nm, 1D numpy array

    >>> wavelengths(5)
    array([380., 480., 580., 680., 780.])
    """
    return np.linspace(WL_MIN, WL_MAX, N)


def blackbody(wl: np.ndarray, T: float = 6504.) -> np.ndarray:
    """
    Get spectral radiance of a planck blackbody curve. Unit is W/(sr m³).

    :param wl: wavelength vector in nm (numpy 1D array)
    :param T: blackbody temperature in Kelvin (float)
    :return: blackbody curve values (numpy 1D array)

    >>> blackbody(np.array([380., 500., 600.]), T=5500)
    array([1.54073437e+13, 2.04744373e+13, 1.98272922e+13])
    """

    # physical constants
    c, h, k_B = scipy.constants.c, scipy.constants.h, scipy.constants.k

    # wavelength in meters
    wlm = 1e-9*wl

    # blackbody equation
    # https://en.wikipedia.org/wiki/Planck%27s_law
    return 2 * h * c ** 2 / wlm**5 / (np.exp(h * c / (wlm * k_B * T)) - 1)


def gauss(x: np.ndarray, mu: float, sig: float) -> np.ndarray:
    """
    normalized Gauss Function

    :param x:
    :param mu:
    :param sig:
    :return:

    >>> gauss(np.array([0., 0.5, 1.5]), 0.75, 1)
    array([0.30113743, 0.38666812, 0.30113743])
    """
    return 1/(sig*np.sqrt(2*np.pi)) * np.exp(-0.5 * (x - mu) ** 2 / sig ** 2)


def x_tristimulus(wl: np.ndarray) -> np.ndarray:
    """
    Eye x tristimulus values (CIE 1931 2° Standard Observer)
    :param wl: wavelength array
    :return: value array
    """
    return np.interp(wl, _observers[:, 0], _observers[:, _obs_names.index("x")], left=0, right=0)


def y_tristimulus(wl: np.ndarray) -> np.ndarray:
    """
    Eye y tristimulus values (CIE 1931 2° Standard Observer)
    :param wl: wavelength array
    :return: value array
    """
    return np.interp(wl, _observers[:, 0], _observers[:, _obs_names.index("y")], left=0, right=0)


def z_tristimulus(wl: np.ndarray) -> np.ndarray:
    """
    Eye z tristimulus values (CIE 1931 2° Standard Observer)
    :param wl: wavelength array
    :return: value array
    """
    return np.interp(wl, _observers[:, 0], _observers[:, _obs_names.index("z")], left=0, right=0)


def a_illuminant(wl: np.ndarray) -> np.ndarray:
    """
    A standard illuminant (CIE Colorimetry, 3. Edition, 2004)
    :param wl: wavelength array
    :return: value array
    """
    return np.interp(wl, _illuminants[:, 0], _illuminants[:, _ill_names.index("A")], left=0, right=0)


def c_illuminant(wl: np.ndarray) -> np.ndarray:
    """
    C standard illuminant (CIE Colorimetry, 3. Edition, 2004)
    :param wl: wavelength array
    :return: value array
    """
    return np.interp(wl, _illuminants[:, 0], _illuminants[:, _ill_names.index("C")], left=0, right=0)


def e_illuminant(wl: np.ndarray) -> np.ndarray:
    """
    E standard illuminant (CIE Colorimetry, 3. Edition, 2004)
    :param wl: wavelength array
    :return: value array
    """
    return np.full_like(wl, 100.0, dtype=np.float64)


def d50_illuminant(wl: np.ndarray) -> np.ndarray:
    """
    D50 standard illuminant (CIE Colorimetry, 3. Edition, 2004)
    :param wl: wavelength array
    :return: value array
    """
    return np.interp(wl, _illuminants[:, 0], _illuminants[:, _ill_names.index("D50")], left=0, right=0)


def d55_illuminant(wl: np.ndarray) -> np.ndarray:
    """
    D55 standard illuminant (CIE Colorimetry, 3. Edition, 2004)
    :param wl: wavelength array
    :return: value array
    """
    return np.interp(wl, _illuminants[:, 0], _illuminants[:, _ill_names.index("D55")], left=0, right=0)


def d65_illuminant(wl: np.ndarray) -> np.ndarray:
    """
    D65 standard illuminant (CIE Colorimetry, 3. Edition, 2004)
    :param wl: wavelength array
    :return: value array
    """
    return np.interp(wl, _illuminants[:, 0], _illuminants[:, _ill_names.index("D65")], left=0, right=0)


def d75_illuminant(wl: np.ndarray) -> np.ndarray:
    """
    D75 standard illuminant (CIE Colorimetry, 3. Edition, 2004)
    :param wl: wavelength array
    :return: value array
    """
    return np.interp(wl, _illuminants[:, 0], _illuminants[:, _ill_names.index("D75")], left=0, right=0)


def fl2_illuminant(wl: np.ndarray) -> np.ndarray:
    """
    FL2 standard illuminant (CIE Colorimetry, 3. Edition, 2004)
    :param wl: wavelength array
    :return: value array
    """
    return np.interp(wl, _illuminants[:, 0], _illuminants[:, _ill_names.index("FL2")], left=0, right=0)


def fl7_illuminant(wl: np.ndarray) -> np.ndarray:
    """
    FL7 standard illuminant (CIE Colorimetry, 3. Edition, 2004)
    :param wl: wavelength array
    :return: value array
    """
    return np.interp(wl, _illuminants[:, 0], _illuminants[:, _ill_names.index("FL7")], left=0, right=0)


def fl11_illuminant(wl: np.ndarray) -> np.ndarray:
    """
    FL11 standard illuminant (CIE Colorimetry, 3. Edition, 2004)
    :param wl: wavelength array
    :return: value array
    """
    return np.interp(wl, _illuminants[:, 0], _illuminants[:, _ill_names.index("FL11")], left=0, right=0)


def led_b1_illuminant(wl: np.ndarray) -> np.ndarray:
    """
    LED-B1 standard illuminant (CIE Colorimetry, 4th Edition, 2018)
    :param wl: wavelength array
    :return: value array
    """
    return np.interp(wl, _illuminants[:, 0], _illuminants[:, _ill_names.index("LED-B1")], left=0, right=0)


def led_b2_illuminant(wl: np.ndarray) -> np.ndarray:
    """
    LED-B2 standard illuminant (CIE Colorimetry, 4th Edition, 2018)
    :param wl: wavelength array
    :return: value array
    """
    return np.interp(wl, _illuminants[:, 0], _illuminants[:, _ill_names.index("LED-B2")], left=0, right=0)


def led_b3_illuminant(wl: np.ndarray) -> np.ndarray:
    """
    LED-B3 standard illuminant (CIE Colorimetry, 4th Edition, 2018)
    :param wl: wavelength array
    :return: value array
    """
    return np.interp(wl, _illuminants[:, 0], _illuminants[:, _ill_names.index("LED-B3")], left=0, right=0)


def led_b4_illuminant(wl: np.ndarray) -> np.ndarray:
    """
    LED-B4 standard illuminant (CIE Colorimetry, 4th Edition, 2018)
    :param wl: wavelength array
    :return: value array
    """
    return np.interp(wl, _illuminants[:, 0], _illuminants[:, _ill_names.index("LED-B4")], left=0, right=0)


def led_b5_illuminant(wl: np.ndarray) -> np.ndarray:
    """
    LED-B5 standard illuminant (CIE Colorimetry, 4th Edition, 2018)
    :param wl: wavelength array
    :return: value array
    """
    return np.interp(wl, _illuminants[:, 0], _illuminants[:, _ill_names.index("LED-B5")], left=0, right=0)


def srgb_to_srgb_linear(rgb: np.ndarray) -> np.ndarray:
    """
    Conversion from sRGB to linear RGB values

    :param rgb: RGB values (numpy 1D, 2D or 3D array)
    :return: linear RGB values, array with same shape as input
    """

    RGB = rgb.copy()

    # remove gamma correction (sRGB -> RGBLinear)
    # Source: https://en.wikipedia.org/wiki/SRGB#Specification_of_the_transformation
    a = 0.055
    below = RGB <= 0.04045
    RGB[below] *= 1 / 12.92
    RGB[~below] = ((RGB[~below] + a) / (1 + a)) ** 2.4

    return RGB


def srgb_linear_to_xyz(rgbl: np.ndarray) -> np.ndarray:
    """
    Conversion from linear RGB values to XYZ

    :param rgbl: linear RGB image (numpy 3D array, RGB channels in third dimension)
    :return: XYZ image (numpy 3D array)
    """

    # # transform to XYZ
    # source for matrix: http://brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
    # note that the matrix differs in different sources
    trans = np.array([[0.4124564, 0.3575761, 0.1804375],
                      [0.2126729, 0.7151522, 0.0721750],
                      [0.0193339, 0.1191920, 0.9503041]])

    RGBL_f = rgbl.reshape((rgbl.shape[0] * rgbl.shape[1], 3))
    XYZ = (trans @ RGBL_f.T).T

    return XYZ.reshape(rgbl.shape)


def srgb_to_xyz(rgb: np.ndarray) -> np.ndarray:
    """
    Conversion from sRGB to XYZ

    :param rgb: sRGB image (numpy 3D array, RGB channels in third dimension)
    :return: XYZ image (numpy 3D array)
    """

    # sRGB -> XYZ is just sRGB -> RGBLinear -> XYZ
    RGBL = srgb_to_srgb_linear(rgb)
    XYZ = srgb_linear_to_xyz(RGBL)

    return XYZ


def outside_srgb_gamut(xyz: np.ndarray) -> np.ndarray:
    """

    :param xyz:
    :return:
    """
    return np.any(xyz_to_srgb_linear(xyz, rendering_intent="Ignore") < 0, axis=2)


def xyz_to_xyY(xyz: np.ndarray) -> np.ndarray:
    """
    Convert XYZ to xyY coordinates

    :param xyz:
    :return:
    """
    s = np.sum(xyz, axis=2)
    mask = s > 0

    xyY = xyz.copy()

    xyY[mask, :2] /= s[mask, np.newaxis]
    xyY[~mask, :2] = SRGB_W_XY

    xyY[:, :, 2] = xyz[:, :, 1]

    return xyY


def xyz_to_srgb_linear(xyz: np.ndarray, normalize: bool = True, rendering_intent: str = "Absolute") -> np.ndarray:
    """
    Conversion XYZ to linear RGB values.

    :param xyz: XYZ image (numpy 3D array, XYZ channels in third dimension)
    :param normalize: if image is normalized to highest value before conversion (bool)
    :param rendering_intent:
    :return: linear RGB image (numpy 3D array)
    """

    def _to_srgb(xyz_: np.ndarray) -> np.ndarray:

        # it makes no difference if we normalize before or after the matrix multiplication
        # since X, Y and Z gets scaled by the same value and matrix multiplication is a linear operation
        # normalizing after conversion makes it possible to normalize to the highest RGB value,
        # thus the highest Value (Value as in the V in the HSV model)
        # normalizing after guarantees a fixed range in [0, 1] for all possible  inputs

        # source for conversion matrix: http://brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
        trans = np.array([[3.2404542, -1.5371385, -0.4985314],
                          [-0.9692660, 1.8760108, 0.0415560],
                          [0.0556434, -0.2040259, 1.0572252]])

        XYZ_f = xyz_.reshape((xyz_.shape[0] * xyz_.shape[1], 3))

        RGBL_ = (trans @ XYZ_f.T).T
        RGBL_ = RGBL_.reshape(xyz_.shape)

        # normalize to the highest value
        if normalize:
            nmax = np.nanmax(RGBL_)
            if nmax:
                RGBL_ /= nmax

        return RGBL_

    def _triangle_intersect(r, g, b, w, x, y):

        # sRGB primaries and whitepoint D65 coordinates
        rx, ry, gx, gy, bx, by, wx, wy = *r, *g, *b, *w

        # angles from primaries to whitepoint
        phir = np.arctan2(ry-wy, rx-wx)
        phig = np.arctan2(gy-wy, gx-wx)
        phib = np.arctan2(by-wy, bx-wx) + 2*np.pi  # so range is [0, 2*pi]

        # conditions for this to algorithm to work:
        # whitepoint inside gamut (not on triangle edge)
        # phir > 0, phig < pi, phib > pi and phir < phig < phib
        # rx != bx, gx != bx, rx != gx
        assert np.all(phir > 0)
        assert np.all(phig < np.pi)
        assert np.all(phib > np.pi)
        assert np.all((phir < phig) & (phir  < phib))
        assert rx != bx
        assert gx != bx
        assert rx != gx
        
        phi = ne.evaluate("arctan2(y-wy, x-wx)")
        phi[phi < 0] += 2*np.pi  # so range is [0, 2*pi]

        # slope towards whitepoint and between primaries
        aw = ne.evaluate("(wy-y)/(wx-x)")
        aw[x == wx] = 1e6 # finite slope for x == wx
        abg = (gy-by)/(gx-bx)
        abr = (ry-by)/(rx-bx)
        agr = (ry-gy)/(rx-gx)

        # whitepoint line: line going from (x, y) to whitepoint

        # in the following cases no division by zero can occur,
        # since the whitepoint line and the triangle sides are never parallel
        # (which would for example mean abr = awbr), or if they are, they do so in another intersection case
        # (e.g. the whitepoint line being parallel to the br side occurs only for is_gr)

        # blue-green line intersections
        is_bg = (phi <= phib) & (phi > phig)
        xbg, ybg, awbg = x[is_bg], y[is_bg], aw[is_bg]
        x[is_bg] = t = ne.evaluate("(ybg - by - xbg*awbg + bx*abg) / (abg-awbg)")
        y[is_bg] = ne.evaluate("by + (t - bx)*abg")

        # green-red line intersections
        is_gr = (phi < phig) & (phi > phir)
        xgr, ygr, awgr = x[is_gr], y[is_gr], aw[is_gr]
        x[is_gr] = t = ne.evaluate("(ygr - gy - xgr*awgr + gx*agr) / (agr-awgr)")
        y[is_gr] = ne.evaluate("gy + (t - gx)*agr")

        # blue-red line intersections
        is_br = ~(is_bg | is_gr)
        xbr, ybr, awbr = x[is_br], y[is_br], aw[is_br]
        x[is_br] = t = ne.evaluate("(ybr - by - xbr*awbr + bx*abr) / (abr-awbr)")
        y[is_br] = ne.evaluate("by + (t - bx)*abr")

    # see https://snapshot.canon-asia.com/reg/article/eng/
    # introduction-to-fine-art-printing-part-3-colour-profiles-and-rendering-intents
    # for rendering intents (RI)

    XYZ = xyz.copy()
    RGBL = _to_srgb(XYZ)

    if rendering_intent == "Ignore":
        return RGBL

    # colors outside the gamut
    inv = np.any(RGBL < 0, axis=2)

    if not np.any(inv):
        return RGBL

    if rendering_intent == "Absolute":
        # the following part implements saturation clipping
        # hue and lightness stay untouched, but chroma/saturation is reduced so the color fits inside the sRGB gamut
        # however, in human vision saturation and lightness are not completely independent,
        # see "Helmholtz–Kohlrausch effect". Therefore the perceived lightness still changes slightly.

        xyY = xyz_to_xyY(np.array([XYZ[inv]]))
        x, y, Y = xyY[:, :, 0], xyY[:, :, 1], xyY[:, :, 2]

        # sRGB primaries and whitepoint D65 coordinates
        r, g, b, w = SRGB_R_XY, SRGB_G_XY, SRGB_B_XY, SRGB_W_XY

        _triangle_intersect(r, g, b, w, x, y)

        # rescale x, y, z so the color has the same Y as original color
        k = Y/y
        XYZ[inv, 0] = k*x
        XYZ[inv, 2] = ne.evaluate("k*(1-x-y)")

    if rendering_intent == "Perceptual":
        Luv = xyz_to_luv(XYZ)

        # convert to CIE1976 UCS diagram coordinates
        u_v_L = luv_to_u_v_l(Luv)
        u_, v_ = u_v_L[:, :, 0], u_v_L[:, :, 1]

        # sRGB primaries and D65 uv coordinates
        r, g, b, w = SRGB_R_UV, SRGB_G_UV, SRGB_B_UV, SRGB_W_UV
        un, vn = w

        # squared saturation
        # using the squared saturation saves sqrt calculation
        # we can use the squared saturation since the ratio is minimal at the same point as the saturation ratio
        s0_sq = ne.evaluate("(u_-un)**2 + (v_-vn)**2")

        _triangle_intersect(r, g, b, w, u_, v_)

        # squared saturation after clipping
        s1_sq = ne.evaluate("(u_-un)**2 + (v_-vn)**2")

        # saturation ratio is minimal when squared saturation ratio is minimal
        mask = s0_sq > 0
        s_sq = np.min(s1_sq[mask]/s0_sq[mask])

        Luv[:, :, 1:] *= np.sqrt(s_sq)
        XYZ = luv_to_xyz(Luv)

    # convert another time
    RGB = _to_srgb(XYZ)

    return RGB


def srgb_linear_to_srgb(rgbl: np.ndarray, clip: bool = True) -> np.ndarray:
    """
    Conversion linear RGB to sRGB. sRGBLinear values need to be inside [0, 1]

    :param clip:
    :param rgbl: linear RGB values (numpy 1D, 2D or 3D array)
    :return: sRGB image (same shape as input)
    """
    # return RGBL
    RGB = rgbl.copy()

    # clip values
    if clip:
        np.clip(RGB, 0, 1, out=RGB)

    # gamma correction. RGB -> sRGB
    # Source: https://en.wikipedia.org/wiki/SRGB#Specification_of_the_transformation
    a = 0.055
    below = np.abs(RGB) <= 0.0031308
    RGB[below] *= 12.92
    RGB[~below] = np.sign(RGB[~below])*((1 + a) * np.abs(RGB[~below]) ** (1 / 2.4) - a)

    return RGB


def xyz_to_srgb(xyz: np.ndarray, normalize: bool = True, rendering_intent: str = "Absolute") -> np.ndarray:
    """
    Conversion XYZ to sRGB

    :param xyz: XYZ image (numpy 3D array, XYZ channels in third dimension)
    :param normalize: if values are normalized before conversion (bool)
    :param rendering_intent:
    :return: sRGB image (numpy 3D array)
    """
    # XYZ -> sRGB is just XYZ -> RGBLinear -> sRGB
    RGBL = xyz_to_srgb_linear(xyz, normalize=True, rendering_intent=rendering_intent)
    RGB = srgb_linear_to_srgb(RGBL, normalize)
    return RGB


def xyz_to_luv(xyz: np.ndarray) -> np.ndarray:
    """

    :param xyz:
    :return:
    """

    # all XYZ values need to be below that of D65 reference white,
    # we therefore need to normalize
    Xn, Yn, Zn = WP_D65_XYZ
    _, un, vn = WP_D65_LUV

    # exclude Y = 0 otherwise divisions by zero occur
    mask = xyz[:, :, 1] > 0  # Y > 0 for 3D array
    X, Y, Z = xyz[mask, 0], xyz[mask, 1], xyz[mask, 2]

    if not X.shape[0]:
        return np.zeros_like(xyz)

    XYZm = np.array([np.nanmax(X), np.nanmax(Y), np.nanmax(Z)])
    nmax = max(XYZm/WP_D65_XYZ)
    # we only need to normalize t using nmax (see definition of t below), since u and v consist of XYZ ratios

    # conversion using
    # http://www.brucelindbloom.com/Eqn_XYZ_to_Luv.html
    # with the "actual CIE standard" constants

    Luv = np.zeros_like(xyz)

    k = 903.3
    e = 0.008856
    t = 1/nmax/Yn * Y

    mask2 = t > e  # t > e for L > 0
    mask3 = misc.part_mask(mask, mask2)  # Y > 0 and t > e
    mask4 = misc.part_mask(mask, ~mask2)  # Y > 0 and t <= e

    tm, tn = t[mask2], t[~mask2]
    Luv[mask3, 0] = ne.evaluate("116*tm**(1/3) - 16")
    Luv[mask4, 0] = k * tn

    D = ne.evaluate("1/(X + 15*Y + 3*Z)")
    u = ne.evaluate("4*X*D")
    v = ne.evaluate("9*Y*D")

    L13 = 13*Luv[mask, 0]
    Luv[mask, 1] = ne.evaluate("L13*(u-un)")
    Luv[mask, 2] = ne.evaluate("L13*(v-vn)")

    return Luv


def luv_to_xyz(luv: np.ndarray) -> np.ndarray:
    """

    :param luv:
    :return:
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

    :param luv:
    :return:
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


def get_luv_saturation(luv: np.ndarray) -> np.ndarray:
    """
    Get Chroma from Luv. Calculation using https://en.wikipedia.org/wiki/Colorfulness#Saturation

    :param luv: Luv Image, np.ndarray with shape (Ny, Nx, 3)
    :return: Saturation Image, np.ndarray with shape (Ny, Nx)
    """
    C = get_luv_chroma(luv)

    Sat = np.zeros_like(C)
    mask = luv[:, :, 0] > 0
    Sat[mask] = C[mask] / luv[mask, 0]

    return Sat


def get_luv_chroma(luv: np.ndarray) -> np.ndarray:
    """
    Get Chroma from Luv. Calculation using https://en.wikipedia.org/wiki/Colorfulness#Chroma

    :param luv: Luv Image, np.ndarray with shape (Ny, Nx, 3)
    :return: Chroma Image, np.ndarray with shape (Ny, Nx)
    """
    u, v = luv[:, :, 1], luv[:, :, 2]
    return ne.evaluate("sqrt(u**2 + v**2)")


def get_luv_hue(luv: np.ndarray) -> np.ndarray:
    """
    Get Hue from Luv. Calculation using https://en.wikipedia.org/wiki/Colorfulness#Chroma

    :param luv: Luv Image, np.ndarray with shape (Ny, Nx, 3)
    :return: Hue Image, np.ndarray with shape (Ny, Nx)
    """
    pi = np.pi
    u, v = luv[:, :, 1], luv[:, :, 2]
    hue = ne.evaluate("arctan2(v, u)/pi*180")
    hue[hue < 0] += 360
    return hue


# spectra that lie at the sRGB primaries position in the xy-CIE Diagram
# with the same Y and xy coordinates
# the formulas with rs, gs, bs = 1 have same xy coordinates and spectra maxima at 1
# the formulas with the given values rs, gs, bs have the same Y ratios as the desired primaries,
# but different spectra maxima
# the formulas and coefficients were determined using optimization algorithms
########################################################################################################################


def srgb_r_primary(wl: np.ndarray) -> np.ndarray:
    """
    Possible sRGB r primary curve.

    :param wl: wavelength vector, 1D numpy array
    :return: curve values, 1D numpy array
    """
    rs = 0.951190393
    r = 75.1660756583 * rs * (gauss(wl, 639.854491, 30.0) + 0.0500907584 * gauss(wl, 418.905848, 80.6220465))
    return r


def srgb_g_primary(wl: np.ndarray) -> np.ndarray:
    """
    Possible sRGB g primary curve.

    :param wl: wavelength vector, 1D numpy array
    :return: curve values, 1D numpy array
    """
    gs = 1
    g = 83.4999222966 * gs * gauss(wl, 539.13108974, 33.31164968)
    return g


def srgb_b_primary(wl: np.ndarray) -> np.ndarray:
    """
    Possible sRGB b primary curve.

    :param wl: wavelength vector, 1D numpy array
    :return: curve values, 1D numpy array
    """
    bs = 1.1583866011
    b = 47.7997918851 * bs * (gauss(wl, 454.494831, 20.1804518) + 0.199993596 * gauss(wl, 457.757081, 69.1909777))
    return b


# since we select from the primaries as pdf, the area is the overall probability
# we need to rescale the sRGB Linear channels with the area probabilities of the primary curves
# g is kept constant, factors for the standard wavelength range are given below
_SRGB_R_PRIMARY_POWER_FACTOR = 0.885651229244
_SRGB_G_PRIMARY_POWER_FACTOR = 1.000000000000
_SRGB_B_PRIMARY_POWER_FACTOR = 0.778363814295

########################################################################################################################


def random_wavelengths_from_srgb(rgb: np.ndarray) -> np.ndarray:
    """
    Choose random wavelengths from RGB colors.

    :param rgb: RGB values (numpy 2D array, RGB channels in second dimension)
    :return: random wavelengths for every color (numpy 1D array)
    """
    # since our primaries have the same xyY values as the sRGB primaries, we can use the RGB intensities for mixing.
    # For the RGB intensities we convert the gamma corrected sRGB values to RGBLinear
    RGBL = srgb_to_srgb_linear(rgb)

    # wavelengths to work with
    wl = wavelengths(10000)

    # rescale channel values by probability (=spectrum power factor)
    RGBL[:, 0] *= _SRGB_R_PRIMARY_POWER_FACTOR
    # RGBL[:, 1] *= 1
    RGBL[:, 2] *= _SRGB_B_PRIMARY_POWER_FACTOR

    # in this part we select the r, g or b spectrum depending on the mixing ratios from RGBL
    rgb_sum = np.cumsum(RGBL, axis=-1)
    rgb_sum /= rgb_sum[:, -1, np.newaxis]

    # chose x, y or z depending on in which range rgb_choice fell
    rgb_choice = misc.uniform(0, 1, rgb.shape[0])
    make_r = rgb_choice < rgb_sum[:, 0]
    make_b = rgb_choice > rgb_sum[:, 1]
    make_g = ~make_r & ~make_b

    # select a wavelength from the chosen primaries
    wl_out = np.zeros(rgb.shape[0], dtype=np.float64)
    wl_out[make_r] = misc.random_from_distribution(wl, srgb_r_primary(wl), np.count_nonzero(make_r))
    wl_out[make_g] = misc.random_from_distribution(wl, srgb_g_primary(wl), np.count_nonzero(make_g))
    wl_out[make_b] = misc.random_from_distribution(wl, srgb_b_primary(wl), np.count_nonzero(make_b))

    return wl_out


def power_from_srgb(rgb: np.ndarray) -> np.ndarray:
    """
    relative power for each pixel

    :param rgb:
    :return:
    """
    RGBL = srgb_to_srgb_linear(rgb)  # physical brightness is proportional to RGBLinear signal
    P = _SRGB_R_PRIMARY_POWER_FACTOR * RGBL[:, :, 0] + _SRGB_G_PRIMARY_POWER_FACTOR * RGBL[:, :, 1] \
        + _SRGB_B_PRIMARY_POWER_FACTOR * RGBL[:, :, 2]
    return P

########################################################################################################################

def spectral_colormap(N:    int,
                      wl0:  float = WL_MIN,
                      wl1:  float = WL_MAX) \
        -> np.ndarray:
    """
    Get a spectral colormap with N steps

    :param wl0:
    :param wl1:
    :param N: number of steps (int)
    :return: sRGBA array (numpy 2D array, shape (N, 4))

    >>> spectral_colormap(3, 400, 600)
    array([[5.13881984e+01, 1.52072554e+01, 9.39147181e+01, 2.55000000e+02],
           [1.14346816e-04, 2.53016637e+02, 2.13677273e+02, 2.55000000e+02],
           [2.46742969e+02, 1.10366813e+02, 8.12703774e+01, 2.55000000e+02]])
    """
    # wavelengths to XYZ color
    wl = np.linspace(wl0, wl1, N, dtype=np.float32)  # wavelength vector
    XYZ = np.column_stack((x_tristimulus(wl), y_tristimulus(wl), z_tristimulus(wl)))

    # we want a colorful spectrum with smooth gradients like in reality
    # unfortunately this isn't really possible with sRGB
    # so make a compromise by mixing rendering intents Absolute (=colorful) with Perceptual (=smooth gradients)

    # absolure RI
    RGBa = xyz_to_srgb_linear(np.array([XYZ]), rendering_intent="Absolute")[0]
    nzero = np.any(RGBa != 0, axis=1)
    RGBa[nzero] /= np.max(RGBa[nzero], axis=1)[:, np.newaxis]  # normalize brightness
    
    # perceptual RI
    RGBp = xyz_to_srgb_linear(np.array([XYZ]), rendering_intent="Perceptual")[0]
    nzero = np.any(RGBp != 0, axis=1)
    RGBp[nzero] /= np.max(RGBp[nzero], axis=1)[:, np.newaxis]  # normalize brightness
   
    # mix those two
    RGB = 0.7*RGBa + 0.3*RGBp
    # RGB = RGBa

    # some smoothed rectangle function for brightness fall-off for low and large wavelengths
    # this is for visualization only, there is no physical or perceptual truth behind this
    RGB *= (0.05 + 0.95 / 4 * (1 - np.tanh((wl - 650) / 40))*(1 + np.tanh((wl - 440) / 30)))[:, np.newaxis]

    # convert to sRGB
    RGB = srgb_linear_to_srgb(RGB[:, :3])

    # add alpha channel and rescale to range [0, 255]
    return 255*np.column_stack((RGB, np.ones_like(wl)))
