"""
Color conversion and processing functions

"""

import numpy as np  # calculations
import colorio  # standard illuminants and tristimulus curves
import optrace.tracer.Misc as misc  # calculations
from typing import Callable  # Callable type

# DO NOT CHANGE THIS WAVELENGTH RANGE IF YOU WANT COLORS TO WORK CORRECTLY

WL_MIN: float = 380.
"""lower bound of wavelength range in nm."""

WL_MAX: float = 780.
"""upper bound of wavelength range in nm"""

ILLUMINANTS = ["A", "C", "D50", "D55", "D65", "D75", "E", "F2", "F7", "F11"]
"""List of possible values for Color.Illuminants"""

TRISTIMULI = ["X", "Y", "Z"]
"""List of possible values for Color.Tristimulus"""

sRGB_RI = ["Ignore", "Absolute", "Perceptual"]
"""Rendering intents for XYZ to sRGB conversion"""

# Whitepoints in XYZ and Luv
_WP_D65_XYZ = [0.950489, 1.00000, 1.08840]
_WP_D65_LUV = [100, 0.1978398248, 0.4683363029]

# whitepoints and primary coordinates in xy
_sRGB_r_xy = [0.64, 0.33]  # red
_sRGB_g_xy = [0.30, 0.60]  # green
_sRGB_b_xy = [0.15, 0.06]  # blue
_sRGB_w_xy = [0.3127, 0.3290]  # whitepoint D65

# whitepoints and primary coordinates in uv
_sRGB_r_uv = [0.4507042254, 0.5228873239]  # red
_sRGB_g_uv = [0.125, 0.5625]  # green
_sRGB_b_uv = [0.1754385965, 0.1578947368]  # blue
_sRGB_w_uv = _WP_D65_LUV[1:] # D65


def wavelengths(N: int) -> np.ndarray:
    """
    Get wavelength range array with equal spacing and N points.
    :param N: number of values
    :return: wavelength vector in nm, 1D numpy array
    """
    return np.linspace(WL_MIN, WL_MAX, N)


def Blackbody(wl: np.ndarray, T: float = 6504.) -> np.ndarray:
    """
    Get spectral radiance of a planck blackbody curve. Unit is W/(sr m³).

    :param wl: wavelength vector in nm (numpy 1D array)
    :param T: blackbody temperature in Kelvin (float)
    :return: blackbody curve values (numpy 1D array)
    """
    # this function returns the spectral radiance, but since it is proportional to the spectral power, 
    # we can use this function for our wavelength pronabilities for a blackbody

    # physical constants
    c = 299792458
    h = 6.62607015e-34
    k_B = 1.380649e-23

    # wavelength in meters
    wlm = 1e-9*wl 

    # blackbody equation
    # https://en.wikipedia.org/wiki/Planck%27s_law
    return 2 * h * c ** 2 / wlm** 5 / (np.exp(h * c / (wlm * k_B * T)) - 1)


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

    if name not in TRISTIMULI:
        raise ValueError(f"Invalid Tristimulus Type, must be one of {TRISTIMULI}")

    ind = TRISTIMULI.index(name)
    observer = colorio.observers.cie_1931_2()

    return np.interp(wl, observer.lmbda_nm, observer.data[ind], left=0, right=0)


def Illuminant(wl: np.ndarray, name: str) -> np.ndarray:
    """
    Get Illuminant data at specified wavelengths.
    Uses :obj:`colorio.illuminants` for curve data with additional interpolation.

    :param wl: wavelength vector
    :param name: One of Color.ILLUMINANTS
    :return:    

    >>> Illuminant(np.array([500, 600]), "D50")
    array([95.7237, 97.6878])

    >>> Illuminant(np.array([500, 600]), "D65")
    array([109.3545,  90.0062])
    """

    if name not in ILLUMINANTS:
        raise ValueError(f"Invalid Illuminant Type, must be one of {ILLUMINANTS}.")

    illu = eval(f"colorio.illuminants.{name.lower()}()")
    return np.interp(wl, illu.lmbda_nm, illu.data)     


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


def sRGBLinear_to_XYZ(RGBL: np.ndarray) -> np.ndarray:
    """
    Conversion from linear RGB values to XYZ

    :param RGBL: linear RGB image (numpy 3D array, RGB channels in third dimension)
    :return: XYZ image (numpy 3D array)
    """

    # # transform to XYZ
    # source for matrix: http://brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
    # note that the matrix differs in different sources
    trans = np.array([[0.4124564, 0.3575761, 0.1804375],
                      [0.2126729, 0.7151522, 0.0721750],
                      [0.0193339, 0.1191920, 0.9503041]])

    RGBL_f = RGBL.reshape((RGBL.shape[0] * RGBL.shape[1], 3))
    XYZ = (trans @ RGBL_f.T).T

    return XYZ.reshape(RGBL.shape)


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


def outside_sRGB(XYZ: np.ndarray) -> np.ndarray:
    """"""
    return np.any(XYZ_to_sRGBLinear(XYZ, RI="Ignore") < 0, axis=2)


def XYZ_to_xyY(XYZ: np.ndarray) -> np.ndarray:
    """"""
    s = np.sum(XYZ, axis=2)
    mask = s > 0

    xyY = XYZ.copy()

    xyY[mask, :2] /= s[mask, np.newaxis]
    xyY[~mask, :2] = _sRGB_w_xy

    xyY[:, :, 2] = XYZ[:, :, 1]

    return xyY

def XYZ_to_sRGBLinear(XYZ_in: np.ndarray, normalize: bool = True, RI: str="Absolute") -> np.ndarray:
    """
    Conversion XYZ to linear RGB values.

    :param XYZ_in: XYZ image (numpy 3D array, XYZ channels in third dimension)
    :param normalize: if image is normalized to highest value before conversion (bool)
    :return: linear RGB image (numpy 3D array)
    """

    def _to_sRGB(XYZ):

        # it makes no difference if we normalize before or after the matrix multiplication
        # since X, Y and Z gets scaled by the same value and matrix multiplication is a linear operation
        # normalizing after conversion makes it possible to normalize to the highest RGB value,
        # thus the highest Value (Value as in the V in the HSV model)
        # normalizing after guarantees a fixed range in [0, 1] for all possible  inputs

        # source for conversion matrix: http://brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
        trans = np.array([[3.2404542, -1.5371385, -0.4985314],
                          [-0.9692660, 1.8760108, 0.0415560],
                          [0.0556434, -0.2040259, 1.0572252]])

        XYZ_f = XYZ.reshape((XYZ.shape[0] * XYZ.shape[1], 3))

        RGBL = (trans @ XYZ_f.T).T

        RGBL =  RGBL.reshape(XYZ.shape)
     
        # normalize to highest value
        if normalize:
            nmax = np.nanmax(RGBL)
            if nmax:
                RGBL /= nmax

        return RGBL

    def _triangle_intersect(r, g, b, w, x, y):
        # conditions for this to algorithm to work: 
        # whitepoint inside gamut (not on triangle edge) 
        # phir > 0, phig < pi, phib > pi and phir < phig < phib 
        # rx != bx, gx != bx, rx != gx

        # sRGB primaries and whitepoint D65 coordinates
        rx, ry, gx, gy, bx, by, wx, wy = *r, *g, *b, *w

        # angles from primaries to whitepoint
        phir = np.arctan2(ry-wy, rx-wx)
        phig = np.arctan2(gy-wy, gx-wx)
        phib = np.arctan2(by-wy, bx-wx) + 2*np.pi  # so range is [0, 2*pi]

        phi = misc.calc("arctan2(y-wy, x-wx)")
        phi[phi < 0] += 2*np.pi  # so range is [0, 2*pi]

        # slope towards whitepoint and between primaries
        aw = misc.calc("(wy-y)/(wx-x)")
        aw[x == wx] = 1e6 # finite slope for x == wx
        abg = (gy-by)/(gx-bx)
        abr = (ry-by)/(rx-bx)
        agr = (ry-gy)/(rx-gx)

        # whitepoint line: line going from (x, y) to whitepoint

        # in the following cases no division by zero can occur, since the whitepoint line and the triangle sides are never parallel
        # (which would for example mean abr = awbr), or if they are, they do so in another intersection case 
        # (e.g. the whitepoint line being parallel to the br side occurs only for is_gr)

        # blue-green line intersections
        is_bg = (phi <= phib) & (phi > phig)
        xbg, ybg, awbg = x[is_bg], y[is_bg], aw[is_bg]
        x[is_bg] = misc.calc("(ybg - by - xbg*awbg + bx*abg) / (abg-awbg)")
        y[is_bg] = misc.calc("by + (t - bx)*abg", t=x[is_bg])

        # green-red line intersections
        is_gr = (phi < phig) & (phi > phir)
        xgr, ygr, awgr = x[is_gr], y[is_gr], aw[is_gr]
        x[is_gr] = misc.calc("(ygr - gy - xgr*awgr + gx*agr) / (agr-awgr)")
        y[is_gr] = misc.calc("gy + (t - gx)*agr", t=x[is_gr])
        
        # blue-red line intersections
        is_br = ~(is_bg | is_gr)
        xbr, ybr, awbr = x[is_br], y[is_br], aw[is_br]
        x[is_br] = misc.calc("(ybr - by - xbr*awbr + bx*abr) / (abr-awbr)")
        y[is_br] = misc.calc("by + (t - bx)*abr", t=x[is_br])
    
    # see https://snapshot.canon-asia.com/reg/article/eng/introduction-to-fine-art-printing-part-3-colour-profiles-and-rendering-intents
    # for rendering intents (RI)

    XYZ = XYZ_in.copy()
    RGBL = _to_sRGB(XYZ)

    if RI == "Ignore":
        return RGBL

    # colors outside the gamut
    inv = np.any(RGBL < 0, axis=2)
    
    if not np.any(inv):
        return RGBL
    
    if RI == "Absolute":
        # the following part implements saturation clipping
        # hue and lightness stay untouched, but chroma/saturation is reduced so the color fits inside the sRGB gamut
        # however, in human vision saturation and lightness are not completely independent, 
        # see "Helmholtz–Kohlrausch effect". Therefore perceived lightness still changes slightly.

        xyY = XYZ_to_xyY(np.array([XYZ[inv]]))
        x, y, Y = xyY[:, :, 0], xyY[:, :, 1], xyY[:, :, 2]

        # sRGB primaries and whitepoint D65 coordinates
        r, g, b, w = _sRGB_r_xy, _sRGB_g_xy, _sRGB_b_xy, _sRGB_w_xy

        _triangle_intersect(r, g, b, w, x, y)

        # rescale x, y, z so the color has the same Y as original color
        k = Y/y
        XYZ[inv, 0] = k*x
        XYZ[inv, 2] = misc.calc("k*(1-x-y)")
  
    if RI == "Perceptual":
        Luv = XYZ_to_Luv(XYZ)   

        # convert to CIE1976 UCS diagram coordinates
        u_v_L = Luv_to_u_v_L(Luv)
        u_, v_ = u_v_L[:, :, 0], u_v_L[:, :, 1]

        # sRGB primaries and D65 uv coordinates
        r, g, b, w = _sRGB_r_uv, _sRGB_g_uv, _sRGB_b_uv, _sRGB_w_uv
        un, vn = w

        # squared saturation
        # using the squared saturation saves sqrt calculation
        # we can use the squared saturation since the ratio is minimal at the same point as the saturation ratio
        s0_sq = misc.calc("(u_-un)**2 + (v_-vn)**2")

        _triangle_intersect(r, g, b, w, u_, v_)

        # squared saturation after clipping
        s1_sq = misc.calc("(u_-un)**2 + (v_-vn)**2")

        # saturation ratio is minimal when squared saturation ratio is minimal
        mask = s0_sq > 0
        s_sq = np.min(s1_sq[mask]/s0_sq[mask])

        Luv[:, :, 1:] *= np.sqrt(s_sq)
        XYZ = Luv_to_XYZ(Luv)

    # convert another time
    RGB = _to_sRGB(XYZ)

    return RGB


def sRGBLinear_to_sRGB(RGBL: np.ndarray, clip: bool = True) -> np.ndarray:
    """
    Conversion linear RGB to sRGB. sRGBLinear values need to be inside [0, 1]

    :param RGBL: linear RGB values (numpy 1D, 2D or 3D array)
    :param normalize: if RGB values are normalized before conversion (bool)
    :return: sRGB image (same shape as input)
    """
    # return RGBL 
    RGB = RGBL.copy()

    # clip values
    if clip:
        RGB[RGB < 0] = 0
        RGB[RGB > 1] = 1
  
    # gamma correction. RGB -> sRGB
    # Source: https://en.wikipedia.org/wiki/SRGB#Specification_of_the_transformation
    a = 0.055
    below = np.abs(RGB) <= 0.0031308
    RGB[below] *= 12.92
    RGB[~below] = np.sign(RGB[~below])*((1 + a) * np.abs(RGB[~below]) ** (1 / 2.4) - a)

    return RGB


def XYZ_to_sRGB(XYZ: np.ndarray, normalize: bool = True, RI: str="Absolute") -> np.ndarray:
    """
    Conversion XYZ to sRGB

    :param XYZ: XYZ image (numpy 3D array, XYZ channels in third dimension)
    :param normalize: if values are normalized before conversion (bool)
    :return: sRGB image (numpy 3D array)
    """
    # XYZ -> sRGB is just XYZ -> RGBLinear -> sRGB
    RGBL = XYZ_to_sRGBLinear(XYZ, normalize=True, RI=RI)
    RGB = sRGBLinear_to_sRGB(RGBL, normalize)
    return RGB


def XYZ_to_Luv(XYZ: np.ndarray) -> np.ndarray:
    """"""

    # all XYZ values need to be below that of D65 reference white,
    # we therefore need to normalize
    Xn, Yn, Zn = _WP_D65_XYZ
    _, un, vn = _WP_D65_LUV

    # exclude Y = 0 otherwise divisions by zero occur
    mask = XYZ[:, :, 1] > 0  # Y > 0 for 3D array
    X, Y, Z = XYZ[mask, 0], XYZ[mask, 1], XYZ[mask, 2]

    if not X.shape[0]:
        return np.zeros_like(XYZ)

    XYZm = np.array([np.nanmax(X), np.nanmax(Y), np.nanmax(Z)])
    nmax = max(XYZm/_WP_D65_XYZ)
    # we only need to normalize t using nmax (see definition of t below), since u and v consist of XYZ ratios

    # conversion using 
    # http://www.brucelindbloom.com/Eqn_XYZ_to_Luv.html
    # with the "actual CIE standard" constants

    Luv = np.zeros_like(XYZ)

    k = 903.3
    e = 0.008856
    t = 1/nmax/Yn * Y

    mask2 = t > e  # t > e for L > 0
    mask3 = misc.partMask(mask, mask2)  # Y > 0 and t > e
    mask4 = misc.partMask(mask, ~mask2) # Y > 0 and t <= e

    Luv[mask3, 0] = misc.calc("116*tm**(1/3) - 16", tm=t[mask2])
    Luv[mask4, 0] = k * t[~mask2]

    D = misc.calc("1/(X + 15*Y + 3*Z)")
    u = misc.calc("4*X*D")
    v = misc.calc("9*Y*D")

    L13 = 13*Luv[mask, 0]
    Luv[mask, 1] = misc.calc("L13*(u-un)")
    Luv[mask, 2] = misc.calc("L13*(v-vn)")

    return Luv


def Luv_to_XYZ(Luv: np.ndarray) -> np.ndarray:
    """"""

    # calculations are a rewritten from of
    # http://www.brucelindbloom.com/Eqn_Luv_to_XYZ.html
    # with the "actual CIE standard" constants

    _, un, vn = _WP_D65_LUV

    # exclude L == 0, otherwise divisions by zero
    mask = Luv[:, :, 0] > 0
    L_, u_, v_ = Luv[mask, 0], Luv[mask, 1], Luv[mask, 2]

    XYZ = np.zeros_like(Luv)

    k = 903.3
    e = 0.008856
    mask2 = L_ > k*e
    mask3 = misc.partMask(mask, mask2)
    mask4 = misc.partMask(mask, ~mask2)

    XYZ[mask3, 1] = misc.calc("((Lm+16)/116)**3", Lm=L_[mask2])
    XYZ[mask4, 1] = 1/k * L_[~mask2]

    Y = XYZ[mask, 1]
    L13 = 13 * Luv[mask, 0]

    XYZ[mask, 0] = X = misc.calc("9/4*Y * (u_ + L13*un) / (v_ + L13*vn)")
    XYZ[mask, 2] = misc.calc("3*Y * (L13/(v_ + L13*vn) - 5/3) - X/3")
    
    return XYZ


def Luv_to_u_v_L(Luv: np.ndarray) -> np.ndarray:
    """"""
    _, un, vn = _WP_D65_LUV
    
    mi = Luv[:, :, 0] > 0

    u_v_L = np.zeros_like(Luv)

    u_v_L[:, :, 0] = un
    u_v_L[:, :, 1] = vn
    u_v_L[:, :, 2] = Luv[:, :, 0]

    Lm, um, vm = Luv[mi, 0], Luv[mi, 1], Luv[mi, 2]
    u_v_L[mi, 0] += misc.calc("1/13 * um / Lm")
    u_v_L[mi, 1] += misc.calc("1/13 * vm / Lm")
    
    return u_v_L


def getLuvSaturation(Luv: np.ndarray) -> np.ndarray:
    """
    Get Chroma from Luv. Calculation using https://en.wikipedia.org/wiki/Colorfulness#Saturation
    :param Luv: Luv Image, np.ndarray with shape (Ny, Nx, 3)
    :return: Saturation Image, np.ndarray with shape (Ny, Nx)
    """
    """"""
    C = getLuvChroma(Luv)

    Sat = np.zeros_like(C)
    mask = Luv[:, :, 0] > 0
    Sat[mask] = C[mask] / Luv[mask, 0]

    return Sat


def getLuvChroma(Luv: np.ndarray) -> np.ndarray:
    """
    Get Chroma from Luv. Calculation using https://en.wikipedia.org/wiki/Colorfulness#Chroma
    :param Luv: Luv Image, np.ndarray with shape (Ny, Nx, 3)
    :return: Chroma Image, np.ndarray with shape (Ny, Nx)
    """
    u, v = Luv[:, :, 1], Luv[:, :, 2]
    return misc.calc("sqrt(u**2 + v**2)")


def getLuvHue(Luv: np.ndarray) -> np.ndarray:
    """
    Get Hue from Luv. Calculation using https://en.wikipedia.org/wiki/Colorfulness#Chroma
    :param Luv: Luv Image, np.ndarray with shape (Ny, Nx, 3)
    :return: Hue Image, np.ndarray with shape (Ny, Nx)
    """
    u, v = Luv[:, :, 1], Luv[:, :, 2]
    hue = misc.calc("arctan2(v, u)/pi*180")
    hue[hue < 0] += 360
    return hue

# spectra that lie at the sRGB primaries position in the xy-CIE Diagram
# with the same Y and xy coordinates
# the formulas with rs, gs, bs = 1 have same xy coordinates and spectra maxima at 1
# the formulas with the given values rs, gs, bs have the same Y ratios as the desired primaries, 
# but different spectra maxima
# the formulas and coefficients were determined using optimization algorithms
########################################################################################################################

def _sRGB_r_primary(wl: np.ndarray) -> np.ndarray:
    """
    Possible sRGB r primary curve.
    :param wl: wavelength vector, 1D numpy array
    :return: curve values, 1D numpy array
    """
    rs = 0.94785778
    r =  75.59166249 * rs * (Gauss(wl, 639.899361, 30.1649583) + 0.0479726849 * Gauss(wl, 418.343791, 76.0664758))
    return r

def _sRGB_g_primary(wl: np.ndarray) -> np.ndarray:
    """
    Possible sRGB g primary curve.
    :param wl: wavelength vector, 1D numpy array
    :return: curve values, 1D numpy array
    """
    gs = 1
    g = 83.49992300 * gs * Gauss(wl, 539.131090, 33.3116497)
    return g

def _sRGB_b_primary(wl: np.ndarray) -> np.ndarray:
    """
    Possible sRGB b primary curve.
    :param wl: wavelength vector, 1D numpy array
    :return: curve values, 1D numpy array
    """
    bs = 1.15838660
    b =  47.79979189 * bs * (Gauss(wl, 454.494831, 20.1804518) + 0.199993596 * Gauss(wl, 457.757081, 69.1909777))
    return b


# since we select from the primaries as pdf, the area is the overall probability
# we need to rescale the sRGB Linear channels with the area probabilities of the primary curves
# g is kept constant, factors for the standard wavelength range are given below
_sRGB_r_primary_power_factor = 0.88661241
_sRGB_g_primary_power_factor = 1.0
_sRGB_b_primary_power_factor = 0.77836381

########################################################################################################################

def randomWavelengthFromRGB(RGB: np.ndarray) -> np.ndarray:
    """
    Choose random wavelengths from RGB colors.

    :param RGB: RGB values (numpy 2D array, RGB channels in second dimension)
    :return: random wavelengths for every color (numpy 1D array)
    """
    # since our primaries have the same xyY values as the sRGB primaries, we can use the RGB intensities for mixing.
    # For the RGB intensities we convert the gamma corrected sRGB values to RGBLinear
    RGBL = sRGB_to_sRGBLinear(RGB)

    # wavelengths to work with
    wl = wavelengths(10000)

    # rescale channel values by probability (=spectrum power factor)
    RGBL[:, 0] *= _sRGB_r_primary_power_factor
    # RGBL[:, 1] *= 1
    RGBL[:, 2] *= _sRGB_b_primary_power_factor

    # in this part we select the r, g or b spectrum depending on the mixing ratios from RGBL
    rgb_sum = np.cumsum(RGBL, axis=-1)
    rgb_sum /= rgb_sum[:, -1, np.newaxis]

    # chose x, y or z depending on in which range rgb_choice fell
    rgb_choice = np.random.sample(RGB.shape[0])
    make_r = rgb_choice <= rgb_sum[:, 0]
    make_g = (rgb_choice > rgb_sum[:, 0]) & (rgb_choice <= rgb_sum[:, 1])
    make_b = (rgb_choice > rgb_sum[:, 1]) & (rgb_choice <= rgb_sum[:, 2])

    # select a wavelength from the chosen primaries
    wl_out = np.zeros(RGB.shape[0], dtype=np.float32)
    wl_out[make_r] = misc.random_from_distribution(wl, _sRGB_r_primary(wl), np.count_nonzero(make_r))
    wl_out[make_g] = misc.random_from_distribution(wl, _sRGB_g_primary(wl), np.count_nonzero(make_g))
    wl_out[make_b] = misc.random_from_distribution(wl, _sRGB_b_primary(wl), np.count_nonzero(make_b))

    return wl_out

def _PowerFromSRGB(sRGB: np.ndarray) -> np.ndarray:
    """relative power for each pixel"""
    RGB = sRGB_to_sRGBLinear(sRGB)  # physical brightness is proportional to RGBLinear signal
    P = _sRGB_r_primary_power_factor*RGB[:, :, 0] + _sRGB_g_primary_power_factor*RGB[:, :, 1]\
        +  _sRGB_b_primary_power_factor*RGB[:, :, 2]
    return P 

########################################################################################################################

def spectralCM(N: int, wl0: float = WL_MIN, wl1: float = WL_MAX, RI: str="Absolute") -> np.ndarray:
    """
    Get a spectral colormap with N steps

    :param N: number of steps (int)
    :return: sRGBA array (numpy 2D array, shape (N, 4))
    """

    wl = np.linspace(wl0, wl1, N, dtype=np.float32)  # wavelength vector

    XYZ = np.column_stack((Tristimulus(wl, "X"),
                           Tristimulus(wl, "Y"),
                           Tristimulus(wl, "Z")))

    # XYZ to sRGBLinear
    RGB = XYZ_to_sRGBLinear(np.array([XYZ]), RI=RI)[0] 
    RGB /= np.max(RGB, axis=1)[:, np.newaxis] # normalize brightness

    # some smoothed rectangle function for brightness fall-off for low and large wavelengths
    # this is for visualization only, theres is no physical or perceptual truth behind this
    RGB *= (0.05+0.95/4*(1-np.tanh((wl-650)/50))*(1+np.tanh((wl-450)/30)))[:, np.newaxis]

    # convert to sRGB
    RGB = sRGBLinear_to_sRGB(RGB[:, :3])

    # add alpha channel and rescale to range [0, 255]
    return 255*np.column_stack((RGB, np.ones_like(wl)))

