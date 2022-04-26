"""
Color conversion and processing functions

"""

import numpy as np
import colorio
import optrace.tracer.Misc as misc
from typing import Callable
from optrace.tracer.Misc import timer as timer

WL_MIN: float = 380.
"""lower bound of wavelength range in nm
the wavelength range. Needs to be inside [380, 780] for the Tristimulus and Illuminant functions to work
Note that shrinking that range may lead to color deviations"""

WL_MAX: float = 780.
"""upper bound of wavelength range in nm"""

_WP_D65_XYZ = [0.950489, 1.00000, 1.08840]
_WP_D65_LUV = [100, 0.1978398248, 0.4683363029]

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

def outside_sRGB(XYZ):

    return np.any(XYZ_to_sRGBLinear(XYZ) < 0, axis=2)

# @timer
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

        XYZ_f = XYZ.flatten().reshape((XYZ.shape[0] * XYZ.shape[1], 3))

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
        rx, ry = r
        gx, gy = g
        bx, by = b
        wx, wy = w

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

        # coordinates in CIE xyY 
        XYZs = np.sum(XYZ[inv], axis=1)
        x = XYZ[inv, 0] / XYZs
        y = XYZ[inv, 1] / XYZs
        Y = XYZ[inv, 1]

        # sRGB primaries and whitepoint D65 coordinates
        r = [0.64, 0.33]  # red
        g = [0.30, 0.60]  # green
        b = [0.15, 0.06]  # blue
        w = [0.3127, 0.3290]  # whitepoint

        _triangle_intersect(r, g, b, w, x, y)

        # rescale x, y, z so the color has the same Y as original color
        k = Y/y
        XYZ[inv, 0] = k*x
        XYZ[inv, 2] = misc.calc("k*(1-x-y)")
  
    if RI == "Perceptual":
        Luv = XYZ_to_Luv(XYZ)

        # invalid colors with Lightness above 0
        mi, _ = misc.partMask(inv, Luv[inv, 0] > 0)

        Lm, um, vm = Luv[mi, 0], Luv[mi, 1], Luv[mi, 2]
        u_ = misc.calc("1/13 * um / Lm")
        v_ = misc.calc("1/13 * vm / Lm")

        # squared saturation
        # using the squared saturation saves sqrt calculation
        # we can use the squared saturation since the ratio is minimal at the same point as the saturation ratio
        s0_sq = misc.calc("u_**2 + v_**2")

        # sRGB primaries and D65 uv coordinates
        r = [0.4507042254, 0.5228873239]  # red
        g = [0.125, 0.5625]  # green
        b = [0.1754385965, 0.1578947368]  # blue
        w = _WP_D65_LUV[1:]

        un, vn = w
        u_ += un
        v_ += vn

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


def sRGBLinear_to_sRGB(RGBL: np.ndarray, normalize: bool = True) -> np.ndarray:
    """
    Conversion linear RGB to sRGB. sRGBLinear values need to be inside [0, 1]

    :param RGBL: linear RGB values (numpy 1D, 2D or 3D array)
    :param normalize: if RGB values are normalized before conversion (bool)
    :return: sRGB image (same shape as input)
    """
    # return RGBL 
    RGB = RGBL.copy()

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
    RGBL = XYZ_to_sRGBLinear(XYZ, normalize=True)
    RGB = sRGBLinear_to_sRGB(RGBL, normalize)
    return RGB

# @timer
def XYZ_to_Luv(XYZ) -> np.ndarray:
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
    mask3, _ = misc.partMask(mask, mask2)  # Y > 0 and t > e
    mask4, _ = misc.partMask(mask, ~mask2) # Y > 0 and not t > e

    Luv[mask3, 0] = misc.calc("116*tm**(1/3) - 16", tm=t[mask2])
    Luv[mask4, 0] = k * t[~mask2]

    D = misc.calc("1/(X + 15*Y + 3*Z)")
    u = misc.calc("4*X*D")
    v = misc.calc("9*Y*D")

    L13 = 13*Luv[mask, 0]
    Luv[mask, 1] = misc.calc("L13*(u-un)")
    Luv[mask, 2] = misc.calc("L13*(v-vn)")

    return Luv

def Luv_to_XYZ(Luv) -> np.ndarray:

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
    mask3, _  = misc.partMask(mask, mask2)
    mask4, _ = misc.partMask(mask, ~mask2)

    XYZ[mask3, 1] = misc.calc("((Lm+16)/116)**3", Lm=L_[mask2])
    XYZ[mask4, 1] = 1/k * L_[~mask2]

    Y = XYZ[mask, 1]
    L13 = 13 * Luv[mask, 0]

    XYZ[mask, 0] = X = misc.calc("9/4*Y * (u_ + L13*un) / (v_ + L13*vn)")
    XYZ[mask, 2] = misc.calc("3*Y * (L13/(v_+L13*vn) - 5/3) - X/3")
    
    return XYZ


def getLuvSaturation(Luv) -> np.ndarray:
    # formula from https://en.wikipedia.org/wiki/Colorfulness#Saturation
    C = getLuvChroma(Luv)

    Sat = np.zeros_like(C)
    mask = Luv[:, :, 0] > 0
    Sat[mask] = C[mask] / Luv[mask, 0]

    return Sat


def getLuvChroma(Luv) -> np.ndarray:
    # see https://en.wikipedia.org/wiki/Colorfulness#Chroma
    u, v = Luv[:, :, 1], Luv[:, :, 2]
    return misc.calc("sqrt(u**2 + v**2)") # geom. mean of a and b


def getLuvHue(Luv) -> np.ndarray:
    # see https://en.wikipedia.org/wiki/Colorfulness#Chroma
    u, v = Luv[:, :, 1], Luv[:, :, 2]
    hue = misc.calc("arctan2(v, u)/pi*180")
    hue[hue < 0] += 360
    return hue


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


# TODO doctest
def spectralCM(N: int, wl0: float = WL_MIN, wl1: float = WL_MAX) -> np.ndarray:
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
    RGB = XYZ_to_sRGBLinear(np.array([XYZ]), RI="Absolute")[0] 
    RGB /= np.max(RGB, axis=1)[:, np.newaxis] # normalize brightness

    # some smoothed rectangle function for brightness fall-off for low and large wavelengths
    RGB *= (0.05+0.95/4*(1-np.tanh((wl-650)/50))*(1+np.tanh((wl-450)/30)))[:, np.newaxis]

    # convert to sRGB
    RGB = sRGBLinear_to_sRGB(RGB[:, :3])

    # add alpha channel and rescale to range [0, 255]
    return 255*np.column_stack((RGB, np.ones_like(wl)))

