
import numexpr as ne  # faster calculations
import numpy as np  # matrix calculations

from .. import misc
from . import tools
from .xyz import xyz_to_xyY, WP_D65_XY
from .luv import xyz_to_luv, luv_to_xyz, luv_to_u_v_l, SRGB_R_UV, SRGB_G_UV, SRGB_B_UV, WP_D65_UV
from .observers import x_observer, y_observer, z_observer

from ...global_options import global_options as go


SRGB_RENDERING_INTENTS: list[str, str, str] = ["Ignore", "Absolute", "Perceptual"]
"""Rendering intents for XYZ to sRGB conversion"""

# whitepoints and primary coordinates in xy chromaticity diagram
SRGB_R_XY: list[float, float] = [0.64, 0.33]  #: sRGB red primary in xy coordinates
SRGB_G_XY: list[float, float] = [0.30, 0.60]  #: sRGB green primary in xy coordinates
SRGB_B_XY: list[float, float] = [0.15, 0.06]  #: sRGB blue primary in xy coordinates


def srgb_to_srgb_linear(rgb: np.ndarray) -> np.ndarray:
    """
    Conversion from sRGB to linear RGB values. sRGB values should be inside range [0, 1],
    however this is not checked or enforced.
    For negative values -a  srgb_to_srgb_linear(-a) is the same as -srgb_to_srgb_linear(a)

    :param rgb: RGB values (numpy 1D, 2D or 3D array)
    :return: linear RGB values, array with same shape as input
    """

    RGB = rgb.copy()

    # remove gamma correction (sRGB -> RGBLinear)
    # Source: https://en.wikipedia.org/wiki/SRGB#Specification_of_the_transformation
    a = 0.055
    below = np.abs(RGB) <= 0.04045
    RGB[below] *= 1 / 12.92
    RGBnb = RGB[~below]
    RGB[~below] = np.sign(RGBnb)*ne.evaluate("((abs(RGBnb) + a) / (1 + a)) ** 2.4")

    return RGB


def srgb_linear_to_xyz(rgbl: np.ndarray) -> np.ndarray:
    """
    Conversion from linear sRGB values to XYZ

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
    Conversion from sRGB to XYZ.

    :param rgb: sRGB image (numpy 3D array, shape (Ny, Nx, 3) )
    :return: XYZ image (shape (Ny, Nx, 3))
    """
    RGBL = srgb_to_srgb_linear(rgb)
    XYZ = srgb_linear_to_xyz(RGBL)

    return XYZ


def outside_srgb_gamut(xyz: np.ndarray) -> np.ndarray:
    """
    Checks if the XYZ colors produce valid colors inside the sRGB gamut

    :param xyz: XYZ values, shape (Ny, Nx, 3)
    :return: boolean 2D image, shape (Ny, Nx)
    """
    # small negative tolerance because of numeric precision
    return np.any(xyz_to_srgb_linear(xyz, normalize=True, rendering_intent="Ignore") < -1e-6, axis=2)


def _to_srgb(xyz_: np.ndarray, normalize: bool) -> np.ndarray:
    """
    Helper function for conversion of XYZ to sRGB linear

    :param xyz_: XYZ array (shape (Ny, Nx, 3))
    :param normalize: if output values should be normalized by the highest value
    :return: sRGB Linear array (shape (Ny, Nx, 3))
    """
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
    """
    Helper function.
    Projects chromaticity values x,y towards the whitepoint w on the triangular gamut edge defined by r,g,b.
    Values already inside the gamut are unchanged.
    x, y are changed inplace (no return value)

    :param r: xy red coordinates (tuple of two floats)
    :param g: xy green coordinates (tuple of two floats)
    :param b: xy blue coordinates (tuple of two floats)
    :param b: xy whitepoint coordinates (tuple of two floats)
    :param x: x chromaticity values input vector
    :param y: y chromaticity values input vector
    """
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
    aw[x == wx] = 1e10 # finite slope for x == wx
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
    is_gr = (phi <= phig) & (phi > phir)
    xgr, ygr, awgr = x[is_gr], y[is_gr], aw[is_gr]
    x[is_gr] = t = ne.evaluate("(ygr - gy - xgr*awgr + gx*agr) / (agr-awgr)")
    y[is_gr] = ne.evaluate("gy + (t - gx)*agr")

    # blue-red line intersections
    is_br = ~(is_bg | is_gr)
    xbr, ybr, awbr = x[is_br], y[is_br], aw[is_br]
    x[is_br] = t = ne.evaluate("(ybr - by - xbr*awbr + bx*abr) / (abr-awbr)")
    y[is_br] = ne.evaluate("by + (t - bx)*abr")


def _get_chroma_scale(Luv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate chroma scaling factors for each pixel so it fits into the sRGB gamut.

    :param Luv: CIELUV image values (shape (Ny, Nx, 3))
    :return: array of valid colors (human vision), array of chroma scaling factors
    """

    # convert to CIE1976 UCS diagram coordinates
    u_v_L = luv_to_u_v_l(Luv)
    u_, v_ = u_v_L[:, :, 0], u_v_L[:, :, 1]

    # sRGB primaries and D65 uv coordinates
    r, g, b, w = SRGB_R_UV, SRGB_G_UV, SRGB_B_UV, WP_D65_UV
    un, vn = w

    # due to numerical problems some colors could be outside of the gamut
    # exclude these for the calculation of the chroma scaling
    # note that these linear equations don't follow the gamut edge precisely
    l1 = v_ > (0.5065 - 0.013)/(0.6235-0.255)*(u_-0.2555) + 0.01373
    l2 = v_ < (0.5065 - 0.6)/(0.6235-0.0)*u_ + 0.6
    l3 = u_ > 0
    l4 = v_ > (0.013-0.28)/(0.255-0)*u_ + 0.28
    l5 = v_ > (0.0-0.48)/(0.18-0)*u_ + 0.48
    in_gamut = l1 & l2 & l3 & l4 & l5

    # only invalid colors
    if not np.any(in_gamut):
        return in_gamut, np.ones_like(u_)

    else:
        # squared chroma
        # using the squared saturation saves sqrt calculation
        # we can use the squared saturation since the ratio is minimal at the same point as the saturation ratio
        cr0_sq = ne.evaluate("(u_-un)**2 + (v_-vn)**2")

        _triangle_intersect(r, g, b, w, u_, v_)

        # squared chroma after clipping
        cr1_sq = ne.evaluate("(u_-un)**2 + (v_-vn)**2")

        # chroma scaling factor
        cr_fact2 = cr1_sq/(cr0_sq+1e-9)

        return in_gamut, cr_fact2


def get_chroma_scale(Luv: np.ndarray, L_th = 0.0):
    """
    Calculate the chroma scaling factor for xyz_to_srgb_linear with mode "Perceptual"
    Ignore values below L_th*np.max(L) with L being the lightness.
    Impossible colors are also ignored.

    This functions determines the needed scaling factor so all image colors are inside the sRGB gamut.

    :param Luv: CIELUV image values (shape (Ny, Nx, 3))
    :param L_th: optional lightness threshold as fraction of the peak lightness (range 0 - 1)
    :return: saturation scaling factor (0-1)
    """

    mask_valid, cr_fact2 = _get_chroma_scale(Luv)
    mask_th = Luv[:, :, 0] > L_th*np.max(Luv[:, :, 0])
    return _min_factor_from_chroma_factors(cr_fact2[mask_valid & mask_th])


def _min_factor_from_chroma_factors(cr_fact2):
    """
    Calculate the smallest scaling factor from chroma factors.
    Enfoce range 0.32-1

    :param cr_fact2: list of quadratic chroma factors
    :return: chroma factor
    """
    if not cr_fact2.size:
        return 1.0

    cr_sq = np.min(cr_fact2)
    cr = np.sqrt(cr_sq)
    
    # due to numerical issues cr could be above 1 or below some worst case
    return float(np.clip(cr, 0.32, 1.0))


def xyz_to_srgb_linear(xyz:                 np.ndarray,
                       normalize:           bool = True,
                       rendering_intent:    str = "Absolute",
                       L_th:                float = 0.,
                       chroma_scale:        float = None)\
        -> np.ndarray:
    """
    Conversion XYZ to linear RGB values.

    Ignore: Leave out of gamut colors as is
    Absolute: Clip the saturation of out-of-bound colors, but preserve hue and brightness

    Perceptual: Scale the chroma of all pixels so saturation ratios stay the same.
    Scaling factor is determined so all colors are representable.
    Additional lightness L_th threshold:
    L_th = 0.01 ignores all colors below 1% of the peak brightness for saturation detection.
    Prevents dark, merely visible pixels from scaling down the saturation of the whole image.
    Impossible colors are left untouched by this mode, so clipping the output values is recommended.
    Alternatively the chroma_scale parameter can be provided that scales the chroma down by a fixed amount.
    The colors still outside the gamut after the operation are chroma clipped like in Absolute RI.
    So Perceptual RI with L_th or chroma_scale can be seen as hybrid method, perceptual for colors meeting
    the criteria, Absolute RI for colors not doing so.

    :param xyz: XYZ image (shape (Ny, Nx, 3))
    :param normalize: if image is normalized to highest value before conversion (bool)
    :param rendering_intent: "Absolute", "Perceptual" or "Ignore", see above
    :param L_th: lightness threshold for mode "Perceptual". 
    :param chroma_scale: chroma_scale option for mode "Perceptual"
    :return: linear sRGB image (shape (Ny, Nx, 3))
    """

    # see https://snapshot.canon-asia.com/reg/article/eng/
    # introduction-to-fine-art-printing-part-3-colour-profiles-and-rendering-intents
    # for rendering intents (RI)

    XYZ = xyz.copy()
    RGBL = _to_srgb(XYZ, normalize)

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
        # see "Helmholtz–Kohlrausch effect". Therefore, the perceived lightness still changes slightly.

        xyY = xyz_to_xyY(np.array([XYZ[inv]]))
        x, y, Y = xyY[:, :, 0], xyY[:, :, 1], xyY[:, :, 2]

        # sRGB primaries and whitepoint D65 coordinates
        r, g, b, w = SRGB_R_XY, SRGB_G_XY, SRGB_B_XY, WP_D65_XY

        _triangle_intersect(r, g, b, w, x, y)

        # rescale x, y, z so the color has the same Y as original color
        k = Y/np.where(y > 0, y, np.inf)
        XYZ[inv, 0] = k*x
        XYZ[inv, 2] = ne.evaluate("k*(1-x-y)")

    if rendering_intent == "Perceptual":
        XYZ[XYZ < 0] = 0
        Luv = xyz_to_luv(XYZ, normalize=False)  
        # ^-- don't normalize, since we transform back and forth and expect the same L

        mask_th = Luv[:, :, 0] > L_th*np.max(Luv[:, :, 0])
        mask_valid, cr_fact2 =  _get_chroma_scale(Luv)

        if chroma_scale is None:
            chroma_scale = _min_factor_from_chroma_factors(cr_fact2[mask_th & mask_valid])

        cr_fact = np.sqrt(cr_fact2)
        cr_fact[cr_fact > chroma_scale] = chroma_scale
        Luv[:, :, 1:] *= cr_fact[:, :, np.newaxis]

        XYZ = luv_to_xyz(Luv)

    # convert another time
    RGB = _to_srgb(XYZ, normalize)

    return RGB


def srgb_linear_to_srgb(rgbl: np.ndarray) -> np.ndarray:
    """
    Conversion linear RGB to sRGB. Values should be inside range [0, 1], however this is not enforced.
    For negative values srgb(-a) is the same as -srgb(a)

    :param rgbl: linear RGB values (numpy 1D, 2D or 3D array)
    :return: sRGB image (same shape as input)
    """
    # return RGBL
    RGB = rgbl.copy()

    # gamma correction. RGB -> sRGB
    # Source: https://en.wikipedia.org/wiki/SRGB#Specification_of_the_transformation
    a = 0.055
    below = np.abs(RGB) <= 0.0031308
    RGB[below] *= 12.92
    RGBnb = RGB[~below]
    RGB[~below] = np.sign(RGBnb)*ne.evaluate("((1 + a) * abs(RGBnb) ** (1 / 2.4) - a)")

    return RGB


def xyz_to_srgb(xyz:                np.ndarray,
                normalize:          bool = True,
                clip:               bool = True,
                rendering_intent:   str = "Absolute",
                L_th:               float = 0,
                chroma_scale:       float = None)\
        -> np.ndarray:
    """
    Conversion of XYZ to sRGB Linear to sRGB.

    see function :srgb_linear_to_srgb() for detail on rendering intents.

    :param xyz: XYZ image (numpy 3D array, XYZ channels in third dimension)
    :param normalize: if values are normalized before conversion (bool)
    :param rendering_intent: one of SRGB_RENDERING_INTENTS ("Ignore", "Absolute", "Perceptual")
    :param clip: if sRGB values are clipped before gamma correction
    :param L_th: lightness threshold for mode "Perceptual". 
    :param chroma_scale: chroma_scale option for mode "sRGB (Perceptual RI)"
    :return: sRGB image (numpy 3D array)
    """
    # XYZ -> sRGB is just XYZ -> RGBLinear -> sRGB
    RGBL = xyz_to_srgb_linear(xyz, normalize=normalize, rendering_intent=rendering_intent,
                              L_th=L_th, chroma_scale=chroma_scale)

    if clip:
        RGBL = np.clip(RGBL, 0, 1)

    RGB = srgb_linear_to_srgb(RGBL)
    return RGB


def log_srgb(img: np.ndarray) -> np.ndarray:
    """
    Logarithmically scale sRGB values.
    This is done by rescaling the lightness of the colors (in CIELUV) while keeping the chromaticities the same.

    :param img: input image, shape (Ny, Nx, 3) in sRGB
    :return: logarithmically scaled sRGB values
    """
    if not np.any(img > 0):
        return img.copy()

    # convert to luv
    xyz = srgb_to_xyz(img)
    luv = xyz_to_luv(xyz)
       
    # get lightness bounds (except zero)
    L = luv[:, :, 0]
    L0 = L[L > 0]
    lmax = np.max(L0)
    lmin = np.min(L0)

    if lmin == lmax:
        return img.copy()
   
    # rescale lightness logarithmically
    luv2 = luv.copy()
    luv2[L > 0, 0] = L02 = ne.evaluate("100 - 99.5*log(L0/ lmax) / log(lmin / lmax)")
    
    # rescale uv so chromaticity stays the same
    chroma_scale = L02 / L0
    luv2[L > 0, 1:] *= chroma_scale[:, np.newaxis]

    # convert back to srgb
    xyz = luv_to_xyz(luv2)
    return xyz_to_srgb(xyz)


def gauss(x: np.ndarray, mu: float, sig: float) -> np.ndarray:
    """
    Gauss Function / Normal distribution.
    Normaklized so peak value is 1.

    :param x: 1D value vector
    :param mu: mean value
    :param sig: standard deviation
    :return: function values with same shape as x

    >>> gauss(np.array([0., 0.5, 1.5]), 0.75, 1)
    array([ 0.30113743, 0.38666812, 0.30113743])
    """
    return 1/(sig*np.sqrt(2*np.pi)) * np.exp(-0.5 * (x - mu) ** 2 / sig ** 2)


# spectra that lie at the sRGB primaries position in the xy-CIE Diagram
# with the same Y and xy coordinates
# the formulas with rs, gs, bs = 1 have same xy coordinates and spectra maxima at 1
# the formulas with the given values rs, gs, bs have the same Y ratios as the desired primaries,
# but different spectra maxima
# the formulas and coefficients were determined using optimization algorithms
########################################################################################################################


def srgb_r_primary(wl: np.ndarray) -> np.ndarray:
    """
    Exemplary sRGB r primary curve.

    :param wl: wavelength vector, 1D numpy array
    :return: curve values, 1D numpy array
    """
    r = np.zeros_like(wl)
    m = (wl >= tools._WL_MIN0) & (wl <= tools._WL_MAX0)
    wlm = wl[m]

    rs = 0.951190393
    r[m] = 75.1660756583 * rs * (gauss(wlm, 639.854491, 30.0) + 0.0500907584 * gauss(wlm, 418.905848, 80.6220465))
    return r 


def srgb_g_primary(wl: np.ndarray) -> np.ndarray:
    """
    Exemplary sRGB g primary curve.

    :param wl: wavelength vector, 1D numpy array
    :return: curve values, 1D numpy array
    """
    g = np.zeros_like(wl)
    m = (wl >= tools._WL_MIN0) & (wl <= tools._WL_MAX0)
    wlm = wl[m]
    
    gs = 1
    g[m] = 83.4999222966 * gs * gauss(wlm, 539.13108974, 33.31164968)
    return g


def srgb_b_primary(wl: np.ndarray) -> np.ndarray:
    """
    Exemplary sRGB b primary curve.

    :param wl: wavelength vector, 1D numpy array
    :return: curve values, 1D numpy array
    """
    b = np.zeros_like(wl)
    m = (wl >= tools._WL_MIN0) & (wl <= tools._WL_MAX0)
    wlm = wl[m]
    
    bs = 1.16364585503
    b[m] = 47.99521746361 * bs * (gauss(wlm, 454.833119, 20.1460206) + 0.184484176 * gauss(wlm, 459.658190, 71.0927568))
    return b


# since we select from the primaries as pdf, the area is the overall probability
# we need to rescale the sRGB Linear channels with the area probabilities of the primary curves
# g is kept constant, factors for the standard wavelength range are given below
_SRGB_R_PRIMARY_POWER_FACTOR = 0.885651229244
_SRGB_G_PRIMARY_POWER_FACTOR = 1.000000000000
_SRGB_B_PRIMARY_POWER_FACTOR = 0.775993481741

########################################################################################################################


def random_wavelengths_from_srgb(rgb: np.ndarray) -> np.ndarray:
    """
    Choose random wavelengths from sRGB colors.

    :param rgb: RGB values (numpy 2D array, RGB channels in second dimension)
    :return: random wavelengths for every color (numpy 1D array)
    """
    # since our primaries have the same xyY values as the sRGB primaries, we can use the RGB intensities for mixing.
    # For the RGB intensities we convert the gamma corrected sRGB values to RGBLinear
    RGBL = srgb_to_srgb_linear(rgb)

    if tools._WL_MIN0 < go.wavelength_range[0] or tools._WL_MAX0 > go.wavelength_range[1]:
        raise RuntimeError(f"Wavelength range {go.wavelength_range} does not include range "
                           f"[{tools._WL_MIN0}, {tools._WL_MAX0}] needed for this feature.")

    wl = np.linspace(tools._WL_MIN0, tools._WL_MAX0, 10000)

    # wavelengths to work with
    wl = tools.wavelengths(10000)

    # rescale channel values by probability (=spectrum power factor)
    RGBL[:, 0] *= _SRGB_R_PRIMARY_POWER_FACTOR
    # RGBL[:, 1] *= 1
    RGBL[:, 2] *= _SRGB_B_PRIMARY_POWER_FACTOR

    # in this part we select the r, g or b spectrum depending on the mixing ratios from RGBL
    rgb_sum = np.cumsum(RGBL, axis=-1)
    rgb_sum_last = rgb_sum[:, -1, np.newaxis]
    rgb_sum /= np.where(rgb_sum_last, rgb_sum_last, 1)

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


def _power_from_srgb(rgb: np.ndarray) -> np.ndarray:
    """
    Get a measure of pixel power/probability with the sRGB primaries above.

    :param rgb: sRGB image (2D, with channels in third dimension)
    :return: power/probability image with two dimensions
    """
    RGBL = srgb_to_srgb_linear(rgb)  # physical brightness is proportional to RGBLinear signal
    P = _SRGB_R_PRIMARY_POWER_FACTOR * RGBL[:, :, 0] + _SRGB_G_PRIMARY_POWER_FACTOR * RGBL[:, :, 1] \
        + _SRGB_B_PRIMARY_POWER_FACTOR * RGBL[:, :, 2]
    return P

########################################################################################################################

def spectral_colormap(wl: np.ndarray) \
        -> np.ndarray:
    """
    Get a spectral colormap in sRGB for wavelength values.
    The Hue is rendered physically correct, however the lightness and saturation are set to be visually pleasing

    :param wl: wavelength array. Values must be inside global_options.wavelength_ramge
    :return: sRGBA array (numpy 2D array, shape (N, 4)) with values ranging 0-1
    """
    # wavelengths to XYZ color
    XYZ = np.column_stack((x_observer(wl), y_observer(wl), z_observer(wl)))

    # we want a colorful spectrum with smooth gradients like in reality
    # unfortunately this isn't really possible with sRGB
    # so make a compromise by mixing rendering intents Absolute (=colorful) with Perceptual (=smooth gradients)

    # absolute RI
    RGBa = xyz_to_srgb_linear(np.array([XYZ]), rendering_intent="Absolute")[0]
    nzero = np.any(RGBa != 0, axis=1)
    RGBa[nzero] /= np.max(RGBa[nzero], axis=1)[:, np.newaxis]  # normalize brightness
    
    # perceptual RI
    RGBp = xyz_to_srgb_linear(np.array([XYZ]), rendering_intent="Perceptual")[0]
    nzero = np.any(RGBp != 0, axis=1)
    RGBp[nzero] /= np.max(RGBp[nzero], axis=1)[:, np.newaxis]  # normalize brightness
   
    # mix those two
    RGB = 0.5*RGBa + 0.5*RGBp

    # some smoothed rectangle function for brightness fall-off for low and large wavelengths
    # this is for visualization only, there is no physical or perceptual truth behind this
    RGB *= (1 / 4 * (1 - np.tanh((wl - 650) / 50))*(1 + np.tanh((wl - 440) / 30)))[:, np.newaxis]

    # convert to sRGB
    RGB = srgb_linear_to_srgb(RGB[:, :3])

    # add alpha channel and rescale to range [0, 255]
    return np.column_stack((RGB, np.ones_like(wl)))

