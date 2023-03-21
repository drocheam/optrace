
import numexpr as ne  # faster calculations
import numpy as np  # matrix calculations

from .. import misc
from . import tools
from .xyz import xyz_to_xyY, WP_D65_XY
from .luv import xyz_to_luv, luv_to_xyz, luv_to_u_v_l, SRGB_R_UV, SRGB_G_UV, SRGB_B_UV, WP_D65_UV
from .observers import x_observer, y_observer, z_observer


SRGB_RENDERING_INTENTS: list[str, str, str] = ["Ignore", "Absolute", "Perceptual"]
"""Rendering intents for XYZ to sRGB conversion"""

# whitepoints and primary coordinates in xy
SRGB_R_XY: list[float, float] = [0.64, 0.33]  #: sRGB red primary in xy coordinates
SRGB_G_XY: list[float, float] = [0.30, 0.60]  #: sRGB green primary in xy coordinates
SRGB_B_XY: list[float, float] = [0.15, 0.06]  #: sRGB blue primary in xy coordinates


def srgb_to_srgb_linear(rgb: np.ndarray) -> np.ndarray:
    """
    Conversion from sRGB to linear RGB values. sRGB values should be inside [0, 1], however this is not checked or enforced.
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

    :param rgb: sRGB image (numpy 3D array, RGB channels in third dimension)
    :return: XYZ image (numpy 3D array)
    """
    RGBL = srgb_to_srgb_linear(rgb)
    XYZ = srgb_linear_to_xyz(RGBL)

    return XYZ


def outside_srgb_gamut(xyz: np.ndarray) -> np.ndarray:
    """
    Checks if the XYZ colors produce valid colors inside the sRGB gamut

    :param xyz: XYZ values, 2D image with channels in the third dimension
    :return: boolean 2D image
    """
    return np.any(xyz_to_srgb_linear(xyz, rendering_intent="Ignore") < 0, axis=2)


def xyz_to_srgb_linear(xyz:                 np.ndarray, 
                       normalize:           bool = True, 
                       rendering_intent:    str = "Absolute")\
        -> np.ndarray:
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
        r, g, b, w = SRGB_R_XY, SRGB_G_XY, SRGB_B_XY, WP_D65_XY

        _triangle_intersect(r, g, b, w, x, y)

        # rescale x, y, z so the color has the same Y as original color
        k = Y/np.where(y > 0, y, np.inf)
        XYZ[inv, 0] = k*x
        XYZ[inv, 2] = ne.evaluate("k*(1-x-y)")

    if rendering_intent == "Perceptual":
        Luv = xyz_to_luv(XYZ, normalize=False)  
        # ^-- don't normalize, since we transform back and forth and expect the same L

        # convert to CIE1976 UCS diagram coordinates
        u_v_L = luv_to_u_v_l(Luv)
        u_, v_ = u_v_L[:, :, 0], u_v_L[:, :, 1]

        # sRGB primaries and D65 uv coordinates
        r, g, b, w = SRGB_R_UV, SRGB_G_UV, SRGB_B_UV, WP_D65_UV
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
                rendering_intent:   str = "Absolute")\
        -> np.ndarray:
    """
    Conversion of XYZ to sRGB.

    :param xyz: XYZ image (numpy 3D array, XYZ channels in third dimension)
    :param normalize: if values are normalized before conversion (bool)
    :param rendering_intent: one of SRGB_RENDERING_INTENTS ("Ignore", "Absolute", "Perceptual")
    :param clip: if sRGB values are clipped before gamma correction
    :return: sRGB image (numpy 3D array)
    """
    # XYZ -> sRGB is just XYZ -> RGBLinear -> sRGB
    RGBL = xyz_to_srgb_linear(xyz, normalize=normalize, rendering_intent=rendering_intent)

    if clip:
        RGBL = np.clip(RGBL, 0, 1)

    RGB = srgb_linear_to_srgb(RGBL)
    return RGB


def log_srgb_linear(img: np.ndarray, exp: float = 1) -> np.ndarray:
    """
    Logarithmically scale linear sRGB components and additionally exponentiate by exp

    :param img: input image, shape (Ny, Nx, 3) in linear sRGB
    :param exp: scaling exponent, optional
    :return: logarithmically scaled and exponated linear sRGB values
    """

    # addition, multiplication etc. only work correctly in the linear color space
    # otherwise we would change the color ratios, but we only want the brightness to change
    if np.any(img > 0):
        rgbs = np.sum(img, axis=2)  # assume RGB channel sum as brightness
        nz = rgbs > 0
        rgbsnz = rgbs[nz]
        wmin = np.min(rgbsnz)  # minimum nonzero brightness
        wmax = np.max(rgbsnz)  # minimum brightness

        # constant image or image differences due to numerical errors
        if wmin > (1 - 1e-6)*wmax:
            return img.copy()

        maxrgb = np.max(img, axis=2)  # highest rgb value for each pixel

        # normalize pixel so highest channel value is 1, then rescale logarithmically.
        # Highest value is 1, lowest 0. Exclude all zero channels (maxrgb = 0) for calculation
        mrgb, exp_ = maxrgb[nz], exp
        fact = np.zeros(img.shape[:2])
        fact[nz] = ne.evaluate("1/mrgb * (1 - 0.995*log(rgbsnz/ wmax) / log(wmin / wmax)) ** exp_")

        return img * fact[..., np.newaxis]

    else:
        return img.copy()


def gauss(x: np.ndarray, mu: float, sig: float) -> np.ndarray:
    """
    Normalized Gauss Function

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
    Possible sRGB r primary curve.

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
    Possible sRGB g primary curve.

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
    Possible sRGB b primary curve.

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
    Choose random wavelengths from RGB colors.

    :param rgb: RGB values (numpy 2D array, RGB channels in second dimension)
    :return: random wavelengths for every color (numpy 1D array)
    """
    # since our primaries have the same xyY values as the sRGB primaries, we can use the RGB intensities for mixing.
    # For the RGB intensities we convert the gamma corrected sRGB values to RGBLinear
    RGBL = srgb_to_srgb_linear(rgb)

    if tools._WL_MIN0 < tools.WL_BOUNDS[0] or tools._WL_MAX0 > tools.WL_BOUNDS[1]:
        raise RuntimeError(f"Wavelength range {tools.WL_BOUNDS} does not include range "
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

def spectral_colormap(N:    int,
                      wl0:  float = None,
                      wl1:  float = None) \
        -> np.ndarray:
    """
    Get a spectral colormap with N steps in sRGB.
    The Hue is rendered physically correct, however the lightness and saturation are set to be visually pleasing

    :param wl0: lower wavelength
    :param wl1: upper wavelength
    :param N: number of steps (int)
    :return: sRGBA array (numpy 2D array, shape (N, 4))

    >>> spectral_colormap(3, 400, 600)
    array([[ 3.81626223e+01, 9.30946402e+00, 7.20900561e+01, 2.55000000e+02],
           [ 1.14017400e-04, 2.52697905e+02, 2.13400035e+02, 2.55000000e+02],
           [ 2.41139375e+02, 1.07694238e+02, 7.92225931e+01, 2.55000000e+02]])
    """
    # wavelengths to XYZ color
    wl0 = wl0 if wl0 is not None else tools.WL_BOUNDS[0]
    wl1 = wl1 if wl1 is not None else tools.WL_BOUNDS[1]
    wl = np.linspace(wl0, wl1, N, dtype=np.float32)  # wavelength vector
    XYZ = np.column_stack((x_observer(wl), y_observer(wl), z_observer(wl)))

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

    # some smoothed rectangle function for brightness fall-off for low and large wavelengths
    # this is for visualization only, there is no physical or perceptual truth behind this
    RGB *= (1 / 4 * (1 - np.tanh((wl - 650) / 50))*(1 + np.tanh((wl - 440) / 30)))[:, np.newaxis]

    # convert to sRGB
    RGB = srgb_linear_to_srgb(RGB[:, :3])

    # add alpha channel and rescale to range [0, 255]
    return 255*np.column_stack((RGB, np.ones_like(wl)))
