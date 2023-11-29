
from typing import Callable  # Callable typing hints

import numpy as np  # calculations
import matplotlib.pyplot as plt  # actual plotting
import matplotlib.patheffects as path_effects  # path effects for plot lines

from ..tracer.r_image import RImage
from ..tracer.spectrum import LightSpectrum
from ..tracer import color  # color conversions for chromaticity plots
from ..tracer.misc import PropertyChecker as pc
from .misc_plots import _save_or_show


chromaticity_norms: list[str, str, str] = ["Largest", "Sum"]
"""possible norms for the chromaticity diagrams"""


# when looking at the XYZ to sRGB Matrix, the r channel has by far the highest coefficients
# therefore the red region leads to the highest brightness clipping
# normalizing to the redmost point, the primary, saves all values inside the gamut from brightness clipping
# coefficients are from Color.XYZ_to_sRGBLinear()
# we do this since we can't normalize on the highest value in the chromaticity image,
# since the brightest values come from impossible colors (colors outside human vision) clipped towards the gamut
_red_xyz = np.array([[[*color.SRGB_R_XY, 1 - color.SRGB_R_XY[0] - color.SRGB_R_XY[1]]]])  # red in xyz chromaticities
_CONV_XYZ_NORM = color.xyz_to_srgb_linear(_red_xyz, normalize=False)[0, 0, 0]  # srgb linear red channel primary


def chromaticities_cie_1931(img:                  RImage | LightSpectrum | list[LightSpectrum] = None,
                            rendering_intent:     str = "Ignore",
                            title:                str = "CIE 1931 Chromaticity Diagram",
                            fargs:                dict = None,
                            **kwargs)\
        -> None:
    """
    Draw a CIE 1931 xy chromaticity diagram and mark spectrum/image colors inside of it.

    :param img: RImage, LightSpectrum or a list of LightSpectrum
    :param rendering_intent: rendering_intent for the sRGB conversion
    :param title: title of the plot
    :param fargs: keyword argument dictionary for pyplot.figure() (e.g. figsize)
    :param kwargs: additional plotting parameters, including:
     block: if the plot should block the execution of the program
     norm: brightness norm for the chromaticity diagram, one of "chromaticity_norms"
     path: if provided, the plot is saved at this location instead of displaying a plot. Specify with file ending.
     sargs: option dictionary for pyplot.savefig
    """

    # coordinates of whitepoint and sRGB primaries in chromaticity diagram coordinates (xy)
    r, g, b, w = color.SRGB_R_XY, color.SRGB_G_XY, color.SRGB_B_XY, color.WP_D65_XY

    # XYZ to chromaticity Diagram coordinates (xy in this case)
    def conv(XYZ: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xyY = color.xyz_to_xyY(XYZ)
        return xyY[:, :, 0].flatten(), xyY[:, :, 1].flatten()

    # inverse operation of conv
    def i_conv(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        xg, yg = np.meshgrid(x, y)
        xyY = np.dstack((xg, yg, yg))
        return 1/_CONV_XYZ_NORM * color.xyY_to_xyz(xyY)

    ext = [0, 0.83, 0, 0.9]  # extent of diagram
    _chromaticity_plot(img, conv, i_conv, rendering_intent, r, g, b, w, ext,
                       title, "x", "y", fargs=fargs, **kwargs)


def chromaticities_cie_1976(img:                  RImage | LightSpectrum | list[LightSpectrum] = None,
                            rendering_intent:     str = "Ignore",
                            title:                str = "CIE 1976 UCS Diagram",
                            fargs:                dict = None,
                            **kwargs)\
        -> None:
    """
    Draw a CIE 1976 xy chromaticity diagram and mark spectrum/image colors inside of it.

    :param img: RImage, LightSpectrum or a list of LightSpectrum
    :param rendering_intent: rendering_intent for the sRGB conversion
    :param title: title of the plot
    :param fargs: keyword argument dictionary for pyplot.figure() (e.g. figsize)
    :param kwargs: additional plotting parameters, including:
     block: if the plot should block the execution of the program
     norm: brightness norm for the chromaticity diagram, one of "chromaticity_norms"
     path: if provided, the plot is saved at this location instead of displaying a plot. Specify with file ending.
     sargs: option dictionary for pyplot.savefig
    """

    # coordinates of whitepoint and sRGB primaries in chromaticity diagram coordinates (u'v')
    r, g, b, w = color.SRGB_R_UV, color.SRGB_G_UV, color.SRGB_B_UV, color.WP_D65_UV

    # XYZ to chromaticity diagram coordinates (u'v' in this case)
    def conv(XYZ):
        Luv = color.xyz_to_luv(XYZ)
        u_v_L = color.luv_to_u_v_l(Luv)
        return u_v_L[:, :, 0].flatten(), u_v_L[:, :, 1].flatten()

    # inverse operation of conv
    def i_conv(u_, v_):
        u_g, v_g = np.meshgrid(u_, v_)
        # see https://en.wikipedia.org/wiki/CIELUV
        xg = 9*u_g / (6*u_g - 16*v_g + 12)
        yg = 4*v_g / (6*u_g - 16*v_g + 12)
        zg = 1 - xg - yg
        return 1/_CONV_XYZ_NORM * np.dstack((xg, yg, zg))

    ext = [0, 0.7, 0, 0.7]  # extent of diagram
    _chromaticity_plot(img, conv, i_conv, rendering_intent, r, g, b, w, ext,
                       title, "u'", "v'", fargs=fargs, **kwargs)


def _chromaticity_plot(img:                     RImage | LightSpectrum | list[LightSpectrum],
                       conv:                    Callable,
                       i_conv:                  Callable,
                       rendering_intent:        str,
                       r:                       list,
                       g:                       list,
                       b:                       list,
                       w:                       list,
                       ext:                     list,
                       title:                   str,
                       xl:                      str,
                       yl:                      str,
                       block:                   bool = False,
                       fargs:                   dict = None,
                       norm:                    str = "Sum",
                       path:                    str = None,
                       sargs:                   dict = None)\
        -> None:
    """Lower level plotting function. Don't use directly"""

    pc.check_type("title", title, str)
    pc.check_type("block", block, bool)
    pc.check_if_element("rendering_intent", rendering_intent, color.srgb.SRGB_RENDERING_INTENTS)
    pc.check_if_element("norm", norm, chromaticity_norms)
    
    # RImage -> plot Pixel colors (RI != "Ignore" means sRGB conversion with said rendering intent)
    if isinstance(img, RImage):
        XYZ = img.xyz() if rendering_intent == "Ignore" \
            else color.srgb_to_xyz(img.rgb(rendering_intent=rendering_intent))
        labels = []  # no labels, otherwise many labels would cover the whole diagram
        legend3 = "Image Colors"
        point_alpha = 0.1  # low alpha since we have a lot of pixels

    # LightSpectrum -> single point in diagram
    elif isinstance(img, LightSpectrum):
        XYZ = np.array([[[*img.xyz()]]])
        labels = [img.get_desc()]
        legend3 = "Spectrum Colors"
        point_alpha = 1

    # list of LightSpectrum -> multiple points in diagram
    # works when list is empty or no img parameter provided
    elif isinstance(img, list) or img is None:

        # make sure they are of LightSpectrum type
        if img is not None:
            for Imi in img:
                if not isinstance(Imi, LightSpectrum):
                    raise TypeError(f"Expected list of LightSpectrum, got one element of {type(Imi)}.")

        labels = []
        XYZ = np.zeros((0, 1, 3), dtype=np.float64)  # will hold XYZ coordinates of spectra

        # fill legend and color array
        if img is not None:
            for i, Imi in enumerate(img):
                labels.append(Imi.get_desc())
                XYZ = np.vstack((XYZ, np.array([[[*Imi.xyz()]]])))

        legend3 = "Spectrum Colors"
        point_alpha = 1

    else:
        raise TypeError(f"Invalid parameter of type {type(img)}.")

    # convert wavelength to coordinates in diagram
    def wl_to_xy(wl):
        XYZ = np.column_stack((color.x_observer(wl),
                               color.y_observer(wl),
                               color.z_observer(wl)))

        XYZ = np.array([XYZ])
        return conv(XYZ)

    # spectral curve
    wl = np.linspace(380, 780, 1001)
    xs, ys = wl_to_xy(wl)

    # points for scatters (image pixel colors or spectrum positions)
    wls = np.linspace(380, 780, 41)
    xss, yss = wl_to_xy(wls)

    # coordinates for some spectral curve labels
    wls2 = np.array([380, 460, 480, 500, 520, 540, 560, 580, 600, 620, 780], dtype=np.float64)
    xss2, yss2 = wl_to_xy(wls2)

    # calculate chromaticity shoe area
    x = np.linspace(ext[0], ext[1], 100)
    y = np.linspace(ext[2], ext[3], 100)
    XYZ_shoe = i_conv(x, y)
    RGB = color.xyz_to_srgb_linear(XYZ_shoe, rendering_intent="Absolute", normalize=False)
    RGB = np.clip(RGB, 0, 1)

    # "bright" chromaticity diagram -> normalize such that one of the sRGB coordinates is 1
    if norm == "Largest":
        mask = ~np.all(RGB == 0, axis=2)
        RGB[mask] /= np.max(RGB[mask], axis=1)[:, np.newaxis]  # normalize brightness

    # "smooth gradient" chromaticity diagram -> normalize colors such that R + G + B = 1
    elif norm == "Sum":
        mask = ~np.all(RGB == 0, axis=2)
        RGB[mask] /= np.sum(RGB[mask], axis=1)[:, np.newaxis]  # normalize brightness

    # convert to sRGB
    sRGB = color.srgb_linear_to_srgb(RGB)

    fargs = dict() if fargs is None else fargs
    plt.figure(**fargs)
    plt.xlim(ext[0], ext[1])
    plt.ylim(ext[2], ext[3])
    plt.minorticks_on()

    # plot colored area, this also includes colors outside the gamut
    plt.imshow(sRGB, extent=[x[0], x[-1], y[0], y[-1]], interpolation="bilinear", label='_nolegend_', origin="lower")

    # we want to fill the regions outside the gamut black
    # the edge can be divided into a lower and upper part
    # both go from the leftmost point xsm to the rightmost x[-1]
    # position and wavelength of leftmost point in gamut
    xsm = np.argmin(xs)
    wlleft, ylleft = wl[xsm], ys[xsm]

    # fill upper edge part of gamut
    xf1 = np.concatenate(([x[0]],   xs[wl >= wlleft], [x[-1]]))
    yf1 = np.concatenate(([ylleft], ys[wl >= wlleft], [ys[-1]]))
    plt.fill_between(xf1, yf1, np.ones_like(xf1)*y[-1], color="0.2", label='_nolegend_')  # fill region outside gamut

    # fill lower edge part of gamut
    xf2 = np.concatenate(([x[0]],   np.flip(xs[wl <= wlleft]), [xs[-1]], [x[-1]]))
    yf2 = np.concatenate(([ylleft], np.flip(ys[wl <= wlleft]), [ys[-1]], [ys[-1]]))
    plt.fill_between(xf2, yf2, np.ones_like(xf2)*y[0], color="0.2", label='_nolegend_')

    # plot gamut edge line
    plt.plot(xs, ys, color="k", zorder=4, linewidth=1, label='_nolegend_')
    plt.plot([xs[0], xs[-1]], [ys[0], ys[-1]], color="k", linewidth=1, zorder=4, label='_nolegend_')

    # add spectral wavelength labels and markers
    plt.scatter(xss, yss, marker="+", color="0.9", linewidth=1, s=15, zorder=5, label='_nolegend_')
    for i, wli in enumerate(wls2.tolist()):
        text = plt.text(xss2[i], yss2[i], str(int(wli))+"nm", fontsize=10, color="w", zorder=10, label='_nolegend_')
        text.set_path_effects([path_effects.Stroke(linewidth=1, foreground='black'), path_effects.Normal()])

    # draw sRGB gamut and whitepoint
    plt.plot([r[0], g[0], b[0], r[0]], [r[1], g[1], b[1], r[1]], "k-.", linewidth=1)
    plt.scatter(w[0], w[1], color="w", marker="o", s=20, edgecolor="k")

    # plot image/spectrum points and labels
    xi, yi = conv(XYZ)
    plt.scatter(xi, yi, color="k", marker="x", s=10, alpha=point_alpha)
    for i, l in enumerate(labels):
        text = plt.text(xi[i], yi[i], l, fontsize=9, color="k")

    plt.minorticks_on()

    # labels and legends
    plt.xlabel(xl)
    plt.ylabel(yl)
    plt.title(title)
    plt.legend(["sRGB Gamut", "Whitepoint D65", legend3])
    plt.tight_layout()
    _save_or_show(block, path, sargs)
