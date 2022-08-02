
# plotting library
import matplotlib  # plotting library
import matplotlib.pyplot as plt  # actual plotting
import matplotlib.patheffects as path_effects  # path effects for plot lines

import numpy as np  # calculations
import scipy.optimize  # optimize result type
from typing import Callable  # Callable typing hints

import optrace.tracer.color as Color  # color conversions for chromacity plots

# only needed for typing and plotting
from optrace.tracer.r_image import RImage
from optrace.tracer.spectrum import Spectrum, LightSpectrum
from optrace.tracer.refraction_index import RefractionIndex
from optrace.tracer.geometry import Surface

import optrace.tracer.presets.spectral_lines as Lines  # spectral lines for AbbePlot


chromacity_norms = ["Ignore", "Largest", "Sum"]
"""possible norms for the chromacity diagrams"""


def autofocus_cost_plot(r:       np.ndarray,
                        vals:    np.ndarray,
                        res:     scipy.optimize.OptimizeResult,
                        title:   str = "Autofocus",
                        block:   bool = False)\
        -> None:
    """

    :param r:
    :param vals:
    :param res:
    :param title:
    :param block:
    """

    _set_font()
    plt.figure()

    # evaluation points and connection line of cost function
    plt.plot(r, vals)
    plt.plot(r, vals, "r.")

    # found minimum x and y coordinate
    plt.axvline(res.x, ls="--", color="y")
    plt.axhline(res.fun, ls="--", color="y")

    _show_grid()
    plt.xlabel("z in mm")
    plt.ylabel("cost function")
    plt.legend(["cost estimation", "cost values", "found minimum"])
    plt.title(title)
    plt.show(block=block)
    plt.pause(0.1)


def refraction_index_plot(ri:         RefractionIndex | list[RefractionIndex],
                          title:      str = "Refraction Index",
                          **kwargs)\
        -> None:
    """

    :param ri:
    :param title:
    :param kwargs:
    """
    spectrum_plot(ri, title=title, **kwargs)


def spectrum_plot(spectrum:  Spectrum | list[Spectrum],
                  title:     str = "Spectrum",
                  **kwargs)\
        -> None:
    """

    :param spectrum:
    :param title:
    :param kwargs:
    """
    # set ylabel
    Spec0 = spectrum[0] if isinstance(spectrum, list) else spectrum
    ylabel = Spec0.quantity if Spec0.quantity != "" else "value"
    ylabel += f" in {Spec0.unit}" if Spec0.unit != "" else ""

    _spectrum_plot(spectrum, r"$\lambda$ in nm", ylabel, title=title, **kwargs)


def abbe_plot(ri:     list[RefractionIndex],
              title:  str = "Abbe Diagram",
              lines:  list = None,
              block:  bool = False,
              silent: bool = False)\
        -> None:
    """

    :param ru:
    :param title:
    :param lines:
    :param block:
    :param silent:
    :return:
    """

    lines = Lines.FdC if lines is None else lines
    _set_font()
    plt.figure()

    for i, RIi in enumerate(ri):

        # get refraction index and abbe number
        nd = RIi(lines[1])
        Vd = RIi.get_abbe_number(lines)

        # check if dispersive
        if not np.isfinite(Vd):
            if not silent:
                print(f"Ignoring non dispersive material '{RIi.get_desc()}'")
            continue  # skip plotting

        # plot point and label
        sc = plt.scatter(Vd, nd, marker="x")
        col = sc.get_facecolors()[0].tolist()
        plt.text(Vd, nd, RIi.get_desc(), color=col)
    
    plt.xlim([plt.xlim()[1], plt.xlim()[0]])  # reverse direction of x-axis
    _show_grid()
    plt.xlabel("Abbe Number V")
    plt.ylabel(f"Refraction Index n ($\lambda$ = {lines[1]}nm)")
    plt.title(title)
    plt.show(block=block)
    plt.pause(0.1)


def chromacities_cie_1931(img:                  RImage | LightSpectrum | list[LightSpectrum],
                          rendering_intent:     str = "Ignore",
                          **kwargs)\
        -> None:
    """

    :param img:
    :param rendering_intent:
    :param kwargs:
    :return:
    """
    # coordinates of whitepoint and sRGB primaries in Chromacity diagramm coordinates (xy)
    r, g, b, w = Color.SRGB_R_XY, Color.SRGB_G_XY, Color.SRGB_B_XY, Color.SRGB_W_XY

    # when looking at the XYZ to sRGB Matrix, the r channel has by far the highest coefficients
    # therefore the red region leads to the highest brightness clipping
    # normalizing to the redmost point, the primary, saves all values inside the gamut from brightness clipping
    # coefficients are from Color.XYZ_to_sRGBLinear()
    # we do this since we can't normalize on the highest value in the chromacity image, 
    # since the brightest values come from impossible colors (colors outside human vision) clipped towards the gamut
    norm = 3.2404542*r[0] - 1.5371385*r[1] - 0.4985314*(1 - r[0] - r[1])

    # XYZ to Chromacity Diagram coordinates (xy in this case)
    def conv(XYZ):
        xyY = Color.XYZ_to_xyY(XYZ)
        return xyY[:, :, 0].flatten(), xyY[:, :, 1].flatten()

    # inverse operation of conv
    def i_conv(x, y):
        xg, yg = np.meshgrid(x, y)
        zg = 1 - xg - yg
        return 1/norm*np.dstack((xg, yg, zg))

    ext = [0, 0.8, 0, 0.9]  # extent of diagram
    _chromaticity_plot(img, conv, i_conv, rendering_intent, r, g, b, w, ext,
                       "CIE 1931 Chromaticity Diagram", "x", "y", **kwargs)


def chromacities_cie_1976(img:                  RImage | LightSpectrum | list[LightSpectrum],
                          rendering_intent:     str = "Ignore",
                          **kwargs)\
        -> None:
    """

    :param img:
    :param rendering_intent:
    :param kwargs:
    :return:
    """

    # coordinates of whitepoint and sRGB primaries in Chromacity diagramm coordinates (u'v')
    r, g, b, w = Color.SRGB_R_UV, Color.SRGB_G_UV, Color.SRGB_B_UV, Color.SRGB_W_UV

    # see notes in ChromacitiesCIE1931()
    r_xy = Color.SRGB_R_XY
    norm = 3.2404542*r_xy[0] - 1.5371385*r_xy[1] - 0.4985314*(1 - r_xy[0] - r_xy[1])
    
    # XYZ to Chromacity Diagram coordinates (u'v' in this case)
    def conv(XYZ):
        Luv = Color.XYZ_to_Luv(XYZ)
        u_v_L = Color.Luv_to_u_v_L(Luv)
        return u_v_L[:, :, 0].flatten(), u_v_L[:, :, 1].flatten()

    # inverse operation of conv
    def i_conv(u_, v_):
        u_g, v_g = np.meshgrid(u_, v_)
        xg = 9*u_g / (6*u_g - 16*v_g + 12)
        yg = 4*v_g / (6*u_g - 16*v_g + 12)
        zg = 1 - xg - yg

        return 1/norm*np.dstack((xg, yg, zg))

    ext = [0, 0.7, 0, 0.7]  # extent of diagram
    _chromaticity_plot(img, conv, i_conv, rendering_intent, r, g, b, w, ext,
                       "CIE 1976 UCS Diagram", "u'", "v'", **kwargs)


def surface_profile_plot(surface:          Surface | list[Surface],
                         r0:               float = 0.,
                         re:               float = None,
                         remove_offset:    bool = False,
                         title:            str = "Surface Profile",
                         block:            bool = False)\
        -> None:
    """

    :param surface:
    :param r0:
    :param re:
    :param remove_offset:
    :param title:
    :param block:
    :return:
    """

    Surf_list = [surface] if isinstance(surface, Surface) else surface  # enforce list even for one element
    legends = []  # legend entries

    _set_font()
    plt.figure()

    for i, Surfi in enumerate(Surf_list):
        # create r range and get Surface values (in x direction)
        if re is None:
            re = (Surfi.pos[0] + Surfi.r) if Surfi.r is not None else Surfi.get_extent()[1]
        r = np.linspace(Surfi.pos[0] + r0, re, 2000)  # 2000 values should be enough
        vals = Surfi.get_values(r, np.full_like(r, Surfi.pos[1]))  # from center in x direction
       
        # remove position offset
        if remove_offset:
            vals -= Surfi.pos[2]

        plt.plot(r, vals)
        # legend entry is long_desc, desc or fallback
        legends.append(Surfi.get_long_desc(fallback=f"Surface {i}"))

    _show_grid()
    plt.xlabel("r in mm")
    plt.ylabel("z in mm")
    plt.title(title)
    plt.legend(legends)
    plt.show(block=block)
    plt.pause(0.1)

    
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
                       norm:                    str = "Sum")\
        -> None:
    """Lower level plotting function. Don't use directly"""

    # RImage -> plot Pixel colors (RI != "Ignore" means sRGB conversion with said rendering intent)
    if isinstance(img, RImage):
        XYZ = img.get_xyz() if rendering_intent == "Ignore" \
            else Color.sRGB_to_XYZ(img.getRGB(rendering_intent=rendering_intent))
        labels = []  # no labels, otherwise many labels would cover the whole diagram
        legend3 = "Image Colors"
        point_alpha = 0.1  # low alpha since we have a lot of pixels

    # LightSpectrum -> single point in diagram
    elif isinstance(img, LightSpectrum):
        XYZ = img.get_xyz()
        labels = [img.get_desc()]
        legend3 = "Spectrum Colors"
        point_alpha = 1

    # list of LightSpectrum -> multiple points in diagram
    elif isinstance(img, list):

        # make sure they are of LightSpectrum type
        for Imi in img:
            if not isinstance(Imi, LightSpectrum):
                raise RuntimeError(f"Expected list of LightSpectrum, got one element of {type(Imi)}.")

        labels = []
        XYZ = np.zeros((0, 1, 3), dtype=np.float64)  # will hold XYZ coordinates of spectra

        # fill legend and color array
        for i, Imi in enumerate(img):
            labels.append(Imi.get_desc())
            XYZ = np.vstack((XYZ, Imi.get_xyz()))

        legend3 = "Spectrum Colors"
        point_alpha = 1

    else:
        raise RuntimeError(f"Invalid parameter of type {type(Im)}.")

    # convert wavelength to coordinates in diagram
    def wl_to_xy(wl):
        XYZ = np.column_stack((Color.tristimulus(wl, "X"),
                               Color.tristimulus(wl, "Y"),
                               Color.tristimulus(wl, "Z")))

        XYZ = np.array([XYZ])
        return conv(XYZ)

    # spectral curve
    wl = np.linspace(380, 780, 1001)
    xs, ys = wl_to_xy(wl)

    # points for scatters (image pixel colors or spectrum positions)
    wls = np.linspace(380, 780, 41)
    xss, yss = wl_to_xy(wls)

    # coordinates for some spectral curve labels
    wls2 = np.array([380, 470, 480, 490, 500, 510, 520, 540,  560, 580, 600, 620, 780], dtype=np.float64)
    xss2, yss2 = wl_to_xy(wls2)

    # calculate chromacity shoe area
    x = np.linspace(ext[0], ext[1], 100)
    y = np.linspace(ext[2], ext[3], 100)
    XYZ_shoe = i_conv(x, y)
    RGB = Color.XYZ_to_sRGBLinear(XYZ_shoe, rendering_intent="Absolute", normalize=False)

    # "bright" chromacity diagram -> normalize such that one of the sRGB coordinates is 1
    if norm == "Largest":
        mask = ~np.all(RGB == 0, axis=2)
        RGB[mask] /= np.max(RGB[mask], axis=1)[:, np.newaxis]  # normalize brightness

    # "smooth gradient" chromacity diagram -> normalize colors such that R + G + B = 1
    elif norm == "Sum":
        mask = ~np.all(RGB == 0, axis=2)
        RGB[mask] /= np.sum(RGB[mask], axis=1)[:, np.newaxis]  # normalize brightness

    # convert to sRGB and flip such that element [0, 0] is in the lower left of the diagram
    sRGB = Color.sRGBLinear_to_sRGB(RGB)
    sRGB = np.flipud(sRGB)

    _set_font()
    plt.figure()
    plt.minorticks_on()
   
    # plot colored area, this also includes colors outside the gamut
    plt.imshow(sRGB, extent=[x[0], x[-1], y[0], y[-1]], interpolation="bilinear", label='_nolegend_')

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
    plt.scatter(xss, yss, marker="+", color="0.7", linewidth=1, s=15, zorder=5, label='_nolegend_')
    for i, wli in enumerate(wls2.tolist()):
        text = plt.text(xss2[i], yss2[i], str(int(wli)), fontsize=9, color="w", zorder=10, label='_nolegend_')
        text.set_path_effects([path_effects.Stroke(linewidth=1, foreground='black'), path_effects.Normal()])
    
    # draw sRGB gamut and whitepoint
    # effect = [path_effects.Stroke(linewidth=2, foreground='w'), path_effects.Normal()]
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
    plt.show(block=block)
    plt.pause(0.1)


def _spectrum_plot(obj:          Spectrum | list[Spectrum],
                   xlabel:       str,
                   ylabel:       str,
                   title:        str,
                   steps:        int = 5000,
                   legend_off:   bool = False,
                   labels_off:   bool = False,
                   colors:       str | list[str] = None,
                   block:        bool = False)\
        -> None:
    """Lower level plotting function. Don't use directly"""

    # wavelength range
    wl0 = Color.WL_MIN 
    wl1 = Color.WL_MAX 
   
    _set_font()
    fig, (ax1, ax2) = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': [15, 1]})
 
    # get spectrum values
    def get_val(obj):
        # discrete wavelengths -> not plottable
        if not obj.is_continuous():
            print(f"Can't plot discontinuous spectrum_type '{obj.spectrum_type}'.")
        # spectrum_type="Data" -> show actual data values and positions
        # otherwise we would have interpolation issues or fake accuracy
        # this could also lead to an incorrect power for a LightSpectrum in the next part
        elif obj.spectrum_type == "Data":
            return obj._wls, obj._vals
        # default mode, crate wavelength range and just call the object
        else:
            wl = np.linspace(wl0, wl1, steps)
            return wl, obj(wl)

    # single Spectrum
    if not isinstance(obj, list):
        wlp, val = get_val(obj)
        ax1.plot(wlp, val, color=colors)

        # assign title. add total power if it is a LightSpectrum
        if isinstance(obj, LightSpectrum):
            total = np.sum(val)*(wlp[1]-wlp[0])
            fig.suptitle("\n" + obj.get_long_desc(fallback=title) + f"\nTotal Power: {total:.5g}W")
        else:
            fig.suptitle("\n" + obj.get_long_desc(fallback=title))

    # multiple spectra
    else:
        lg = []
        for i, obji in enumerate(obj):
            wlp, val = get_val(obji)

            cl = colors[i] if colors is not None else None
            axp = ax1.plot(wlp, val, color=cl)
            lg.append(obji.get_long_desc())
           
            # labels for each spectrum
            if not labels_off:
                tp = int(i / len(obj) * wlp.shape[0] / 10)
                ax1.text(wlp[tp], val[tp], obji.get_desc(), color=axp[0].get_color())

        # add legend and title
        if not legend_off:
            ax1.legend(lg)
        fig.suptitle("\n" + title)
    
    _show_grid(ax1)

    # add wavelength color bar
    # enforce image extent of 1:10 for every wavelength range, 
    # othwerwise the colorbar size changes for different wavlength ranges
    colors = np.array([Color.spectral_colormap(wl0=plt.xlim()[0], wl1=plt.xlim()[1], N=1000)[:, :3]])/255
    ax2.imshow(colors, extent=[*plt.xlim(), 0.1*plt.xlim()[0], 0.1*plt.xlim()[1]], aspect="auto")

    ax1.set(ylabel=ylabel)
    ax2.set(xlabel=xlabel)
    ax2.get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.show(block=block)
    plt.pause(0.1)


def _show_grid(what=plt) -> None:
    """active major and minor grid lines, while minor are dashed and less visible"""
    what.grid(visible=True, which='major')
    what.grid(visible=True, which='minor', color='gainsboro', linestyle='--')
    what.minorticks_on()


def _set_font() -> None:
    """set the font to something professionally looking (similar to Times New Roman)"""
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'
