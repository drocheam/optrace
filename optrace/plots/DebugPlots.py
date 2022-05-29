
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np

import optrace.tracer.Color as Color
import optrace.tracer.Misc as misc
from optrace.tracer.RImage import *
from optrace.tracer.spectrum.LightSpectrum import *
from optrace.tracer.spectrum.RefractionIndex import *
from optrace.tracer.geometry.Surface import *
import optrace.tracer.presets.Lines as Lines

def AutoFocusDebugPlot(r, vals, rf, ff, title="Focus Finding", block=False):
    """

    :param r:
    :param vals:
    :param rf:
    :param ff:
    :param title:
    :param block:
    """

    _set_font()
    
    plt.figure()
    plt.plot(r, vals)
    plt.plot(r, vals, "r.")
    plt.axvline(rf, ls="--", color="y")
    plt.axhline(ff, ls="--", color="y")

    _show_grid()
    plt.xlabel("z in mm")
    plt.ylabel("cost function")
    plt.legend(["cost estimation", "cost values", "found minimum"])
    plt.title(title)
    plt.show(block=block)
    plt.pause(0.1)


def RefractionIndexPlot(RI, title="Refraction Index", **kwargs):
    """
    """
    SpectrumPlot(RI, title=title, **kwargs)

def SpectrumPlot(Spec, title="Spectrum", **kwargs):
    """
    """
    Spec0 = Spec[0] if isinstance(Spec, list) else Spec
    ylabel = Spec0.quantity if Spec0.quantity != "" else "value"
    ylabel += f" in {Spec0.unit}" if Spec0.unit != "" else ""

    _SpectrumPlot(Spec, r"$\lambda$ in nm", ylabel, title=title, **kwargs)


def AbbePlot(RI:    list[RefractionIndex], 
             title: str = "Abbe Diagram", 
             lines: list = Lines.preset_lines_FdC,
             block: bool = False):
    """
    """

    _set_font()
    plt.figure()

    for i, RIi in enumerate(RI):

        nd = RIi(lines[1])
        Vd = RIi.getAbbeNumber(lines)

        if not np.isfinite(Vd):
            print(f"Ignoring non dispersive material '{RIi.getDesc()}'")
            continue

        sc = plt.scatter(Vd, nd, marker="x")
        col = sc.get_facecolors()[0].tolist()
        plt.text(Vd, nd, RIi.getDesc(), color=col)
    
    plt.xlim([plt.xlim()[1], plt.xlim()[0]])
    _show_grid()
    plt.xlabel("Abbe Number V")
    plt.ylabel(f"Refraction Index n ($\lambda$ = {lines[1]}nm)")
    plt.title(title)
    plt.show(block=block)
    plt.pause(0.1)
    

def ChromacitiesCIE1931(Im: RImage | Spectrum | list[Spectrum], RI="Ignore", **kwargs):

    r, g, b, w = Color._sRGB_r_xy, Color._sRGB_g_xy, Color._sRGB_b_xy, Color._sRGB_w_xy

    # when looking at the XYZ to sRGB Matrix, the r channel has by far the highest coefficients
    # therefore the red region leads to the highest brightness clipping
    # normalizing to the redmost point, the primary, saves all values inside the gamut from brightness clipping
    # coefficients are from Color.XYZ_to_sRGBLinear()
    # we do this since we can't normalize on the highest value in the chromacity image, 
    # since the brightest values come from impossible colors (colors outside human vision) clipped towards the gamut
    norm = 3.2404542*r[0] - 1.5371385*r[1] - 0.4985314*(1 - r[0] - r[1])

    def conv(XYZ):
        xyY = Color.XYZ_to_xyY(XYZ)
        return xyY[:, :, 0].flatten(), xyY[:, :, 1].flatten()

    def i_conv(x, y):
        xg, yg = np.meshgrid(x, y)
        zg = 1 - xg - yg
        return 1/norm*np.dstack((xg, yg, zg))

    ext = [0, 0.8, 0, 0.9]

    _ChromaticityPlot(Im, conv, i_conv, RI, r, g, b, w, ext, "CIE 1931 Chromaticity Diagram", "x", "y", **kwargs)


def ChromacitiesCIE1976(Im: RImage | LightSpectrum | list[LightSpectrum], RI="Ignore", **kwargs):

    r, g, b, w = Color._sRGB_r_uv, Color._sRGB_g_uv, Color._sRGB_b_uv, Color._sRGB_w_uv

    # see notes in ChromacitiesCIE1931()
    r_xy = Color._sRGB_r_xy
    norm = 3.2404542*r_xy[0] - 1.5371385*r_xy[1] - 0.4985314*(1 - r_xy[0] - r_xy[1])
    
    def conv(XYZ):
        Luv = Color.XYZ_to_Luv(XYZ)
        u_v_L = Color.Luv_to_u_v_L(Luv)
        return u_v_L[:, :, 0].flatten(), u_v_L[:, :, 1].flatten()

    def i_conv(u_, v_):
       
        u_g, v_g = np.meshgrid(u_, v_)
        xg = 9*u_g / (6*u_g - 16*v_g + 12)
        yg = 4*v_g / (6*u_g - 16*v_g + 12)
        zg = 1 - xg - yg

        return 1/norm*np.dstack((xg, yg, zg))

    ext = [0, 0.7, 0, 0.7]
    _ChromaticityPlot(Im, conv, i_conv, RI, r, g, b, w, ext, "CIE 1976 UCS Diagram", "u'", "v'", **kwargs)

# TODO improve. Multiple Surfaces?
# How to handle r and re?
# def SurfaceProfilePlot(Surf:  Surface, 
                       # r0:    float = 0, 
                       # re:    float = None, 
                       # title: str = "Surface Profile", 
                       # block: bool = False)
        # -> None:
    # """"""

    # if re is None:
        # re = (surf.pos[0] + Surf.r) if Surf.r is not None else Surf.getExtent()[1]
    # r = np.linspace(surf.pos[0] + r0, re, 2000)
    # vals = Surf.getValues(r, np.zeros_like(r))

    # _set_font()
    # plt.figure()
    # plt.plot(r, vals)

    # _show_grid()
    # plt.xlabel("r in mm")
    # plt.ylabel("h in mm")
    # plt.title(Surf.getLongDesc(fallback=title))
    # plt.show(block=block)

    
# TODO list of Images. Enables RI for Images
def _ChromaticityPlot(Im, conv, i_conv, RI, r, g, b, w, ext, title, xl, yl, block=False, norm="Sum"):
    """"""

    if isinstance(Im, RImage):
        XYZ = Im.getXYZ() if RI == "Ignore" else Color.sRGB_to_XYZ(Im.getRGB(RI=RI))
        labels = []
        legend3 = "Image Colors"

    elif isinstance(Im, LightSpectrum):
        XYZ = Im.getXYZ()
        labels = [Im.getDesc()]
        legend3 = "Spectrum Colors"
   
    # TODO only allow list of LightSpectrum
    elif isinstance(Im, list):
        labels = []
        XYZ = np.zeros((0, 1, 3), dtype=np.float64)

        for i, Imi in enumerate(Im):
            labels.append(Imi.getDesc())
            XYZ = np.vstack((XYZ, Imi.getXYZ()))

        legend3 = "Spectrum Colors"

    def wl_to_xy(wl):
        XYZ = np.column_stack((Color.Tristimulus(wl, "X"),\
                               Color.Tristimulus(wl, "Y"),\
                               Color.Tristimulus(wl, "Z")))

        XYZ = np.array([XYZ])
        return conv(XYZ)

    # spectral curve
    wl = np.linspace(380, 780, 1001)
    xs, ys = wl_to_xy(wl)

    # points for scatters
    wls = np.linspace(380, 780, 41)
    xss, yss = wl_to_xy(wls)

    # points for labels
    wls2 = np.array([380, 470, 480, 490, 500, 510, 520, 540,  560, 580, 600, 620, 780], dtype=np.float64)
    xss2, yss2 = wl_to_xy(wls2)

    x = np.linspace(ext[0], ext[1], 100)
    y = np.linspace(ext[2], ext[3], 100)

    XYZ_shoe = i_conv(x, y)
    RGB = Color.XYZ_to_sRGBLinear(XYZ_shoe, RI="Absolute", normalize=False)
    if norm == "Largest":
        mask = ~np.all(RGB == 0, axis=2)
        RGB[mask] /= np.max(RGB[mask], axis=1)[:, np.newaxis] # normalize brightness
    elif norm == "Sum":
        mask = ~np.all(RGB == 0, axis=2)
        RGB[mask] /= np.sum(RGB[mask], axis=1)[:, np.newaxis] # normalize brightness
    # elif norm == "Luminance":
        # mask = ~np.all(RGB == 0, axis=2)
        # L = 0.2126729*RGB[mask, 0] +  0.7151522*RGB[mask, 1] + 0.0721750*RGB[mask, 2] # matrix from sRGB to XYZ for Y
        # RGB[mask] /= L[:, np.newaxis] / 0.0721750 # divide by smallest coefficient to ensure range [0, 1]

    sRGB = Color.sRGBLinear_to_sRGB(RGB)
    sRGB = np.flipud(sRGB)

    _set_font()
    plt.figure()
    plt.minorticks_on()
    
    plt.imshow(sRGB, extent=[x[0], x[-1], y[0], y[-1]], interpolation="bilinear", label='_nolegend_')

    # position and wavelength of leftmost point in gamut
    xsm = np.argmin(xs)
    wlleft, ylleft = wl[xsm], ys[xsm]

    fillargs = dict(color="0.2", label='_nolegend_')

    # line going from x-edges of chromacity diagram around the upper part of gamut
    xf1 = np.concatenate(([x[0]],   xs[wl>=wlleft], [x[-1]]))
    yf1 = np.concatenate(([ylleft], ys[wl>=wlleft], [ys[-1]]))
    plt.fill_between(xf1, yf1, np.ones_like(xf1)*y[-1], **fillargs) # fill region outside gamut

    # line going from x-edges of chromacity diagram around the lower part of gamut
    xf2 = np.concatenate(([x[0]],   np.flip(xs[wl<=wlleft]), [xs[-1]], [x[-1]]))
    yf2 = np.concatenate(([ylleft], np.flip(ys[wl<=wlleft]), [ys[-1]], [ys[-1]]))
    plt.fill_between(xf2, yf2, np.ones_like(xf2)*y[0], **fillargs)

    # plot gamut edge
    plt.plot(xs, ys, color="k", zorder=4, linewidth=1, label='_nolegend_')
    plt.plot([xs[0], xs[-1]], [ys[0], ys[-1]], color="k", linewidth=1, zorder=4, label='_nolegend_')

    # add spectral wavelength labels and markers
    plt.scatter(xss, yss, marker="+", color="0.7", linewidth=1, s=15, zorder=5, label='_nolegend_')
    for i, wli in enumerate(wls2.tolist()):
        text = plt.text(xss2[i], yss2[i], str(int(wli)), fontsize=9, color="w", zorder=10, label='_nolegend_')
        text.set_path_effects([path_effects.Stroke(linewidth=1, foreground='black'),
                       path_effects.Normal()])
    
    # draw sRGB gamut and whitepoint
    # effect = [path_effects.Stroke(linewidth=2, foreground='w'), path_effects.Normal()]
    plt.plot([r[0], g[0], b[0], r[0]], [r[1], g[1], b[1], r[1]], "k-.", linewidth=1)
    plt.scatter(w[0], w[1], color="w", marker="o", s=20, edgecolor="k")

    # plot image/spectrum points and labels
    xi, yi = conv(XYZ)
    plt.scatter(xi, yi, color="k", marker="x", s=10)
    for i, l in enumerate(labels):
        text = plt.text(xi[i], yi[i], l, fontsize=9, color="k")

    plt.xlabel(xl)
    plt.ylabel(yl)
    plt.title(title)
    plt.legend(["sRGB Gamut", "Whitepoint D65", legend3])
    plt.show(block=block)
    plt.pause(0.1)

# TODO make Lines and Monochromatic plottable        
def _SpectrumPlot(obj, xlabel, ylabel, title, wl0=Color.WL_MIN, wl1=Color.WL_MAX, steps=5000, 
                  legend_off=False, labels_off=False, colors=None, block=False):

    # TODO isContinuous not needed after this?
    # if not obj.isContinuous():
        # raise RuntimeError(f"Can't plot discontinuous spectrum_type '{obj.spectrum_type}'.")
    
    wl = np.linspace(wl0, wl1, steps)
    
    _set_font()
   
    fig, (ax1, ax2) = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': [15, 1]})
   
    if not isinstance(obj, list):
        val = obj(wl)
        ax1.plot(wl, val, color=colors)
        fig.suptitle("\n" + obj.getLongDesc(fallback=title))
    else:
        wl = np.linspace(wl0, wl1, steps)
        lg = []
        for i, obji in enumerate(obj):
            val = obji(wl)

            cl = colors[i] if colors is not None else None
            axp = ax1.plot(wl, obji(wl), color=cl)
            lg.append(obji.getLongDesc())
            
            if not labels_off:
                tp = int(i / len(obj) * wl.shape[0] / 10)
                ax1.text(wl[tp], val[tp], obji.getDesc(), color=axp[0].get_color())

        if not legend_off:
            ax1.legend(lg)
        fig.suptitle("\n" + title)
    
    _show_grid(ax1)

    colors = np.array([Color.spectralCM(wl0=plt.xlim()[0], wl1=plt.xlim()[1], N=1000)[:, :3]])/255

    #enforce image extent of 1:10 for every wavelength range
    ax2.imshow(colors, extent=[*plt.xlim(), 0.1*plt.xlim()[0], 0.1*plt.xlim()[1]], aspect="auto")

    ax1.set(ylabel=ylabel)
    ax2.set(xlabel=xlabel)
    ax2.get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.show(block=block)
    plt.pause(0.1)


def _show_grid(what=plt):
    what.grid(visible=True, which='major')
    what.grid(visible=True, which='minor', color='gainsboro', linestyle='--')
    what.minorticks_on()


def _set_font():
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'

