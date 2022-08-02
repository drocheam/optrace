"""
Functions for plotting of Source/Detector Images.
Plotting Modes include Irradiance, Illuminance and RGB.
All Modes can be shown in linear or logarithmic scaling

"""

# plotting library
import matplotlib  # plotting library
import matplotlib.pyplot as plt  # actual plotting

import numpy as np  # calculations

from optrace.tracer.r_image import RImage  # RImage type and RImage displaying
from optrace.tracer.misc import PropertyChecker as pc  # check types and values


def r_image_plot(im:       RImage,
                 imc:      np.ndarray = None,
                 block:    bool = False,
                 log:      bool = False,
                 flip:     bool = False,
                 mode:     str = RImage.display_modes[0])\
        -> None:
    """

    :param im:
    :param flip:
    :param imc: precalculated Image (np.ndarray) to display. If not specified it is calculated by parameter 'mode'
    :param block: if plot is blocking (bool)
    :param log: if logarithmic values are shown (bool)
    :param mode: "sRGB", "Illuminance" or "Irradiance" (string)
    """
    pc.checkType("im", im, RImage)
    pc.checkIfIn("mode", mode, RImage.display_modes)

    text = im.get_long_desc(fallback="")

    match mode:
        case "Irradiance":
            clabel = "Irradiance in W/mm²"
            text += f"\n Total Radiant Flux: {im.get_power():.5g} W"

        case "Illuminance":
            clabel = "Illuminance in lm/mm²"
            text += f"\n Total Luminous Flux: {im.get_luminous_power():.5g} lm"

        case _:
            clabel = mode
            text += f"\nMode: {mode}"

    Imd = imc.copy() if imc is not None else im.get_by_display_mode(mode, log=log)
    Imd = np.flipud(Imd)  # flup upside down so element [0, 0] is in the bottom left

    # fall back to linear values when all pixels have the same value
    if log and (np.max(Imd) == np.min(Imd) or mode == "Outside sRGB Gamut"):
        log = False

    extent = im.extent

    # rotate 180 deg
    if flip:
        Imd = np.fliplr(np.flipud(Imd))
        extent = extent[[1, 0, 3, 2]]

    # better fonts to make everything look more professional
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'

    # set colormap and color norm
    current_cmap = matplotlib.cm.get_cmap("Greys_r").copy()
    current_cmap.set_bad(color='black')
    norm = matplotlib.colors.LogNorm() if log else None

    # make image black if all content is zero
    vmin, vmax = None, None
    if np.max(Imd) == np.min(Imd) == 0:
        vmin, vmax = 0, 1e-16
    elif not log and not mode.startswith("sRGB"):
        vmin = 0

    # plot image
    plt.figure()
    plt.imshow(Imd, extent=extent, cmap=current_cmap, aspect="equal", norm=norm, vmin=vmin, vmax=vmax)

    # plot labels
    if im.coordinate_type == "Polar":
        plt.xlabel(r"$\theta_x$ / °")
        plt.ylabel(r"$\theta_y$ / °")
    else:
        plt.xlabel("x / mm")
        plt.ylabel("y / mm")

    if mode.find("sRGB") == -1 and mode != "Lightness (CIELUV)":
        clb = plt.colorbar(orientation='horizontal', shrink=0.6)
        clb.ax.set_xlabel(clabel)
    plt.title(text)

    # show image
    plt.show(block=block)
    plt.pause(0.1)


def r_image_cut_plot(im:       RImage,
                     imc:      np.ndarray = None,
                     block:    bool = False,
                     log:      bool = False,
                     flip:     bool = False,
                     mode:     str = RImage.display_modes[0],
                     **kwargs)\
        -> None:
    """

    :param im:
    :param imc:
    :param kwargs:
    :param block: if plot is blocking (bool)
    :param log: if logarithmic values are shown (bool)
    :param flip:
    :param mode: "sRGB", "Illuminance" or "Irradiance" (string)
    """
    pc.checkType("im", im, RImage)
    pc.checkIfIn("mode", mode, RImage.display_modes)

    if "x" not in kwargs and "y" not in kwargs:
        raise RuntimeError("Provide an x or y parameter to the RImageCutPlot function.")

    yim = "x" in kwargs

    text = im.get_long_desc(fallback="")
    if im.coordinate_type == "Polar":
        text += "\nCut at " + ((r"$\theta_x$ = " + f'{kwargs["x"]:.5g}') if yim 
                                else (r"$\theta_y$ = " + f'{kwargs["y"]:.5g}')) + " °"
    else:
        text += "\nCut at " + (f'x = {kwargs["x"]:.5g}' if yim else f'y = {kwargs["y"]:.5g}') + " mm"

    match mode:
        case "Irradiance":      clabel = "Irradiance in W/mm²"
        case "Illuminance":     clabel = "Illuminance in lm/mm²"
        case _:                 clabel = mode

    s, Imd = im.cut(mode, log=log, imc=imc, **kwargs)

    # better fonts to make everything look more professional
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'

    # plot image
    plt.figure()

    colors = ["r", "g", "b"] if mode.startswith("sRGB") else [None, None, None]
    [plt.plot(s, Imd[i], color=colors[i]) for i, _ in enumerate(Imd)]

    # plot labels
    if im.coordinate_type == "Polar":
        plt.xlabel(r"$\theta_y$ / °") if yim else plt.xlabel(r"$\theta_x$ / °")
    else:
        plt.xlabel("y / mm") if yim else plt.xlabel("x / mm")

    if log and any(np.max(Imdi) > 0 for Imdi in Imd):
        plt.yscale('log')

    if mode.startswith("sRGB"):
        plt.legend(["R", "G", "B"])

    plt.grid(visible=True, which='major')
    plt.grid(visible=True, which='minor', color='gainsboro', linestyle='--')
    plt.minorticks_on()
    plt.ylabel(clabel)
    plt.title(text)

    if flip:
        plt.gca().invert_xaxis()

    # show image
    plt.show(block=block)
    plt.pause(0.1)
