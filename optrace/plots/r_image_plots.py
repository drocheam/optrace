"""
Functions for plotting of Source/Detector Images.
Plotting Modes include Irradiance, Illuminance and RGB.
All Modes can be shown in linear or logarithmic scaling

"""

# plotting library
import matplotlib  # plotting library
import matplotlib.pyplot as plt  # actual plotting

import numpy as np  # calculations

from ..tracer.r_image import RImage  # RImage type and RImage displaying
from ..tracer.misc import PropertyChecker as pc  # check types and values



def r_image_plot(im:       RImage,
                 imc:      np.ndarray = None,
                 block:    bool = False,
                 log:      bool = False,
                 flip:     bool = False,
                 title:    str = None,
                 mode:     str = RImage.display_modes[0])\
        -> None:
    """

    :param im:
    :param flip:
    :param imc: precalculated Image (np.ndarray) to display. If not specified it is calculated by parameter 'mode'
    :param block: if plot is blocking (bool)
    :param log: if logarithmic values are shown (only for images with 1 channel)
    :param title:
    :param mode: "sRGB", "Illuminance" or "Irradiance" (string)
    """
    _check_types(im, imc, block, log, flip, title, mode)

    Imd = imc.copy() if imc is not None else im.get_by_display_mode(mode, log=log)
    Imd = np.flipud(Imd)  # flip upside down so element [0, 0] is in the bottom left

    _, _, xlabel, _, _, ylabel, _, _, zlabel, text = _get_labels(im, mode, log)

    if im.projection not in ["Equal-Area", None] and mode in ["Irradiance", "Illuminance"]:
        imax = np.max(Imd)
        if imax:
            Imd /= imax

    if title is not None:
        text = title

    # fall back to linear values when all pixels have the same value
    if log and (np.max(Imd) == np.min(Imd) or mode == "Outside sRGB Gamut"):
        log = False

    # get extent. Convert to degrees with projection "equidistant"
    extent = im.extent
    if im.projection == "Equidistant":
        extent = np.rad2deg(extent)

    # rotate 180 deg
    if flip:
        Imd = np.fliplr(np.flipud(Imd))
        extent = extent[[1, 0, 3, 2]]

    # better fonts to make everything look more professional
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'

    # set colormap and color norm
    current_cmap = matplotlib.colormaps["Greys_r"].copy()
    current_cmap.set_bad(color='black')
    norm = matplotlib.colors.LogNorm() if log and Imd.ndim == 2 else None

    # make image black if all content is zero
    vmin, vmax = None, None
    if np.max(Imd) == np.min(Imd) == 0:
        vmin, vmax = 0, 1e-16
    elif not log and not mode.startswith("sRGB"):
        vmin = 0

    # plot image
    fig = plt.figure()
    plt.imshow(Imd, extent=extent, cmap=current_cmap, aspect="equal", norm=norm, vmin=vmin, vmax=vmax)

    # plot labels
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # hide numbers from axis if the projection is not linear
    if im.projection not in ["Equidistant", None]:
        fig.axes[0].set_xticklabels([])
        fig.axes[0].set_yticklabels([])

    # add colorbar for some modes
    if mode.find("sRGB") == -1 and mode != "Lightness (CIELUV)":
        clb = plt.colorbar(orientation='horizontal', shrink=0.6)
        clb.ax.set_xlabel(zlabel)
   
    # title
    plt.title(text)

    # show image
    plt.show(block=block)
    plt.tight_layout()
    plt.pause(0.1)


def r_image_cut_plot(im:       RImage,
                     imc:      np.ndarray = None,
                     block:    bool = False,
                     log:      bool = False,
                     flip:     bool = False,
                     title:    str = None,
                     mode:     str = RImage.display_modes[0],
                     **kwargs)\
        -> None:
    """

    :param im:
    :param imc:
    :param kwargs:
    :param block: if plot is blocking (bool)
    :param log: if logarithmic values are shown (only for images with one channel)
    :param flip:
    :param title:
    :param mode: "sRGB", "Illuminance" or "Irradiance" (string)
    """
    _check_types(im, imc, block, log, flip, title, mode)

    # check if cut parameter provided
    if "x" not in kwargs and "y" not in kwargs:
        raise RuntimeError("Provide an x or y parameter to the RImageCutPlot function.")

    # make cut
    s, Imd = im.cut(mode, log=log, imc=imc, **kwargs)

    # get labels
    xname, xunit, xlabel, yname, yunit, ylabel, _, _, zlabel, text = _get_labels(im, mode, log)
    cut_val = kwargs["x" if "x" in kwargs else "y"]
    text += "\nCut at " + (f"{xname} = {cut_val:.5g} {xunit}" if "x" in kwargs else f"{yname} = {cut_val:.5g} {yunit}")

    # normalize values for sphere projections that are not Equal-Area
    # (since the values are incorrect anyway)
    if im.projection not in ["Equal-Area", None] and mode in ["Irradiance", "Illuminance"]:
        imax = np.max(Imd)
        if imax:
            Imd /= imax

    # overwrite title if provided
    if title is not None:
        text = title
    
    # better fonts to make everything look more professional
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'

    # convert angles to degrees with projection mode equidistant
    if im.projection == "Equidistant":
        s = np.rad2deg(s)

    # new figure
    fig = plt.figure()

    # enforce rgb colors for rgb modes
    colors = ["r", "g", "b"] if mode.startswith("sRGB") else [None, None, None]

    # plot curve(s)
    [plt.stairs(Imd[i], s, color=colors[i]) for i, _ in enumerate(Imd)]

    # flip if desired
    if flip:
        plt.gca().invert_xaxis()
    
    # plot labels
    plt.xlabel(xlabel if "y" in kwargs else ylabel)
    plt.ylabel(zlabel)

    # turn off labels with non linear axes
    if im.projection not in ["Equidistant", None]:
        fig.axes[0].set_xticklabels([])

    # major and finer grid
    plt.grid(visible=True, which='major')
    plt.grid(visible=True, which='minor', color='gainsboro', linestyle='--')
    plt.minorticks_on()

    # toggle log mode
    if log and any(np.any(Imdi > 0) for Imdi in Imd) and len(Imd) == 1:
        plt.yscale('log')

    # add RGB legend if needed
    if mode.startswith("sRGB"):
        plt.legend(["R", "G", "B"])

    # set title
    plt.title(text)

    # show image
    plt.show(block=block)
    plt.tight_layout()
    plt.pause(0.1)


def _check_types(im, imc, block, log, flip, title, mode) -> None:
    """check types for r_image plots"""
    pc.check_type("im", im, RImage)
    pc.check_type("imc", imc, np.ndarray | None)
    pc.check_type("block", block, bool)
    pc.check_type("flip", flip, bool)
    pc.check_type("log", log, bool)
    pc.check_type("mode", mode, str)
    pc.check_type("title", title, str | None)
    pc.check_if_element("mode", mode, RImage.display_modes)


def _get_labels(im: RImage, mode: str, log: bool) -> tuple[str, str, str, str, str, str, str, str, str, str]:
    """get plot labels and title"""

    text = im.get_long_desc(fallback="")
    
    # x, y labels
    match im.projection:
        case None:
            xname, xunit, xlabel = "x", "mm", "x / mm"
            yname, yunit, ylabel = "y", "mm", "y / mm"
        case "Equidistant":
            xname, xunit, xlabel = r"$\theta_x$", "°", r"$\theta_x$ / °"
            yname, yunit, ylabel = r"$\theta_y$", "°", r"$\theta_y$ / °"
        case _:
            xname, xunit, xlabel = fr"Nonlinear Projection $p_x$", "", fr"Nonlinear Projection $p_x$"
            yname, yunit, ylabel = fr"Nonlinear Projection $p_y$", "", fr"Nonlinear Projection $p_y$"

    # z-label
    if mode in ["Irradiance", "Illuminance"]:
        punit, aname, srname, p = ("W", "Irradiance", "Radiant", im.get_power()) if mode == "Irradiance"\
                                  else ("lm", "Illuminance", "Luminous", im.get_luminous_power())
    
        match im.projection:
            case "Equal-Area":
                zname, zunit, zlabel = f"{srname} Intensity", f"{punit}/sr", f"{srname} Intensity in {punit}/sr"
            case None:
                zname, zunit, zlabel = aname, f"{punit}/mm²", f"{aname} in {punit}/mm²"
            case _:
                zname, zunit, zlabel = f"Normalized Projected {srname} Intensity", "",\
                                       f"Normalized Projected {srname} Intensity"
        
        text += f"\n Total {srname} Flux: {p:.5g} {punit}"
        
    else:
        zname, zunit, zlabel = mode, "", mode
        zlabel += ", Logarithmic" if log and mode.startswith("sRGB") else ""
        text += f"\nMode: {zlabel}"

    if im.projection is not None:
        text += f"\n{im.projection} Projection"

    return xname, xunit, xlabel, yname, yunit, ylabel, zname, zunit, zlabel, text
