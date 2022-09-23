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
    :param log: if logarithmic values are shown (bool)
    :param title:
    :param mode: "sRGB", "Illuminance" or "Irradiance" (string)
    """
    pc.check_type("im", im, RImage)
    pc.check_type("block", block, bool)
    pc.check_type("flip", flip, bool)
    pc.check_type("log", log, bool)
    pc.check_type("mode", mode, str)
    pc.check_type("title", title, str | None)
    pc.check_if_element("mode", mode, RImage.display_modes)

    Imd = imc.copy() if imc is not None else im.get_by_display_mode(mode, log=log)
    Imd = np.flipud(Imd)  # flip upside down so element [0, 0] is in the bottom left
    
    text = im.get_long_desc(fallback="")

    if mode in ["Irradiance", "Illuminance"]:
        punit, aname, srname, p = ("W", "Irradiance", "Radiant", im.get_power()) if mode == "Irradiance"\
                                  else ("lm", "Illuminance", "Luminous", im.get_luminous_power())
    
        if im.coordinate_type == "Cartesian":
            clabel = f"{aname} in {punit}/mm²"
        elif im.projection_method == "Equal-Area":
            clabel = f"{srname} Intensity in {punit}/sr"
        else:
            clabel = f"Normalized Projected {srname} Intensity"
        
        text += f"\n Total {srname} Flux: {p:.5g} {punit}"
        
        if im.coordinate_type == "Polar" and im.projection_method != "Equal-Area":
            imax = np.max(Imd)
            if imax:
                Imd /= imax
    else:
        clabel = mode
        text += f"\nMode: {mode}"

    if title is not None:
        text = title

    if im.coordinate_type == "Polar":
        text += f"\n{im.projection_method} Projection"

    # fall back to linear values when all pixels have the same value
    if log and (np.max(Imd) == np.min(Imd) or mode == "Outside sRGB Gamut"):
        log = False

    # get extent. Convert to degrees with projection method equidistant
    extent = im.extent
    if im.coordinate_type == "Polar" and im.projection_method == "Equidistant":
        extent = np.rad2deg(extent)

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
    fig = plt.figure()
    plt.imshow(Imd, extent=extent, cmap=current_cmap, aspect="equal", norm=norm, vmin=vmin, vmax=vmax)

    # plot labels
    if im.coordinate_type == "Polar":
        if im.projection_method != "Equidistant":
            fig.axes[0].set_xticklabels([])
            fig.axes[0].set_yticklabels([])
        plt.xlabel(r"$\theta_x$ / °" if im.projection_method == "Equidistant" else fr"Nonlinear Projection $p_x$")
        plt.ylabel(r"$\theta_y$ / °" if im.projection_method == "Equidistant" else fr"Nonlinear Projection $p_y$")
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
                     title:    str = None,
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
    :param title:
    :param mode: "sRGB", "Illuminance" or "Irradiance" (string)
    """
    # check types
    pc.check_type("im", im, RImage)
    pc.check_type("block", block, bool)
    pc.check_type("flip", flip, bool)
    pc.check_type("log", log, bool)
    pc.check_type("mode", mode, str)
    pc.check_type("title", title, str | None)
    pc.check_if_element("mode", mode, RImage.display_modes)

    # check if cut parameter provided
    if "x" not in kwargs and "y" not in kwargs:
        raise RuntimeError("Provide an x or y parameter to the RImageCutPlot function.")

    # make cut
    s, Imd = im.cut(mode, log=log, imc=imc, **kwargs)

    # title from image object
    text = im.get_long_desc(fallback="")

    # y label
    if mode in ["Irradiance", "Illuminance"]:
        punit, aname, srname, p = ("W", "Irradiance", "Radiant", im.get_power()) if mode == "Irradiance"\
                                  else ("lm", "Illuminance", "Luminous", im.get_luminous_power())
    
        if im.coordinate_type == "Cartesian":
            ylabel = f"{aname} in {punit}/mm²"
        elif im.projection_method == "Equal-Area":
            ylabel = f"{srname} Intensity in {punit}/sr"
        else:
            ylabel = f"Normalized Projected {srname} Intensity"
        
    else:
        ylabel = mode

    # normalize values for sphere projections that are not Equal-Area
    # (since the values are incorrect anyway)
    if im.coordinate_type == "Polar" and im.projection_method != "Equal-Area"\
            and mode in ["Irradiance", "Illuminance"]:
        imax = np.max(Imd)
        if imax:
            Imd /= imax
    
    # assign labels and units
    cut = "x" if "x" in kwargs else "y"
    dir_ = "y" if "x" in kwargs else "x"
    cut_val = kwargs[cut]
    sunit = "rad" if im.projection_method == "Equidistant" else ""
    qnt = fr"$\theta_{dir_}$" if im.projection_method == "Equidistant" else fr"Nonlinear Projection $p_{dir_}$"
    text += f"\nCut at {qnt} = {cut_val:.5g} {sunit}" if im.coordinate_type == "Polar"\
            else f"\nCut at {cut} = {cut_val:.5g} mm"

    # overwrite title if provided
    if title is not None:
        text = title
    
    # better fonts to make everything look more professional
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'

    # convert angles to degrees with projection mode equidistant
    if im.coordinate_type == "Polar" and im.projection_method == "Equidistant":
        s = np.rad2deg(s)

    # new figure
    fig = plt.figure()

    # enforce rgb colors for rgb modes
    colors = ["r", "g", "b"] if mode.startswith("sRGB") else [None, None, None]

    # plot curve(s)
    [plt.plot(s, Imd[i], color=colors[i]) for i, _ in enumerate(Imd)]

    # flip if desired
    if flip:
        plt.gca().invert_xaxis()
    
    # plot labels
    plt.ylabel(ylabel)
    if im.coordinate_type == "Polar":
        plt.xlabel(fr"{qnt} / °" if im.projection_method == "Equidistant" else qnt)
    else:
        plt.xlabel(f"{dir_} / mm")

    # turn off labels with non linear axes
    if im.coordinate_type == "Polar" and im.projection_method != "Equidistant":
        fig.axes[0].set_xticklabels([])

    # major and finer grid
    plt.grid(visible=True, which='major')
    plt.grid(visible=True, which='minor', color='gainsboro', linestyle='--')
    plt.minorticks_on()

    # toggle log mode
    if log and any(np.any(Imdi > 0) for Imdi in Imd):
        plt.yscale('log')

    # add RGB legend if needed
    if mode.startswith("sRGB"):
        plt.legend(["R", "G", "B"])

    # set title
    plt.title(text)

    # show image
    plt.show(block=block)
    plt.pause(0.1)
