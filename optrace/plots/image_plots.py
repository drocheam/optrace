
import matplotlib  # plotting library
import matplotlib.pyplot as plt  # actual plotting

import numpy as np  # calculations

from .misc_plots import _show_grid, _save_or_show

from ..tracer.image import RGBImage, LinearImage
from ..tracer.misc import PropertyChecker as pc  # check types and values
from ..tracer import color
from .. import global_options


def image_plot(im:       LinearImage | RGBImage,
               log:      bool = False,
               flip:     bool = False,
               title:    str = None,
               path:     str = None,
               sargs:    dict = {})\
        -> None:
    """

    :param im: Image to plot
    :param flip: if the image should be flipped
    :param log: if logarithmic values are shown
    :param title: title of the plot
    :param path: if provided, the plot is saved at this location instead of displaying a plot. 
                 Specify a path with file ending.
    :param sargs: option dictionary for pyplot.savefig
    """
    _check_types(im, log, flip, title)

    if isinstance(im, RGBImage) and log:
        Imd = color.log_srgb(im.data)
    else:
        Imd = im.data

    _, _, xlabel, _, _, ylabel, _, _, zlabel, text = _get_labels(im, im.quantity, log)

    if im.projection not in ["Equal-Area", None] and im.quantity in ["Irradiance", "Illuminance"]:
        imax = np.max(Imd)
        if imax:
            Imd /= imax

    if title is not None:
        text = title

    # fall back to linear values when all pixels have the same value
    if log and (np.max(Imd) == np.min(Imd) or im.quantity == "Outside sRGB Gamut"):
        log = False

    # get extent. Convert to degrees with projection "equidistant"
    extent = im.extent
    if im.projection == "Equidistant":
        extent = np.rad2deg(extent)

    # rotate 180 deg
    if flip:
        Imd = np.fliplr(np.flipud(Imd))
        extent = extent[[1, 0, 3, 2]]

    # set colormap and color norm
    current_cmap = matplotlib.colormaps["Greys_r"].copy()
    current_cmap.set_bad(color='black')
    norm = matplotlib.colors.LogNorm() if log and Imd.ndim == 2 else None

    # make image black if all content is zero
    vmin, vmax = None, None
    if np.max(Imd) == np.min(Imd) == 0:
        vmin, vmax = 0, 1e-16
    elif not log and not im.quantity.startswith("sRGB"):
        vmin = 0

    # plot image
    fig = plt.figure()

    # show ticks but hide grid
    _show_grid()
    plt.grid(visible=False, which="major")
    plt.grid(visible=False, which="minor")
   
    plt.imshow(Imd, extent=extent, cmap=current_cmap, aspect="equal", norm=norm, 
               vmin=vmin, vmax=vmax, origin="lower")

    # plot labels
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # hide numbers from axis if the projection is not linear
    if im.projection not in ["Equidistant", "Orthographic", None]:
        fig.axes[0].set_xticklabels([])
        fig.axes[0].set_yticklabels([])

    # add colorbar for some modes
    if not isinstance(im, RGBImage) and im.quantity not in ["Lightness (CIELUV)", "Outside sRGB Gamut"]\
            and im.quantity != "":
        clb = plt.colorbar(orientation='horizontal', shrink=0.6)
        clb.ax.set_xlabel(zlabel)
   
    # title
    plt.title(text)

    # show image
    plt.tight_layout()
    _save_or_show(path, sargs)


def image_profile_plot(im:       RGBImage | LinearImage,
                       log:      bool = False,
                       flip:     bool = False,
                       title:    str = None,
                       path:     str = None,
                       sargs:    dict = {},
                       **kwargs)\
        -> None:
    """

    :param im: Image to plot
    :param log: if logarithmic values are shown
    :param flip: if the image should be flipped
    :param title: title of the plot
    :param path: if provided, the plot is saved at this location instead of displaying a plot. 
                 Specify a path with file ending.
    :param sargs: option dictionary for pyplot.savefig
    :param kwargs: additional keyword arguments for Image.profile
    """
    _check_types(im, log, flip, title)
    
    # make profile
    if isinstance(im, RGBImage) and log:
        im2 = RGBImage(color.log_srgb(im.data), extent=im.extent)
        s, Imd = im2.profile(**kwargs)
    else:
        s, Imd = im.profile(**kwargs)

    # get labels
    cut_dim, cut_val = ("x", kwargs["x"]) if "x" in kwargs else ("y", kwargs["y"])
    xname, xunit, xlabel, yname, yunit, ylabel, _, _, zlabel, text = _get_labels(im, im.quantity, log, cut_dim, cut_val)

    # normalize values for sphere projections that are not Equal-Area
    # (since the values are incorrect anyway)
    if im.projection not in ["Equal-Area", None] and im.quantity in ["Irradiance", "Illuminance"]:
        imax = np.max(Imd)
        if imax:
            Imd /= imax

    # overwrite title if provided
    if title is not None:
        text = title

    # convert angles to degrees with projection mode equidistant
    if im.projection == "Equidistant":
        s = np.rad2deg(s)

    # new figure
    fig = plt.figure()
    _show_grid()

    # enforce rgb colors for rgb modes
    if isinstance(im, RGBImage):
        colors = ["#f30", "#2b3", "#08e"] if global_options.plot_dark_mode else ["r", "g", "b"] 
    else:
        colors = [None, None, None]

    # plot curve(s)
    [plt.stairs(Imd[i], s, color=colors[i], zorder=10) for i, _ in enumerate(Imd)]

    # flip if desired
    if flip:
        plt.gca().invert_xaxis()
    
    # plot labels
    plt.xlabel(xlabel if "y" in kwargs else ylabel)
    plt.ylabel(zlabel)

    # turn off labels with non linear axes
    if im.projection not in ["Equidistant", None, "Orthographic"]:
        fig.axes[0].set_xticklabels([])

    # toggle log mode
    if log and any(np.any(Imdi > 0) for Imdi in Imd) and len(Imd) == 1:
        plt.yscale('log')

    # add RGB legend if needed
    if isinstance(im, RGBImage):
        plt.legend(["R", "G", "B"])

    # set title
    plt.title(text)

    # show image
    plt.tight_layout()
    _save_or_show(path, sargs)


def _check_types(im, log, flip, title) -> None:
    """check types for r_image plots"""
    pc.check_type("im", im, LinearImage | RGBImage)
    pc.check_type("flip", flip, bool)
    pc.check_type("log", log, bool)
    pc.check_type("title", title, str | None)


def _get_labels(im:             RGBImage | LinearImage, 
                mode:           str, 
                log:            bool, 
                cut_pos_dim:    str = None, 
                cut_pos_val:    float = None)\
        -> tuple[str, str, str, str, str, str, str, str, str, str]:
    """get plot labels and title"""

    text = im.get_long_desc(fallback="")
    
    # x, y labels
    match im.projection:
        case None | "Orthographic":
            xname, xunit, xlabel = "x", "mm", "x / mm"
            yname, yunit, ylabel = "y", "mm", "y / mm"
        case "Equidistant":
            xname, xunit, xlabel = r"$\theta_x$", "°", r"$\theta_x$ / °"
            yname, yunit, ylabel = r"$\theta_y$", "°", r"$\theta_y$ / °"
        case _:
            xname, xunit, xlabel = fr"Nonlinear Projection $p_x$", "", fr"Nonlinear Projection $p_x$"
            yname, yunit, ylabel = fr"Nonlinear Projection $p_y$", "", fr"Nonlinear Projection $p_y$"
    
    if cut_pos_dim is not None:
        text += ", Profile at " + (f"{xname} = {cut_pos_val:.5g} {xunit}" if cut_pos_dim == "x" \
                else f"{yname} = {cut_pos_val:.5g} {yunit}")
    
    zname, zunit, zlabel = mode, "", mode
    zlabel += ", Logarithmic" if log and mode.startswith("sRGB") else ""

    if zlabel != "":
        text += f"\nMode: {zlabel}"
        if im.projection is not None:
            text += ", " 

    if im.projection is not None:
        text += f"{im.projection} Projection"
    
    if im.limit is not None:
        text += f", {im.limit:.2f}µm Resolution Filter"

    if mode in ["Irradiance", "Illuminance"]:
        punit, aname, srname, p = ("W", "Irradiance", "Radiant", np.sum(im.data)*im.Apx) if mode == "Irradiance"\
                                  else ("lm", "Illuminance", "Luminous", np.sum(im.data)*im.Apx)
    
        match im.projection:
            case "Equal-Area":
                zname, zunit, zlabel = f"{srname} Intensity", f"{punit}/sr", f"{srname} Intensity in {punit}/sr"
            case None:
                zname, zunit, zlabel = aname, f"{punit}/mm²", f"{aname} in {punit}/mm²"
            case _:
                zname, zunit, zlabel = f"Normalized Projected {srname} Intensity", "",\
                                       f"Normalized Projected {srname} Intensity"
        
        text += f"\n Total {srname} Flux: {p:.5g} {punit}"

    return xname, xunit, xlabel, yname, yunit, ylabel, zname, zunit, zlabel, text
