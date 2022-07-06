
"""
Functions for plotting of Source/Detector Images.
Plotting Modes include Irradiance, Illuminance and RGB.
All Modes can be shown in linear or logarithmic scaling

"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import copy

from optrace.tracer.RImage import RImage


def RImagePlot(Im:       RImage,
               Imc:      np.ndarray=None,
               block:    bool = False,
               log:      bool = False,
               flip:     bool = False,
               text:     str = "",
               clabel:   str = "",
               mode:     str = RImage.display_modes[0])\
        -> None:
    """

    :param Im_in: Image from Raytracer SourceImage/DetectorImage function, numpy 3D array shape (N, N, 5)
    :param Imc: precalculated Image (np.ndarray) to display. If not specified it is calculated by parameter 'mode'
    :param block: if plot is blocking (bool)
    :param log: if logarithmic values are shown (bool)
    :param text: Title text to display (string)
    :param clabel: label for colorbar (string)
    :param mode: "sRGB", "Illuminance" or "Irradiance" (string)
    """

    text = Im.getLongDesc(fallback="")

    match mode:
        case "Irradiance":      
            clabel = "Irradiance in W/mm²"
            text += f"\n Total Radiant Flux: {Im.getPower():.5g} W"

        case "Illuminance":    
            clabel = "Illuminance in lm/mm²"
            text += f"\n Total Luminous Flux: {Im.getLuminousPower():.5g} lm"

        case _:                 
            clabel = mode
            text += f"\nMode: {mode}"

    Imd = Imc.copy() if Imc is not None else Im.getByDisplayMode(mode, log=log)

    # fall back to linear values when all pixels have the same value
    if log and (np.max(Imd) == np.min(Imd) or mode == "Outside sRGB Gamut"):
        log = False

    # rotate 180 deg
    if flip:
        Imd = np.fliplr(np.flipud(Imd))
        extent = Im.extent[[1, 0, 3, 2]]
    else:
        extent = Im.extent

    # better fonts to make everything look more professional
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'

    # set colormap and color norm
    current_cmap = copy.copy(matplotlib.cm.get_cmap("Greys_r"))
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
    if Im.coordinate_type == "Polar":
        plt.xlabel(r"$\theta_x$ / °")
        plt.ylabel(r"$\theta_y$ / °")
    else:
        plt.xlabel("x / mm")
        plt.ylabel("y / mm")

    if not mode.startswith("sRGB") and mode != "Outside sRGB Gamut":
        clb = plt.colorbar(orientation='horizontal', shrink=0.6)
        clb.ax.set_xlabel(clabel)
    plt.title(text)

    # show image
    plt.show(block=block)
    plt.pause(0.1)

# TODO test
def RImageCutPlot(Im:       RImage,
                  block:    bool = False,
                  log:      bool = False,
                  flip:     bool = False,
                  text:     str = "",
                  clabel:   str = "",
                  mode:     str = RImage.display_modes[0],
                  **kwargs)\
        -> None:
    """

    :param Im_in: Image from Raytracer SourceImage/DetectorImage function, numpy 3D array shape (N, N, 5)
    :param block: if plot is blocking (bool)
    :param log: if logarithmic values are shown (bool)
    :param text: Title text to display (string)
    :param clabel: label for colorbar (string)
    :param mode: "sRGB", "Illuminance" or "Irradiance" (string)
    """

    text = Im.getLongDesc(fallback="")

    match mode:
        case "Irradiance":      clabel = "Irradiance in W/mm²"
        case "Illuminance":     clabel = "Illuminance in lm/mm²"
        case _:                 clabel = mode

    s, Imd = Im.cut(mode, log=log, **(kwargs))

    # better fonts to make everything look more professional
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'

    yim = "x" in kwargs
    
    # plot image
    plt.figure()

    colors = ["r", "g", "b"] if mode.startswith("sRGB") else [None, None, None]
    [plt.plot(s, Imd[i], color=colors[i]) for i, _ in enumerate(Imd)]

    # plot labels
    if Im.coordinate_type == "Polar":
        plt.xlabel(r"$\theta_x$ / °") if yim else plt.ylabel(r"$\theta_y$ / °")
    else:
        plt.xlabel("x / mm") if yim else plt.ylabel("y / mm")

    if log:
        plt.yscale('log')

    if mode.startswith("sRGB"):
        plt.legend(["R", "G", "B"])

    plt.grid(visible=True, which='major')
    plt.grid(visible=True, which='minor', color='gainsboro', linestyle='--')
    plt.minorticks_on()
    plt.ylabel(clabel)
    plt.title(text)

    # show image
    plt.show(block=block)
    plt.pause(0.1)

