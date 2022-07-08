
"""
Functions for plotting of Source/Detector Images.
Plotting Modes include Irradiance, Illuminance and RGB.
All Modes can be shown in linear or logarithmic scaling

"""

# plotting library
import matplotlib  # plotting library
import matplotlib.pyplot as plt  # actual plotting

import numpy as np  # calculations
import copy  # copying for classes without copy function

from optrace.tracer.RImage import RImage  # RImage type and RImage displaying
from optrace.tracer.BaseClass import BaseClass  # type checking 


def RImagePlot(Im:       RImage,
               Imc:      np.ndarray=None,
               block:    bool = False,
               log:      bool = False,
               flip:     bool = False,
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
    BaseClass._checkType("Im", Im, RImage)
    BaseClass._checkIfIn("mode", mode, RImage.display_modes)

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
    BaseClass._checkType("Im", Im, RImage)
    BaseClass._checkIfIn("mode", mode, RImage.display_modes)

    if "x" not in kwargs and "y" not in kwargs:
        raise RuntimeError("Provide an x or y parameter to the RImageCutPlot function.")

    yim = "x" in kwargs
   
    text = Im.getLongDesc(fallback="")
    text += "\nCut at " + (f'x = {kwargs["x"]:.5g}' if yim else f'y = {kwargs["y"]:.5g}') + " mm"

    match mode:
        case "Irradiance":      clabel = "Irradiance in W/mm²"
        case "Illuminance":     clabel = "Illuminance in lm/mm²"
        case _:                 clabel = mode

    s, Imd = Im.cut(mode, log=log, **(kwargs))

    # better fonts to make everything look more professional
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'

    # plot image
    plt.figure()

    colors = ["r", "g", "b"] if mode.startswith("sRGB") else [None, None, None]
    [plt.plot(s, Imd[i], color=colors[i]) for i, _ in enumerate(Imd)]

    # plot labels
    if Im.coordinate_type == "Polar":
        plt.xlabel(r"$\theta_y$ / °") if yim else plt.xlabel(r"$\theta_x$ / °")
    else:
        plt.xlabel("y / mm") if yim else plt.xlabel("x / mm")

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

