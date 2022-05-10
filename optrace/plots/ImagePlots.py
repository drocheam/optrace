
"""
Functions for plotting of Source/Detector Images.
Plotting Modes include Irradiance, Illuminance and RGB.
All Modes can be shown in linear or logarithmic scaling

"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import copy

from optrace.tracer.Image import *
from optrace.tracer.Detector import *
from optrace.tracer.RaySource import *

def DetectorPlot(Im: Image, mode: str = Image.display_modes[0], **kwargs) -> None:
    """
    Plot an Detector Image in Irradiance, Illuminance or RGB Mode.
    Also displays image position and flux on detector.

    :param Im: Image object
    :param mode: mode from Image.modes (string)
    """
    # index = f"{Im.index}" if Im.index is not None else ""
    # text = f"{Detector.abbr}{index} at z = {Im.z:.5g} mm"
   
    text = Im.desc

    match mode:
        case "Irradiance":      
            clabel = "Irradiance in W/mm²"
            text += f"\n Total Radiant Flux at Detector: {Im.getPower():.5g} W"

        case "Illuminance":    
            clabel = "Illuminance in lm/mm²"
            text += f"\n Total Luminous Flux at Detector: {Im.getLuminousPower():.5g} lm"

        case _:                 
            clabel = mode
            text += f"\nMode: {mode}"

    showImage(Im, clabel=clabel, text=text, mode=mode, **kwargs)


def SourcePlot(Im: Image, mode: str = Image.display_modes[0], **kwargs) -> None:
    """
    Plot an Source Image in Irradiance, Illuminance or RGB Mode.
    Also displays image position and flux on detector.

    :param Im: Image object
    :param mode: mode from Image.modes (string)
    """
    # index = f"{Im.index}" if Im.index is not None else ""
    # text = f"{RaySource.abbr}{index} at z = {Im.z:.5g} mm"
    text = Im.desc
    
    match mode:
        case "Irradiance":      
            clabel = "Radiant Emittance in W/mm²"
            text += f"\n Total Radiant Flux from Source: {Im.getPower():.5g} W"

        case "Illuminance":    
            clabel = "Luminous Emittance in lm/mm²"
            text += f"\n Total Luminous Flux from Source: {Im.getLuminousPower():.5g} lm"

        case _:                 
            clabel = mode
            text += f"\nMode: {mode}"

    showImage(Im, clabel=clabel, text=text, mode=mode, **kwargs)


def showImage(Im_in:    Image,
              Imc:      np.ndarray=None,
              block:    bool = False,
              log:      bool = False,
              flip:     bool = False,
              text:     str = "",
              clabel:   str = "",
              mode:     str = "sRGB")\
        -> None:
    """
    Shared plotting function for DetectorImage() and SourceImage(), call these functions for plotting instead.

    :param Im_in: Image from Raytracer SourceImage/DetectorImage function, numpy 3D array shape (N, N, 5)
    :param Imc: precalculated Image (np.ndarray) to display. If not specified it is calculated by parameter 'mode'
    :param block: if plot is blocking (bool)
    :param log: if logarithmic values are shown (bool)
    :param text: Title text to display (string)
    :param clabel: label for colorbar (string)
    :param mode: "sRGB", "Illuminance" or "Irradiance" (string)
    """

    Im = Imc.copy() if Imc is not None else Im_in.getByDisplayMode(mode, log=log)

    # fall back to linear values when all pixels have the same value
    if log and (np.max(Im) == np.min(Im) or mode == "Outside sRGB Gamut"):
        log = False

    # rotate 180 deg
    if flip:
        Im = np.fliplr(np.flipud(Im))
        extent = Im_in.extent[[1, 0, 3, 2]]
    else:
        extent = Im_in.extent

    # better fonts to make everything look more professional
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'

    # set colormap and color norm
    current_cmap = copy.copy(matplotlib.cm.get_cmap("Greys_r"))
    current_cmap.set_bad(color='black')
    norm = matplotlib.colors.LogNorm() if log else None

    # make image black if all content is zero
    vmin, vmax = None, None
    if np.max(Im) == np.min(Im) == 0:
        vmin, vmax = 0, 1e-16
    elif not log and not mode.startswith("sRGB"):
        vmin = 0

    # plot image
    plt.figure()
    plt.imshow(Im, extent=extent, cmap=current_cmap, aspect="equal", norm=norm, vmin=vmin, vmax=vmax)

    # plot labels
    if Im_in.coordinate_type == "Polar":
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

    # wait a little to render the plot
    if not block:
        plt.pause(0.05)

