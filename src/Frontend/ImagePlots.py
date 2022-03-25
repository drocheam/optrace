
"""
Functions for plotting of Source/Detector Images.
Plotting Modes include Irradiance, Illuminance and RGB.
All Modes can be shown in linear or logarithmic scaling

"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import copy
import PyQt5

from Backend.Image import Image


def DetectorPlot(Im:        Image,
                 block:     bool = False,
                 log:       bool = False,
                 flip:      bool = False,
                 mode:      str = "sRGB")\
        -> None:
    """
    Plot an Detector Image in Irradiance, Illuminance or RGB Mode.
    Also displays image position and flux on detector.

    :param Im: Image object
    :param block: if plot is blocking (bool)
    :param log: if logarithmic values are shown (bool)
    :param mode: "sRGB", "Illuminance" or "Irradiance" (string)
    """
    clabel = "Irradiance in W/mm²" if mode == "Irradiance" else "Illuminance in lm/mm²"
    text = f"Detector at z = {Im.z:.5g} mm"

    if mode == "Irradiance":
        text += f"\n Total Radiant Flux at Detector: {Im.getPower():.5g} W"
    else:
        text += f"\n Total Luminous Flux at Detector: {Im.getLuminousPower():.5g} lm"

    showImage(Im, block=block, log=log, flip=flip, clabel=clabel, text=text, mode=mode)


def SourcePlot(Im:      Image,
               block:   bool = False,
               log:     bool = False,
               flip:    bool = False,
               mode:    str = "sRGB")\
        -> None:
    """
    Plot an Source Image in Irradiance, Illuminance or RGB Mode.
    Also displays image position and flux on detector.

    :param Im: Image object
    :param block: if plot is blocking (bool)
    :param log: if logarithmic values are shown (bool)
    :param mode: "sRGB", "Illuminance" or "Irradiance" (string)
    """

    clabel = "Radiant Emittance in W/mm²" if mode == "Irradiance" else "Luminous Emittance in lm/mm²"
    index = f" {Im.index}" if Im.index else ""
    text = f"Source{index} at z = {Im.z:.5g} mm"

    if mode == "Irradiance":
        text += f"\n Total Radiant Flux from Source: {Im.getPower():.5g} W"
    else:
        text += f"\n Total Luminous Flux from Source: {Im.getLuminousPower():.5g} lm"

    showImage(Im, block=block, log=log, flip=flip, clabel=clabel, text=text, mode=mode)


def showImage(Im_in:    Image,
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
    :param block: if plot is blocking (bool)
    :param log: if logarithmic values are shown (bool)
    :param text: Title text to display (string)
    :param clabel: label for colorbar (string)
    :param mode: "sRGB", "Illuminance" or "Irradiance" (string)
    """

    match mode:
        case "Irradiance":
            Im = Im_in.getIrradiance()

        case "Illuminance":
            Im = Im_in.getIlluminance()

        case "sRGB":
            Im = Im_in.getRGB(log=log)

        case _:
            raise ValueError("Invalid image mode.")

    # fall back to linear values when all pixels have the same value
    if log and np.max(Im) == np.min(Im):
        log = False

    # rotate 180 deg
    if flip:
        Im = np.fliplr(np.flipud(Im))
        # extent needs to be adapted so points are in the correct position
        extent = Im_in.extent[[1, 0, 3, 2]]
    else:
        extent = Im_in.extent

    # enforce plotting backend to show plots interactive and in separate windows
    matplotlib.use('Qt5Agg')

    # better fonts to make everything look more professional
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'

    # raise window to front when showing plot
    matplotlib.rcParams['figure.raise_window'] = True

    plt.figure()

    # set colormap and color norm
    current_cmap = copy.copy(matplotlib.cm.get_cmap("Greys_r"))
    current_cmap.set_bad(color='black')
    norm = matplotlib.colors.LogNorm() if log else None

    # make image black if all content is zero
    vmin, vmax = None, None
    if np.max(Im) == np.min(Im) == 0:
        vmin, vmax = 0, 1e-16

    # plot image
    plt.imshow(Im, extent=extent, cmap=current_cmap, aspect="equal", norm=norm, vmin=vmin, vmax=vmax)

    # plot labels
    if Im_in.image_type == "Polar":
        plt.xlabel(r"$\theta_x$ / °")
        plt.ylabel(r"$\theta_y$ / °")
    else:
        plt.xlabel("x / mm")
        plt.ylabel("y / mm")

    if mode != "sRGB":
        clb = plt.colorbar(orientation='horizontal', shrink=0.6)
        clb.ax.set_xlabel(clabel)
    plt.title(text)

    # show image
    plt.show(block=block)

    # wait a little to render the plot
    if not block:
        plt.pause(0.05)
