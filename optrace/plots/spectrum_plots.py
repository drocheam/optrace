
import numpy as np  # calculations
import matplotlib.pyplot as plt  # actual plotting

from ..tracer import color as mcolor  # color conversions for chromaticity plots
from ..tracer.spectrum import Spectrum, LightSpectrum
from ..tracer.refraction_index import RefractionIndex
from .misc_plots import _show_grid, _save_or_show

from ..property_checker import PropertyChecker as pc
from ..global_options import global_options as go


def refraction_index_plot(ri:         RefractionIndex | list[RefractionIndex],
                          title:      str = "Refraction Index",
                          **kwargs)\
        -> None:
    """
    Plot a single or a list of RefractionIndex

    :param ri: RefractionIndex or list of RefractionIndex
    :param title: title of the plot
    :param kwargs: additional plotting arguments, including:
     legend_off: if the legend should be turned off
     label_off: if the labels should be turned off
     color: a single or a list of colors
     path: if provided, the plot is saved at this location instead of displaying a plot. Specify with file ending.
     sargs: option dictionary for pyplot.savefig
    """
    spectrum_plot(ri, title=title, **kwargs)


def spectrum_plot(spectrum:  Spectrum | list[Spectrum],
                  title:     str = "Spectrum",
                  **kwargs)\
        -> None:
    """
    Plot a single or multiple spectra (LightSpectrum, Spectrum, TransmissionSpectrum)

    :param spectrum: spectrum or a list of spectra
    :param title: title of the plot
    :param kwargs: additional plotting arguments, including:
     legend_off: if the legend should be turned off
     label_off: if the labels should be turned off
     color: a single or a list of colors
     path: if provided, the plot is saved at this location instead of displaying a plot. Specify with file ending.
     sargs: option dictionary for pyplot.savefig
    """
    pc.check_type("title", title, str)
    pc.check_type("spectrum", spectrum, Spectrum | list)
    # set ylabel
    Spec0 = spectrum[0] if isinstance(spectrum, list) and len(spectrum) else spectrum
    ylabel = Spec0.quantity if Spec0 and Spec0.quantity != "" else "value"
    ylabel += f" in {Spec0.unit}" if Spec0 and Spec0.unit != "" else ""

    _spectrum_plot(spectrum, r"$\lambda$ in nm", ylabel, title=title, **kwargs)


def _spectrum_plot(obj:          Spectrum | list[Spectrum],
                   xlabel:       str,
                   ylabel:       str,
                   title:        str,
                   steps:        int = 5000,
                   legend_off:   bool = False,
                   labels_off:   bool = False,
                   color:        str | list[str] = None,
                   path:         str = None,
                   sargs:        dict = {})\
        -> None:
    """Lower level plotting function. Don't use directly"""

    # type checks
    pc.check_type("xlabel", xlabel, str)
    pc.check_type("ylabel", ylabel, str)
    pc.check_type("title", title, str)
    pc.check_type("steps", steps, int)
    pc.check_type("legend_off", legend_off, bool)
    pc.check_type("labels_off", labels_off, bool)
    pc.check_type("color", color, str | list | None)

    # wavelength range
    wl0, wl1 = go.wavelength_range

    fig, (ax1, ax2) = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': [15, 1]})
    _show_grid(ax1)

    # get spectrum values
    def get_val(obj):
        # discrete wavelengths -> not plottable
        if not obj.is_continuous():
            plt.close(fig)  # otherwise a window would stay open
            raise RuntimeError(f"Can't plot discontinuous spectrum_type '{obj.spectrum_type}'.")
        # spectrum_type="Data" -> show actual data values and positions
        # otherwise we would have interpolation issues or fake accuracy
        # this could also lead to an incorrect power for a LightSpectrum in the next part
        elif obj.spectrum_type in ["Data", "Histogram"]:
            return obj._wls, obj._vals
        # default mode, crate wavelength range and just call the object
        else:
            wl = np.linspace(wl0, wl1, steps)
            return wl, obj(wl)

    # single Spectrum
    if not isinstance(obj, list):
        wlp, val = get_val(obj)
        if obj.spectrum_type == "Histogram":
            ax1.stairs(val, wlp, color=color, zorder=10)
        else:
            ax1.plot(wlp, val, color=color)

        # assign title. add total power if it is a LightSpectrum
        if isinstance(obj, LightSpectrum):
            ax1.set_title(obj.get_long_desc(fallback=title) + f"\nTotal Power: {obj.power():.5g}W")
        else:
            ax1.set_title(obj.get_long_desc(fallback=title))

    # multiple spectra
    else:
        for i, obji in enumerate(obj):
            wlp, val = get_val(obji)

            cl = color[i] if color is not None else None

            if obji.spectrum_type == "Histogram":
                axp = ax1.stairs(val, wlp, color=cl, label=obji.get_long_desc(), zorder=10)
                tcolor = axp._original_edgecolor
            else:
                axp = ax1.plot(wlp, val, color=cl, label=obji.get_long_desc())
                tcolor = axp[0]._color

            # labels for each spectrum
            if not labels_off:
                tp = int(i / len(obj) * wlp.shape[0] / 10)
                ax1.text(wlp[tp], val[tp], obji.get_desc(), color=tcolor)

        # add legend and title
        if not legend_off:
            ax1.legend()
        ax1.set_title(title)

    # add wavelength color bar
    # enforce image extent of 1:10 for every wavelength range,
    # otherwise the color bar size changes for different wavelength ranges
    spectral_colormap = go.spectral_colormap if go.spectral_colormap is not None else mcolor.spectral_colormap
    colors = np.array([spectral_colormap(np.linspace(plt.xlim()[0], plt.xlim()[1], 500))[:, :3]])
    ax2.imshow(colors, extent=[*plt.xlim(), 0.1*plt.xlim()[0], 0.1*plt.xlim()[1]], aspect="auto", interpolation="gaussian")

    ax1.set(ylabel=ylabel)
    ax2.set(xlabel=xlabel)
    ax2.get_yaxis().set_visible(False)
    plt.tight_layout()
    _save_or_show(path, sargs)
