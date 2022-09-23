
import numpy as np  # calculations
import matplotlib.pyplot as plt  # actual plotting

from ..tracer import color  # color conversions for chromacity plots
from ..tracer.spectrum import Spectrum, LightSpectrum
from ..tracer.refraction_index import RefractionIndex
from .misc_plots import _set_font, _show_grid

from ..tracer.misc import PropertyChecker as pc


def refraction_index_plot(ri:         RefractionIndex | list[RefractionIndex],
                          title:      str = "Refraction Index",
                          **kwargs)\
        -> None:
    """

    :param ri:
    :param title:
    :param kwargs:
    """
    spectrum_plot(ri, title=title, **kwargs)


def spectrum_plot(spectrum:  Spectrum | list[Spectrum],
                  title:     str = "Spectrum",
                  **kwargs)\
        -> None:
    """

    :param spectrum:
    :param title:
    :param kwargs:
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
                   colors:       str | list[str] = None,
                   block:        bool = False)\
        -> None:
    """Lower level plotting function. Don't use directly"""

    # type checks
    pc.check_type("xlabel", xlabel, str)
    pc.check_type("ylabel", ylabel, str)
    pc.check_type("title", title, str)
    pc.check_type("steps", steps, int)
    pc.check_type("legend_off", legend_off, bool)
    pc.check_type("labels_off", labels_off, bool)
    pc.check_type("block", block, bool)
    pc.check_type("colors", colors, str | list | None)

    # wavelength range
    wl0 = color.WL_MIN
    wl1 = color.WL_MAX

    _set_font()
    fig, (ax1, ax2) = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': [15, 1]})

    # get spectrum values
    def get_val(obj):
        # discrete wavelengths -> not plottable
        if not obj.is_continuous():
            plt.close(fig)  # otherwise a window would stay open
            raise RuntimeError(f"Can't plot discontinuous spectrum_type '{obj.spectrum_type}'.")
        # spectrum_type="Data" -> show actual data values and positions
        # otherwise we would have interpolation issues or fake accuracy
        # this could also lead to an incorrect power for a LightSpectrum in the next part
        elif obj.spectrum_type == "Data":
            return obj._wls, obj._vals
        # default mode, crate wavelength range and just call the object
        else:
            wl = np.linspace(wl0, wl1, steps)
            return wl, obj(wl)

    # single Spectrum
    if not isinstance(obj, list):
        wlp, val = get_val(obj)
        ax1.plot(wlp, val, color=colors)

        # assign title. add total power if it is a LightSpectrum
        if isinstance(obj, LightSpectrum):
            total = np.sum(val)*(wlp[1]-wlp[0])
            fig.suptitle("\n" + obj.get_long_desc(fallback=title) + f"\nTotal Power: {total:.5g}W")
        else:
            fig.suptitle("\n" + obj.get_long_desc(fallback=title))

    # multiple spectra
    else:
        lg = []
        for i, obji in enumerate(obj):
            wlp, val = get_val(obji)

            cl = colors[i] if colors is not None else None
            axp = ax1.plot(wlp, val, color=cl)
            lg.append(obji.get_long_desc())

            # labels for each spectrum
            if not labels_off:
                tp = int(i / len(obj) * wlp.shape[0] / 10)
                ax1.text(wlp[tp], val[tp], obji.get_desc(), color=axp[0].get_color())

        # add legend and title
        if not legend_off:
            ax1.legend(lg)
        fig.suptitle("\n" + title)

    _show_grid(ax1)

    # add wavelength color bar
    # enforce image extent of 1:10 for every wavelength range,
    # otherwise the color bar size changes for different wavelength ranges
    colors = np.array([color.spectral_colormap(wl0=plt.xlim()[0], wl1=plt.xlim()[1], N=1000)[:, :3]]) / 255
    ax2.imshow(colors, extent=[*plt.xlim(), 0.1*plt.xlim()[0], 0.1*plt.xlim()[1]], aspect="auto")

    ax1.set(ylabel=ylabel)
    ax2.set(xlabel=xlabel)
    ax2.get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.show(block=block)
    plt.pause(0.1)
