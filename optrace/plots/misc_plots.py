
import copy

import numpy as np  # calculations
import scipy.optimize  # optimize result type
import matplotlib.pyplot as plt  # actual plotting

# only needed for typing and plotting
from ..tracer.refraction_index import RefractionIndex
from ..tracer.geometry import Surface
from ..tracer.presets import spectral_lines as Lines  # spectral lines for AbbePlot
from ..tracer.misc import PropertyChecker as pc
from ..warnings import warning
from ..global_options import global_options


def autofocus_cost_plot(res:     scipy.optimize.OptimizeResult,
                        afdict:  dict,
                        title:   str = "Autofocus Cost Function",
                        path:    str = None,
                        sargs:   dict = {})\
        -> None:
    """
    Plot a cost function plot for the autofocus results.

    :param res: optimize result from Raytracer.autofocus()
    :param afdict: dictionary from Raytracer.autofocus()
    :param title: title of the plot
    :param path: if provided, the plot is saved at this location instead of displaying a plot. 
                 Specify a path with file ending.
    :param sargs: option dictionary for pyplot.savefig
    """

    # type checking
    pc.check_type("res", res, scipy.optimize.OptimizeResult)
    pc.check_type("afdict", afdict, dict)
    pc.check_type("title", title, str)

    if afdict["z"] is None or afdict["cost"] is None:
        warning('Parameters missing in focus dict. For mode "Position Variance" set '
                'autofocus("Position Variance", ..., return_cost=True) when'
                ' wanting to plot the debug plot.')
        return

    r, vals = afdict["z"], afdict["cost"]

    plt.figure()

    # evaluation points and connection line of cost function
    plt.plot(r, vals, label="cost estimation")
    plt.plot(r, vals, "r.", label="cost values")

    # found minimum x and y coordinate
    plt.axvline(res.x, ls="--", color="y", label="found minimum")
    plt.axhline(res.fun, ls="--", color="y")

    _show_grid()
    plt.xlabel("z in mm")
    plt.ylabel("cost")
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    _save_or_show(path, sargs)

def abbe_plot(ri:     list[RefractionIndex],
              title:  str = "Abbe Diagram",
              lines:  list = None,
              path:   str = None,
              sargs:  dict = {})\
        -> None:
    """
    Create an Abbe Plot for different refractive media.

    :param ri: list of RefractionIndex
    :param title: title of the plot
    :param lines: spectral lines to use for the Abbe number calculation
    :param path: if provided, the plot is saved at this location instead of displaying a plot. 
                 Specify a path with file ending.
    :param sargs: option dictionary for pyplot.savefig
    """

    # type checking
    pc.check_type("title", title, str)
    pc.check_type("lines", lines, list | None)
    pc.check_type("ri", ri, list)

    lines = Lines.FdC if lines is None else lines
   
    plt.figure()
    _show_grid()

    for i, RIi in enumerate(ri):

        # get refraction index and abbe number
        nd = RIi(lines[1])
        Vd = RIi.abbe_number(lines)

        # check if dispersive
        if not np.isfinite(Vd):
            warning(f"Ignoring non dispersive material '{RIi.get_desc()}'")
            continue  # skip plotting

        # plot point and label
        sc = plt.scatter(Vd, nd, marker="x", zorder=100)
        col = sc.get_facecolors()[0].tolist()
        plt.text(Vd, nd, RIi.get_desc(), color=col)

    plt.xlim([plt.xlim()[1], plt.xlim()[0]])  # reverse direction of x-axis
    plt.xlabel("Abbe Number V")
    plt.ylabel(r"Refraction Index n ($\lambda$" + f" = {lines[1]}nm)")
    plt.title(title)
    plt.tight_layout()
    _save_or_show(path, sargs)


def surface_profile_plot(surface:          Surface | list[Surface],
                         x0:               float = None,
                         xe:               float = None,
                         remove_offset:    bool = False,
                         title:            str = "Surface Profile",
                         path:             str = None,
                         sargs:            dict = {})\
        -> None:
    """
    Plot surface profiles in x direction

    if x0 and xe are not provided, it plots each surface over its extent
    Note that it plots each surface at y=y0, its center position, even if it is different for each surface

    :param surface: Surface or List of Surface to plot
    :param x0: x start value for the plot, defaults to the beginning of the surface
    :param xe: x end value for the plot, defaults to the end of the surface
    :param remove_offset: remove the height offset for each surface
    :param title: title of the plot
    :param path: if provided, the plot is saved at this location instead of displaying a plot. 
                 Specify a path with file ending.
    :param sargs: option dictionary for pyplot.savefig
    """

    # type checking
    pc.check_type("surface", surface, Surface | list)
    pc.check_type("x0", x0, float | int | None)
    pc.check_type("xe", xe, float | int | None)
    pc.check_type("title", title, str)
    pc.check_type("remove_offset", remove_offset, bool)

    Surf_list = [surface] if isinstance(surface, Surface) else surface  # enforce list even for one element

    plt.figure()

    plottable = False  # if at least one plot is plottable

    for i, Surfi in enumerate(Surf_list):
        # create x range and get Surface values
        xe = Surfi.extent[1] if xe is None else xe
        x0 = Surfi.extent[0] if x0 is None else x0
        x = np.linspace(x0, xe, 2000)  # 2000 values should be enough

        vals = Surfi.values(x, np.full_like(x, Surfi.pos[1]))  # from center in x direction
        vals[~Surfi.mask(x, np.full_like(x, Surfi.pos[1]))] = np.nan  # mask invalid values

        # remove position offset
        if remove_offset:
            vals -= Surfi.pos[2]

        if np.any(np.isfinite(vals)):
            plottable = True

        plt.plot(x, vals, label=Surfi.get_long_desc(fallback=f"Surface {i}"))

    # print info message
    if Surf_list and not plottable:
        warning("no plottable surface for this x region")

    _show_grid()
    plt.xlabel("x in mm")
    plt.ylabel("z in mm")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    _save_or_show(path, sargs)


def block() -> None:
    """show all plots and block application"""
    plt.show(block=True)

def _show_grid(what=plt) -> None:
    """active major and minor grid lines, while minor are dashed and less visible"""
    if global_options.plot_dark_mode:
        what.grid(visible=True, which='major', color="#555")
        what.grid(visible=True, which='minor', color='#333', linestyle='--')
    else:
        what.grid(visible=True, which='major')
        what.grid(visible=True, which='minor', color="gainsboro", linestyle='--')
    what.minorticks_on()


def _save_or_show(path: str = None, sargs: dict = {}):
    """show a plot (path is None) or store the image of a plot at file given as 'path'"""
    pc.check_type("path", path, str | None)
               
    if path is None:
        plt.show(block=False)
        plt.pause(0.1)

    else:
        plt.savefig(path, **sargs)

