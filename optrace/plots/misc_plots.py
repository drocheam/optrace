

import numpy as np  # calculations
import scipy.optimize  # optimize result type
import matplotlib  # plotting library
import matplotlib.pyplot as plt  # actual plotting

# only needed for typing and plotting
from ..tracer.refraction_index import RefractionIndex
from ..tracer.geometry import Surface
from ..tracer.presets import spectral_lines as Lines  # spectral lines for AbbePlot
from ..tracer.misc import PropertyChecker as pc


def autofocus_cost_plot(res:     scipy.optimize.OptimizeResult,
                        afdict:  dict,
                        title:   str = "Autofocus Cost Function",
                        block:   bool = False)\
        -> None:
    """

    :param res:
    :param afdict:
    :param title:
    :param block:
    """

    # type checking
    pc.check_type("res", res, scipy.optimize.OptimizeResult)
    pc.check_type("afdict", afdict, dict)
    pc.check_type("title", title, str)
    pc.check_type("block", block, bool)

    r, vals = afdict["z"], afdict["cost"]

    _set_font()
    plt.figure()

    # evaluation points and connection line of cost function
    plt.plot(r, vals)
    plt.plot(r, vals, "r.")

    # found minimum x and y coordinate
    plt.axvline(res.x, ls="--", color="y")
    plt.axhline(res.fun, ls="--", color="y")

    _show_grid()
    plt.xlabel("z in mm")
    plt.ylabel("cost")
    plt.legend(["cost estimation", "cost values", "found minimum"])
    plt.title(title)
    plt.show(block=block)
    plt.pause(0.1)


def abbe_plot(ri:     list[RefractionIndex],
              title:  str = "Abbe Diagram",
              lines:  list = None,
              block:  bool = False,
              silent: bool = False)\
        -> None:
    """

    :param ri:
    :param title:
    :param lines:
    :param block:
    :param silent:
    :return:
    """

    # type checking
    pc.check_type("block", block, bool)
    pc.check_type("silent", silent, bool)
    pc.check_type("title", title, str)
    pc.check_type("lines", lines, list | None)
    pc.check_type("ri", ri, list)

    lines = Lines.FdC if lines is None else lines
    _set_font()
    plt.figure()
    _show_grid()

    for i, RIi in enumerate(ri):

        # get refraction index and abbe number
        nd = RIi(lines[1])
        Vd = RIi.get_abbe_number(lines)

        # check if dispersive
        if not np.isfinite(Vd):
            if not silent:
                print(f"Ignoring non dispersive material '{RIi.get_desc()}'")
            continue  # skip plotting

        # plot point and label
        sc = plt.scatter(Vd, nd, marker="x", zorder=100)
        col = sc.get_facecolors()[0].tolist()
        plt.text(Vd, nd, RIi.get_desc(), color=col)

    plt.xlim([plt.xlim()[1], plt.xlim()[0]])  # reverse direction of x-axis
    plt.xlabel("Abbe Number V")
    plt.ylabel(r"Refraction Index n ($\lambda$" + f" = {lines[1]}nm)")
    plt.title(title)
    plt.show(block=block)
    plt.pause(0.1)


def surface_profile_plot(surface:          Surface | list[Surface],
                         x0:               float = None,
                         xe:               float = None,
                         remove_offset:    bool = False,
                         title:            str = "Surface Profile",
                         silent:           bool = False,
                         block:            bool = False)\
        -> None:
    """
    plot surface profiles in x direction

    if x0 and xe are not provided it plots each surface over its extent
    Note that it plots each surface at y=y0, its center position, even if it is different for each surface

    :param surface:
    :param x0:
    :param xe:
    :param remove_offset:
    :param title:
    :param silent:
    :param block:
    :return:
    """

    # type checking
    pc.check_type("surface", surface, Surface | list)
    pc.check_type("x0", x0, float | int | None)
    pc.check_type("xe", xe, float | int | None)
    pc.check_type("title", title, str)
    pc.check_type("remove_offset", remove_offset, bool)
    pc.check_type("block", block, bool)
    pc.check_type("silent", silent, bool)

    Surf_list = [surface] if isinstance(surface, Surface) else surface  # enforce list even for one element
    legends = []  # legend entries

    _set_font()
    plt.figure()

    plottable = False  # if at least one plot is plottable

    for i, Surfi in enumerate(Surf_list):
        # create x range and get Surface values
        xe = Surfi.extent[1] if xe is None else xe
        x0 = Surfi.extent[0] if x0 is None else x0
        x = np.linspace(x0, xe, 2000)  # 2000 values should be enough

        vals = Surfi.get_values(x, np.full_like(x, Surfi.pos[1]))  # from center in x direction
        vals[~Surfi.get_mask(x, np.full_like(x, Surfi.pos[1]))] = np.nan  # mask invalid values

        # remove position offset
        if remove_offset:
            vals -= Surfi.pos[2]

        if np.any(np.isfinite(vals)):
            plottable = True

        plt.plot(x, vals)
        # legend entry is long_desc, desc or fallback
        legends.append(Surfi.get_long_desc(fallback=f"Surface {i}"))

    # print info message
    if Surf_list and not plottable and not silent:
        print("no plottable surface for this x region")

    _show_grid()
    plt.xlabel("x in mm")
    plt.ylabel("z in mm")
    plt.title(title)
    plt.legend(legends)
    plt.show(block=block)
    plt.pause(0.1)


def _show_grid(what=plt) -> None:
    """active major and minor grid lines, while minor are dashed and less visible"""
    what.grid(visible=True, which='major')
    what.grid(visible=True, which='minor', color='gainsboro', linestyle='--')
    what.minorticks_on()


def _set_font() -> None:
    """set the font to something professionally looking (similar to Times New Roman)"""
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'
