
import copy

import numpy as np  # calculations
import scipy.optimize  # optimize result type
import matplotlib  # plotting library
import matplotlib.pyplot as plt  # actual plotting

# only needed for typing and plotting
from ..tracer.refraction_index import RefractionIndex
from ..tracer.geometry import Surface
from ..tracer.presets import spectral_lines as Lines  # spectral lines for AbbePlot
from ..tracer.misc import PropertyChecker as pc
from ..tracer import misc as misc
from ..tracer import color as color
from ..tracer import RImage


def autofocus_cost_plot(res:     scipy.optimize.OptimizeResult,
                        afdict:  dict,
                        title:   str = "Autofocus Cost Function",
                        fargs:   dict = None,
                        block:   bool = False,
                        silent:  bool = False,
                        path:    str = None,
                        sargs:   dict = None)\
        -> None:
    """
    Plot a cost function plot for the autofocus results.

    :param res: optimize result from Raytracer.autofocus()
    :param afdict: dictionary from Raytracer.autofocus()
    :param title: title of the plot
    :param fargs: keyword argument dictionary for pyplot.figure() (e.g. figsize)
    :param block: if the plot should be blocking the execution of the program
    :param silent: if all standard output should be muted
    :param path: if provided, the plot is saved at this location instead of displaying a plot. 
                 Specify a path with file ending.
    :param sargs: option dictionary for pyplot.savefig
    """

    # type checking
    pc.check_type("res", res, scipy.optimize.OptimizeResult)
    pc.check_type("afdict", afdict, dict)
    pc.check_type("title", title, str)
    pc.check_type("block", block, bool)

    if afdict["z"] is None or afdict["cost"] is None:
        if not silent:
            print('Parameters missing in focus dict. For mode "Position Variance" set '
                  'autofocus("Position Variance", ..., return_cost=True) when'
                  ' wanting to plot the debug plot.')
        return

    r, vals = afdict["z"], afdict["cost"]

    fargs = dict() if fargs is None else fargs
    plt.figure(**fargs)

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
    plt.tight_layout()
    _save_or_show(block, path, sargs)


def image_plot(img:     np.ndarray | RImage | str, 
               s:       list[float, float],
               flip:    bool = False,
               fargs:   dict = None,
               title:   str = "",
               block:   bool = False,
               path:    str = None,
               sargs:   dict = None)\
        -> None:
    """
    Plot an image (array or path).

    img needs to be a valid path or a sRGB numpy array with value range [0, 1].
    Note that image arrays will get normalized to use the whole range.
    There will be always a sRGB image shown and not values linear to the intensity.

    :param img: image path or image array
    :param s: image side lengths in mms (list of two elements, with first element being the x-length)
    :param flip: flip the image (rotate by 180 degrees)
    :param fargs: keyword argument dictionary for pyplot.figure() (e.g. figsize)
    :param title: optional title of the plot
    :param block: if the plot window should be blocking
    :param path: if provided, the plot is saved at this location instead of displaying a plot. 
                 Specify a path with file ending.
    :param sargs: option dictionary for pyplot.savefig
    """
    pc.check_type("img", img, np.ndarray | str | RImage)
    pc.check_type("s", s, list | tuple)
    pc.check_type("title", title, str)
    pc.check_type("flip", flip, bool)
    pc.check_type("block", block, bool)

    if isinstance(img, str):
        img_ = misc.load_image(img)
    elif isinstance(img, RImage):
        img_ = img.get("sRGB (Absolute RI)")
    else:
        img_ = img.copy()

        if img_.ndim == 2:
            img_ = np.repeat(img_[:, :, np.newaxis], 3, axis=2)
            
            if (imax := np.max(img_)):
                img_ /= imax

            img_ = color.srgb_linear_to_srgb(img_)

    # adapt extent so the coordinates are at the center of pixels
    extent = np.array([-s[0]/2, s[0]/2, -s[1]/2, s[1]/2])
    dy, dx = s[1] / img_.shape[0], s[0] / img_.shape[1]
    extent += [-dx/2, dx/2, -dy/2, dy/2]

    # rotate 180 deg
    if flip:
        img_ = np.fliplr(np.flipud(img_))
        extent = extent[[1, 0, 3, 2]]
    
    fargs = dict() if fargs is None else fargs
    plt.figure(**fargs)
    
    _show_grid()
    plt.title(title)
    plt.xlabel("x in mm")
    plt.ylabel("y in mm")

    plt.imshow(img_, extent=extent, zorder=10, aspect="equal", origin="lower")

    plt.tight_layout()
    _save_or_show(block, path, sargs)


def abbe_plot(ri:     list[RefractionIndex],
              title:  str = "Abbe Diagram",
              lines:  list = None,
              fargs:  dict = None,
              block:  bool = False,
              silent: bool = False,
              path:   str = None,
              sargs:  dict = None)\
        -> None:
    """
    Create an Abbe Plot for different refractive media.

    :param ri: list of RefractionIndex
    :param title: title of the plot
    :param lines: spectral lines to use for the Abbe number calculation
    :param fargs: keyword argument dictionary for pyplot.figure() (e.g. figsize)
    :param block: if the plot should block the execution of the program
    :param silent: if all standard output of this function should be muted
    :param path: if provided, the plot is saved at this location instead of displaying a plot. 
                 Specify a path with file ending.
    :param sargs: option dictionary for pyplot.savefig
    """

    # type checking
    pc.check_type("block", block, bool)
    pc.check_type("silent", silent, bool)
    pc.check_type("title", title, str)
    pc.check_type("lines", lines, list | None)
    pc.check_type("ri", ri, list)

    lines = Lines.FdC if lines is None else lines
   
    fargs = dict() if fargs is None else fargs
    plt.figure(**fargs)
    _show_grid()

    for i, RIi in enumerate(ri):

        # get refraction index and abbe number
        nd = RIi(lines[1])
        Vd = RIi.abbe_number(lines)

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
    plt.tight_layout()
    _save_or_show(block, path, sargs)


def surface_profile_plot(surface:          Surface | list[Surface],
                         x0:               float = None,
                         xe:               float = None,
                         remove_offset:    bool = False,
                         fargs:            dict = None,
                         title:            str = "Surface Profile",
                         silent:           bool = False,
                         block:            bool = False,
                         path:             str = None,
                         sargs:            dict = None)\
        -> None:
    """
    Plot surface profiles in x direction

    if x0 and xe are not provided, it plots each surface over its extent
    Note that it plots each surface at y=y0, its center position, even if it is different for each surface

    :param surface: Surface or List of Surface to plot
    :param x0: x start value for the plot, defaults to the beginning of the surface
    :param xe: x end value for the plot, defaults to the end of the surface
    :param remove_offset: remove the height offset for each surface
    :param fargs: keyword argument dictionary for pyplot.figure() (e.g. figsize)
    :param title: title of the plot
    :param silent: if all standard output of this function should be muted
    :param block: if the plot should block the execution of the program
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
    pc.check_type("block", block, bool)
    pc.check_type("silent", silent, bool)

    Surf_list = [surface] if isinstance(surface, Surface) else surface  # enforce list even for one element
    legends = []  # legend entries

    fargs = dict() if fargs is None else fargs
    plt.figure(**fargs)

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
    plt.tight_layout()
    _save_or_show(block, path, sargs)


def _show_grid(what=plt) -> None:
    """active major and minor grid lines, while minor are dashed and less visible"""
    what.grid(visible=True, which='major')
    what.grid(visible=True, which='minor', color='gainsboro', linestyle='--')
    what.minorticks_on()


def _save_or_show(block: bool, path: str = None, sargs: dict = None):
    """show a plot (path is None) or store the image of a plot at file given as 'path'"""
    pc.check_type("path", path, str | None)

    if path is None:
        plt.show(block=block)
        plt.pause(0.1)

    else:
        sargs = sargs or dict() 
        plt.savefig(path, **sargs)

