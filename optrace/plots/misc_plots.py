

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


def autofocus_cost_plot(res:     scipy.optimize.OptimizeResult,
                        afdict:  dict,
                        title:   str = "Autofocus Cost Function",
                        block:   bool = False,
                        silent:  bool = False)\
        -> None:
    """
    Plot a cost function plot for the autofocus results.

    :param res: optimize result from Raytracer.autofocus()
    :param afdict: dictionary from Raytracer.autofocus()
    :param title: title of the plot
    :param block: if the plot should be blocking the execution of the program
    :param silent: if all standard output should be muted
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

# TODO simple plot image function

# TODO test

def convolve_debug_plots(img2:      np.ndarray, 
                         s:         list[float, float], 
                         dbg:       dict,
                         log:       bool = True,
                         log_exp:   float = 3,
                         block:     bool = False)\
        -> None:
    """

    """
    s_psf, psf, s_img, img = dbg["s_psf"], dbg["psf"], dbg["s_img"], dbg["img"]

    # adapt fourier images
    dbg["F_img2"] = dbg["F_img"]*dbg["F_psf"]
    for key in ["F_img2", "F_img", "F_psf"]:
        val = dbg[key]
        val /= np.max(val)
        val = np.clip(val, 0, 1)
        if log:
            val = color.log_srgb_linear(val, exp=log_exp)
        dbg[key] = color.srgb_linear_to_srgb(val)

    # calculate min and max frequencies, 
    # see for x: https://numpy.org/doc/stable/reference/generated/numpy.fft.rfftfreq.html
    # see for y: https://numpy.org/doc/stable/reference/generated/numpy.fft.fftfreq.html
    ny, nx, = img2.shape[:2]
    dx, dy = s[0]/nx, s[1]/ny
    fx0, fx1 = (0, nx/2/dx/nx) if not nx % 2 else (0, (nx-1)/2/dx/nx)
    fy0, fy1 = (-ny/2/dy/ny, (ny/2 - 1)/dy/ny) if not ny % 2 else (-(ny-1)/2/dy/ny, (ny-1)/2/dy/ny)

    extf = [fx0, fx1, fy0, fy1]
    extp = [-s_psf[0]/2, s_psf[0]/2, -s_psf[1]/2, s_psf[1]/2]
    exti = [-s_img[0]/2, s_img[0]/2, -s_img[1]/2, s_img[1]/2]
    exti2 = [-s[0]/2, s[0]/2, -s[1]/2, s[1]/2]

    plt.figure()
    _show_grid()
    plt.title("FT of Input Image")
    plt.imshow(np.fft.ifftshift(dbg["F_img"], axes=0), extent=extf, zorder=10)
    plt.xlabel(r"$f_x$ in 1/mm")
    plt.ylabel(r"$f_y$ in 1/mm")
    plt.show(block=False)

    plt.figure()
    _show_grid()
    plt.title("Cut and interpolated FT of PSF")
    plt.imshow(np.fft.ifftshift(dbg["F_psf"], axes=0), extent=extf, zorder=10)
    plt.xlabel(r"$f_x$ in 1/mm")
    plt.ylabel(r"$f_y$ in 1/mm")
    plt.show(block=False)
    #
    plt.figure()
    _show_grid()
    plt.title("Product of both FTs")
    plt.imshow(np.fft.ifftshift(dbg["F_img2"], axes=0), extent=extf, zorder=10)
    plt.xlabel(r"$f_x$ in 1/mm")
    plt.ylabel(r"$f_y$ in 1/mm")
    plt.show(block=False)

    plt.figure()
    _show_grid()
    plt.title("PSF")
    plt.imshow(psf, extent=extp, zorder=10)
    plt.xlabel("x in mm")
    plt.ylabel("y in mm")
    plt.show(block=False)

    plt.figure()
    _show_grid()
    plt.title("Convoluted Image Result")
    plt.imshow(img2, extent=exti2, zorder=10)
    plt.xlabel("x in mm")
    plt.ylabel("y in mm")
    plt.show(block=False)

    plt.figure()
    _show_grid()
    plt.title("Input Image")
    plt.imshow(img, extent=exti, zorder=10)
    plt.xlabel("x in mm")
    plt.ylabel("y in mm")
    plt.show(block=block)
    plt.pause(0.1)


def abbe_plot(ri:     list[RefractionIndex],
              title:  str = "Abbe Diagram",
              lines:  list = None,
              block:  bool = False,
              silent: bool = False)\
        -> None:
    """
    Create an Abbe Plot for different refractive media.

    :param ri: list of RefractionIndex
    :param title: title of the plot
    :param lines: spectral lines to use for the Abbe number calculation
    :param block: if the plot should block the execution of the program
    :param silent: if all standard output of this function should be muted
    """

    # type checking
    pc.check_type("block", block, bool)
    pc.check_type("silent", silent, bool)
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
    Plot surface profiles in x direction

    if x0 and xe are not provided, it plots each surface over its extent
    Note that it plots each surface at y=y0, its center position, even if it is different for each surface

    :param surface: Surface or List of Surface to plot
    :param x0: x start value for the plot, defaults to the beginning of the surface
    :param xe: x end value for the plot, defaults to the end of the surface
    :param remove_offset: remove the height offset for each surface
    :param title: title of the plot
    :param silent: if all standard output of this function should be muted
    :param block: if the plot should block the execution of the program
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

