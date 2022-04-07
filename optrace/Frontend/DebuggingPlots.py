
import matplotlib
import matplotlib.pyplot as plt


def AutoFocusDebugPlot(r, vals, rf, ff, title="Focus Finding", block=False):
    """

    :param r:
    :param vals:
    :param rf:
    :param ff:
    :param title:
    :param block:
    """

    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'
    
    plt.figure()
    plt.plot(r, vals)
    plt.plot(r, vals, "r.")
    plt.axvline(rf, ls="--", color="y")
    plt.axhline(ff, ls="--", color="y")
    plt.grid(visible=True, which='major')
    plt.grid(visible=True, which='minor', color='gainsboro', linestyle='--')
    plt.minorticks_on()
    plt.xlabel("z in mm")
    plt.ylabel("cost function")
    plt.legend(["cost estimation", "cost values", "found minimum"])
    plt.title(title)
    plt.show(block=block)

