
import matplotlib.pyplot as plt
import matplotlib


# TODO mark found minimum in plot
def AutoFocusDebugPlot(r, vals, block=False):
    """

    """

    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'
    
    plt.figure()
    plt.plot(r, vals)
    plt.plot(r, vals, "r.")
    plt.grid(visible=True, which='major')
    plt.grid(visible=True, which='minor', color='gainsboro', linestyle='--')
    plt.minorticks_on()
    plt.xlabel("z in mm")
    plt.ylabel("cost function")
    plt.show(block=block)

