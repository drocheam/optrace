import matplotlib
import matplotlib.style
import pyface.gui
from pyface.qt import QtGui, QtCore
import qdarktheme

from .image_plots import image_plot, image_profile_plot
from .misc_plots import focus_search_cost_plot, abbe_plot, surface_profile_plot, block
from .spectrum_plots import spectrum_plot, refraction_index_plot
from .chromaticity_plots import chromaticity_norms, chromaticities_cie_1931, chromaticities_cie_1976

from ..global_options import global_options

# increase dpi if in "inline" backend, otherwise the resolution is way too low in IDEs like spyder
if "inline" in matplotlib.rcParams["backend"]:  # pragma: no cover
    matplotlib.rcParams["figure.dpi"] = 300  # pragma: no cover
else:
    matplotlib.rcParams["backend"] = "qtagg"

# create and assign handler for Qt GUI and pyplot style light/dark mode changes
# set by global_options.plot_dark_mode and global_options.ui_dark_mode

def _plot_dark_mode_handler(val: bool):

    if val:
        matplotlib.style.use("dark_background")
        matplotlib.rcParams['figure.facecolor'] ='#333'
        matplotlib.rcParams['savefig.facecolor'] ='#333'
        matplotlib.rcParams['axes.facecolor'] ='#1f2023'
        matplotlib.rcParams['axes.edgecolor'] ='#555'
        matplotlib.rcParams['legend.facecolor'] ='#333333'
    else:
        matplotlib.style.use("default")

    # without this the settings get unset for some reason (bug?)
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'

def _ui_dark_mode_handler(val: bool):

    if val:
        qdarktheme.setup_theme("dark")
        # set button palette explicitly so pyplot qt toolbar icons get colored correctly
        pal = QtCore.QCoreApplication.instance().palette()
        pal.setColor(QtGui.QPalette.Button, QtGui.QColor("black"))
        pal.setColor(QtGui.QPalette.ButtonText, QtGui.QColor("white"))
        QtCore.QCoreApplication.instance().setPalette(pal)
    else:
        # reset the style to default fusion
        qdarktheme.setup_theme("light", custom_colors={"foreground": "#000000", "primary": "#155bb6"})
        # set button palette explicitly so pyplot qt toolbar icons get colored correctly
        pal = QtCore.QCoreApplication.instance().palette()
        pal.setColor(QtGui.QPalette.Button, QtGui.QColor("white"))
        pal.setColor(QtGui.QPalette.ButtonText, QtGui.QColor("black"))
        QtCore.QCoreApplication.instance().setPalette(pal)

global_options.ui_dark_mode_handler = _ui_dark_mode_handler
global_options.ui_dark_mode_handler(global_options.ui_dark_mode)

global_options.plot_dark_mode_handler = _plot_dark_mode_handler
global_options.plot_dark_mode_handler(global_options.plot_dark_mode)

