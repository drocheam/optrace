import pathlib
import matplotlib
import matplotlib.style
import pyface.gui  # required here so QApplication instance is created
from pyface.qt import QtGui, QtCore

from ..global_options import global_options

# increase dpi if in "inline" backend, otherwise the resolution is way too low in IDEs like spyder
if "inline" in matplotlib.rcParams["backend"]:  # pragma: no cover
    matplotlib.rcParams["figure.dpi"] = 300  # pragma: no cover
else:
    matplotlib.rcParams["backend"] = "qtagg"

# path for icon and stylesheet
ui_data_dir = pathlib.Path(__file__).resolve().parent.parent / "resources" / "ui"

# get stylesheet
with open(ui_data_dir / "stylesheet.css", "r", encoding="utf-8") as file:
    _SHEET = file.read()

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


def _ui_dark_mode_handler(dark: bool):

    # get app instance
    app = QtGui.QApplication.instance()
    if app is None:
        raise RuntimeError("No QApplication instance detected.")

    # define colors
    style = {}
    style["bg"] =               "#202124" if dark else "#f0f0f0"
    style["panel"] =            "#37373a" if dark else "#ffffff"
    style["fg"] =               "#eeeeee" if dark else "#000000"
    style["fg_disabled"] =      "#999999" if dark else "#555555"
    style["border"] =           "#555555" if dark else "#b0b0b0"
    style["button"] =           "#333333" if dark else "#f0f0f0"
    style["hover"] =            "#3f3f3f" if dark else "#dadada"
    style["hover_selection"] =  "#63a3fe" if dark else "#369ddd"
    style["selection"] =        "#558bdb" if dark else "#308cc6"
    style["code"] =             "#a2a2a2" if dark else "#ffffff"
    style["slider"] =           "#4a4a4a" if dark else "#f0f0f0"

    # define icon paths. Qt expects POSIX strings
    mode = "dark" if dark else "light"
    style["checkmark"] = (ui_data_dir / f"ui_check_{mode}_mode.svg").as_posix()
    style["upmark"] = (ui_data_dir / f"ui_up_{mode}_mode.svg").as_posix()
    style["downmark"] = (ui_data_dir / f"ui_down_{mode}_mode.svg").as_posix()

    # replace variables
    sheet = _SHEET[:]
    for key, val in style.items():
        sheet = sheet.replace(f"<{key}>", val)

    # apply stylesheet to Fusion
    app.setStyle("Fusion")
    app.setStyleSheet(sheet)

    # set button palette explicitly so pyplot qt toolbar icons get colored correctly
    pal = QtCore.QCoreApplication.instance().palette()
    pal.setColor(QtGui.QPalette.Button, QtGui.QColor("black" if dark else "white"))
    pal.setColor(QtGui.QPalette.ButtonText, QtGui.QColor("white" if dark else "black"))
    QtCore.QCoreApplication.instance().setPalette(pal)

global_options.ui_dark_mode_handler = _ui_dark_mode_handler
global_options.ui_dark_mode_handler(global_options.ui_dark_mode)

global_options.plot_dark_mode_handler = _plot_dark_mode_handler
global_options.plot_dark_mode_handler(global_options.plot_dark_mode)

