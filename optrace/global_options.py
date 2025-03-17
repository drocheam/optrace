from typing import Callable
from contextlib import contextmanager  # context managers

import numpy as np
import threading


class ClassGlobalOptions:

    multithreading: bool = True
    """enable/disable multithreading in the backend. For the GUI threading is always enabled"""

    show_progress_bar: bool = True
    """show a progressbar for backend activities (rendering, tracing, convolving, focussing, ...)"""

    show_warnings: bool = True
    """if optrace should print warnings"""

    wavelength_range: tuple[float, float] = [380., 780.]
    """wavelength range in nm for light generation and simulation. Must include the range 380-780nm"""

    spectral_colormap: Callable[[np.ndarray], np.ndarray] = None
    """spectral colormap function that takes wavelengths in nm as numpy.ndarray and returns a (N, 4) 
       shaped numpy array with RGBA colors with values in range 0-1.
       By default (Value = None) uses a colormap with hues that would be visible by the human eye.
       It is useful to define a different colormap when working in UV or IR."""

    ui_dark_mode: bool = True
    """dark mode for GUI elements (TraceGui, matplotlib windows etc."""
    
    plot_dark_mode: bool = True
    """dark mode for the pyplots. Background is dark and black and white colors are inverted."""

    def __init__(self):
        self.plot_dark_mode_handler = None
        self.ui_dark_mode_handler = None

    @contextmanager
    def no_progress_bar(self) -> None:
        """context manager that toggles the progressbar off"""
        old_setting = self.show_progress_bar
        self.show_progress_bar = False
        try:
            yield
        finally:
            self.show_progress_bar = old_setting
    
    @contextmanager
    def no_warnings(self) -> None:
        """context manager that toggles the warning off. NOT RECOMMENDED AS WARNINGS CAN BE IMPORTANT"""
        old_setting = self.show_warnings
        self.show_warnings = False
        try:
            yield
        finally:
            self.show_warnings = old_setting

    def __setattr__(self, key, val):

        if key in ["show_progress_bar", "show_warnings", "multithreading"]:
            if not isinstance(val, bool):
                raise TypeError(f"Property '{key}' needs to be of type bool, but is {type(val)}.")

        if key == "ui_dark_mode":
            if not isinstance(val, bool):
                raise TypeError(f"Property '{key}' needs to be of type bool, but is {type(val)}.")

            assert threading.current_thread() is threading.main_thread(),\
                "Change of ui_dark_mode must be done in main thread!!!"

            if self.ui_dark_mode_handler is not None:
                self.ui_dark_mode_handler(val)

        if key == "plot_dark_mode":

            if not isinstance(val, bool):
                raise TypeError(f"Property '{key}' needs to be of type bool, but is {type(val)}.")

            if self.plot_dark_mode_handler is not None:
                self.plot_dark_mode_handler(val)

        elif key == "wavelength_range":
            if not isinstance(val, list | tuple):
                raise TypeError(f"Property '{key}' needs to be of types list or tuple, but is {type(val)}.")

            if len(val) != 2:
                raise ValueError(f"{key} must have two elements.")
        
            if val[0] > 380.:
                raise ValueError(f"Property '{key}' needs to be below or equal to 380, but is {val[0]}.")

            if val[1] < 780.:
                raise ValueError(f"Property '{key}' needs to be above or equal to 780, but is {val[1]}.")

        self.__dict__[key] = val

global_options = ClassGlobalOptions()
