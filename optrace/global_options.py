from typing import Callable
from contextlib import contextmanager  # context managers

import numpy as np



class ClassGlobalOptions:

    multithreading: bool = True
    """enable/disable multithreading in the backend. For the GUI threading is always enabled"""

    show_progressbar: bool = True
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
    
    @contextmanager
    def no_progressbar(self) -> None:
        """context manager that toggles the progressbar off"""
        old_setting = self.show_progressbar
        self.show_progressbar = False
        try:
            yield
        finally:
            self.show_progressbar = old_setting
    
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

        if key in ["show_progressbar", "show_warnings", "multithreading"]:
            if not isinstance(val, bool):
                raise TypeError(f"Property '{key}' needs to be of type bool, but is {type(val)}.")

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
