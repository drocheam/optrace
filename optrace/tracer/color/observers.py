import pathlib
import numpy as np  # calculations

# CIE 1931 2° colorimetric standard observers
# see optrace/ressources/SOURCE.txt for more information


# load observers from file
_obs_names = ["wl", "x", "y", "z"]
_obs_path = pathlib.Path(__file__).resolve().parent.parent.parent / "ressources" / "observers.csv"
_observers = np.genfromtxt(_obs_path, skip_header=1, delimiter=",", filling_values=0, dtype=np.float64)


def x_observer(wl: np.ndarray) -> np.ndarray:
    """
    CIE 1931 2° colorimetric standard observer x.

    :param wl: wavelength array in nm
    :return: value array
    """
    return np.interp(wl, _observers[:, 0], _observers[:, _obs_names.index("x")], left=0, right=0)


def y_observer(wl: np.ndarray) -> np.ndarray:
    """
    CIE 1931 2° colorimetric standard observer y.

    :param wl: wavelength array in nm
    :return: value array
    """
    return np.interp(wl, _observers[:, 0], _observers[:, _obs_names.index("y")], left=0, right=0)


def z_observer(wl: np.ndarray) -> np.ndarray:
    """
    CIE 1931 2° colorimetric standard observer z.

    :param wl: wavelength array in nm
    :return: value array
    """
    return np.interp(wl, _observers[:, 0], _observers[:, _obs_names.index("z")], left=0, right=0)

