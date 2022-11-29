import pathlib
import numpy as np  # calculations


# load observers
_obs_names = ["wl", "x", "y", "z"]
_obs_path = pathlib.Path(__file__).resolve().parent.parent.parent / "ressources" / "observers.csv"
_observers = np.genfromtxt(_obs_path, skip_header=1, delimiter=",", filling_values=0, dtype=np.float64)

# Sources xyz_observers values:
# https://cie.co.at/datatable/cie-1931-colour-matching-functions-2-degree-observer


def x_observer(wl: np.ndarray) -> np.ndarray:
    """
    Eye x observer values (CIE 1931 2° Standard Observer)

    :param wl: wavelength array in nm
    :return: value array
    """
    return np.interp(wl, _observers[:, 0], _observers[:, _obs_names.index("x")], left=0, right=0)


def y_observer(wl: np.ndarray) -> np.ndarray:
    """
    Eye y observer values (CIE 1931 2° Standard Observer)

    :param wl: wavelength array in nm
    :return: value array
    """
    return np.interp(wl, _observers[:, 0], _observers[:, _obs_names.index("y")], left=0, right=0)


def z_observer(wl: np.ndarray) -> np.ndarray:
    """
    Eye z observer values (CIE 1931 2° Standard Observer)

    :param wl: wavelength array in nm
    :return: value array
    """
    return np.interp(wl, _observers[:, 0], _observers[:, _obs_names.index("z")], left=0, right=0)
