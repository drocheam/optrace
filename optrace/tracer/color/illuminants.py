
import pathlib

import numpy as np  # calculations

# Sources A, C, E, D50, D55, D65, D75, F2, F7, F11:
# CIE Colorimetry, 3. Edition, 2004

# Sources xyz_observers values:
# https://cie.co.at/datatable/cie-1931-colour-matching-functions-2-degree-observer

# Sources LED Series:
# eigentlich CIE Colorimetry, 4th Edition, 2018, Werte aber von
# https://github.com/colour-science/colour/blob/develop/colour/colorimetry/datasets/illuminants/sds.py


# load illuminants
_ill_names = ["wl", "A", "C", "D50", "D55", "D65", "D75", "FL2", "FL7", "FL11",
              "LED-B1", "LED-B2", "LED-B3", "LED-B4", "LED-B5"]
_ill_path = pathlib.Path(__file__).resolve().parent.parent.parent / "ressources" / "illuminants.csv"
_illuminants = np.genfromtxt(_ill_path, skip_header=1, delimiter=",", filling_values=0, dtype=np.float64)


def a_illuminant(wl: np.ndarray) -> np.ndarray:
    """
    A standard illuminant (CIE Colorimetry, 3. Edition, 2004)

    :param wl: wavelength array in nm
    :return: value array
    """
    return np.interp(wl, _illuminants[:, 0], _illuminants[:, _ill_names.index("A")], left=0, right=0)


def c_illuminant(wl: np.ndarray) -> np.ndarray:
    """
    C standard illuminant (CIE Colorimetry, 3. Edition, 2004)

    :param wl: wavelength array in nm
    :return: value array
    """
    return np.interp(wl, _illuminants[:, 0], _illuminants[:, _ill_names.index("C")], left=0, right=0)


def e_illuminant(wl: np.ndarray) -> np.ndarray:
    """
    E standard illuminant (CIE Colorimetry, 3. Edition, 2004)

    :param wl: wavelength array in nm
    :return: value array
    """
    return np.full_like(wl, 100.0, dtype=np.float64)


def d50_illuminant(wl: np.ndarray) -> np.ndarray:
    """
    D50 standard illuminant (CIE Colorimetry, 3. Edition, 2004)

    :param wl: wavelength array in nm
    :return: value array
    """
    return np.interp(wl, _illuminants[:, 0], _illuminants[:, _ill_names.index("D50")], left=0, right=0)


def d55_illuminant(wl: np.ndarray) -> np.ndarray:
    """
    D55 standard illuminant (CIE Colorimetry, 3. Edition, 2004)

    :param wl: wavelength array in nm
    :return: value array
    """
    return np.interp(wl, _illuminants[:, 0], _illuminants[:, _ill_names.index("D55")], left=0, right=0)


def d65_illuminant(wl: np.ndarray) -> np.ndarray:
    """
    D65 standard illuminant (CIE Colorimetry, 3. Edition, 2004)

    :param wl: wavelength array in nm
    :return: value array
    """
    return np.interp(wl, _illuminants[:, 0], _illuminants[:, _ill_names.index("D65")], left=0, right=0)


def d75_illuminant(wl: np.ndarray) -> np.ndarray:
    """
    D75 standard illuminant (CIE Colorimetry, 3. Edition, 2004)

    :param wl: wavelength array in nm
    :return: value array
    """
    return np.interp(wl, _illuminants[:, 0], _illuminants[:, _ill_names.index("D75")], left=0, right=0)


def fl2_illuminant(wl: np.ndarray) -> np.ndarray:
    """
    FL2 standard illuminant (CIE Colorimetry, 3. Edition, 2004)

    :param wl: wavelength array in nm
    :return: value array
    """
    return np.interp(wl, _illuminants[:, 0], _illuminants[:, _ill_names.index("FL2")], left=0, right=0)


def fl7_illuminant(wl: np.ndarray) -> np.ndarray:
    """
    FL7 standard illuminant (CIE Colorimetry, 3. Edition, 2004)

    :param wl: wavelength array in nm
    :return: value array
    """
    return np.interp(wl, _illuminants[:, 0], _illuminants[:, _ill_names.index("FL7")], left=0, right=0)


def fl11_illuminant(wl: np.ndarray) -> np.ndarray:
    """
    FL11 standard illuminant (CIE Colorimetry, 3. Edition, 2004)

    :param wl: wavelength array in nm
    :return: value array
    """
    return np.interp(wl, _illuminants[:, 0], _illuminants[:, _ill_names.index("FL11")], left=0, right=0)


def led_b1_illuminant(wl: np.ndarray) -> np.ndarray:
    """
    LED-B1 standard illuminant (CIE Colorimetry, 4th Edition, 2018)

    :param wl: wavelength array in nm
    :return: value array
    """
    return np.interp(wl, _illuminants[:, 0], _illuminants[:, _ill_names.index("LED-B1")], left=0, right=0)


def led_b2_illuminant(wl: np.ndarray) -> np.ndarray:
    """
    LED-B2 standard illuminant (CIE Colorimetry, 4th Edition, 2018)

    :param wl: wavelength array in nm
    :return: value array
    """
    return np.interp(wl, _illuminants[:, 0], _illuminants[:, _ill_names.index("LED-B2")], left=0, right=0)


def led_b3_illuminant(wl: np.ndarray) -> np.ndarray:
    """
    LED-B3 standard illuminant (CIE Colorimetry, 4th Edition, 2018)

    :param wl: wavelength array in nm
    :return: value array
    """
    return np.interp(wl, _illuminants[:, 0], _illuminants[:, _ill_names.index("LED-B3")], left=0, right=0)


def led_b4_illuminant(wl: np.ndarray) -> np.ndarray:
    """
    LED-B4 standard illuminant (CIE Colorimetry, 4th Edition, 2018)

    :param wl: wavelength array in nm
    :return: value array
    """
    return np.interp(wl, _illuminants[:, 0], _illuminants[:, _ill_names.index("LED-B4")], left=0, right=0)


def led_b5_illuminant(wl: np.ndarray) -> np.ndarray:
    """
    LED-B5 standard illuminant (CIE Colorimetry, 4th Edition, 2018)

    :param wl: wavelength array in nm
    :return: value array
    """
    return np.interp(wl, _illuminants[:, 0], _illuminants[:, _ill_names.index("LED-B5")], left=0, right=0)
