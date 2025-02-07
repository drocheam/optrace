
import pathlib

import numpy as np  # calculations

# CIE standard illuminants
# see optrace/resources/SOURCE.txt for more information

# load illuminants from file
_ill_names = ["wl", "A", "C", "D50", "D55", "D65", "D75", "F2", "F7", "F11",
              "LED-B1", "LED-B2", "LED-B3", "LED-B4", "LED-B5", "LED-BH1", "LED-RGB1", "LED-V1", "LED-V2"]
_ill_path = pathlib.Path(__file__).resolve().parent.parent.parent / "resources" / "illuminants.csv"
_illuminants = np.genfromtxt(_ill_path, skip_header=1, delimiter=",", filling_values=0, dtype=np.float64)


def a_illuminant(wl: np.ndarray) -> np.ndarray:
    """
    A standard illuminant (CIE Colorimetry, 3. Edition, 2004).
    Typical, domestic, tungsten-filament lighting. Color temperature of 2856K.

    :param wl: wavelength array in nm
    :return: value array
    """
    return np.interp(wl, _illuminants[:, 0], _illuminants[:, _ill_names.index("A")], left=0, right=0)


def c_illuminant(wl: np.ndarray) -> np.ndarray:
    """
    C standard illuminant (CIE Colorimetry, 3. Edition, 2004).
    Obsolete, average / north sky daylight. Color temperature of 6774K.

    :param wl: wavelength array in nm
    :return: value array
    """
    return np.interp(wl, _illuminants[:, 0], _illuminants[:, _ill_names.index("C")], left=0, right=0)


def e_illuminant(wl: np.ndarray) -> np.ndarray:
    """
    E standard illuminant (CIE Colorimetry, 3. Edition, 2004).
    Equal energy radiator with a color temperature of 5455K.

    :param wl: wavelength array in nm
    :return: value array
    """
    return np.full_like(wl, 100.0, dtype=np.float64)


def d50_illuminant(wl: np.ndarray) -> np.ndarray:
    """
    D50 standard illuminant (CIE Colorimetry, 3. Edition, 2004).
    Horizon light. Color temperature of 5003K.

    :param wl: wavelength array in nm
    :return: value array
    """
    return np.interp(wl, _illuminants[:, 0], _illuminants[:, _ill_names.index("D50")], left=0, right=0)


def d55_illuminant(wl: np.ndarray) -> np.ndarray:
    """
    D55 standard illuminant (CIE Colorimetry, 3. Edition, 2004).
    Mid-morning/mid-afternoon daylight. Color temperature of 5503K.

    :param wl: wavelength array in nm
    :return: value array
    """
    return np.interp(wl, _illuminants[:, 0], _illuminants[:, _ill_names.index("D55")], left=0, right=0)


def d65_illuminant(wl: np.ndarray) -> np.ndarray:
    """
    D65 standard illuminant (CIE Colorimetry, 3. Edition, 2004).
    Noon daylight. Color temperature of 6504K.

    :param wl: wavelength array in nm
    :return: value array
    """
    return np.interp(wl, _illuminants[:, 0], _illuminants[:, _ill_names.index("D65")], left=0, right=0)


def d75_illuminant(wl: np.ndarray) -> np.ndarray:
    """
    D75 standard illuminant (CIE Colorimetry, 3. Edition, 2004).
    North sky daylight. Color temperature of 7504K.

    :param wl: wavelength array in nm
    :return: value array
    """
    return np.interp(wl, _illuminants[:, 0], _illuminants[:, _ill_names.index("D75")], left=0, right=0)


def f2_illuminant(wl: np.ndarray) -> np.ndarray:
    """
    F2 standard illuminant (CIE Colorimetry, 3. Edition, 2004).
    Fluorescent lamp with two semi-broadband emissions. Color temperature of 4230K.

    :param wl: wavelength array in nm
    :return: value array
    """
    return np.interp(wl, _illuminants[:, 0], _illuminants[:, _ill_names.index("F2")], left=0, right=0)


def f7_illuminant(wl: np.ndarray) -> np.ndarray:
    """
    F7 standard illuminant (CIE Colorimetry, 3. Edition, 2004).
    Broadband fluorescent lamp with multiple phosphors.
    Color temperature of 6500K.

    :param wl: wavelength array in nm
    :return: value array
    """
    return np.interp(wl, _illuminants[:, 0], _illuminants[:, _ill_names.index("F7")], left=0, right=0)


def f11_illuminant(wl: np.ndarray) -> np.ndarray:
    """
    F11 standard illuminant (CIE Colorimetry, 3. Edition, 2004).
    Narrowband triband fluorescent lamp in R, G, B regions.
    Color temperature of 4000K.

    :param wl: wavelength array in nm
    :return: value array
    """
    return np.interp(wl, _illuminants[:, 0], _illuminants[:, _ill_names.index("F11")], left=0, right=0)


def led_b1_illuminant(wl: np.ndarray) -> np.ndarray:
    """
    LED-B1 standard illuminant (CIE Colorimetry, 4th Edition, 2018)
    Blue excited phosphor type LED with a color temperature of 2733K.

    :param wl: wavelength array in nm
    :return: value array
    """
    return np.interp(wl, _illuminants[:, 0], _illuminants[:, _ill_names.index("LED-B1")], left=0, right=0)


def led_b2_illuminant(wl: np.ndarray) -> np.ndarray:
    """
    LED-B2 standard illuminant (CIE Colorimetry, 4th Edition, 2018)
    Blue excited phosphor type LED with a color temperature of 2998K.

    :param wl: wavelength array in nm
    :return: value array
    """
    return np.interp(wl, _illuminants[:, 0], _illuminants[:, _ill_names.index("LED-B2")], left=0, right=0)


def led_b3_illuminant(wl: np.ndarray) -> np.ndarray:
    """
    LED-B3 standard illuminant (CIE Colorimetry, 4th Edition, 2018)
    Blue excited phosphor type LED with a color temperature of 4103K.

    :param wl: wavelength array in nm
    :return: value array
    """
    return np.interp(wl, _illuminants[:, 0], _illuminants[:, _ill_names.index("LED-B3")], left=0, right=0)


def led_b4_illuminant(wl: np.ndarray) -> np.ndarray:
    """
    LED-B4 standard illuminant (CIE Colorimetry, 4th Edition, 2018)
    Blue excited phosphor type LED with a color temperature of 5109K.

    :param wl: wavelength array in nm
    :return: value array
    """
    return np.interp(wl, _illuminants[:, 0], _illuminants[:, _ill_names.index("LED-B4")], left=0, right=0)


def led_b5_illuminant(wl: np.ndarray) -> np.ndarray:
    """
    LED-B5 standard illuminant (CIE Colorimetry, 4th Edition, 2018)
    Blue excited phosphor type LED with a color temperature of 6598K.

    :param wl: wavelength array in nm
    :return: value array
    """
    return np.interp(wl, _illuminants[:, 0], _illuminants[:, _ill_names.index("LED-B5")], left=0, right=0)


def led_bh1_illuminant(wl: np.ndarray) -> np.ndarray:
    """
    LED-BH1 standard illuminant (CIE Colorimetry, 4th Edition, 2018)
    Hybrid type white LED with added red. Color temperature of 2851K.

    :param wl: wavelength array in nm
    :return: value array
    """
    return np.interp(wl, _illuminants[:, 0], _illuminants[:, _ill_names.index("LED-BH1")], left=0, right=0)


def led_rgb1_illuminant(wl: np.ndarray) -> np.ndarray:
    """
    LED-RGB1 standard illuminant (CIE Colorimetry, 4th Edition, 2018)
    Tri-led RGB source with a color temperature of 2840K.

    :param wl: wavelength array in nm
    :return: value array
    """
    return np.interp(wl, _illuminants[:, 0], _illuminants[:, _ill_names.index("LED-RGB1")], left=0, right=0)


def led_v1_illuminant(wl: np.ndarray) -> np.ndarray:
    """
    LED-V1 standard illuminant (CIE Colorimetry, 4th Edition, 2018)
    Violet enhanced blue excited phosphor LED with a color temperature of 2724K.

    :param wl: wavelength array in nm
    :return: value array
    """
    return np.interp(wl, _illuminants[:, 0], _illuminants[:, _ill_names.index("LED-V1")], left=0, right=0)


def led_v2_illuminant(wl: np.ndarray) -> np.ndarray:
    """
    LED-V2 standard illuminant (CIE Colorimetry, 4th Edition, 2018)
    Violet enhanced blue excited phosphor LED with a color temperature of 4070K.

    :param wl: wavelength array in nm
    :return: value array
    """
    return np.interp(wl, _illuminants[:, 0], _illuminants[:, _ill_names.index("LED-V2")], left=0, right=0)

