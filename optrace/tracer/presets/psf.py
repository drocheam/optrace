import numpy as np
import numexpr as ne
import scipy.special

from ..misc import PropertyChecker as pc
from .. import color



def circle(d: float = 1.0) -> tuple[np.ndarray, list[float, float]]:
    """
    Two dimensional circle kernel with diameter d.

    :param d: circle diameter in µm
    :return: kernel image array, image side lengths (mm)
    """
    pc.check_above("d", d, 0)
   
    ds = 1.05/2  # 5% larger so we have a black edge
    sz = 601

    # calculate radial data
    Y, X = np.mgrid[-ds:ds:sz*1j, -ds:ds:sz*1j]
    R2 = ne.evaluate("X**2 + Y**2")

    Z = np.zeros((sz, sz), dtype=np.float64)
    Z[R2 <= (0.5 + ds/Y.shape[0])**2] = 0.25  # smoother edge
    Z[R2 <= 0.5**2] = 0.75  # circle area
    Z[R2 <= (0.5 - ds/Y.shape[0])**2] = 1.0  # smoother edge

    s = [2*ds*d/1000, 2*ds*d/1000]  # scale size with d

    return Z, s


def gaussian(d: float = 1.0) -> tuple[np.ndarray, list[float, float]]:
    """
    Two dimensional gaussian kernel.
    d describes the diameter in µm of the function that approximately matches the zeroth order of an airy disk.

    :param d: gaussian diameter / resolution limit in µm
    :return: kernel image array, image side lengths (mm)
    """
    pc.check_above("d", d, 0)

    sig = 0.175  # sigma so approximating zeroth order of airy disk
    ds = 5*sig  # plot 5 sigma
    sz = 401

    Y, X = np.mgrid[-ds:ds:sz*1j, -ds:ds:sz*1j]
    Z = ne.evaluate("exp(-(X**2 + Y**2) / 2 / sig**2)")

    s = [2*ds*d/1000, 2*ds*d/1000]  # scale size with d

    return Z, s


def airy(d: float = 1.0) -> tuple[np.ndarray, list[float, float]]:
    """
    Airy disk kernel, where d is the diameter of the zeroth order.

    :param d: resolution limit (diameter) in µm
    :return: kernel image array, image side lengths (mm)
    """
    pc.check_above("d", d, 0)

    ds = 5/2  # calculate diameter 5*d
    sz = 401

    Z = np.ones((sz, sz), dtype=np.float64)

    # normalized r
    Y, X = np.mgrid[-ds:ds:sz*1j, -ds:ds:sz*1j]
    R = ne.evaluate("sqrt(X**2 + Y**2) * 3.8317 * 2")

    Rnz = R[R!=0]

    # calculate airy function intensity
    j1 = scipy.special.j1(Rnz)
    Z[R != 0] = ne.evaluate("(2*j1 / Rnz) ** 2")
    
    s = [5*d/1000, 5*d/1000]  # scale size with d

    return Z, s


def glare(d1: float = 1.0, d2: float = 7.0, a: float = 0.15):
    """
    Glare kernel. This glare consists of two gaussian kernels, one with diameter d1, the other with a larger diameter d2.
    See gaussian() for details on the diameter.
    Factor a describes the relative amplitude of the larger kernel to the smaller one.

    :param d1: diameter in µm of first gaussian, the focus
    :param d2: diameter in µm of second gaussian, the glare
    :param a: relative brightness of the second one compared to the first one
    :return: kernel image array, image side lengths (mm)
    """
    pc.check_above("d1", d1, 0)
    pc.check_above("d2", d1, 0)
    pc.check_not_below("a", a, 0)

    if d2 <= d1:
        raise ValueError("d2 must be larger than d1.")

    sig = 0.175  # sigma so approximating zeroth order of airy disk
    ds = 5*sig  # plot 5 sigma
    sz = 801

    Y, X = np.mgrid[-ds:ds:sz*1j, -ds:ds:sz*1j]
    R2 = ne.evaluate("X**2 + Y**2")
    Z = ne.evaluate("a/(1+a)*exp(-R2 / 2 / sig**2) +"
                    "1/(1+a)*exp(-R2 / 2 / (sig*d1/d2)**2)")

    s = [2*ds*d2/1000, 2*ds*d2/1000]  # scale size with d

    return Z, s


def halo(d1: float = 1.0, d2: float = 4.0, a: float = 0.3, w: float = 0.2):
    """
    Halo kernel. It consists of a central 2D gaussian and an outer gaussian ring.

    :param d1: diameter of gaussian focus in µm
    :param d2: radial position of ring center in µm
    :param a: relative brightness of ring compared to focus
    :param w: radial ring size 
    :return: kernel image array, image side lengths (mm)
    """
    pc.check_above("d1", d1, 0)
    pc.check_above("d2", d1, 0)
    pc.check_not_below("a", a, 0)
    pc.check_not_below("w", w, 0)

    if d2 <= d1:
        raise ValueError("d2 must be larger than d1.")

    sig = 0.175*d1 
    sig2 = w/2.14597/2  # approximate radial value so gaussian falls at 10%
    ds = d2/2 + 5*sig2 
    sz = 801

    Y, X = np.mgrid[-ds:ds:sz*1j, -ds:ds:sz*1j]
    R = ne.evaluate("sqrt(X**2 + Y**2)")
    Z = ne.evaluate("exp(-R**2 / 2 / sig**2) + a*exp(-(R - d2/2)**2 / 2 / sig2**2)")

    s = [2*ds/1000, 2*ds/1000]  # scale size with d

    return Z, s

