import numpy as np
import scipy.special

from ...property_checker import PropertyChecker as pc
from ..image.grayscale_image import GrayscaleImage
from ..color.srgb import srgb_linear_to_srgb


def circle(d: float = 1.0) -> GrayscaleImage:
    """
    Two dimensional circle kernel with diameter d.

    :param d: circle diameter in µm
    :return: image object
    """
    pc.check_above("d", d, 0)
   
    ds = 1.05/2  # 5% larger so we have a black edge
    sz = 601

    # calculate radial data
    Y, X = np.mgrid[-ds:ds:sz*1j, -ds:ds:sz*1j]
    R2 = X**2 + Y**2

    Z = np.zeros((sz, sz), dtype=np.float64)
    Z[R2 <= (0.5 + ds/Y.shape[0])**2] = 0.25  # smoother edge
    Z[R2 <= 0.5**2] = 0.75  # circle area
    Z[R2 <= (0.5 - ds/Y.shape[0])**2] = 1.0  # smoother edge
    Z = srgb_linear_to_srgb(Z)

    s = [2*ds*d/1000, 2*ds*d/1000]  # scale size with d

    return GrayscaleImage(Z, s)


def gaussian(sig: float = 0.5) -> GrayscaleImage:
    """
    Two dimensional gaussian kernel.
    d describes the diameter in µm of the function that approximately matches the zeroth order of an airy disk.

    :param sig: gaussian sigma in µm
    :return: image object
    """
    pc.check_above("sig", sig, 0)

    ds = 5*sig  # plot 5 sigma
    sz = 401

    Y, X = np.mgrid[-ds:ds:sz*1j, -ds:ds:sz*1j]
    Z = np.exp(-(X**2 + Y**2) / 2 / sig**2)
    Z = srgb_linear_to_srgb(Z)

    s = [2*ds/1000, 2*ds/1000]  # scale size with d

    return GrayscaleImage(Z, s)


def airy(r: float = 1.0) -> GrayscaleImage:
    """
    Airy disk kernel, where d is the diameter of the zeroth order.

    :param r: resolution limit in µm (half the core airy disk diameter) in µm
    :return: image object
    """
    pc.check_above("r", r, 0)

    ds = 10.1735 / 3.8317 # calculate to third zero (so first two rings)
    sz = 401

    Z = np.ones((sz, sz), dtype=np.float64)

    # normalized r
    Y, X = np.mgrid[-ds:ds:sz*1j, -ds:ds:sz*1j]
    R = np.sqrt(X**2 + Y**2) * 3.8317

    Rnz = R[R!=0]

    # calculate airy function intensity
    Z[R != 0] = (2*scipy.special.j1(Rnz) / Rnz) ** 2
    Z[R > 10.1735] = 0  # deleted values after third zero crossing
    Z = srgb_linear_to_srgb(Z)

    s = [2*ds*r/1000, 2*ds*r/1000]  # scale size with d

    return GrayscaleImage(Z, s)


def glare(sig1: float = 0.5, sig2: float = 3.0, a: float = 0.15) -> GrayscaleImage:
    """
    Glare kernel. This glare consists of two gaussian kernels.
    See gaussian() for details on the diameter.
    Factor a describes the relative amplitude of the larger kernel to the smaller one.

    :param sig1: sigma in µm of first gaussian, the focus
    :param sig2: sigma in µm of second gaussian, the glare
    :param a: relative brightness of the second one compared to the first one
    :return: image object
    """
    pc.check_above("sig1", sig1, 0)
    pc.check_above("sig2", sig2, 0)
    pc.check_not_below("a", a, 0)
    pc.check_not_above("a", a, 1)

    if sig2 <= sig1:
        raise ValueError("d2 must be larger than d1.")

    ds = 5*sig2  # plot 5 sigma
    sz = 801

    Y, X = np.mgrid[-ds:ds:sz*1j, -ds:ds:sz*1j]
    R2 = X**2 + Y**2
    Z = a*np.exp(-R2 / 2 / sig2**2) + (1-a)*np.exp(-R2 / 2 / sig1**2)
    Z /= Z.max()
    Z = srgb_linear_to_srgb(Z)

    s = [2*ds/1000, 2*ds/1000]  # scale size with d

    return GrayscaleImage(Z, s)


def halo(sig1: float = 0.5, sig2: float = 0.25, r: float = 4.0, a: float = 0.3) -> GrayscaleImage:
    """
    Halo kernel. It consists of a central 2D gaussian and an outer gaussian ring.

    :param sig1: sigma of gaussian focus in µm
    :param sig2: sigma of radial ring in µm
    :param r: radial position of ring center in µm
    :param a: relative brightness of ring compared to focus
    :return: image object
    """
    pc.check_above("sig1", sig1, 0)
    pc.check_above("sig2", sig2, 0)
    pc.check_not_below("a", a, 0)
    pc.check_not_above("a", a, 1)
    pc.check_not_below("r", r, 0)

    ds = r + 5*sig2
    sz = 801

    Y, X = np.mgrid[-ds:ds:sz*1j, -ds:ds:sz*1j]
    R = np.sqrt(X**2 + Y**2)
    Z = np.exp(-R**2 / 2 / sig1**2) + a*np.exp(-(R - r)**2 / 2 / sig2**2)
    Z /= Z.max()
    Z = srgb_linear_to_srgb(Z)

    s = [2*ds/1000, 2*ds/1000]  # scale size with d

    return GrayscaleImage(Z, s)

