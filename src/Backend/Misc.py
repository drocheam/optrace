"""
helper
"""

import scipy.interpolate
import numpy as np
import numexpr as ne
import scipy.special

from functools import wraps
import time
import sys

from typing import Callable, Any, Dict


def interp1d(x: np.ndarray, y: np.ndarray, xs: np.ndarray) -> np.ndarray:
    """ 
    fast alternative to :obj:`scipy.interpolate.interp1d` with equally spaced x-values 

    >>> interp1d(np.array([1, 2, 3, 4]), np.array([5, 6, 5, 4]), np.array([1, 1.5, 3.5, 2.5]))
    array([5. , 5.5, 4.5, 5.5])
    """
    x0, x1 = x[0], x[1]
    ind0 = ne.evaluate("(1-1e-12)/(x1-x0)*(xs-x0)")
    ind1 = ind0.astype(int)

    # multiplication with (1-1e-12) to avoid case xs = x[-1], 
    # which would lead to y[ind1 + 1] being an access violation
    # we could circumvent this using comparisons, masks or cases, but this would decreas performance

    ya = y[ind1]
    yb = y[ind1+1]

    return ne.evaluate("(1-ind0+ind1)*ya + (ind0-ind1)*yb")

def calc(expr: str, out: np.ndarray=None, **kwargs) -> np.ndarray:
    """"""

    base = dict(pi=np.pi, nan=np.nan, inf=np.inf, ninf=np.NINF, euler_gamma=np.euler_gamma, euler=np.e)

    # get variables from local frame of caller
    # this is the way numexpr does it too (see https://github.com/pydata/numexpr/blob/master/numexpr/necompiler.py)
    loc = sys._getframe(2).f_locals | sys._getframe(1).f_locals

    # dictonary from additional keyword arguments
    kw = dict(**kwargs) if kwargs is not None else dict()

    # join local dict and keyword dict
    dict_ = base | loc | kw

    return ne.evaluate(expr, local_dict=dict_, out=out)

def getCoreCount() -> int:
    """get CPU Core Count"""
    return ne.detect_number_of_cores()

def random_from_distribution(x: np.ndarray, pdf: np.ndarray, N: int) -> np.ndarray:
    """
    Get randomly distributed values with pdf(x) as probability distribution function
    using inverse transform sampling

    :param x: pdf x values
    :param pdf: pdf y values
    :param N: number of random values
    :return: N randomly distributed values

    >>> ch = random_from_distribution(np.array([0., 1.]), np.array([0., 1.]), 10000)
    >>> (0.49 < np.mean(ch) < 0.51) and (np.max(ch) > 0.99) and (np.min(ch) < 0.01)
    True
    """
    # unnormalized cdf
    cdf = np.cumsum(pdf)
    cdf /= cdf[-1]

    # normalize cdf and append 0 at the beginning
    xc = np.concatenate(([x[0]], x))
    cdf = np.concatenate(([0], cdf))

    X = np.random.sample(N)

    # unfortunately we can't use out own interp1d, since cdf is not equally spaced
    icdf = scipy.interpolate.interp1d(cdf, xc)
    return icdf(X)


# with the help of https://stackoverflow.com/questions/51503672/decorator-for-timeit-timeit-method/51503837#51503837
# can be used as decorator @timer around a function
def timer(func: Callable) -> Any:
    """

    :param func:
    :return:
    """
    @wraps(func)
    def _time_it(*args, **kwargs) -> Any:
        start = time.time()
        try:
            return func(*args, **kwargs)
        finally:
            diff = time.time() - start
            if diff < 0.1:
                print(f"Timing: {1000*diff:.3f} ms for {func}")
            else:
                print(f"Timing: {diff:.3f} s for {func}")

    return _time_it


def rdot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """ 
    row wise scalar product. 
    Result c = a[:, 0]*b[:, 0] + a[:, 1]*b[:, 1] + a[:, 2]*b[:, 2] 

    >>> rdot(np.array([[1., 2., 3.], [4., 5., 6.]]), np.array([[-1., 2., -3.], [7., 8., 9.]]))
    array([ -6., 122.])
    """
    x, y, z    = a[:, 0], a[:, 1], a[:, 2]
    x2, y2, z2 = b[:, 0], b[:, 1], b[:, 2]
    return ne.evaluate("x*x2 + y*y2 + z*z2")


def partMask(cond1: np.ndarray, cond2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ 
    set True values of bool mask cond1 to values of cond2, returns resulting mask and cond2 

    >>> partMask(np.array([True, False, False, True]), np.array([True, False]))
    (array([ True, False, False, False]), array([ True, False]))
    """
    wc = np.zeros(cond1.shape, dtype=bool)
    wc[cond1] = cond2
    return wc, cond2.copy()


def normalize(a: np.ndarray) -> None:
    """ 
    faster vector normalization for vectors in axis=1.
    Zero length vectors can normalized to np.nan.
    
    >>> a = np.array([[1., 2., 3.], [4., 5., 6.]])
    >>> normalize(a)
    >>> a
    array([[0.26726124, 0.53452248, 0.80178373],
           [0.45584231, 0.56980288, 0.68376346]])
    """

    x, y, z = a[:, 0], a[:, 1], a[:, 2]

    norms = ne.evaluate("sqrt(x**2 + y**2 + z**2)")[:, np.newaxis]

    # save normalization to initial array a
    nan = np.nan
    ne.evaluate("a/where(norms>0, norms, nan)", out=a)

def cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """ 
    faster alternative to :func:`numpy.cross` for 3 element vectors and axis=1

    >>> cross(np.array([[1., 2., 3.], [4., 5., 6.]]), np.array([[-1., 2., -3.], [7., 8., 9.]]))
    array([[-12.,   0.,   4.],
           [ -3.,   6.,  -3.]])
    """
    x, y, z    = a[:, 0], a[:, 1], a[:, 2]
    x2, y2, z2 = b[:, 0], b[:, 1], b[:, 2]

    n = np.zeros((a.shape[0], 3), dtype=np.float64, order='F')

    # using ne is ~2x faster than np.cross
    ne.evaluate("y*z2 - z*y2", out=n[:, 0])
    ne.evaluate("z*x2 - x*z2", out=n[:, 1])
    ne.evaluate("x*y2 - y*x2", out=n[: ,2])

    return n

# def blueNoise(a, b, N):
#     # create random phase and amplitude
#     phase = np.random.uniform(0, 2 * np.pi, N)
#     amp = np.random.uniform(0, 1, N)
#
#     # frequency vector
#     f = np.arange(-np.ceil(N / 2), np.floor(N / 2))
#
#     # create blue noise spectrum (blue noise: spectrum ~ f^0.5, since power ~ f)
#     spec = np.abs(f)**5 * amp * np.exp(-1j * phase)
#
#     # tranform to original domain
#     noise = np.real(np.fft.ifft(np.fft.ifftshift(spec)))
#
#     # remove mean and normalize to standard deviation
#     noise = (noise - np.mean(noise)) / np.std(noise)
#
#     # due to the central limit theorem (https://en.wikipedia.org/wiki/Central_limit_theorem)
#     # the noise amplitude is normal distributed. For uniform distribution we need to insert it
#     # in the normal distribution function inverse
#     noise = scipy.special.erf(noise / np.sqrt(2))
#
#     # rescale to range [a, b]
#     noise = a + (b - a) / 2 * (1 + noise)
#
#     return noise


def ValueAt(x:  np.ndarray,
            y:  np.ndarray,
            Z:  np.ndarray,
            x0: float,
            y0: float) \
        -> float:
    """
    Interpolate value at given coordinate.
    Use :obj:`Backend.Misc.interp2f` for list of coordinates.

    :param x: x values (numpy 1D array)
    :param y: y values (numpy 1D array)
    :param Z: z values (numpy 2D array)
    :param x0: x coordinate (float)
    :param y0: y coordinate (float)
    :return: z value

    >>> ValueAt(np.array([2., 3.]), np.array([1., 2.]), np.array([[0., 1.], [2., 3.]]), 2.75, 1.5)
    1.75
    """
    # try interpolation
    val = interp2d(x, y, Z, np.array([x0]), np.array([y0]))[0]

    # interpolation failed, nan values in proximity
    if np.isnan(val):

        # interpolate Nans before retry
        Z = interpolateNan(Z)
        val = interp2d(x, y, Z, np.array([x0]), np.array([y0]))[0]

        # still no good
        if np.isnan(val):
            raise RuntimeError(f"Nan value at x0={x0}, y0={y0}")

    return val


# TODO improve speed
# TODO Bug: Remaining "nan islands". Example: h_data with isnan = [[0, 1, 1, 1], [0, 0, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0]]
#  creates remaining nan at isnan[3,3]. Execute function multiple times to solve this issue?
def interpolateNan(h_data: np.ndarray) -> np.ndarray:
    """
    2D interpolation of gridded data with missing points (nan values). 2D interpolation of not gridded data
    is extremly slow, so we instead interpolate row and columnwise and use the average of the results at these points.
    This speeds up interpolation by a factor of 10x-100x compared to the function griddata

    :param h_data: z values (2D array)
    :return: h_data with invalid data interpolated (2D array)

    >>> interpolateNan(np.array([[1., np.nan, 3.], [4., 5., np.nan], [7., np.nan, 9.]]))
    array([[1., 2., 3.],
           [4., 5., 6.],
           [7., 8., 9.]])
    """
    
    # copy initial data
    h_data_ix = h_data.copy()
    h_data_iy = h_data.copy()

    x = np.arange(h_data.shape[1])
    y = np.arange(h_data.shape[0])

    bad_mask = ~np.isfinite(h_data)

    # in x direction
    for n in y:
        bad_mask_n = bad_mask[n, :]

        # only interpolate if more than 10 valid data points and there is missing data to interpolate
        if np.any(bad_mask_n) and np.count_nonzero(~bad_mask_n) > 1:

            # interpolation
            f = scipy.interpolate.interp1d(x[~bad_mask_n], h_data_ix[n, ~bad_mask_n])

            # exclude outlying invalid data points, so no extrapolation takes place
            finite = (~bad_mask_n).nonzero()[0]
            bad_mask_n[:finite.min()] = False
            bad_mask_n[finite.max():] = False

            # fill interpolated data
            h_data_ix[n, bad_mask_n] = f(x[bad_mask_n])

    bad_mask = ~np.isfinite(h_data)

    # do the same in y direction
    for n in x:
        bad_mask_n = bad_mask[:, n]

        if np.any(bad_mask_n) and np.count_nonzero(~bad_mask_n) > 1:
            f = scipy.interpolate.interp1d(y[~bad_mask_n], h_data_iy[~bad_mask_n, n])

            finite = (~bad_mask_n).nonzero()[0]
            bad_mask_n[:finite.min()] = False
            bad_mask_n[finite.max():] = False

            h_data_iy[bad_mask_n, n] = f(y[bad_mask_n])

    # initial bad data
    bad_mask = ~np.isfinite(h_data)

    # unfixed data points
    still_bad_x = ~np.isfinite(h_data_ix)
    still_bad_y = ~np.isfinite(h_data_iy)

    # use mean of fixed data in x and y direction where possible
    mean_mask = bad_mask & ~still_bad_x & ~still_bad_y
    h_data_ix[mean_mask] = h_data_ix[mean_mask]/2 + h_data_iy[mean_mask]/2

    # copy data that is fixed in h_data_y but not in h_data_x to h_data_x
    cp_y_mask = bad_mask & still_bad_x & ~still_bad_y
    h_data_ix[cp_y_mask] = h_data_iy[cp_y_mask]

    return h_data_ix


def interp2d(x:         np.ndarray,
             y:         np.ndarray,
             z:         np.ndarray,
             xp:        np.ndarray,
             yp:        np.ndarray,
             method:    str = 'linear')\
        -> np.ndarray:
    """
    Faster alternative for :obj:`scipy.interpolate.interp2d` for gridded and equally spaced 2D data
    linear or nearest neighbor interpolation

    :param x: x coordinate vector (numpy 1D array)
    :param y: y coordinate vector (numpy 1D array)
    :param z: z values (numpy 2D array)
    :param xp: numpy 1D arrays holding the interpolation points x coordinates
    :param yp: numpy 1D arrays holding the interpolation points y coordinates
    :param method: "linear" or "nearest" (string)
    :return: interpolated values as 1D array
    
    >>> interp2d(np.array([2, 3]), np.array([1, 2]), np.array([[0, 1], [2, 3]]), np.array([2.25, 2.75]), np.array([1, 1.5]), method="linear")
    array([0.25, 1.75])

    >>> interp2d(np.array([2, 3]), np.array([1, 2]), np.array([[0, 1], [2, 3]]), np.array([2.25, 2.75]), np.array([1, 1.5]), method="nearest")
    array([0, 3])
    """

    # check input shapes
    if x.ndim != 1 or y.ndim != 1:
        raise TypeError("x and y need to be vectors")
    if (z.ndim != 2) or (z.shape[0] != y.shape) != (z.shape[1] != x.shape):
        raise TypeError("z needs to be 2 dimensional with size y*x")
    if xp.ndim != 1 or yp.ndim != 1:
        raise TypeError("xp and yp need to be vectors")

    # check values of xp and yp:
    if np.min(xp) < x[0] or np.max(xp) > x[-1]:
        raise ValueError("xp value outside data range")
    if np.min(yp) < y[0] or np.max(yp) > y[-1]:
        raise ValueError("yp value outside data range")

    # bilinear interpolation: https://en.wikipedia.org/wiki/Bilinear_interpolation
    # rewrite the equation to:
    # zi = (1 - yr)*(xr*z[yc, xc+1] + z[yc, xc]*(1-xr)) + yr*(xr*z[yc+1, xc+1]+ z[yc+1, xc]*(1-xr))
    #
    # with:
    # xt, yt: sample points (x, y in Wikipedia)
    # xc, yc: integer part of xt, yt coordinates (x1, y1 in Wikipedia)
    # xr, yr: float part of xt, yt coordinates ((x-x1)/(x2-x1), (y-y1)/(y2-y1) in Wikipedia)

    xt =  1 / (x[1] - x[0]) * (xp - x[0])
    yt =  1 / (y[1] - y[0]) * (yp - y[0])

    if method == 'linear':
        # this part is faster than using np.divmod
        xc = xt.astype(int)
        yc = yt.astype(int)
        xr = xt - xc
        yr = yt - yc

        a = z[yc, xc]
        b = z[yc+1, xc]
        c = z[yc, xc+1]
        d = z[yc+1, xc+1]

        # rearranged form with only 4 multiplications for speedup
        return ne.evaluate("(1 - yr) * (xr * (c - a) + a) + yr * (xr * (d - b) + b)")

    elif method == 'nearest':
        # fast rounding for positive numbers
        xc = (xt + 0.5).astype(int)
        yc = (yt + 0.5).astype(int)

        return z[yc, xc]
    else:
        raise ValueError("Invalid method")

