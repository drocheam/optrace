"""
helper
"""

import scipy.interpolate  # linear interpolation
import numpy as np  # calculations
import numexpr as ne  # faster calculations

from typing import Callable, Any  # Callable and Any types
from functools import wraps  # wrapping of functions
import time  # timing


# with the help of https://stackoverflow.com/questions/51503672/decorator-for-timeit-timeit-method/51503837#51503837
# can be used as decorator @timer around a function
def timer(func: Callable) -> Any:
    """
    timing wrapper for function timings
    write a @timer decorator around a function to time it

    :param func: function to wrap
    :return: function return value
    """
    @wraps(func)
    def _time_it(*args, **kwargs) -> Any:
        start = time.time()
        try:
            return func(*args, **kwargs)
        finally:
            diff = time.time() - start

            unit = "ms" if diff < 0.1 else "s"
            time_ = 1000*diff if diff < 0.1 else diff

            print(f"Timing: {time_:.3f} {unit} for {func}")

    return _time_it


# class only used for separate namespace
class PropertyChecker:
    
    @staticmethod
    def checkType(key, val, type_) -> None:
        if not isinstance(val, type_):
            types = str(type_).replace("|", "or")
            raise TypeError(f"Property '{key}' needs to be of type(s) {types}, but is {type(val)}.")

    @staticmethod
    def checkNotAbove(key, val, cmp) -> None:
        if val > cmp:
            raise ValueError(f"Property '{key}' needs to be below or equal to {cmp}, but is {val}.")

    @staticmethod
    def checkNotBelow(key, val, cmp) -> None:
        if val < cmp:
            raise ValueError(f"Property '{key}' needs to be above or equal to {cmp}, but is {val}.")

    @staticmethod
    def checkAbove(key, val, cmp) -> None:
        if val <= cmp:
            raise ValueError(f"Property '{key}' needs to be above {cmp}, but is {val}.")
    
    @staticmethod
    def checkBelow(key, val, cmp) -> None:
        if val >= cmp:
            raise ValueError(f"Property '{key}' needs to be below {cmp}, but is {val}.")
    
    @staticmethod
    def checkNoneOrCallable(key, val) -> None:
        if val is not None and not callable(val):
            raise TypeError(f"{key} needs to be callable or None, but is {type(val)}.")
    
    @staticmethod
    def checkIfIn(key, val, list_) -> None:
        if val not in list_:
            raise ValueError(f"Invalid value '{val}' for property '{key}'. Needs to be one of {list_}, but is {val}.")


def random_from_distribution(x: np.ndarray, pdf: np.ndarray, N: int) -> np.ndarray:
    """
    Get randomly distributed values with pdf(x) as probability distribution function
    using inverse transform sampling

    :param x: pdf x values
    :param pdf: pdf y values
    :param N: number of random values
    :return: N randomly distributed values according to pdf

    >>> ch = random_from_distribution(np.array([0., 1.]), np.array([0., 1.]), 10000)
    >>> (0.49 < np.mean(ch) < 0.51) and (np.max(ch) > 0.99) and (np.min(ch) < 0.01)
    True
    """
    cdf = np.cumsum(pdf)  # unnormalized cdf

    if not cdf[-1]:
        raise RuntimeError("Cumulated probability is zero.")
    elif np.min(pdf) < 0:
        raise RuntimeError("Got negative value in pdf.")

    # alternatively we could use 
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.html#scipy.stats.rv_continuous
    # or something similar

    # unfortunately we can't use np.interp1d, since cdf is not equally spaced
    icdf = scipy.interpolate.interp1d(cdf, x, assume_sorted=True, kind='linear')

    # random variable
    X = np.random.uniform(cdf[0], cdf[-1], N)
    
    return icdf(X)  # sample to get random values


def rdot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """ 
    row wise scalar product. 
    Result c = a[:, 0]*b[:, 0] + a[:, 1]*b[:, 1] + a[:, 2]*b[:, 2] 

    >>> rdot(np.array([[1., 2., 3.], [4., 5., 6.]]), np.array([[-1., 2., -3.], [7., 8., 9.]]))
    array([ -6., 122.])
    """
    x, y, z = a[:, 0], a[:, 1], a[:, 2]
    x2, y2, z2 = b[:, 0], b[:, 1], b[:, 2]
    
    return ne.evaluate("x*x2 + y*y2 + z*z2")


def part_mask(cond1: np.ndarray, cond2: np.ndarray) -> np.ndarray:
    """ 
    set True values of bool mask cond1 to values of cond2, returns resulting mask

    >>> part_mask(np.array([True, False, False, True]), np.array([True, False]))
    array([ True, False, False, False])
    """
    wc = np.zeros(cond1.shape, dtype=bool)
    wc[cond1] = cond2
    return wc


def uniform_resample(x: list | np.ndarray, y: list | np.ndarray, N: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Resample irregular 1D function data to regular data with N values.
    This is done using linear interpolation.

    :param x: x values (1D numpy vector)
    :param y: y values (1D numpy vector)
    :param N: number of points
    :return: resampled x vector, resampled y vector

    >>> uniform_resample([0, 1, 2.5, 4, 6], [1, 2, 5, 5, 1], 7)
    (array([0., 1., 2., 3., 4., 5., 6.]), array([1., 2., 4., 5., 5., 3., 1.]))
    """
    xs = np.linspace(x[0], x[-1], N)
    interp = scipy.interpolate.interp1d(np.array(x, dtype=np.float64), np.array(y, dtype=np.float64))
    return xs, interp(xs)


def normalize(a: np.ndarray) -> np.ndarray:
    """ 
    faster vector normalization for vectors in axis=1.
    Zero length vectors can normalized to np.nan.
    
    >>> a = np.array([[1., 2., 3.], [4., 5., 6.]])
    >>> normalize(a)
    array([[0.26726124, 0.53452248, 0.80178373],
           [0.45584231, 0.56980288, 0.68376346]])
    """
    x, y, z = a[:, 0, np.newaxis], a[:, 1, np.newaxis], a[:, 2, np.newaxis]
    valid = ~((x == 0) & (y == 0) & (z == 0))
    nan = np.nan
    return ne.evaluate("a/where(valid, sqrt(x**2 + y**2 + z**2), nan)")


def cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """ 
    faster alternative to :func:`numpy.cross` for 3 element vectors and axis=1

    >>> cross(np.array([[1., 2., 3.], [4., 5., 6.]]), np.array([[-1., 2., -3.], [7., 8., 9.]]))
    array([[-12.,   0.,   4.],
           [ -3.,   6.,  -3.]])
    """
    x, y, z = a[:, 0], a[:, 1], a[:, 2]
    x2, y2, z2 = b[:, 0], b[:, 1], b[:, 2]

    n = np.zeros_like(a, dtype=np.float64, order='F')

    # using ne is ~2x faster than np.cross
    ne.evaluate("y*z2 - z*y2", out=n[:, 0])
    ne.evaluate("z*x2 - x*z2", out=n[:, 1])
    ne.evaluate("x*y2 - y*x2", out=n[:, 2])

    return n


def interp2d(x: np.ndarray, y: np.ndarray, z: np.ndarray, xp: np.ndarray, yp: np.ndarray, method="linear")\
        -> np.ndarray:
    """
    Faster alternative for :obj:`scipy.interpolate.interp2d` for gridded and equally spaced 2D data
    and kind="linear"

    :param x: x coordinate vector (numpy 1D array)
    :param y: y coordinate vector (numpy 1D array)
    :param z: z values (numpy 2D array)
    :param xp: numpy 1D arrays holding the interpolation points x coordinates
    :param yp: numpy 1D arrays holding the interpolation points y coordinates
    :param method: "linear" or "nearest" (string)
    :return: interpolated values as 1D array
    
    >>> interp2d(np.array([2, 3]), np.array([1, 2]), np.array([[0, 1], [2, 3]]),\
        np.array([2.25, 2.75]), np.array([1, 1.5]))
    array([0.25, 1.75])

    >>> interp2d(np.array([2, 3]), np.array([1, 2]), np.array([[0, 1], [2, 3]]),\
        np.array([2.25, 2.75]), np.array([1, 1.5]), method="nearest")
    array([0, 3])
    """
    # check input shapes
    if x.ndim != 1 or y.ndim != 1:
        raise TypeError("x and y need to be vectors")
    if (z.ndim != 2) or (z.shape[0] != y.shape) != (z.shape[1] != x.shape):
        raise TypeError("z needs to be 2 dimensional with size y*x")
    if xp.ndim != 1 or yp.ndim != 1:
        raise TypeError("xp and yp need to be vectors")
    if xp.shape != yp.shape:
        raise TypeError("xp and yp need to have the same shape.")

    # check values of xp and yp:
    if np.min(xp) < x[0] or np.max(xp) > x[-1]:
        raise ValueError("xp value outside data range")
    if np.min(yp) < y[0] or np.max(yp) > y[-1]:
        raise ValueError("yp value outside data range")

    x0, x1, y0, y1 = x[0], x[1], y[0], y[1]
    xt = ne.evaluate("1 / (x1 - x0) * (xp - x0)")
    yt = ne.evaluate("1 / (y1 - y0) * (yp - y0)")

    if method == 'linear':
        # bi-linear interpolation: https://en.wikipedia.org/wiki/Bilinear_interpolation
        # rewrite the equation to:
        # zi = (1 - yr)*(xr*z[yc, xc+1] + z[yc, xc]*(1-xr)) + yr*(xr*z[yc+1, xc+1]+ z[yc+1, xc]*(1-xr))
        #
        # with:
        # xt, yt: sample points (x, y in Wikipedia)
        # xc, yc: integer part of xt, yt coordinates (x1, y1 in Wikipedia)
        # xr, yr: float part of xt, yt coordinates ((x-x1)/(x2-x1), (y-y1)/(y2-y1) in Wikipedia)

        # this part is faster than using np.divmod
        xc = xt.astype(int)
        yc = yt.astype(int)
        xr = xt - xc
        yr = yt - yc

        # handle case where we are exactly add the edge of the data, the index is
        xcp = np.where(xc < x.shape[0]-1, xc+1, xc)
        ycp = np.where(yc < y.shape[0]-1, yc+1, yc)

        a, b, c, d = z[yc, xc], z[ycp, xc], z[yc, xcp], z[ycp, xcp]

        # rearranged form with only 4 multiplications for speedup
        return ne.evaluate("(1 - yr) * (xr * (c - a) + a) + yr * (xr * (d - b) + b)")
    
    elif method == 'nearest':
        # fast rounding for positive numbers
        xc = (xt + 0.5).astype(int)
        yc = (yt + 0.5).astype(int)

        return z[yc, xc]
    else:
        raise ValueError("Invalid method")
