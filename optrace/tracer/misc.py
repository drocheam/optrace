
from datetime import datetime  # date for fallback file naming
from pathlib import Path  # path handling for image saving

from typing import Callable, Any  # Callable and Any types
from functools import wraps  # wrapping of functions
import time  # timing

from PIL import Image as PILImage
import scipy.interpolate  # linear interpolation
import scipy.integrate
import numpy as np  # calculations
import numexpr as ne  # faster calculations


def random():
    """less secure (= lower quality of randomness), but faster random number generator"""
    return np.random.Generator(np.random.SFC64())

def load_image(path: str) -> np.ndarray:
    return np.asarray(PILImage.open(path).convert("RGB"), dtype=float) / 2**8

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


def uniform2(a: float, b: float, c: float, d: float, N: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Stratified Sampling 2D.
    
    :param a: lower x-bound
    :param b: upper x-bound
    :param c: lower y-bound
    :param d: upper y-bound
    :param N: number of values
    :return: tuple of random values inside [a, b] and [c, d]
    """
    # return np.random.uniform(a, b, N), np.random.uniform(c, d, N)

    if not N:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    
    # side length and number of remaining elements
    N2 = int(np.sqrt(N))
    dN = N - N2**2

    # grid spacing
    dba = (b-a)/N2
    ddc = (d-c)/N2

    # create rectangular grid and add dither
    X, Y = np.mgrid[a:b-dba:N2*1j, c:d-ddc:N2*1j]
    X += random().uniform(0, dba, X.shape)
    Y += random().uniform(0, ddc, Y.shape)

    # add remaining elements and shuffle everything
    x = np.concatenate((X.ravel(), random().uniform(a, b, dN)))
    y = np.concatenate((Y.ravel(), random().uniform(c, d, dN)))

    # shuffle order
    ind = np.arange(N)
    random().shuffle(ind)

    return x[ind], y[ind]


def uniform(a: float, b: float, N: int) -> np.ndarray:   
    """
    Stratified Sampling 1D.

    :param a: lower bound
    :param b: upper bound
    :param N: number of values
    :return: random values
    """
    # return np.random.uniform(a, b, N)
    if not N:
        return np.array([], dtype=np.float64)

    dba = (b-a)/N  # grid spacing
    x = np.linspace(a, b - dba, N) + random().uniform(0., dba, N)  # grid + dither

    random().shuffle(x)
    return x


def ring_uniform(ri: float, r: float, N: int, polar: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate uniform random positions inside a ring area (annulus) with stratified sampling.
    This is done by equi-areal mapping from grid to disc to annulus.
    Set ri=0 so you get disc sampling.

    :param ri: inner radius in mm
    :param r: outer radius in mm
    :param N: number of points
    :param polar: if polar coordinates should be returned instead of cartesian ones
    :return: random x and y positions (or r, phi)
    """
    x, y = uniform2(-r, r, -r, r, N)

    r_ = np.zeros_like(x)
    theta = np.zeros_like(x)
    x2, y2 = x**2, y**2  # store since used multiple times

    # Shirley's Equal-Area Mapping
    # see https://jcgt.org/published/0005/02/01/paper.pdf

    m = x2 > y2
    r_[m] = xm = x[m]
    ym, pi = y[m], np.pi
    theta[m] = ne.evaluate("pi/4 * ym/xm")

    m = (x2 <= y2) & (y2 > 0)
    r_[m] = ym = y[m]
    xm = x[m]
    theta[m] = ne.evaluate("pi/2 - pi/4 * xm/ym")

    # equi-areal disc to ring mapping
    if ri:
        # (1 - ((r < 0) << 1)) is a fast inline method for: 1 if r >= 0, -1 if r < 0
        r_ = ne.evaluate("(1 - ((r_ < 0) << 1)) * sqrt(ri**2 + r_**2*(1 - (ri/r)**2))")

    if not polar:
        return ne.evaluate("r_*cos(theta)"), ne.evaluate("r_*sin(theta)")

    else:
        # r is signed, remove sign and convert it to an angle
        theta[r_ < 0] -= np.pi
        return np.abs(r_), theta

# class only used for separate namespace
class PropertyChecker:

    @staticmethod
    def check_type(key, val, type_) -> None:
        if not isinstance(val, type_):
            types = str(type_).replace("|", "or")
            raise TypeError(f"Property '{key}' needs to be of type(s) {types}, but is {type(val)}.")

    @staticmethod
    def check_not_above(key, val, cmp) -> None:
        if val > cmp:
            raise ValueError(f"Property '{key}' needs to be below or equal to {cmp}, but is {val}.")

    @staticmethod
    def check_not_below(key, val, cmp) -> None:
        if val < cmp:
            raise ValueError(f"Property '{key}' needs to be above or equal to {cmp}, but is {val}.")

    @staticmethod
    def check_above(key, val, cmp) -> None:
        if val <= cmp:
            raise ValueError(f"Property '{key}' needs to be above {cmp}, but is {val}.")

    @staticmethod
    def check_below(key, val, cmp) -> None:
        if val >= cmp:
            raise ValueError(f"Property '{key}' needs to be below {cmp}, but is {val}.")

    @staticmethod
    def check_callable(key, val) -> None:
        if not callable(val):
            raise TypeError(f"{key} needs to be callable, but is {type(val)}.")
    
    @staticmethod
    def check_none_or_callable(key, val) -> None:
        if val is not None and not callable(val):
            raise TypeError(f"{key} needs to be callable or None, but is '{type(val)}'.")
    
    @staticmethod
    def check_if_element(key, val, list_) -> None:
        if val not in list_:
            raise ValueError(f"Invalid value '{val}' for property '{key}'. Needs to be one of {list_}, but is '{val}'.")


def random_from_distribution(x: np.ndarray, f: np.ndarray, S: int | np.ndarray, kind="continuous") -> np.ndarray:
    """
    Get randomly distributed values with f(x) as distribution using inverse transform sampling.
    kind specifies whether f(x) is interpreted as continuous or discrete distribution.
    Linear interpolation between data points in continous mode

    f does not need to be normalized

    :param x: pdf x values
    :param f: pdf y values
    :param S: sampler. Can be a number of samples or a list of samples in [0, 1]
    :param kind: "continuous" or "discrete"
    :return: N randomly distributed values according to pdf

    >>> ch = random_from_distribution(np.array([0., 1.]), np.array([0., 1.]), 10000)
    >>> (0.49 < np.mean(ch) < 0.51) and (np.max(ch) > 0.99) and (np.min(ch) < 0.01)
    True
    """

    # check if cdf is non-zero and monotonically increasing
    if not np.sum(f):
        raise RuntimeError("Cumulated probability is zero.")
    elif np.min(f) < 0:
        raise RuntimeError("Got negative value in pdf.")

    # exclude zeros values for discrete distribution, set interpolation to nearest
    if kind == "discrete":
        x_ = x[f > 0]
        f_ = f[f > 0]

        F = np.cumsum(f_)  # we can't use cumtrapz, since we need the actual sum

        # make inverse of cdf
        iF = scipy.interpolate.interp1d(F, x_, assume_sorted=True, kind="next", fill_value="extrapolate")

        # random variable, rescaled S or create new uniform variable
        X = F[-1]*S if isinstance(S, np.ndarray) else uniform(0, F[-1], S)

    # use all values, linear interpolation
    else:
        F = scipy.integrate.cumtrapz(f, initial=0)  # much more precise than np.cumsum

        # make inverse of cdf
        # don't use higher orders than linear, since function could be noisy
        iF = scipy.interpolate.interp1d(F, x, assume_sorted=True, kind="linear")

        # random variable, rescaled S or create new uniform variable
        X = (F[0] + S*(F[-1] - F[0])) if isinstance(S, np.ndarray) else uniform(F[0], F[-1], S)

    return iF(X)  # sample to get random values


def rdot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    row wise scalar product for two or three dimension.
    Coordinate values in matrix columns.

    :param a: 2D matrix, with shape (N, 2) or (N,  3)
    :param b: 2D matrix, with shape (N, 2) or (N,  3)
    :return: result vector, shape (N, )

    >>> rdot(np.array([[1., 2., 3.], [4., 5., 6.]]), np.array([[-1., 2., -3.], [7., 8., 9.]]))
    array([ -6., 122.])
    
    >>> rdot(np.array([[2., 3.], [4., 5.]]), np.array([[-1., 2.], [8., 9.]]))
    array([ 4., 77.])
    """
    if a.shape[1] == 3:
        return a[:, 0]*b[:, 0] + a[:, 1]*b[:, 1] + a[:, 2]*b[:, 2]

    elif a.shape[1] == 2:
        return a[:, 0]*b[:, 0] + a[:, 1]*b[:, 1]

    else:
        raise RuntimeError("Invalid number of dimensions.")


def part_mask(cond1: np.ndarray, cond2: np.ndarray) -> np.ndarray:
    """
    set True values of bool mask cond1 to values of cond2, returns resulting mask

    :param cond1: 1D array
    :param con2: 1D array
    :return: 1D array with shape of cond1

    >>> part_mask(np.array([True, False, False, True]), np.array([True, False]))
    array([ True, False, False, False])
    """
    wc = np.zeros_like(cond1)
    wc[cond1] = cond2
    return wc


def normalize(a: np.ndarray) -> np.ndarray:
    """
    faster vector normalization for vectors in axis=1.
    Zero length vectors can normalized to np.nan.

    :param a: array to normalize, shape (N, 3)
    :return: normalized values, shape (N, 3)

    >>> a = np.array([[1., 2., 3.], [4., 5., 6.]])
    >>> normalize(a)
    array([[...0.26726124, 0.53452248, 0.80178373],
           [...0.45584231, 0.56980288, 0.68376346]])
    """
    nan = np.nan
    valid = np.any(a != 0, axis=1)[:, np.newaxis]
    x, y, z = a[:, 0, np.newaxis], a[:, 1, np.newaxis], a[:, 2, np.newaxis]
    return ne.evaluate("a/where(valid, sqrt(x**2 + y**2 + z**2), nan)")


def cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    faster alternative to :func:`numpy.cross` for 3 element vectors and axis=1

    :param a: first array, shape (N, 3)
    :param b: second array, shape (N, 3)
    :return: cross product, shape (N, 3)

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

def save_with_fallback(path:        str,
                       sfunc:       Callable,
                       fname:       str,
                       ending:      str,
                       overwrite:   bool = False,
                       silent:      bool = False)\
        -> str:
    """saving with fallback path if the file exists and overwrite=False"""

    # called when invalid path or file exists but overwrite=False
    def fallback():
        # create a valid path and filename
        wd = Path.cwd()
        filename = f"{fname}_" + datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f%z') + ending
        path_ = str(wd / filename)

        # resave
        if not silent:
            print(f"Failed saving {fname}, resaving as \"{path_}\".")
        sfunc(path_)

        return path_

    # append file ending if the path provided has none
    if path[-len(ending):] != ending:
        path += ending

    # check if file already exists
    exists = Path(path).exists()

    if overwrite or not exists:
        try:
            sfunc(path)
            if not silent:
                print(f"Saved {fname} as \"{path}\".")
            return path
        except:
            return fallback()
    else:
        return fallback()

