
from typing import Callable, Any  # Callable and Any types
from functools import wraps  # wrapping of functions
import os  # cpu count
import time  # timing
import sys

import scipy.interpolate  # linear interpolation
import scipy.integrate
import numpy as np  # calculations
import tqdm  # progressbar

from .. import global_options

random = np.random.Generator(np.random.SFC64())
"""less secure (= lower quality of randomness), but slightly faster random number generator"""


# TODO separate files for random functions, property checker and progressbar?
# TODO progressbar and property checker in main optrace mainspace?

def cpu_count() -> int:
    """
    Number of logical cpu cores assigned to this process (Python >= 3.13)
    Number of logical cpu cores (Python < 3.13)

    :return: cpu count
    """
    count = os.process_cpu_count() if hasattr(os, "process_cpu_count") else os.cpu_count()
    return count or 1  # count could be None, set to 1


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


def stratified_rectangle_sampling(a: float, b: float, c: float, d: float, N: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Stratified Sampling 2D. 
    Lower discrepancy than scipy.stats.qmc.LatinHypercube.
    
    :param a: lower x-bound
    :param b: upper x-bound
    :param c: lower y-bound
    :param d: upper y-bound
    :param N: number of values
    :return: tuple of random values inside [a, b] and [c, d]
    """

    if not N:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    
    # side length and number of remaining elements
    N2 = int(np.sqrt(N))
    dN = N - N2**2

    # grid spacing
    dba = (b-a)/N2
    ddc = (d-c)/N2

    # create rectangular grid and add dither
    Y, X = np.mgrid[c:d-ddc:N2*1j, a:b-dba:N2*1j]
    X += random.uniform(0, dba, X.shape)
    Y += random.uniform(0, ddc, Y.shape)

    # add remaining elements and shuffle everything
    x = np.concatenate((X.ravel(), random.uniform(a, b, dN)))
    y = np.concatenate((Y.ravel(), random.uniform(c, d, dN)))

    # shuffle order
    ind = np.arange(N)
    random.shuffle(ind)

    return x[ind], y[ind]


def stratified_interval_sampling(a: float, b: float, N: int, shuffle: bool = True) -> np.ndarray:
    """
    Stratified Sampling 1D.

    :param a: lower bound
    :param b: upper bound
    :param N: number of values
    :param shuffle: shuffle order of values
    :return: random values
    """
    # return np.random.uniform(a, b, N)
    if not N:
        return np.array([], dtype=np.float64)

    dba = (b-a)/N  # grid spacing
    x = np.linspace(a, b - dba, N) + random.uniform(0., dba, N)  # grid + dither

    if shuffle:
        random.shuffle(x)
    return x


def stratified_ring_sampling(ri: float, r: float, N: int, polar: bool = False) -> tuple[np.ndarray, np.ndarray]:
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
    x, y = stratified_rectangle_sampling(-r, r, -r, r, N)

    r_ = np.zeros_like(x)
    theta = np.zeros_like(x)
    x2, y2 = x**2, y**2  # store since used multiple times

    # Shirley's Equal-Area Mapping
    # see https://jcgt.org/published/0005/02/01/paper.pdf

    m = x2 > y2
    r_[m] = x[m]
    theta[m] = np.pi/4 * y[m]/x[m]

    m = (x2 <= y2) & (y2 > 0)
    r_[m] = y[m]
    theta[m] = np.pi/2 - np.pi/4 * x[m] / y[m]

    # equi-areal disc to ring mapping
    if ri:
        # (1 - ((r < 0) << 1)) is a fast inline method for: 1 if r >= 0, -1 if r < 0
        r_ = (1 - ((r_ < 0) << 1)) * np.sqrt(ri**2 + r_**2*(1 - (ri/r)**2))

    if not polar:
        return r_*np.cos(theta), r_*np.sin(theta)

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


class progressbar:

    def __init__(self, text: str, steps: int, **kwargs):
        """
        progressbar wrapper class. Uses tqdm internally

        :param text: text to display at front of progressbar
        :param steps: number of steps/iterations
        :param kwargs: additional parameters to tqdm
        """

        if global_options.show_progressbar:
            self.bar = tqdm.tqdm(desc=text, total=steps, disable=None, 
                                 bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]', **kwargs)
        else:
            self.bar = None

    def update(self, condition: bool = True) -> None:
        """
        Increment/update the state by one.
        
        :param condition: only update if condition is met
        """
        if self.bar is not None and condition:
            self.bar.update(1)

    def finish(self) -> None:
        """finish and close the progressbar"""
        if self.bar is not None:
            self.bar.close()


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
        X = F[-1]*S if isinstance(S, np.ndarray) else stratified_interval_sampling(0, F[-1], S)

    # use all values, linear interpolation
    else:
        F = scipy.integrate.cumulative_trapezoid(f, initial=0)  # much more precise than np.cumsum

        # make inverse of cdf
        # don't use higher orders than linear, since function could be noisy
        iF = scipy.interpolate.interp1d(F, x, assume_sorted=True, kind="linear")

        # random variable, rescaled S or create new uniform variable
        X = (F[0] + S*(F[-1] - F[0])) if isinstance(S, np.ndarray) else stratified_interval_sampling(F[0], F[-1], S)

    return iF(X)  # sample to get random values


def rdot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    row wise scalar product for two or three dimension.
    Coordinate values in matrix columns.
    For (N, 2) and (N, 3) faster than np.einsum('ij,ij->i', a, b) and (a*b).sum(axis=1)

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
    with np.errstate(invalid="ignore"):  # we use nan as zero division indicator, suppress warnings
        return a / np.sqrt(a[:, 0]**2 + a[:, 1]**2 + a[:, 2]**2)[:, np.newaxis]

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
    n = np.zeros_like(a, dtype=np.float64, order='F')
    n[:, 0] = a[:, 1]*b[:, 2] - a[:, 2]*b[:, 1]
    n[:, 1] = a[:, 2]*b[:, 0] - a[:, 0]*b[:, 2]
    n[:, 2] = a[:, 0]*b[:, 1] - a[:, 1]*b[:, 0]

    return n

