
from typing import Callable, Any  # Callable and Any types
from functools import wraps  # wrapping of functions
import os  # cpu count
import time  # timing

import numpy as np  # calculations


def cpu_count() -> int:
    """
    Number of logical cpu cores assigned to this process (Python >= 3.13)
    Number of logical cpu cores (Python < 3.13)
    Can be overridden by setting the PYTHON_CPU_COUNT environment variable or running python -X cpucores
    Setting by PYTHON_CPU_COUNT must be between 1-64.

    :return: cpu count
    """
    count = os.process_cpu_count() if hasattr(os, "process_cpu_count") else os.cpu_count()
    count = count or 1

    # while cpu_count and process_cpu_count handle the PYTHON_CPU_COUNT by now,
    # this was not the case for Python < 3.13
    # additionally, it also does not seem to work when the env variable is set at runtime?
    if "PYTHON_CPU_COUNT" in os.environ:
        count = int(os.environ["PYTHON_CPU_COUNT"])

    if not (1 <= count <= 64):
        raise RuntimeError(f"Invalid core count {count}, must be between 1 and 64.")

    return count


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


def binning_indices_2d(x: np.ndarray, y: np.ndarray, w: np.ndarray, Nx: int, Ny: int, extent: list)\
        -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate x and y bin indices for a 2D histogram from provided positions x, y.
    Positions outside the extent are set to indices (0, 0) and their weight is set to zero in result wm.
    Binning includes all edges.

    :param x: x position array
    :param y: y position array
    :param w: weight array
    :param Nx: number of x pixels
    :param Ny: number of y pixels
    :param extent: image position extent as list [x0, x1, y0, y1]
    :return: y bin indices, x bin indices, weight array with values at invalid position indices set to zero
    """
    # extent side lengths
    s = extent[1] - extent[0], extent[3] - extent[2]

    # calculate relative pixel positions
    xi = np.floor(Nx / s[0] * (x - extent[0])).astype(np.int32)
    yi = np.floor(Ny / s[1] * (y - extent[2])).astype(np.int32)

    # include positive edge values
    yi[y == extent[3]] = Ny - 1
    xi[x == extent[1]] = Nx - 1

    # set weights of outside rays to zero and their position to (0, 0)
    outside = (xi < 0) | (yi < 0) | (yi >= Ny) | (xi >= Nx)
    wm = np.where(outside, 0, w)
    yi[outside] = 0
    xi[outside] = 0

    return xi, yi, wm


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


def masked_assign(cond1: np.ndarray, cond2: np.ndarray) -> np.ndarray:
    """
    set True values of bool mask cond1 to values of cond2, returns resulting mask

    :param cond1: 1D array
    :param con2: 1D array
    :return: 1D array with shape of cond1

    >>> masked_assign(np.array([True, False, False, True]), np.array([True, False]))
    array([ True, False, False, False], dtype=bool)
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

