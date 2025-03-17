import scipy.interpolate  # linear interpolation
import scipy.integrate
import numpy as np  # calculations

_random = np.random.Generator(np.random.SFC64())
"""less secure (= lower quality of randomness), but slightly faster random number generator"""

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
    X += _random.uniform(0, dba, X.shape)
    Y += _random.uniform(0, ddc, Y.shape)

    # add remaining elements and shuffle everything
    x = np.concatenate((X.ravel(), _random.uniform(a, b, dN)))
    y = np.concatenate((Y.ravel(), _random.uniform(c, d, dN)))

    # shuffle order
    ind = np.arange(N)
    _random.shuffle(ind)

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
    x = np.linspace(a, b - dba, N) + _random.uniform(0., dba, N)  # grid + dither

    if shuffle:
        _random.shuffle(x)
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


def inverse_transform_sampling(x: np.ndarray, f: np.ndarray, S: int | np.ndarray, kind="continuous") -> np.ndarray:
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


