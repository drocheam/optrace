
import numpy as np
from typing import Callable
import warnings


# TODO check if working
class SurfaceFunction:

    def __init__(self,
                 func:          Callable[[np.ndarray, np.ndarray], np.ndarray],
                 r:             float,
                 mask:          Callable[[np.ndarray, np.ndarray], np.ndarray] = None,
                 derivative:    Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]] = None,
                 hits:          Callable[[np.ndarray, np.ndarray], np.ndarray] = None,
                 minz:          float = None,
                 maxz:          float = None)\
            -> None:
        """

        :param func:
        :param r:
        :param mask:
        :param derivative:
        :param hits:
        :param minz:
        :param maxz:
        """
        self.func = func
        self.r = float(r)
        self.mask = mask
        self.derivative = derivative
        self.hits = hits

        # get offset at (0, 0), gets removed later
        self.off = func(np.array([0.]), np.array([0.]))[0]

        if maxz is None or minz is None:
            warnings.warn("WARNING: minz or maxz missing, the values will be determined automatically."
                  "This is however less accurate than specifying them.")
            self.minz, self.maxz = self.__findBounds()
            self.minz, self.maxz = self.minz - self.off, self.maxz - self.off
        else:
            self.minz, self.maxz = minz - self.off, maxz - self.off

        self._lock = True

    def hasDerivative(self) -> bool:
        """

        :return:
        """
        return self.derivative is not None

    def hasHits(self) -> bool:
        """

        :return:
        """
        return self.hits is not None

    def getHits(self, p: np.ndarray, s: np.ndarray) -> np.ndarray:
        """

        :param p:
        :param s:
        :return:
        """
        return self.hits(p, s)

    def getDerivative(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """

        :param x:
        :param y:
        :return:
        """
        return self.derivative(x, y)

    def getMask(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """

        :param x:
        :param y:
        :return:
        """
        # values outside circle are masked out
        m = np.zeros_like(x, dtype=bool)
        inside = x**2 + y**2 <= self.r**2

        m[inside] = self.mask(x[inside], y[inside]) if self.mask is not None else True

        return m

    def getValues(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """

        :param x:
        :param y:
        :return:
        """
        inside = x**2 + y**2 <= self.r**2

        z = np.full_like(x, self.maxz, dtype=np.float64)
        z[inside] = self.func(x[inside], y[inside]) - self.off
    
        return z

    def __findBounds(self) -> tuple[float, float]:
        """

        :return:
        """
        # how to regularly sample a circle area, while sampling 
        # as much different phi and r values as possible?
        # => sunflower sampling for surface area
        # see https://stackoverflow.com/a/44164075

        N = 10000
        ind = np.arange(0, N, dtype=np.float64) + 0.5

        r = np.sqrt(ind/N) * self.r
        phi = 2*np.pi * (1 + 5**0.5)/2 * ind

        vals = self.func(r*np.cos(phi), r*np.sin(phi))
        
        # mask out invalid values
        mask = self.getMask(r*np.cos(phi), r*np.sin(phi))
        vals[~mask] = np.nan
      
        # in many cases the minimum and maximum are at the center or edge of the surface
        # => sample them additionally

        # values at surface edge
        phi2 = np.linspace(0, 2*np.pi, 1001)  # N is odd, since step size is 1/(N-1) * 2*pi, 
        r2 = np.full_like(phi2, self.r, dtype=np.float64)
        vals2 = self.func(r2*np.cos(phi2), r2*np.sin(phi2))
        mask = self.getMask(r2*np.cos(phi2), r2*np.sin(phi2))
        vals2[~mask] = np.nan

        # surface center
        vals3 = self.func(np.array([0.]), np.array([0.]))
        if not self.getMask(np.array([0.]), np.array([0.])):
            vals3 = np.nan

        # add all surface values into one array
        vals = np.concatenate((vals, vals2, vals3))

        # find minimum and maximum value
        minz = np.nanmin(vals)
        maxz = np.nanmax(vals)

        return minz, maxz

    def crepr(self):
        """ Compact state representation using only lists and immutable types """
        return [self.r, self.off, self.minz, self.maxz, id(self.derivative), id(self.func), id(self.hits), id(self.mask)]

    def __setattr__(self, key, val):

        if "_lock" in self.__dict__ and self._lock: # and key != "_lock":
            raise RuntimeError("Changing SurfaceFunction properties after initialization is prohibited."\
                               "Create a new SurfaceFunction and assign it to the parent Surface.")
        
        self.__dict__[key] = val
