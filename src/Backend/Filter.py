
"""
Filter class:
A filter is a surface with wavelength dependent or constant transmittance.
Useful for color filters or apertures.
"""


import numpy as np

from Backend.Surface import Surface  # for the Filter surface
import Backend.Color as Color  # for the calculation of the filter color 

from typing import Callable  # for function type hints
import copy  # for copy.deepcopy


class Filter:

    def __init__(self, 
                 Surface:       Surface, 
                 pos:           (list | np.ndarray),
                 filter_type:   str = "Constant",
                 tau:           float = 0.,
                 func:          Callable[[np.ndarray], np.ndarray] = None)\
            -> None:
        """
        Create a Filter object.

        :param Surface: Surface object
        :param pos: 3D position of Filter center (numpy array or list)
        :param filter_type: "Constant" or "Function" (string)
        :param tau: transmittance (float between 0 and 1), used for filter_type="Constant"
        :param func: transmittance function, used for filter_type="Function"
        """

        # use a Surface copy, since we change its position in 3D space
        self.Surface = Surface.copy()

        self.filter_type = filter_type
        self.tau = float(tau)
        self.func = func

        self.moveTo(pos)

        match filter_type:
            case "Constant": 
                if not (0 <= tau <= 1):
                    raise ValueError("Transmittance tau needs to be inside range [0, 1]")

            case "Function":
                if func is None:
                    raise ValueError("filter_type='Function', but Function not specified")

            case _:
                raise ValueError(f"Invalid filter_type '{filter_type}'.")



    def setSurface(self, surf: Surface) -> None:
        """
        Assign a new Surface to the Filter.

        :param surf: Surface to assign
        """
        pos = self.Surface.pos
        self.Surface = surf.copy()
        self.Surface.moveTo(pos)

    def moveTo(self, pos: (list | np.ndarray)) -> None:
        """
        Moves the filter in 3D space.

        :param pos: new 3D position of filter center (list or numpy array)
        """
        self.Surface.moveTo(pos)
    
    def copy(self) -> 'Filter':
        """
        Return a fully independent copy of the Filter object.

        :return: copy
        """
        return copy.deepcopy(self)

    @property
    def pos(self) -> np.ndarray:
        """ position of the Filter center """
        return self.Surface.pos

    @property
    def extent(self) -> tuple[float, float, float, float, float, float]:
        """ 3D extent of the Filter"""
        return self.Surface.getExtent()

    def __call__(self, wl: np.ndarray) -> np.ndarray:        
        """
        Return filter transmittance in range [0, 1] for specified wavelengths.

        :param wl: wavelengths
        :return: transmittance
        """
        match self.filter_type:
            case "Constant":
                return np.full_like(wl, self.tau, dtype=np.float32)

            case "Function":
                return self.func(wl)

            case _:
                raise ValueError(f"filter_type '{filter_type}' not implemented.")


    def getColor(self) -> tuple[float, float, float]:
        """
        Get sRGB color tuple from filter transmission curve

        :return: sRGB color tuple, with each channel in range [0, 1]
        """
        return tuple(Color.ColorUnderDaylight(self.__call__))

    def getCylinderSurface(self, nc: int = 100, d: float = 0.1) \
            -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get a 3D surface representation of the filter cylinder for plotting.

        :param nc: number of surface edge points (int)
        :param d: thickness for visualization (float)
        :return: tuple of coordinate arrays X, Y, Z (2D numpy arrays)
        """

        # get Surface edge. The edge is the same for both cylinder sides
        X1, Y1, Z1 = self.Surface.getEdge(nc)

        # create coordinates for cylinder front edge and back edge
        X = np.column_stack((X1, X1))
        Y = np.column_stack((Y1, Y1))
        Z = np.column_stack((Z1, Z1 + d))  # shift back edge by d

        return X, Y, Z
