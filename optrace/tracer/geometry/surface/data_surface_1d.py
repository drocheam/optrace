
import numpy as np  # ndarray type

from .data_surface_2d import DataSurface2D  # parent class


class DataSurface1D(DataSurface2D):
    
    rotational_symmetry: bool = True  #: has the surface rotational symmetry?

    _1D: bool = True
    """1D or 2D data surface"""
    
    def __init__(self,
                 r:                 float,
                 data:              np.ndarray,
                 parax_roc:         float = None,
                 **kwargs)\
            -> None:
        """
        Define a data surface with rotational symmetry.

        :param r: surface radius
        :param parax_roc: paraxial radius of curvature, optional
        :param data: equi-spaced data array, going from 0 to r
        :param kwargs: additional keyword arguments for parent classes
        """
        self._lock = False

        super().__init__(r=r, data=data, parax_roc=parax_roc, **kwargs)
