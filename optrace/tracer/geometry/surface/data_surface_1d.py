
import numpy as np  # calculations

from ... import misc  # calculations
from ...misc import PropertyChecker as pc  # check types and values
from .data_surface_2d import DataSurface2D


class DataSurface1D(DataSurface2D):
    
    rotational_symmetry: bool = True

    _1D: bool = True
    """1D or 2D data surface"""
    
    def __init__(self,
                 r:                 float,
                 data:              np.ndarray,
                 parax_roc:         float = None,
                 **kwargs)\
            -> None:
        """

        :param r:
        :param parax_roc:
        :param data: 
        """
        self._lock = False

        super().__init__(r=r, data=data, parax_roc=parax_roc, **kwargs)
