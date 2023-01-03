
import numpy as np  # calculations and ndarray type

from ..refraction_index import RefractionIndex  # material refraction index
from .surface import CircularSurface  # Surface type
from .lens import Lens  # parent class

from ..misc import PropertyChecker as pc  # check types and values


# TODO handle D = 0

class IdealLens(Lens):

    is_ideal: bool = True

    def __init__(self,
                 r:       float,
                 D:       float,
                 pos:     (list | np.ndarray),
                 n2:      RefractionIndex = None,
                 **kwargs)\
            -> None:
        """

        :param n2: refraction index behind lens (positive z direction) (RefractionIndex object)
        :param pos: 3D position of lens center (list or numpy array)
        """

        pc.check_type("D", D, int | float)
        np.asarray_chkfinite(D)
        self.D = float(D)

        if not D:
            raise ValueError("Optical Power needs to be non-zero")

        super().__init__(front=CircularSurface(r=r),
                         back=CircularSurface(r=r),
                         n=RefractionIndex("Constant", n=1),  # actually not meaningful in any way
                         pos=pos, 
                         d=0, 
                         n2=n2, 
                         **kwargs)

