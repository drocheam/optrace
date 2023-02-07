
import numpy as np  # for ndarray type

from ..surface.rectangular_surface import RectangularSurface
from .volume import Volume


class BoxVolume(Volume):

    def __init__(self,
                 dim:           list | np.ndarray,
                 length:        float,
                 pos:           (list | np.ndarray),
                 color:         tuple[float] = None,
                 opacity:       float = 0.3,
                 **kwargs)\
            -> None:
        """
        Create a box volume.

        :param dim: side lengths of the front surface, see class RectangularSurface
        :param length: length of the box in z-direction
        :param pos: position of front surface center
        :param color: sRGB color tuple, optional
        :param opacity: plotting opacity, value range 0.0 - 1.0
        :param kwargs: additional keyword args for class Volume, Element and BaseClass
        """
        front = RectangularSurface(dim=dim)
        back = RectangularSurface(dim=dim)
        super().__init__(front, back, pos, d1=0, d2=length, color=color, opacity=opacity, **kwargs)

