
import numpy as np  # for ndarray type

from .volume import Volume
from ..surface.circular_surface import CircularSurface


class CylinderVolume(Volume):

    def __init__(self,
                 r:             float,
                 length:        float,
                 pos:           (list | np.ndarray),
                 color:         tuple[float] = None,
                 opacity:       float = 0.3, 
                 **kwargs)\
            -> None:
        """
        Create a cylinder volume.

        :param r: radius of the cylinder
        :param length: length of the cylinder
        :param pos: position of front disk center
        :param opacity: plotting opacity, value range 0.0 - 1.0
        :param color: sRGB color tuple (optional)
        :param kwargs: additional keyword args for class Volume, Element and BaseClass
        """
        front = CircularSurface(r=r)
        back = CircularSurface(r=r)

        super().__init__(front, back, pos, d1=0, d2=length, color=color, opacity=opacity, **kwargs)

