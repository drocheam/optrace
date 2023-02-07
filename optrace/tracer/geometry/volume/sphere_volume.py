
import numpy as np  # for ndarray type

from ..surface.spherical_surface import SphericalSurface
from .volume import Volume


class SphereVolume(Volume):

    def __init__(self,
                 R:             float,
                 pos:           (list | np.ndarray),
                 color:         tuple[float] = None,
                 opacity:       float = 0.3, 
                 **kwargs)\
            -> None:
        """
        Create a sphere volume.

        :param R: sphere radius
        :param pos: position of sphere center
        :param color: sRGB color tuple, optional
        :param opacity: plotting opacity, value range 0.0 - 1.0
        :param kwargs: additional keyword args for class Volume, Element and BaseClass
        """
        front = SphericalSurface(R=R, r=0.9999*R)
        back = SphericalSurface(R=-R, r=0.9999*R)
        super().__init__(front, back, pos, d1=R, d2=R, opacity=opacity, color=color, **kwargs)
    
