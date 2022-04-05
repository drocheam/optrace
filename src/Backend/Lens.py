
"""
Lens class:
A lens is an geometrical object with two refractive surfaces.
A refractive index is specified for the material and one for the area behind the lens (optional)
"""

import numpy as np
import copy

from Backend.RefractionIndex import *
from Backend.Surface import *
from Backend.SObject import *

# TODO error message when surface is point or line

# TODO setattr handling

class Lens(SObject):

    def __init__(self, 
                 front: Surface, 
                 back:  Surface,
                 n:     RefractionIndex,
                 pos:   (list | np.ndarray), 
                 de:    float = 0,
                 d1:    float = None,
                 d2:    float = None,
                 n2:    RefractionIndex = None)\
            -> None:
        """
        Creates a lens object using 2 surfaces and additional properties.
        Of the thickness parameters only one of de or (d1, d2) needs to be specified.

        :param front: front surface (smaller z-position) (Surface object)
        :param back: back surface (higher z-position) (Surface object)
        :param de: edge thickness (distance between highest point of front and smallest point of back surface), (float)
        :param d1: thickness of front surface, relative to position (float)
        :param d2: thickness of back surface, relative to position (float)
        :param n: material refraction index (RefractionIndex object)
        :param n2: refraction index behind lens (positive z direction) (RefractionIndex object)
        :param pos: 3D position of lens center (list or numpy array)
        """

        self.n = n
        self.n2 = n2
        d1 = float(d1) if d1 is not None else d1
        d2 = float(d2) if d2 is not None else d2

        if de is not None and d1 is None and d2 is None:
            d1 = de / 2. + front.maxz - front.pos[2]
            d2 = de / 2. + back.pos[2] - back.minz

        elif d1 is None or d2 is None:
            raise ValueError("Both thicknesses d1, d2 need to be specified")

        self.name = "Lens"
        self.short_name = "L"
        
        super().__init__(front, pos, back, d1, d2)

        self._new_lock = True

    def estimateFocalLength(self, wl: float=555., n0: RefractionIndex=RefractionIndex("Constant", n=1.)) -> float:
        """
        Estimate the lens focal length using the lensmaker equation.
        Only works if both surfaces are of type "Sphere", "Asphere", or "Circle".

        :param wl: wavelength
        :param n0: ambient refraction index
        :return: focal length
        """
        match self.FrontSurface.surface_type:
            case ("Sphere" | "Asphere"):
                R1 = 1/self.FrontSurface.rho
            case "Circle":
                R1 = np.inf
            case _:
                raise RuntimeError("Calculation only possible with surface_type 'Circle', 'Sphere' or 'Asphere'.")

        match self.BackSurface.surface_type:
            case ("Sphere" | "Asphere"):
                R2 = 1/self.BackSurface.rho
            case "Circle":
                R2 = np.inf
            case _:
                raise RuntimeError("Calculation only possible with surface_type 'Circle', 'Sphere' or 'Asphere'.")

        n = self.n(wl)
        n0_ = n0(wl)
        d = self.BackSurface.pos[2] - self.FrontSurface.pos[2]  # thickness along the optical axis

        # lensmaker equation
        D = (n-n0_)/n0_ * (1/R1 - 1/R2 + (n - n0_) * d /(n*R1*R2))

        return 1 / D    


    def crepr(self):

        """

        """
        return [self.FrontSurface.crepr(), self.BackSurface.crepr(), self.d1, self.d2, self.n.crepr(), (self.n2.crepr() if self.n2 is not None else None)]
