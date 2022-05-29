
"""
Lens class:
A lens is an geometrical object with two refractive surfaces.
A refractive index is specified for the material and one for the area behind the lens (optional)
"""

import numpy as np

from optrace.tracer.spectrum.RefractionIndex import *
from optrace.tracer.geometry.Surface import *
from optrace.tracer.geometry.SObject import *


class Lens(SObject):

    abbr = "L"  # object abbreviation
    _allow_non_2D = False  # don't allow points or lines as surfaces

    def __init__(self, 
                 front:   Surface, 
                 back:    Surface,
                 n:       RefractionIndex,
                 pos:     (list | np.ndarray), 
                 de:      float = 0,
                 d1:      float = None,
                 d2:      float = None,
                 n2:      RefractionIndex = None,
                 **kwargs)\
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

        super().__init__(front, pos, back, d1, d2, **kwargs)

        self._new_lock = True

    def estimateFocalLength(self, wl: float=555., n0: RefractionIndex=RefractionIndex("Constant", n=1.)) -> float:
        """
        Estimate the lens focal length using the lensmaker equation.
        Only works if both surfaces are of type "Sphere", "Asphere" or planar.

        :param wl: wavelength
        :param n0: ambient refraction index, defaults to vacuum
        :return: focal length
        """

        def getR(Surf):
            if Surf.surface_type in ["Sphere", "Asphere"]:
                return 1/Surf.rho
            elif Surf.isPlanar():
                return np.inf
            else:
                raise RuntimeError("Calculation only possible with surface_type 'Circle', 'Sphere' or 'Asphere'.")

        R1 = getR(self.FrontSurface)
        R2 = getR(self.BackSurface)

        n = self.n(wl)
        n0_ = n0(wl)
        d = self.BackSurface.pos[2] - self.FrontSurface.pos[2]  # thickness along the optical axis

        # lensmaker equation
        D = (n-n0_)/n0_ * (1/R1 - 1/R2 + (n - n0_) * d /(n*R1*R2))

        return 1 / D    

    def __setattr__(self, key, val):

        if key == "n2":
            self._checkType(key, val, RefractionIndex | None)

        if key == "n":
            self._checkType(key, val, RefractionIndex)

        super().__setattr__(key, val)

