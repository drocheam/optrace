
"""
Lens class:
A lens is an geometrical object with two refractive surfaces.
A refractive index is specified for the material and one for the area behind the lens (optional)
"""

import numpy as np
import copy

from Backend.RefractionIndex import RefractionIndex as RefractionIndex
from Backend.Surface import Surface


# TODO Function for estimation of focal length
# TODO error message when surface is point or line

class Lens:

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

        self.front = front.copy()
        self.back = back.copy()
        self.n = n
        self.n2 = n2
        self.d1 = float(d1) if d1 is not None else d1
        self.d2 = float(d2) if d2 is not None else d2

        if de is not None and self.d1 is None and self.d2 is None:
            self.d1 = de / 2. + self.front.maxz - self.front.pos[2]
            self.d2 = de / 2. + self.back.pos[2] - self.back.minz

        elif d1 is None or d2 is None:
            raise ValueError("Both thicknesses d1, d2 need to be specified")

        if self.d1 < 0 or self.d2 < 0:
            raise ValueError("Thicknesses de, d1, d2 need to be non-negative.")

        self.moveTo(pos)

    # TODO include d1 parameter to move surface
    def setFrontSurface(self, surf: Surface) -> None:
        """

        :param surf:
        """
        pos = self.front.pos
        self.front = surf.copy()
        self.front.moveTo(pos)

    # TODO include d2 parameter to move surface
    def setBackSurface(self, surf: Surface) -> None:
        """

        :param surf:
        """
        pos = self.back.pos
        self.back = surf.copy()
        self.back.moveTo(pos)

    def moveTo(self, pos: (list | np.ndarray)) -> None:
        """
        Moves the lens in 3D space.

        :param pos: new 3D position of filter center (list or numpy array)
        """
        pos = np.array(pos, dtype=np.float64)

        self.front.moveTo(pos - [0, 0, self.d1])
        self.back.moveTo(pos + [0, 0, self.d2])

    def copy(self) -> 'Lens':
        """
        Return a fully independent copy of the Lens object.

        :return: copy
        """
        return copy.deepcopy(self)

    @property
    def pos(self) -> np.ndarray:
        """ position of lens center """
        return self.front.pos + [0, 0, self.d1]

    @property
    def extent(self) -> tuple[float, float, float, float, float, float]:
        """ 3D extent of Lens"""
        front_ext = self.front.getExtent()[:4]
        back_ext = self.back.getExtent()[:4]

        return min(front_ext[0], back_ext[0]),\
               max(front_ext[1], back_ext[1]),\
               min(front_ext[2], back_ext[2]),\
               max(front_ext[3], back_ext[3]),\
               self.front.minz,\
               self.back.maxz
    
    def getCylinderSurface(self, nc: int = 100) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get a 3D surface representation of the lens side cylinder for plotting.

        :param nc: number of surface edge points (int)
        :return: coordinate arrays X, Y, Z (2D numpy arrays)
        """

        X1, Y1, Z1 = self.front.getEdge(nc)
        X2, Y2, Z2 = self.back.getEdge(nc)

        X = np.column_stack((X1, X2))
        Y = np.column_stack((Y1, Y2))
        Z = np.column_stack((Z1, Z2))

        return X, Y, Z


    def estimateFocalLength(self, wl: float=555., n0: RefractionIndex=RefractionIndex("Constant", n=1.)) -> float:
        """
        Estimate the lens focal length using the lensmaker equation.
        Only works if both surfaces are of type "Sphere", "Asphere", or "Circle".

        :param wl: wavelength
        :param n0: ambient refraction index
        :return: focal length
        """
        match self.front.surface_type:
            case ("Sphere" | "Asphere"):
                R1 = 1/self.front.rho
            case "Circle":
                R1 = np.inf
            case _:
                raise RuntimeError("Calculation only possible with surface_type 'Circle', 'Sphere' or 'Asphere'.")

        match self.back.surface_type:
            case ("Sphere" | "Asphere"):
                R2 = 1/self.back.rho
            case "Circle":
                R2 = np.inf
            case _:
                raise RuntimeError("Calculation only possible with surface_type 'Circle', 'Sphere' or 'Asphere'.")

        n = self.n(wl)
        n0_ = n0(wl)
        d = self.back.pos[2] - self.front.pos[2]  # thickness along the optical axis

        # lensmaker equation
        D = (n-n0_)/n0_ * (1/R1 - 1/R2 + (n - n0_) * d /(n*R1*R2))

        return 1 / D
