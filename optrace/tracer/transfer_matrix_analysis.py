import numpy as np

from .refraction_index import RefractionIndex  # media of and media between lenses
from .base_class import BaseClass  # parent class
from . import color  # for wavelength bounds
from .misc import PropertyChecker as pc  # type checking


# sources for sign conventions:
# https://www.edmundoptics.de/knowledge-center/application-notes/optics/understanding-optical-lens-geometries/
# https://www.montana.edu/jshaw/documents/1%20EELE_481_582_S15_GeoSignConventions.pdf

# resources TMA
# https://indico.cern.ch/event/266133/attachments/474621/656940/Gillespie_Aachen_CP_vs_P_optics.pdf
# https://www.montana.edu/ddickensheets/documents/abcdCardinal%202.pdf


# transfer matrix analysis class
class TMA(BaseClass):

    def __init__(self,
                 lenses:    list,
                 wl:        float = 555.,
                 n0:        RefractionIndex = None,
                 **kwargs)\
            -> None:
        """
        
        :param lenses:
        :param wl:
        :param n0:
        :param kwargs:
        """
        # type checks
        pc.check_type("lenses", lenses, list)
        pc.check_type("wl", wl, float | int)
        pc.check_not_below("wl", wl, color.WL_BOUNDS[0])
        pc.check_not_above("wl", wl, color.WL_BOUNDS[1])
        pc.check_type("n0", n0, RefractionIndex | None)

        self.wl = wl
        L = sorted(lenses, key=lambda el: el.front.pos[2])

        # cardinal points from ABCD matrix:
        # https://www.montana.edu/ddickensheets/documents/abcdCardinal%202.pdf
        # bfl and ffl from
        # https://www.edmundoptics.com/knowledge-center/tech-tools/focal-length/

        # with no lenses the abcd matrix is a unity matrix and all other properties are set to nan

        self.vertex_point: tuple[float, float] = (L[0].front.pos[2], L[-1].back.pos[2]) if len(lenses)\
                                                 else (np.nan, np.nan)
        """z-position of vertex points"""

        self.n1: float = n0(self.wl) if n0 is not None else 1.0
        """refraction index value before the lens setup"""
        
        self.n2: float = L[-1].n2(self.wl) if len(lenses) and L[-1].n2 is not None else self.n1
        """refraction index value after the lens setup"""

        _1, _2 = self._1, self._2 = self.vertex_point
        
        self.abcd = self._gen_abcd(L)
        """abcd matrix for matrix ray optics calculations """

        n1_, n2_ = self.n1, self.n2
        A, B, C, D = tuple(self.abcd.ravel())

        self.principal_point: tuple[float, float] = (_1-(n1_-n2_*D)/(n2_*C), _2+(1-A)/C) if C else (np.nan, np.nan)
        """z-position of principal points"""

        p1, p2 = self.principal_point

        self.nodal_point: tuple[float, float] = (_1-(1-D)/C, _2+(n1_-n2_*A)/(n2_*C)) if C else (np.nan, np.nan)
        """z-position of nodal points"""
        
        self.focal_point: tuple[float, float] = (p1+n1_/n2_/C, p2-1/C) if C else (np.nan, np.nan)
        """z-position of focal points"""

        f1p, f2p = self.focal_point

        self.focal_length: tuple[float, float] = (f1p-p1, f2p-p2) if C else (np.nan, np.nan)
        """focal lengths of the lens """
        
        f1, f2 = self.focal_length

        self.ffl: float = f1p - _1 if C else np.nan
        """back focal length, Distance between back focal point and back surface vertex point """
        
        self.bfl: float = f2p - _2 if C else np.nan
        """front focal length, Distance between front focal point and front surface vertex"""

        self.d: float = self._2 - self._1
        """center thickness of the lens """

        self.efl: float = f2
        """effective focal length"""
        
        self.efl_n: float = f2 / self.n2
        """effective focal length"""
    
        self.focal_length_n: tuple[float, float] = f1/self.n1, f2/self.n2
        """"""
   
        self.power: tuple[float, float] = 1000/f1, 1000/f2
        """optical powers of the lens, inverse of focal length"""
        
        self.power_n: tuple[float, float] = 1000*self.n1/f1, 1000*self.n2/f2
        """
        different definition for the optical power. The optical power is scaled with the ambient index for each side.
        Mainly used in ophthalmic optics.
        This definition has the advantage, that both powers always have the same magnitude, but only different signs.
        """

        super().__init__(**kwargs)

        # lock object
        self.lock()
        self._new_lock = True

    def _gen_abcd(self, L) -> np.ndarray:
        
        # calculate abcd matrix
        mat = np.eye(2)
        for i in np.arange(len(L)-1, -1, -1):
            
            if i > 0 and not np.isclose(L[i].pos[0], L[i-1].pos[0]) or not np.isclose(L[i].pos[1], L[i-1].pos[1]):
                raise RuntimeError("Lenses don't share one axis")
       
            if L[i].is_ideal: # IdealLens
                l_matrix = np.array([[1, 0], [-L[i].D/1000, 1]])

            else:
                if L[i].front.parax_roc is None or L[i].back.parax_roc is None:
                    raise RuntimeError("Lens without rotational symmetry in transfer matrix analysis.")
                
                # lens properties
                n_ = L[i].n(self.wl)
                n1_ = L[i-1].n2(self.wl) if i and L[i-1].n2 is not None else self.n1
                n2_ = L[i].n2(self.wl) if L[i].n2 is not None else self.n1
                R2 = L[i].front.parax_roc
                R1 = L[i].back.parax_roc

                # component matrices
                front = np.array([[1, 0], [-(n_-n1_)/R2/n_, n1_/n_]])  # front surface
                thickness = np.array([[1, L[i].d], [0, 1]])  # space between surfaces
                back = np.array([[1, 0], [-(n2_-n_)/R1/n2_, n_/n2_]])  # back surface

                # lens matrix
                l_matrix = back @ thickness @ front

            # matrix product
            mat = mat @ l_matrix

            # multiply with distance matrix to next lens
            if i:
                dz = L[i].front.pos[2] - L[i-1].back.pos[2]
                d_matrix = np.array([[1, dz], [0, 1]])
                mat = mat @ d_matrix
            
                if dz < 0:
                    raise RuntimeError("Negative distance between lenses. Maybe there are object collisions?")

        return mat

    def image_position(self, z_g) -> float:
        """
        get image position
        :param z_g: z-position of object
        :return:
        """
        # overall matrix:
        # mat = [[1, b], [0, 1]] * [[A, B], [C, D]] * [[1, g], [0, 1]]
        # element mat[0, 1] needs to be zero for imaging (dx / dtheta = 0)
        # which is mat[0, 1] = g*(A + C*b) + B + D*b = 0
        # => b = - (B + g*A) / (D + C*g)
        # for A = D = 1, B = 0, C := -1/f (thin lens case) this is equivalent
        # to solving 1/f = 1/g + 1/b

        if self._1 < z_g < self._2:
            raise ValueError(f"Object inside lens with z-extent at optical axis of {self.vertex_point}")

        A, B, C, D = tuple(self.abcd.ravel())
        g = self._1 - z_g

        if np.isfinite(g):
            b = - (B + g*A) / (D + C*g) if D + C*g else np.nan
        else:
            b = -A/C if C else np.nan

        return b + self._2

    def object_position(self, z_b) -> float:
        """
        get image position
        :param z_b: z-position of image
        :return:
        """
        # overall matrix:
        # mat = [[1, b], [0, 1]] * [[A, B], [C, D]] * [[1, g], [0, 1]]
        # element mat[0, 1] needs to be zero for imaging (dx / dtheta = 0)
        # which is mat[0, 1] = g*(A + C*b) + B + D*b = 0
        # => g = - (B + b*D) / (A + C*b)
        # for A = D = 1, B = 0, C := -1/f (thin lens case) this is equivalent
        # to solving 1/f = 1/g + 1/b

        if self._1 < z_b < self._2:
            raise ValueError(f"Image inside lens with z-extent at optical axis of {self.vertex_point}")
        
        A, B, C, D = tuple(self.abcd.ravel())
        b = z_b - self._2

        if np.isfinite(b):
            g = - (B + D*b) / (A + C*b) if A + C*b else np.nan
        else:
            g = -D/C if C else np.nan

        return self._1 - g

    def matrix_at(self, z_g: float, z_b: float) -> np.ndarray:
        """
        Calculate the abcd matrix for an object position z_g and image position z_b

        :param z_g:
        :param z_b:
        :return:
        """
        d_b_matrix = np.array([[1, z_b - self._2], [0, 1]])  # matrix for distance to first lens
        d_g_matrix = np.array([[1, self._1 - z_g], [0, 1]])  # matrix for distance from last lens
        mat = d_b_matrix @ self.abcd @ d_g_matrix

        return mat

    def trace(self, pos: list[float, float] | np.ndarray):
        """
        Paraxial tracing. Calculates r and theta values.

        :param pos: either two element list or array with [r, theta] 
                or two-dimensional array with r and theta in columns
        :return: resulting r and theta in same shape as pos
        """
        in_arr = np.array(pos, dtype=float)

        if in_arr.ndim == 1:
            return self.abcd @ in_arr
        else:
            return (self.abcd @ in_arr.T).T
