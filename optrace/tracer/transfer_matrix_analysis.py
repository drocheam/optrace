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
                 lens_list:  list[RefractionIndex], 
                 wl:        float = 555., 
                 n0:        RefractionIndex = None,
                 **kwargs)\
            -> None:
        """
        
        :param lens_list:
        :param wl:
        :param n0:
        :param kwargs:
        """
        # type checks
        pc.check_type("lens_list", lens_list, list)
        pc.check_type("wl", wl, float | int)
        pc.check_not_below("wl", wl, color.WL_MIN)
        pc.check_not_above("wl", wl, color.WL_MAX)
        pc.check_type("n0", n0, RefractionIndex | None)

        if not len(lens_list):
            raise ValueError("Empty lens_list.")

        self.lens_list = sorted(lens_list, key=lambda el: el.front.pos[2])
        self.wl = wl
        self.n1 = n0(self.wl) if n0 is not None else 1.0
        self.n2 = self.lens_list[-1].n2(self.wl) if self.lens_list[-1].n2 is not None else self.n1

        self._gen_abcd()
        self._gen_properties()

        super().__init__(**kwargs)

        # lock object
        self.lock()
        self._new_lock = True

    def _gen_abcd(self) -> np.ndarray:
        """

        :return:
        """
        L = self.lens_list
        mat = np.eye(2)

        for i in np.arange(len(L)-1, -1, -1):
            
            if L[i].front.curvature_circle is None or L[i].back.curvature_circle is None:
                raise RuntimeError("Lens without rotational symmetry in transfer matrix analysis.")
        
            if i > 0 and not np.isclose(L[i].pos[0], L[i-1].pos[0]) or not np.isclose(L[i].pos[1], L[i-1].pos[1]):
                raise RuntimeError("Lenses don't share one axis")
        
            # lens properties
            n_ = L[i].n(self.wl)
            n1_ = L[i-1].n2(self.wl) if i and L[i-1].n2 is not None else self.n1
            n2_ = L[i].n2(self.wl) if L[i].n2 is not None else self.n1
            R2 = L[i].front.curvature_circle
            R1 = L[i].back.curvature_circle

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

            self._abcd = mat

    def _gen_properties(self)\
            -> tuple[list, list, list, list, float, float]:
        """

        :return:
        """
        # cardinal points from ABCD matrix:
        # https://www.montana.edu/ddickensheets/documents/abcdCardinal%202.pdf
        # bfl and ffl from
        # https://www.edmundoptics.com/knowledge-center/tech-tools/focal-length/

        n1_, n2_ = self.n1, self.n2
        _1, _2 = self._vertices = self.lens_list[0].front.pos[2], self.lens_list[-1].back.pos[2]

        A, B, C, D = tuple(self._abcd.ravel())

        if C:
            self._principals = _1-(n1_-n2_*D)/(n2_*C), _2+(1-A)/C
            self._nodals = _1-(1-D)/C, _2+(n1_-n2_*A)/(n2_*C)
            self._focals = self._principals[0]+n1_/n2_/C, self._principals[1]-1/C
            self._focal_lengths = self._focals[0]-self._principals[0], self._focals[1]-self._principals[1]
            self._ffl, self._bfl = self._focals[0]-_1, self._focals[1]-_2
        # C == 0: planar surfaces with infinite focal length, set everything to nan
        else:
            self._focal_lengths = self._ffl, self._bfl = self._principals\
                = self._nodals = self._focals = np.nan, np.nan

    @property
    def abcd(self) -> np.ndarray:
        """abcd matrix for matrix ray optics calculations """
        return self._abcd

    @property
    def bfl(self) -> float:
        """back focal length, Distance between back focal point and back surface vertex point """
        return self._bfl
   
    @property
    def ffl(self) -> float:
        """front focal length, Distance between front focal point and front surface vertex"""
        return self._ffl
   
    @property
    def efl(self) -> float:
        """effective focal length. """
        return self._focal_lengths[1]
    
    @property
    def efl_n(self) -> float:
        """effective focal length. """
        return self._focal_lengths[1] / self.n2

    @property
    def focal_length(self) -> tuple[float, float]:
        """focal lengths of the lens """
        return self._focal_lengths
    
    @property
    def d(self) -> float:
        """center thickness of the lens """
        return self._vertices[1] - self._vertices[0]
     
    @property
    def principal_point(self) -> tuple[float, float]:
        """z-position of principal points """
        return self._principals
    
    @property
    def nodal_point(self) -> tuple[float, float]:
        """z-position of nodal points """
        return self._nodals
    
    @property
    def vertex_point(self) -> tuple[float, float]:
        """z-position of vertex points """
        return self._vertices
    
    @property
    def focal_point(self) -> tuple[float, float]:
        """z-position of focal points """
        return self._focals

    @property
    def focal_length_n(self) -> tuple[float, float]:
        """"""
        f1, f2 = self._focal_lengths
        return f1/self.n1, f2/self.n2
   
    @property
    def power(self) -> tuple[float, float]:
        """optical powers of the lens, inverse of focal length"""
        f1, f2 = self._focal_lengths
        return 1000/f1, 1000/f2
    
    @property
    def power_n(self) -> tuple[float, float]:
        """
        different definition for the optical power. The optical power is scaled with the ambient index for each side.
        Mainly used in ophthalmic optics.
        This definition has the advantage, that both powers always have the same magnitude, but only different signs.
        """
        f1, f2 = self._focal_lengths
        return 1000*self.n1/f1, 1000*self.n2/f2

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

        if self.vertex_point[0] < z_g < self.vertex_point[1]:
            raise ValueError(f"Object inside lens with z-extent at optical axis of {self.vertex_point}")

        A, B, C, D = tuple(self._abcd.ravel())
        g = self._vertices[0] - z_g

        if np.isfinite(g):
            b = - (B + g*A) / (D + C*g) if D + C*g else np.nan
        else:
            b = -A/C if C else np.nan

        return b + self.lens_list[-1].back.pos[2]

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

        if self.vertex_point[0] < z_b < self.vertex_point[1]:
            raise ValueError(f"Image inside lens with z-extent at optical axis of {self.vertex_point}")
        
        A, B, C, D = tuple(self._abcd.ravel())
        b = z_b - self._vertices[1]

        if np.isfinite(b):
            g = - (B + D*b) / (A + C*b) if A + C*b else np.nan
        else:
            g = -D/C if C else np.nan

        return self.lens_list[0].front.pos[2] - g

    def matrix_at(self, z_g: float, z_b: float) -> np.ndarray:
        """
        Calculate the abcd matrix for an object position z_g and image position z_b

        :param z_g:
        :param z_b:
        :return:
        """
        d_b_matrix = np.array([[1, z_b - self.vertex_point[1]], [0, 1]])  # matrix for distance to first lens
        d_g_matrix = np.array([[1, self.vertex_point[0] - z_g], [0, 1]])  # matrix for distance ffrom last lens
        mat = d_b_matrix @ self._abcd @ d_g_matrix

        return mat

    def trace(self, in_: list[float, float] | np.ndarray):
        """
        Paraxial tracing. Calculates r and theta values.

        :param in_: either two element list or array with [r, theta] 
                or two dimensional array with r and theta in columns
        :return: resulting r and theta in same shape as in_
        """
        in_arr = np.array(in_, dtype=float)

        if in_arr.ndim == 1:
            return self._abcd @ in_arr
        else:
            return (self._abcd @ in_arr.T).T
