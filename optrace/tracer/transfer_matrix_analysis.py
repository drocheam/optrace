import numpy as np  # matrix calculations

from .refraction_index import RefractionIndex  # media of and media between lenses
from .base_class import BaseClass  # parent class
from . import color  # for wavelength bounds
from .misc import PropertyChecker as pc  # type checking


# Some Sources
# https://www.edmundoptics.de/knowledge-center/application-notes/optics/understanding-optical-lens-geometries/
# https://www.montana.edu/jshaw/documents/1%20EELE_481_582_S15_GeoSignConventions.pdf
# https://indico.cern.ch/event/266133/attachments/474621/656940/Gillespie_Aachen_CP_vs_P_optics.pdf
# https://www.montana.edu/ddickensheets/documents/abcdCardinal%202.pdf
# https://www.edmundoptics.com/knowledge-center/tech-tools/focal-length/


class TMA(BaseClass):

    def __init__(self,
                 lenses:    list,
                 wl:        float = 555.,
                 n0:        RefractionIndex = None,
                 **kwargs)\
            -> None:
        """
        Create an ray transfer matrix analysis object.
        This is a snapshot of properties for when the object gets created, nothing is updated after that.
        
        With no lenses the abcd matrix is a unity matrix and all other properties are set to nan

        :param lenses: list of Lens
        :param wl: wavelength to create the analysis for
        :param n0: ambient medium before the lens setup
        :param kwargs: additional keyword arguments for the parent class
        """
        # type checks
        pc.check_type("lenses", lenses, list)
        pc.check_type("n0", n0, RefractionIndex | None)
        pc.check_type("wl", wl, float | int)
        pc.check_not_below("wl", wl, color.WL_BOUNDS[0])
        pc.check_not_above("wl", wl, color.WL_BOUNDS[1])

        self.wl = wl
        """wavelength for the analysis"""

        L = sorted(lenses, key=lambda el: el.front.pos[2])

        self.vertex_point: tuple[float, float] = (L[0].front.pos[2], L[-1].back.pos[2]) if len(lenses)\
                                                 else (np.nan, np.nan)
        """z-position of vertex points"""

        self.n1: float = n0(self.wl) if n0 is not None else 1.0
        """refraction index value before the lens setup"""
        
        self.n2: float = L[-1].n2(self.wl) if len(lenses) and L[-1].n2 is not None else self.n1
        """refraction index value after the lens setup"""

        _1, _2 = self._1, self._2 = self.vertex_point
        
        self._ds = []  # position list
        self._mats = []  # ABCD matrix list

        self.abcd = self._gen_abcd(L)
        """abcd matrix for matrix ray optics calculations """

        n1_, n2_ = self.n1, self.n2
        A, B, C, D = tuple(self.abcd.ravel())

        self.principal_points: tuple[float, float] = (_1 - (n1_ - n2_ * D) / (n2_ * C), _2 + (1 - A) / C) if C else (np.nan, np.nan)
        """z-position of principal points"""

        p1, p2 = self.principal_points

        self.nodal_points: tuple[float, float] = (_1 - (1 - D) / C, _2 + (n1_ - n2_ * A) / (n2_ * C)) if C else (np.nan, np.nan)
        """z-position of nodal points"""
        
        self.focal_points: tuple[float, float] = (p1 + n1_ / n2_ / C, p2 - 1 / C) if C else (np.nan, np.nan)
        """z-position of focal points"""

        f1p, f2p = self.focal_points

        self.focal_lengths: tuple[float, float] = (f1p - p1, f2p - p2) if C else (np.nan, np.nan)
        """focal lengths of the lens """
        
        f1, f2 = self.focal_lengths

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
    
        self.focal_lengths_n: tuple[float, float] = f1 / self.n1, f2 / self.n2
        """focal lengths with different definition, see the documentation"""
   
        self.powers: tuple[float, float] = 1000 / f1, 1000 / f2
        """optical powers of the lens, inverse of focal length"""
        
        self.powers_n: tuple[float, float] = 1000 * self.n1 / f1, 1000 * self.n2 / f2
        """
        different definition for the optical powers. The optical powers is scaled with the ambient index for each side.
        Mainly used in ophthalmic optics.
        This definition has the advantage, that both powers always have the same magnitude, but only different signs.
        """

        _oc = 1 - A + B*C / (D - 1) if D - 1 else np.inf
        self.optical_center = _1 + self.d / _oc if _oc else np.nan
        """optical center of the setup"""
        
        super().__init__(**kwargs)

        # lock object (no changes from here)
        self.lock()
        self._new_lock = True

    def _gen_abcd(self, L: list) -> np.ndarray:
        """
        generate the ABCD matrix

        :param L: list of lenses
        :return: 2x2 numpy array
        """
      
        # create sub-matrices
        dz = 0
        for i in np.arange(len(L)):
          
            # check symmetry
            if i+1 < len(L) and (not np.isclose(L[i].pos[0], L[i+1].pos[0])\
                    or not np.isclose(L[i].pos[1], L[i+1].pos[1])):
                raise RuntimeError("Lenses don't share one axis.")
      
            # media
            n1_ = L[i-1].n2(self.wl) if i and L[i-1].n2 is not None else self.n1
            n2_ = L[i].n2(self.wl) if L[i].n2 is not None else self.n1
            
            if L[i].is_ideal: # IdealLens
                z0 = self._ds[-1] if self._ds else 0
                self._ds += [z0]
                self._mats += [np.array([[1, 0], [-L[i].D/1000, n1_/n2_]])]

            else:  # normal lens
                if L[i].front.parax_roc is None or L[i].back.parax_roc is None:
                    raise RuntimeError("Lens without rotational symmetry in transfer matrix analysis.")
                
                # lens properties
                n_ = L[i].n(self.wl)
                R2 = L[i].front.parax_roc
                R1 = L[i].back.parax_roc

                # component matrices
                front = np.array([[1, 0], [-(n_-n1_)/R2/n_, n1_/n_]])  # front surface
                thickness = np.array([[1, L[i].d], [0, 1]])  # space between surfaces
                back = np.array([[1, 0], [-(n2_-n_)/R1/n2_, n_/n2_]])  # back surface

                # add to lists
                z0 = self._ds[-1] if self._ds else 0
                self._ds += [z0, z0+L[i].d, z0+L[i].d]
                self._mats += [front, thickness, back]

            # multiply with distance matrix to next lens
            if i + 1 < len(L):
                dz = L[i+1].front.pos[2] - L[i].back.pos[2]
                self._ds += [self._ds[-1]+dz]
                self._mats += [np.array([[1, dz], [0, 1]])]

                if dz < 0:
                    raise RuntimeError("Negative distance between lenses. Are there object collisions?")

        # calculate ABCD matrix
        mat = np.eye(2)
        n = len(self._mats)
        for i in range(n):
            mat = mat @ self._mats[n-i-1]

        return mat

    @staticmethod
    def _obj_dist(abcd: np.ndarray, z1: float, rev: bool = False) -> float:
        """ 
        Calculates the image distance from a ABCD matrix and object distance (rev=False)
        or object distance from image distance (rev=True)

        :param abcd:  ABCD matrix, shape (2, 2)
        :param z1: distance (image or object)
        :param rev: reverse direction
        :return: other distance
        """
        # in reverse direction (image to object imaging)
        # we need to inverse the matrix and negate the input distance
        # at the end the output distance is also negated
        if rev:
            abcd = np.linalg.inv(abcd)
            z1 *= -1
        
        A, B, C, D = abcd.ravel()

        if np.isfinite(z1):
            z2 = - (B + z1*A) / (D + C*z1) if D + C*z1 else np.nan
        else:
            z2 = -A/C if C else np.nan

        return z2 if not rev else -z2

    def _pupil_props(self, zp: float) -> tuple[float, float, float, float]:
        """
        Internal function, retuns pupil positions and magnifications.

        :param zp: stop position
        :return: pupil positions, pupil magnifications
        """
        # see https://wp.optics.arizona.edu/jgreivenkamp/wp-content/uploads/sites/11/2019/08/502-09-Stops-and-Pupils.pdf
        
        # find the index separating the front and rear group
        i = 0
        while i < len(self._ds) and self._ds[i] + self._1 < zp:
            i += 1
        
        # front group
        if i:
            # calculate group
            lmat = np.eye(2)
            for j in range(i):
                lmat = lmat @ self._mats[i-j-1]

            lmat = np.linalg.inv(lmat) # invert so we can use it right to left
            lz = self._ds[i-1] + self._1  # back vertex of front group
            ze1 = self._1 + self._obj_dist(lmat, lz - zp)  # g < 0 since in negative direction
            m1 = self._dist_mat(lmat, lz - zp, ze1 - self._1)[0, 0]  # magnifications

        # with no front group the entrance pupil is the aperture stop
        else:
            ze1 = zp
            m1 = 1
        
        # rear group
        k = len(self._mats)
        l = k-i
        if l:
            # if followed by something: increase index by one so we don't work on the distance matrix
            # but the next surface matrix
            off = 1 if i+1 < k and self._ds[i] == self._ds[i+1] else 0

            # calculate group
            rmat = np.eye(2)
            for m in range(l-off):
                rmat = rmat @ self._mats[k-m-1]
          
            rz = self._ds[i+off] + self._1  # front vertex of rear group
            ze2 = self._2 + self._obj_dist(rmat, rz - zp)  # pupil position
            m2 = self._dist_mat(rmat, rz - zp, ze2 - self._2)[0, 0]  # magnifications
        
        # with no rear group the exit pupil is the aperture stop
        else:
            ze2 = zp
            m2 = 1

        return ze1, ze2, m1, m2

    def pupil_position(self, z_s: float) -> tuple[float, float]:
        """
        Calculate the entrance and exit pupil positions for a given aperture stop position.

        :param z_p: absolute stop position
        :return: entrance and exit pupil positions
        """
        return self._pupil_props(z_s)[:2]
    
    def pupil_magnification(self, z_s: float) -> tuple[float, float]:
        """
        Calculate the entrance and exit pupil magnifications for a given aperture stop position.

        :param z_p: absolute stop position
        :return: entrance and exit pupil magnifications
        """
        return self._pupil_props(z_s)[2:]

    def image_position(self, z_g) -> float:
        """
        get the image position for a object distance

        :param z_g: z-position of object
        :return: absolute image z-position
        """
        if self._1 < z_g < self._2:
            raise ValueError(f"Object inside lens with z-extent at optical axis of {self.vertex_point}")

        g = self._1 - z_g
        b = self._obj_dist(self.abcd, g, rev=False)

        return b + self._2
    
    def image_magnification(self, z_g) -> float:
        """
        get the image magnification at the image plane for a given object distance

        :param z_g: z-position of object
        :return: magnification factor (image size divided by object size)
        """
        with np.errstate(invalid='ignore'):  # suppresses nan warnings
            z_b = self.image_position(z_g)
            mat = self.matrix_at(z_g, z_b)
            return mat[0, 0]

    def object_position(self, z_b) -> float:
        """
        get the object position for a given image position

        :param z_b: z-position of image
        :return: z-position of object
        """
        if self._1 < z_b < self._2:
            raise ValueError(f"Image inside lens with z-extent at optical axis of {self.vertex_point}")
        
        b = z_b - self._2
        g = self._obj_dist(self.abcd, b, rev=True)

        return self._1 - g

    def object_magnification(self, z_b) -> float:
        """
        get the object magnification at the object plane for a given image distance

        :param z_g: z-position of object
        :return: magnification factor (image size divided by object size)
        """
        with np.errstate(invalid='ignore'):  # suppresses nan warnings
            z_g = self.object_position(z_b)
            mat = self.matrix_at(z_g, z_b)
            return mat[0, 0]

    @staticmethod
    def _dist_mat(abcd: np.ndarray, g: float, b: float) -> np.ndarray:
        """
        Extends an ABCD matrix by a front and rear distance matrix.

        :param abcd: initial ABCD matrix, shape (2, 2)
        :param g: object distance
        :param b: image distance
        :return: new ABCD matrix, shape (2, 2)
        """
        d_b_matrix = np.array([[1, b], [0, 1]])  # matrix for distance to first lens
        d_g_matrix = np.array([[1, g], [0, 1]])  # matrix for distance from last lens
        return d_b_matrix @ abcd @ d_g_matrix

    def matrix_at(self, z_g: float, z_b: float) -> np.ndarray:
        """
        Calculate the ABCD matrix for an object position z_g and image position z_b

        :param z_g: object z-position
        :param z_b: image z-position
        :return: ABCD matrix, 2x2 numpy array
        """
        return self._dist_mat(self.abcd, self._1 - z_g, z_b - self._2)

