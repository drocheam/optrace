
import numpy as np  # np.ndarray type

from ..geometry import Aperture, Detector, Lens, SphericalSurface,\
        ConicSurface, RingSurface, Group  # Elements and Surfaces in the preset geometries
from ..refraction_index import RefractionIndex  # media in the geometries


# Eye models
#######################################################################################################################

def arizona_eye(adaptation:  float = 0.,
                pupil:       float = 5.7,
                r_det:       float = 8,
                pos:         list = None)\
        -> Group:
    """
    Arizona Eye Model from
    Schwiegerling J. Field Guide to Visual and Ophthalmic Optics. SPIE Publications: 2004.
    Includes Eye adaption and chromatic behaviour. Matches different on- and off-axis aberrations of the eye.
    See the source above for more info.

    :param adaptation: adaptation value (optical power), defaults to 0dpt
    :param pupil: pupil diameter in mm, default tos 5.7mm
    :param r_det: radial size of detector (retina), defaults to 8mm
    :param pos: position of the Group
    :return: Group containing all the elements
    """

    # absolute position
    pos0 = np.array(pos if pos is not None else [0, 0, 0])
    geom = Group(long_desc="Arizona Eye Model", desc="Eye")
    
    # rename
    A = adaptation

    # thickness
    d_Aq = 2.97-0.04*A  # thickness Aqueous
    d_Lens = 3.767+0.04*A  # thickness lens

    # media
    n_Cornea = RefractionIndex("Abbe", n=1.377, V=57.1, desc="n_Cornea")
    n_Aqueous = RefractionIndex("Abbe", n=1.337, V=61.3, desc="n_Aqueous")
    n_Lens = RefractionIndex("Abbe", n=1.42+0.00256*A-0.00022*A**2, V=51.9, desc="n_Lens")
    n_Vitreous = RefractionIndex("Abbe", n=1.336, V=61.1, desc="n_Vitreous")

    # add Cornea
    front = ConicSurface(r=5.45, R=7.8, k=-0.25, long_desc="Cornea Anterior")
    back = ConicSurface(r=5.45, R=6.5, k=-0.25, long_desc="Cornea Posterior")
    L0 = Lens(front, back, d1=0, d2=0.55, pos=pos0+[0, 0, 0], n=n_Cornea, n2=n_Aqueous, desc="Cornea")
    geom.add(L0)

    # add Pupil
    ap = RingSurface(r=5.45, ri=pupil/2, desc="Pupil")
    # the pupil position would be closer to something like 3.6, but for large A we would have surface collisions
    AP = Aperture(ap, pos=pos0+[0, 0, 3.3], desc="Pupil")
    geom.add(AP)

    # add Lens
    front = ConicSurface(r=5.1, R=12-0.4*A, k=-7.518749+1.285720*A, long_desc="Lens Anterior")
    back = ConicSurface(r=5.1, R=-5.224557+0.2*A, k=-1.353971-0.431762*A, long_desc="Lens Posterior")
    L1 = Lens(front, back, d1=0, d2=d_Lens, pos=pos0+[0, 0, d_Aq+0.55],
              n=n_Lens, n2=n_Vitreous, desc="Lens")
    geom.add(L1)

    # add Detector
    DetS = SphericalSurface(r=r_det, R=-13.4, desc="Retina")
    Det = Detector(DetS, pos=pos0+[0, 0, 24], desc="Retina")
    geom.add(Det)

    return geom


def legrand_eye(pupil: float = 5.7,
                r_det: float = 8.,
                pos:   list = None)\
        -> Group:
    """
    LeGrand full theoretical eye, a paraxial schematic eye model, taken from
    Schwiegerling J. Field Guide to Visual and Ophthalmic Optics. SPIE Publications: 2004.

    Properties: Infinity adapted eye. Eye approximation by spherical surfaces.
    Only useful for first order optical properties of the eye.

    :param pupil: pupil diameter in mm, defaults to 5.7mm
    :param r_det: radial detector (=retina) size, defaults to 8mm
    :param pos: position of the Group
    :return: Group containing all elements
    """

    pos0 = np.array(pos if pos is not None else [0, 0, 0])
    geom = Group(long_desc="LeGrand Full Theoretical Eye", desc="Eye")

    n_Cornea = RefractionIndex("Constant", n=1.3771, desc="n_Cornea")
    n_Aqueous = RefractionIndex("Constant", n=1.3374, desc="n_Aqueous")
    n_Lens = RefractionIndex("Constant", n=1.4200, desc="n_Lens")
    n_Vitreous = RefractionIndex("Constant", n=1.3360, desc="n_Vitreous")

    # add Cornea
    front = SphericalSurface(r=5.5, R=7.8, long_desc="Cornea Anterior")
    back = SphericalSurface(r=5.5, R=6.5, long_desc="Cornea Posterior")
    L0 = Lens(front, back, d1=0.25, d2=0.30, pos=pos0+[0, 0, 0.25], n=n_Cornea, n2=n_Aqueous, desc="Cornea")
    geom.add(L0)

    # add Aperture
    ap = RingSurface(r=5.5, ri=pupil/2, desc="Pupil")
    # According to https://www.comsol.com/paper/download/682361/Bangalore_2019_Poster_T.Alavanthar.pdf
    # the pupil seems to be at z=3.6mm, which is also the start of the anterior lens surface
    AP = Aperture(ap, pos=pos0+[0, 0, 3.6], desc="Pupil")
    geom.add(AP)

    # add Lens
    front = SphericalSurface(r=4.8, R=10.2, long_desc="Lens Anterior")
    back = SphericalSurface(r=4.8, R=-6, long_desc="Lens Posterior")
    L1 = Lens(front, back, d1=1.5, d2=2.5, pos=pos0+[0, 0, 5.10],
              n=n_Lens, n2=n_Vitreous, desc="Lens")
    geom.add(L1)

    # add Detector
    DetS = SphericalSurface(r=r_det, R=-13.4, desc="Retina")
    Det = Detector(DetS, pos=pos0+[0, 0, 24.197], desc="Retina")
    geom.add(Det)

    return geom


# list with all eye models
eye_models: list = [legrand_eye, arizona_eye]

# list with all geometries
#######################################################################################################################

geometries: list = [*eye_models]
