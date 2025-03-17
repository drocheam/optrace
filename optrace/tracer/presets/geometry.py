
import numpy as np  # np.ndarray type

from ..geometry import Aperture, Detector, Lens, IdealLens, SphericalSurface,\
    RectangularSurface, ConicSurface, RingSurface, Group, BoxVolume, Volume
    # Elements and Surfaces in the preset geometries
from ..refraction_index import RefractionIndex  # media in the geometries

from ...property_checker import PropertyChecker as pc


# helper tools
#######################################################################################################################

def ideal_camera(cam_pos:   np.ndarray, 
                 z_g:       float, 
                 b:         float = 10, 
                 r:         float = 6, 
                 r_det:     float = 6)\
        -> Group:
    """
    Create an ideally imaging camera, consisting of a lens and a detector area.
    Returns a Group of both elements.

    :param cam_pos: position of camera
    :param z_g: position of object
    :param b: image distance (distance between lens and detector)
    :param r: radius of the camera lens
    :param r_det: radius of the detector area
    :return: Group object with Lens and Detector
    """
    pc.check_above("b", b, 0)
    pc.check_above("g = cam_pos[2] - z_g", cam_pos[2] - z_g, 0)

    g = cam_pos[2] - z_g
    D = (1/b + 1/g)*1000   # from imaging equation D = 1/f = 1/b + 1/g

    IL = IdealLens(pos=cam_pos, r=r, D=D, long_desc="Camera Objective", desc="Obj")
    DET = Detector(RectangularSurface([2*r_det, 2*r_det]), pos=np.array(cam_pos)+[0, 0, b],
                   long_desc="Camera Sensor", desc="Sensor")

    # add camera box
    rb = max(r, r_det)
    VOL = BoxVolume(dim=[2*rb, 2*rb], length=b, pos=cam_pos, color=(0.2, 0.2, 0.2))
    
    VOL = Volume(IL.front.copy(), DET.front.copy(), pos=cam_pos, d1=0, d2=b, color=(0, 0, 0))

    return Group([IL, DET, VOL], long_desc="Ideal Camera", desc="Camera")


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
    # place the pupil directly in front of the lens
    AP = Aperture(ap, pos=pos0+[0, 0, L0.back.pos[2]+d_Aq-1e-9], desc="Pupil")
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

    # see notes in legrand_eye() on how these parameters are calculated
    front = ConicSurface(r=12.776270, R=14.8152, k=0.344612)
    back = ConicSurface(r=12.776270, R=-13.4, k=0.1)
    vol = Volume(front, back, pos=Det.pos, d1=front.ds+back.ds, d2=0, color=(1, 1, 0.95))
    geom.add(vol)

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

    # approximate shape for the eye ball so length is correct and last curvature is correct
    # r_e = R / sqrt(k + 1)  # maximum radial distance on conic
    # z_e = abs(R / (k + 1))  # height at maximal radial distance
    # front and back radius are equal:
    # r_e1 = r_e2 =>  R1 / sqrt(k1 + 1) = R2 / sqrt(k2 + 1)
    # both front+back are d in z-direction:
    # z_e1 + z_e2 = d   =>  R1 / (k1 + 1) + R2 / (k2 + 1) = d
    # chosen: R2 = -13.4, k2 = 0.1, d = 23.2

    front = ConicSurface(r=12.776270, R=14.8152, k=0.344612)
    back = ConicSurface(r=12.776270, R=-13.4, k=0.1)
    vol = Volume(front, back, pos=Det.pos, d1=front.ds+back.ds, d2=0, color=(1.0, 1.0, 0.95))
    geom.add(vol)

    return geom


# list with all eye models
eye_models: list = [legrand_eye, arizona_eye]

# list with all geometries
#######################################################################################################################

geometries: list = [ideal_camera, *eye_models]
