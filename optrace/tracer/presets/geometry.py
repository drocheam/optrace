
import numpy as np


# Eye models
#######################################################################################################################

def ArizonaEye(A:           float = 0., 
               dispersion:  bool = False, 
               P:           float = 5.7, 
               DetR:        float = 8, 
               pos:         list = None)\
        -> 'list[Aperture | Lens | Detector]':
    """
    Arizona Eye Model from
    Schwiegerling J. Field Guide to Visual and Ophthalmic Optics. SPIE Publications: 2004.
    Includes Eye adaption and chromatic behaviour. Matches different on- and off-axis aberrations of the eye.
    See the source above for more info.
    """

    # lazy import since it would lead to a circular import otherwise
    from optrace.tracer.geometry import Aperture, Detector, Lens, Surface
    from optrace.tracer.refraction_index import RefractionIndex as RefractionIndex

    pos0 = np.array(pos if pos is not None else [0, 0, 0])
    geom = []

    d_Aq = 2.97-0.04*A  # thickness Aqueous
    d_Lens = 3.767+0.04*A  # thickness lens
    
    if dispersion:
        n_Cornea = RefractionIndex("Abbe", n=1.377, V=57.1, desc="n_Cornea")
        n_Aqueous = RefractionIndex("Abbe", n=1.337, V=61.3, desc="n_Aqueous")
        n_Lens = RefractionIndex("Abbe", n=1.42+0.00256*A-0.00022*A**2, V=51.9, desc="n_Lens")
        n_Vitreous = RefractionIndex("Abbe", n=1.336, V=61.1, desc="n_Vitreous")
    else:
        n_Cornea = RefractionIndex("Constant", n=1.377, desc="n_Cornea")
        n_Aqueous = RefractionIndex("Constant", n=1.337, desc="n_Aqueous")
        n_Lens = RefractionIndex("Constant", n=1.42+0.00256*A-0.00022*A**2, desc="n_Lens")
        n_Vitreous = RefractionIndex("Constant", n=1.336, desc="n_Vitreous")

    # add Cornea
    front = Surface("Asphere", r=5.25, rho=1/7.8, k=-0.25)
    back = Surface("Asphere", r=5.25, rho=1/6.5, k=-0.25)
    L0 = Lens(front, back, d1=0.25, d2=0.30, pos=pos0+[0, 0, 0.25], n=n_Cornea, n2=n_Aqueous, desc="Cornea")
    geom.append(L0)

    # add Pupil
    ap = Surface("Ring", r=5.25, ri=P/2)
    AP = Aperture(ap, pos=pos0+[0, 0, 3.3], desc="Pupil")
    geom.append(AP)

    # add Lens
    front = Surface("Asphere", r=5.25, rho=1/(12-0.4*A), k=-7.518749+1.285720*A)
    back = Surface("Asphere", r=5.25, rho=1/(-5.224557+0.2*A), k=-1.353971-0.431762*A)
    L1 = Lens(front, back, d1=d_Lens/3, d2=d_Lens*2/3, pos=pos0+[0, 0, d_Aq+0.55+d_Lens/3], 
              n=n_Lens, n2=n_Vitreous, desc="Lens")
    geom.append(L1)

    # add Detector
    DetS = Surface("Sphere", r=DetR, rho=-1/13.4)
    Det = Detector(DetS, pos=pos0+[0, 0, 24], desc="Retina")
    geom.append(Det)

    return geom


def ParaxialSchematicEye(P:     float = 5.7, 
                         DetR:  float = 8., 
                         pos:   list = None)\
        -> 'list[Aperture | Lens | Detector]':
    """
    LeGrand full theoretical eye, a paraxial schematic eye model, taken from
    Schwiegerling J. Field Guide to Visual and Ophthalmic Optics. SPIE Publications: 2004.
    
    Properties: Infinity adapted eye. Eye approximation by spherical surfaces.
    Only useful for first order optical properties of the eye.
    """

    # lazy import since it would lead to a circular import otherwise
    from optrace.tracer.geometry import Aperture, Detector, Lens, Surface
    from optrace.tracer.refraction_index import RefractionIndex as RefractionIndex

    pos0 = np.array(pos if pos is not None else [0, 0, 0])
    geom = []

    n_Cornea = RefractionIndex("Constant", n=1.3771, desc="n_Cornea")
    n_Aqueous = RefractionIndex("Constant", n=1.3374, desc="n_Aqueous")
    n_Lens = RefractionIndex("Constant", n=1.4200, desc="n_Lens")
    n_Vitreous = RefractionIndex("Constant", n=1.3360, desc="n_Vitreous")

    # add Cornea
    front = Surface("Sphere", r=5, rho=1/7.8)
    back = Surface("Sphere", r=5, rho=1/6.5)
    L0 = Lens(front, back, d1=0.25, d2=0.30, pos=pos0+[0, 0, 0.25], n=n_Cornea, n2=n_Aqueous, desc="Cornea")
    geom.append(L0)

    # add Aperture
    ap = Surface("Ring", r=5, ri=P/2)
    AP = Aperture(ap, pos=pos0+[0, 0, 3.3], desc="Pupil")
    geom.append(AP)

    # add Lens
    front = Surface("Sphere", r=4.5, rho=1/10.2)
    back = Surface("Sphere", r=4.5, rho=-1/6)
    L1 = Lens(front, back, d1=1.5, d2=2.5, pos=pos0+[0, 0, 5.10], 
              n=n_Lens, n2=n_Vitreous, desc="Lens")
    geom.append(L1)

    # add Detector
    DetS = Surface("Sphere", r=DetR, rho=-1/13.4)
    Det = Detector(DetS, pos=pos0+[0, 0, 24.197], desc="Retina")
    geom.append(Det)

    return geom


# list with all eye models
eye_models: list = [ParaxialSchematicEye, ArizonaEye]

# list with all geometries
#######################################################################################################################

geometries: list = [*eye_models]
