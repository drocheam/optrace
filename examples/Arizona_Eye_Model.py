#!/usr/bin/env python3

import sys
sys.path.append(".")

import optrace as ot
import optrace.gui.TraceGUI as TraceGUI
import numpy as np

# Arizona Eye Model from
# Schwiegerling J. Field Guide to Visual and Ophthalmic Optics. SPIE Publications: 2004.

# options:
P = 5.7 # pupil diameter
A = 0  # adaption in dpt
dispersion = True  # turns on chromatic dispersion

# the eye model geometry, excluding the raysource, can also be loaded using simply
# geom = ot.presets.Geometry.ArizonaEye(P=P, A=A, dispersion=dispersion)
# RT.add(geom)
# for the sake of a learning expericence it is

# create Raytracer
RT = ot.Raytracer(outline=[-6, 6, -6, 6, -10, 28], AbsorbMissing=True, no_pol=False)

# add RaySource
RSS = ot.Surface("Circle", r=3)
RS = ot.RaySource(RSS, direction="Parallel", spectrum=ot.presets.LightSpectrum.D65, pos=[0, 0, -10])
RT.add(RS)

# add refraction indices depending on dispersive behaviour
if dispersion:
    n_Cornea = ot.RefractionIndex("Abbe", n=1.377, V=57.1, desc="n_Cornea")
    n_Aqueous = ot.RefractionIndex("Abbe", n=1.337, V=61.3, desc="n_Aqueous")
    n_Lens = ot.RefractionIndex("Abbe", n=1.42+0.00256*A-0.00022*A**2, V=51.9, desc="n_Lens")
    n_Vitreous = ot.RefractionIndex("Abbe", n=1.336, V=61.1, desc="n_Vitreous")
else:
    n_Cornea = ot.RefractionIndex("Constant", n=1.377, desc="n_Cornea")
    n_Aqueous = ot.RefractionIndex("Constant", n=1.337, desc="n_Aqueous")
    n_Lens = ot.RefractionIndex("Constant", n=1.42+0.00256*A-0.00022*A**2, desc="n_Lens")
    n_Vitreous = ot.RefractionIndex("Constant", n=1.336, desc="n_Vitreous")

d_Aq = 2.97-0.04*A  # thickness Aqueous
d_Lens = 3.767+0.04*A  # thickness lens

# add Cornea
front = ot.Surface("Asphere", r=5.25, rho=1/7.8, k=-0.25)
back = ot.Surface("Asphere", r=5.25, rho=1/6.5, k=-0.25)
L0 = ot.Lens(front, back, d1=0.25, d2=0.30, pos=[0, 0, 0.25], n=n_Cornea, n2=n_Aqueous, desc="Cornea")
RT.add(L0)

# add Pupil
ap = ot.Surface("Ring", r=5.25, ri=P/2)
AP = ot.Aperture(ap, pos=[0, 0, 3.3], desc="Pupil")
RT.add(AP)

# add Lens
front = ot.Surface("Asphere", r=5.25, rho=1/(12-0.4*A), k=-7.518749+1.285720*A)
back = ot.Surface("Asphere", r=5.25, rho=1/(-5.224557+0.2*A), k=-1.353971-0.431762*A)
L1 = ot.Lens(front, back, d1=d_Lens/3, d2=d_Lens*2/3, pos=[0, 0, d_Aq+0.55+d_Lens/3], 
             n=n_Lens, n2=n_Vitreous, desc="Lens")
RT.add(L1)

# add Detector
DetS = ot.Surface("Sphere", r=6, rho=-1/13.4)
Det = ot.Detector(DetS, pos=[0, 0, 24], desc="Retina")
RT.add(Det)

# Instantiate the class and configure its traits.
TG = TraceGUI(RT, LogImage=True, FlipDetImage=True, ImageType="Irradiance")
TG.run()

