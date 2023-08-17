#!/usr/bin/env python3

import numpy as np
import optrace as ot
from optrace.gui import TraceGUI

# Nikkor Wakamiya, 100mm, f1.4, 
# from https://nbviewer.org/github/quartiq/rayopt-notebooks/blob/master/Nikkor-Wakamiya-50mmf1.4_Ex1.ipynb
# and https://patents.google.com/patent/US4448497

# Elements:
# # T   Distance   Rad Curv   Diameter          Material       n      nd      Vd
# 0 S         20        inf        100         basic/air   1.000   1.000   89.30
# 1 S          5      78.36         76                 -   1.797   1.797   45.50
# 2 S     9.8837      469.5         76         basic/air   1.000   1.000   89.30
# 3 S     0.1938       50.3         64                 -   1.773   1.773   49.40
# 4 S     9.1085      74.38         62         basic/air   1.000   1.000   89.30
# 5 S     2.9457      138.1         60                 -   1.673   1.673   32.20
# 6 S     2.3256      34.33         51         basic/air   1.000   1.000   89.30
# 7 S      16.07        inf       49.6         basic/air   1.000   1.000   89.30
# 8 S         13     -34.41       48.8                 -   1.740   1.740   28.30
# 9 S      1.938      -2907         57                 -   1.773   1.773   49.40
# 10 S     12.403     -59.05         60         basic/air   1.000   1.000   89.30
# 11 S     0.3876     -150.9       66.8                 -   1.788   1.788   47.50
# 12 S      8.333     -57.89       67.8         basic/air   1.000   1.000   89.30
# 13 S     0.1938      284.6         66                 -   1.788   1.788   47.50
# 14 S     5.0388     -253.2         66         basic/air   1.000   1.000   89.30
# 15 S     73.839        inf      86.53         basic/air   1.000   1.000   89.30

# create tracer
RT = ot.Raytracer(outline=[-22000, 2000, -2000, 2000, -50000, 180])

# object distance
g = 50000

# create point sources under different angles at -g
for deg in [0, 5, 10, 15, 20]:
    xp = g * np.tan(deg/180*np.pi)
    RSS = ot.Point()
    RS = ot.RaySource(RSS, divergence="Isotropic", orientation="Converging", conv_pos=[0, 0, 0],
                      div_angle=0.03, pos=[-xp, 0, -g], desc=f"{deg}°")
    RT.add(RS)

# Lens 0
S_0 = ot.SphericalSurface(r=76/2, R=78.36)
S_1 = ot.SphericalSurface(r=76/2, R=469.5)
n_0 = ot.RefractionIndex("Abbe", n=1.797, V=45.3)
L_0 = ot.Lens(S_0, S_1, n=n_0, pos=[0, 0, 0], d1=0, d2=9.8837)
RT.add(L_0)

# Lens 1
S_2 = ot.SphericalSurface(r=64/2, R=50.3)
S_3 = ot.SphericalSurface(r=62/2, R=74.38)
n_1 = ot.RefractionIndex("Abbe", n=1.773, V=49.4)
L_1 = ot.Lens(S_2, S_3, n=n_1, pos=[0, 0, L_0.back.pos[2]+0.1938], d1=0, d2=9.1085)
RT.add(L_1)

# Lens 2
S_4 = ot.SphericalSurface(r=59/2, R=138.1)
S_5 = ot.SphericalSurface(r=51/2, R=34.33)
n_2 = ot.RefractionIndex("Abbe", n=1.673, V=32.20)
L_2 = ot.Lens(S_4, S_5, n=n_2, pos=[0, 0, L_1.back.pos[2]+2.9457], d1=0, d2=2.3256)
RT.add(L_2)

# Aperture 0
S_6 = ot.RingSurface(ri=49.6/2, r=76/2)
AP_0 = ot.Aperture(S_6, pos=[0, 0, L_2.back.pos[2]+16.07])
RT.add(AP_0)

# Lens 3
S_7 = ot.SphericalSurface(r=48.8/2, R=-34.41)
S_8 = ot.SphericalSurface(r=57/2, R=-2907)
n_3 = ot.RefractionIndex("Abbe", n=1.740, V=28.30)
L_3 = ot.Lens(S_7, S_8, n=n_3, pos=[0, 0, L_2.back.pos[2]+16.07+13], d1=0, d2=1.938)
RT.add(L_3)

# Lens 4
S_9 = ot.SphericalSurface(r=60/2, R=-59.05)
n_4 = ot.RefractionIndex("Abbe", n=1.773, V=49.40)
L_4 = ot.Lens(S_8, S_9, n=n_4, pos=[0, 0, L_3.back.pos[2]+1e-6], d1=0, d2=12.403)
RT.add(L_4)

# Lens 5
S_10 = ot.SphericalSurface(r=66.8/2, R=-150.9)
S_11 = ot.SphericalSurface(r=67.8/2, R=-57.89)
n_5 = ot.RefractionIndex("Abbe", n=1.788, V=47.50)
L_5 = ot.Lens(S_10, S_11, n=n_5, pos=[0, 0, L_4.back.pos[2]+0.3876], d1=0, d2=8.333)
RT.add(L_5)

# Lens 6
S_12 = ot.SphericalSurface(r=66/2, R=284.6)
S_13 = ot.SphericalSurface(r=66/2, R=-253.2)
n_6 = ot.RefractionIndex("Abbe", n=1.788, V=47.50)
L_6 = ot.Lens(S_12, S_13, n=n_6, pos=[0, 0, L_5.back.pos[2]+0.1938], d1=0, d2=5.0388)
RT.add(L_6)

# add Detector
Det = ot.Detector(ot.RectangularSurface(dim=[86.53, 86.53]), pos=[0, 0, L_6.back.pos[2]+73.839])
RT.add(Det)

# run the simulator
sim = TraceGUI(RT, 
               high_contrast=True, 
               det_image_one_source=True, 
               minimalistic_view=True, 
               ray_opacity=0.05,
               coloring_type="Source")
sim.run()
