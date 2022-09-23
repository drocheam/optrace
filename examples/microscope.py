#!/usr/bin/env python3

import numpy as np
import optrace as ot
from optrace.gui import TraceGUI

# make raytracer
RT = ot.Raytracer(outline=[-10, 10, -10, 10, -10, 300])

# object
RSS = ot.Surface("Rectangle", dim=[200e-3, 200e-3])
RS = ot.RaySource(RSS, divergence="Lambertian", image=ot.presets.image.bacteria,
                  pos=[0, 0, 0], s=[0, 0, 1], div_angle=27, desc="Bacteria")
RT.add(RS)

# UV and deep blue filter. These wavelengths are affected the most by dispersion
F = ot.Filter(ot.Surface("Circle", r=6), spectrum=ot.TransmissionSpectrum("Rectangle", wl0=435, wl1=680), pos=[0, 0, 8])
RT.add(F)

# objective doublet properties
n1 = ot.presets.refraction_index.LAK8
n2 = ot.presets.refraction_index.SF10
R1 = 7.74
R2 = -7.29

# Lens 1 of doublet
front = ot.Surface("Circle", r=5.5)
back = ot.Surface("Sphere", r=5.5, R=-R2)
L01 = ot.Lens(front, back, d1=0.5, d2=0, pos=[0, 0, 0], n=n2, n2=n2)
RT.add(L01)

# Lens 2 of doublet
front = ot.Surface("Sphere", r=5.5, R=-R2)
back = ot.Surface("Conic", r=5.5, R=-R1, k=-0.55)
L02 = ot.Lens(front, back, d1=0, d2=5, pos=[0, 0, 0.001], n=n1)
RT.add(L02)

# move objective lens so that its focal point is 0.6mm behind object
tma_obj = ot.TMA([L01, L02])
L0f0 = tma_obj.focal_point[0]
L01.move_to([0, 0, L01.pos[2] - (L0f0 - RS.pos[2] - 0.6)])
L02.move_to([0, 0, L02.pos[2] - (L0f0 - RS.pos[2] - 0.6)])
tma_obj = ot.TMA([L01, L02])  # update tma

# position of image inside microscope
# this had to be finetuned, since paraxial case was not precise enough
z_img0 = 225

# detector for tubus image
square = ot.Surface("Rectangle", dim=[6, 6])
det = ot.Detector(square, pos=[0, 0, z_img0], desc="Tubus Image")
RT.add(det)

# Eyepiece doublet curvatures
# see the achromat example
R1 = 9.82
R2 = -9.25

# Eyepiece doublet lens 1
front = ot.Surface("Conic", r=4, R=R1, k=-0.55)
back = ot.Surface("Sphere", r=4, R=R2)
L11 = ot.Lens(front, back, de=0, pos=[0, 0, 0], n=n1, n2=n1)
RT.add(L11)

# Eyepiece doublet lens 2
front = ot.Surface("Sphere", r=4, R=R2)
back = ot.Surface("Circle", r=4)
L12 = ot.Lens(front, back, d1=0, d2=0.5, pos=[0, 0, L11.extent[5]+0.001], n=n2)
RT.add(L12)

# move eyepiece so first image is in front focal point
tma_ok = ot.TMA([L11, L12])
L1f0 = tma_ok.focal_point[0]
L11.move_to([0, 0, L11.pos[2] - (L1f0 - z_img0)])
L12.move_to([0, 0, L12.pos[2] - (L1f0 - z_img0)])
tma_ok = ot.TMA([L11, L12])  # update
L1f1 = tma_ok.focal_point[1]

# add eye geometry
geom = ot.presets.geometry.arizona_eye(pos=[0, 0, L1f1+12], pupil=5)
RT.add(geom)

# calculate magnification
L = tma_ok.focal_point[0] - tma_obj.focal_point[1]
f_ok = tma_ok.focal_length[1]
f_obj = tma_ok.focal_length[1]
mu = L/f_ok * 254/f_obj
print(f"Approximate Magnification: {mu:.1f}x")

# add markers
RT.add(ot.Marker("F_obj", [0, 0, tma_obj.focal_point[0]]))
RT.add(ot.Marker("F_obj", [0, 0, tma_obj.focal_point[1]]))
RT.add(ot.Marker("F_ok", [0, 0, tma_ok.focal_point[0]]))
RT.add(ot.Marker("F_ok", [0, 0, tma_ok.focal_point[1]]))

print(RT.tma().matrix_at(RS.pos[2], RT.tma().focal_point[1]))
# start GUI.
sim = TraceGUI(RT, minimalistic_view=True, dark_mode=False, ray_count=2000000)
sim.run()
