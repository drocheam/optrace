#!/usr/bin/env python3

import optrace as ot
from optrace.gui import TraceGUI
import numpy as np


# make raytracer
RT = ot.Raytracer(outline=[-20, 20, -20, 20, -20, 300])

# object
RSS = ot.RectangularSurface(dim=[200e-3, 200e-3])
RS = ot.RaySource(RSS, divergence="Lambertian", image=ot.presets.image.cell,
                  pos=[0, 0, 0], s=[0, 0, 1], div_angle=27, desc="Cell")
RT.add(RS)

# objective doublet properties
n1 = ot.presets.refraction_index.LAK8
n2 = ot.presets.refraction_index.SF10
R1 = 7.74
R2 = -7.29

# objective group
objective = ot.Group(desc="Objective")

# Lens 1 of doublet
front = ot.CircularSurface(r=5.5)
back = ot.SphericalSurface(r=5.5, R=-R2)
L01 = ot.Lens(front, back, d1=0.5, d2=0, pos=[0, 0, 0], n=n2, n2=n2)
objective.add(L01)

# Lens 2 of doublet
front = ot.SphericalSurface(r=5.5, R=-R2)
back = ot.ConicSurface(r=5.5, R=-R1, k=-0.55)
L02 = ot.Lens(front, back, d1=0, d2=5.3, pos=[0, 0, 0.0001], n=n1)
objective.add(L02)

# move objective lens so that its focal point is 0.6mm behind object
L0f0 = objective.tma().focal_points[0]
objective.move_to([0, 0, L01.pos[2] - (L0f0 - RS.pos[2] - 0.6)])

# add group to raytracer
RT.add(objective)

# position of image inside microscope
# this had to be finetuned, since paraxial case was not precise enough
z_img0 = 225

# detector for tubus image
square = ot.RectangularSurface(dim=[6, 6])
det = ot.Detector(square, pos=[0, 0, z_img0], desc="Tubus Image")
RT.add(det)

# Eyepiece doublet curvatures
# see the achromat example
R1 = 9.82
R2 = -9.25

# create Eyepiece Group
eyepiece = ot.Group(desc="Eyepiece")

# Eyepiece doublet lens 1
front = ot.ConicSurface(r=4, R=R1, k=-0.55)
back = ot.SphericalSurface(r=4, R=R2)
L11 = ot.Lens(front, back, de=0.5, pos=[0, 0, 0], n=n1, n2=n1)
eyepiece.add(L11)

# Eyepiece doublet lens 2
front = ot.SphericalSurface(r=4, R=R2)
back = ot.CircularSurface(r=4)
L12 = ot.Lens(front, back, d1=0, d2=0.5, pos=[0, 0, L11.extent[5]+0.0001], n=n2)
eyepiece.add(L12)

# move eyepiece so first image is in front focal point
L1f0 = eyepiece.tma().focal_points[0]
eyepiece.move_to([0, 0, L11.pos[2]-(L1f0-z_img0)])
L1f1 = eyepiece.tma().focal_points[1]

# add group to raytracer
RT.add(eyepiece)

# add eye geometry
eye = ot.presets.geometry.arizona_eye(pos=[0, 0, L1f1+12], pupil=5)
# eye = ot.presets.geometry.ideal_camera(z_g=-np.inf, cam_pos=[0, 0,  L1f1+12], r=5, r_det=5, b=30)
RT.add(eye)

# add cylinder for displaying the microscope
vol = ot.CylinderVolume(r=6, length=eye.extent[4]-objective.extent[4]-3, 
                        pos=[0, 0, objective.extent[4]-3], color=(0, 0, 0))
RT.add(vol)

# calculate magnification
tma_ok = eyepiece.tma()
tma_obj = objective.tma()
L = tma_ok.focal_points[0] - tma_obj.focal_points[1]
f_ok = tma_ok.focal_lengths[1]
f_obj = tma_ok.focal_lengths[1]
mu = L/f_ok * 254/f_obj
print(f"Approximate Magnification: {mu:.1f}x")

# add markers focal points
RT.add(ot.PointMarker("F_obj", [0, 0, tma_obj.focal_points[0]]))
RT.add(ot.PointMarker("F_obj", [0, 0, tma_obj.focal_points[1]]))
RT.add(ot.PointMarker("F_ok", [0, 0, tma_ok.focal_points[0]]))
RT.add(ot.PointMarker("F_ok", [0, 0, tma_ok.focal_points[1]]))

# add misc markers
RT.add(ot.PointMarker("Objective", [-14, 0, L01.pos[2]+2], label_only=True, text_factor=1.2))
RT.add(ot.PointMarker("Body Tube", [-14, 0, (L02.pos[2] + L12.pos[2])/2], label_only=True, text_factor=1.2))
RT.add(ot.PointMarker("Eyepiece", [-14, 0, L11.pos[2]], label_only=True, text_factor=1.2))
RT.add(ot.PointMarker(eye.long_desc, [-14, 0, RT.lenses[-1].pos[2] + 10], label_only=True, text_factor=1.2))

# add line markers
RT.add(ot.LineMarker(r=5, pos=[0, 0, RS.pos[2]], desc="Object\nPlane"))
RT.add(ot.LineMarker(r=5, pos=[0, 0, tma_ok.focal_points[0]], desc="Intermed.\nImage"))
RT.add(ot.LineMarker(r=7, pos=[0, 0, RT.detectors[-1].pos[2]], desc="Image\nPlane"))

# start GUI.
sim = TraceGUI(RT, minimalistic_view=True, ray_count=2000000)
sim.run()
