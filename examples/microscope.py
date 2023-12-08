#!/usr/bin/env python3

import numpy as np
import pathlib
import warnings


import os
import warnings

import numexpr

warnings.warn(str(os.environ))

import optrace as ot
from optrace.gui import TraceGUI


# path for needed resources (materials and eyepiece, objective files)
# we could use a path string for ot.load.agf() and ot.load.zmx(), but using pathlib makes it platform-independent
ressource_dir = pathlib.Path(__file__).resolve().parent / "ressources"

# create tracer
RT = ot.Raytracer(outline=[-50, 50, -50, 50, -30, 430])

# cell image
RSS = ot.RectangularSurface(dim=[100e-3, 100e-3])
RS = ot.RaySource(RSS, divergence="Lambertian", image=ot.presets.image.cell,
                  pos=[0, 0, -0.00000001], s=[0, 0, 1], div_angle=50, desc="Cell")
RT.add(RS)

# load material files (ignore import warnings)
with warnings.catch_warnings():
    warnings.filterwarnings('ignore')
    schott = ot.load.agf(str(ressource_dir / "materials" / "schott.agf"))
    ohara = ot.load.agf(str(ressource_dir / "materials" / "ohara.agf"))
    hikari = ot.load.agf(str(ressource_dir / "materials" / "hikari.agf"))
    hoya = ot.load.agf(str(ressource_dir / "materials" / "hoya.agf"))

# join to one material dictionary
n_dict = schott | ohara | hikari | hoya

# load microscope setup
G = ot.load.zmx(str(ressource_dir / "microscope" / "Nikon_1p25NA_60x_US7889433B2_MultiConfig_v2.zmx"), n_dict=n_dict)

# create objective group
objective = ot.Group(G.lenses[:18])
RT.add(objective)

# create tube group
tube = ot.Group(G.lenses[20:24])
tube.move_to(G.lenses[20].pos-[0, 0, 150])  # lower the distance to the tube lenses
RT.add(tube)

# load eyepiece
eyepiece = ot.load.zmx(str( ressource_dir / "eyepiece" / "UK565851-1.zmx"), n_dict=n_dict)
eyepiece.remove(eyepiece.detectors)

# position of intermediate image
tma = ot.TMA(objective.lenses + tube.lenses, n0=G.n0)
z_img0 = tma.image_position(RS.pos[2])

# move eyepiece so intermediate image is in front focal point of eyepiece
eyep_f0 = eyepiece.tma().focal_points[0]
eyepiece.move_to([0, 0, eyepiece.lenses[0].pos[2] - (eyep_f0 - z_img0)])
RT.add(eyepiece)

# detector for intermediate image
det = ot.Detector(ot.RectangularSurface([20, 20]), pos=[0, 0, z_img0], desc="Intermediate Image")
RT.add(det)

# load eye geometry
eye = ot.presets.geometry.arizona_eye()

# exit pupil of the microscope and entrance pupil of the eye should be the same

# calculate the exit pupil from first surface of objective
exit_pupil_microscope = RT.tma().pupil_position(0.38)[1]

# entrance pupil of the eye
entrance_pupil_eye = eye.tma().pupil_position(eye.apertures[0].pos[2])[0]

# move the eye so both match
eye.move_to([0, 0, exit_pupil_microscope + (eye.pos[2] - entrance_pupil_eye)])

# add eye geometry
RT.add(eye)

# add all groups to raytracer
RT.n0 = G.n0

# add markers for focal points
tma_obj = objective.tma()
tma_ok = eyepiece.tma()
tma_tube = tube.tma()
RT.add(ot.PointMarker("F_obj", [0, 0, tma_obj.focal_points[0]]))
RT.add(ot.PointMarker("F_obj", [0, 0, tma_obj.focal_points[1]]))
RT.add(ot.PointMarker("F_ok", [0, 0, tma_ok.focal_points[0]]))
RT.add(ot.PointMarker("F_tube", [0, 0, tma_tube.focal_points[1]]))

# add group description markers
RT.add(ot.PointMarker("Objective", [-50, 0, np.mean(objective.extent[4:])], label_only=True, text_factor=1.2))
RT.add(ot.PointMarker("Tube Lens", [-50, 0, np.mean(tube.extent[4:])], label_only=True, text_factor=1.2))
RT.add(ot.PointMarker("Eyepiece", [-50, 0, np.mean(eyepiece.extent[4:])], label_only=True, text_factor=1.2))
RT.add(ot.PointMarker("Eye", [-50, 0, np.mean(eye.extent[4:])], label_only=True, text_factor=1.2))

# add cylinder volume around microscope
vol = ot.CylinderVolume(r=15.9, length=eyepiece.extent[5]-objective.extent[4]-1.3, 
                        pos=[0, 0, objective.extent[4]+1.3], color=(0.0, 0.0, 0.0))
RT.add(vol)


# run the simulator
sim = TraceGUI(RT, ray_count=1000000, vertical_labels=True, minimalistic_view=True)
sim.run()
