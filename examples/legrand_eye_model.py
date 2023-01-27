#!/usr/bin/env python3

import optrace as ot
from optrace.gui import TraceGUI
import numpy as np

# this example loads the paraxial eye model 
# and plots the corresponding cardinal points and pupil positions


# create raytracer
RT = ot.Raytracer(outline=[-15, 15, -15, 15, -15, 30])

# add RaySource
RSS = ot.CircularSurface(r=2)
RS = ot.RaySource(RSS, pos=[0, 0, -10])
RT.add(RS)

# load LeGrand Eye model
eye = ot.presets.geometry.legrand_eye()
RT.add(eye)

# plot cardinal points as line markers
tma = eye.tma()
RT.add(ot.LineMarker(r=8, pos=[0, 0, tma.vertex_points[0]], desc="V1"))
RT.add(ot.LineMarker(r=8, pos=[0, 0, tma.vertex_points[1]], desc="V2"))
RT.add(ot.LineMarker(r=8, pos=[0, 0, tma.principal_points[0]], desc="H1"))
RT.add(ot.LineMarker(r=8, pos=[0, 0, tma.principal_points[1]], desc="H2"))
RT.add(ot.LineMarker(r=8, pos=[0, 0, tma.nodal_points[0]], desc="N1"))
RT.add(ot.LineMarker(r=8, pos=[0, 0, tma.nodal_points[1]], desc="N2"))

# calculate entrance and exit pupil positions from position of eye pupil
en, ex = tma.pupil_position(eye.apertures[0].pos[2])
RT.add(ot.LineMarker(r=8, pos=[0, 0, en], desc="EN"))
RT.add(ot.LineMarker(r=8, pos=[0, 0, ex], desc="EX"))

# add optical power label without marker
RT.add(ot.PointMarker(pos=[0, 0, 10], desc=f"Power: {tma.powers_n[1]:.2f} dpt", label_only=True))

# plot focal point as point marker
RT.add(ot.PointMarker(desc="F2", pos=[0, 0, tma.focal_points[1]]))

# Instantiate the class and configure its traits.
TG = TraceGUI(RT, ray_opacity=0.01, vertical_labels=True, high_contrast=True, minimalistic_view=True)
TG.run()
