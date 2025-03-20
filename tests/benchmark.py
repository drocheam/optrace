#!/usr/bin/env python3

import pathlib
import time
import numpy as np

import optrace as ot
from optrace.tracer.misc import cpu_count

# adapted version of examples/microscope.py for benchmarking

# ressources from example folder are required
resource_dir = pathlib.Path(__file__).resolve().parent.parent / "examples" / "resources"

# create tracer
RT = ot.Raytracer(outline=[-50, 50, -50, 50, -30, 430])

# cell image
RSS = ot.presets.image.cell([100e-3, 100e-3])
RS = ot.RaySource(RSS, divergence="Lambertian",
                  pos=[0, 0, -0.00000001], s=[0, 0, 1], div_angle=50, desc="Cell")
RT.add(RS)

# suppress warnings
with ot.global_options.no_warnings():

    # load material files 
    schott = ot.load_agf(str(resource_dir / "materials" / "schott.agf"))
    ohara = ot.load_agf(str(resource_dir / "materials" / "ohara.agf"))
    hikari = ot.load_agf(str(resource_dir / "materials" / "hikari.agf"))
    hoya = ot.load_agf(str(resource_dir / "materials" / "hoya.agf"))

    # join to one material dictionary
    n_dict = schott | ohara | hikari | hoya

    # load microscope setup
    G = ot.load_zmx(str(resource_dir / "microscope" / "Nikon_1p25NA_60x_US7889433B2_MultiConfig_v2.zmx"), n_dict=n_dict)
    RT.n0 = G.n0

    # create objective group
    objective = ot.Group(G.lenses[:18])
    RT.add(objective)

    # create tube group
    tube = ot.Group(G.lenses[20:24])
    tube.move_to(G.lenses[20].pos-[0, 0, 150])  # lower the distance to the tube lenses
    RT.add(tube)

    # load eyepiece
    eyepiece = ot.load_zmx(str(resource_dir / "eyepiece" / "UK565851-1.zmx"), n_dict=n_dict)
    eyepiece.remove(eyepiece.detectors)

# position of intermediate image
# move eyepiece so intermediate image is in front focal point of eyepiece
tma = ot.TMA(objective.lenses + tube.lenses, n0=G.n0)
z_img0 = tma.image_position(RS.pos[2])
eyep_f0 = eyepiece.tma().focal_points[0]
eyepiece.move_to([0, 0, eyepiece.lenses[0].pos[2] - (eyep_f0 - z_img0)])
RT.add(eyepiece)

# load eye geometry and adapted geometry so exit pupil of microscope and entrance pupil of eye are the same
eye = ot.presets.geometry.arizona_eye()
exit_pupil_microscope = RT.tma().pupil_position(0.38)[1]
entrance_pupil_eye = eye.tma().pupil_position(eye.apertures[0].pos[2])[0]
eye.move_to([0, 0, exit_pupil_microscope + (eye.pos[2] - entrance_pupil_eye)])
RT.add(eye)

# benchmark info
snum = len(RT.tracing_surfaces)
N = 1000000
print(f"Number of surfaces:                           {snum}")
print(f"Number of threads for tracing:                {cpu_count()}")
print(f"Number of rays:                               {N}")

# run both polarization cases
for i, no_pol in enumerate([False, True]):
    
    print(f"\nRun {i+1}: " + ("With Polarization Handling" if not no_pol else "Without Polarization Handling"))

    # trace
    time0 = time.time()
    RT.no_pol = no_pol
    RT.trace(N)
    time1 = time.time()

    print(f"Performance (seconds / surf / million rays):  {(time1-time0)/snum/N*1e6:.3f}")

    # cool down before next test
    if not i:
        time.sleep(5)

