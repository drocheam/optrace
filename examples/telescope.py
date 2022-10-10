#!/usr/bin/env python3

import numpy as np
import optrace as ot
from optrace.gui import TraceGUI

r1 = 15  # half lens diameter okular
r2 = 30  # half lens diameter eyepiece
rr = 1  # ray radius
mu = 3  # telescope magnification
f1 = 35  # first focal length
ang = 10  # angle of second ray
n = 1.72


n_ = ot.RefractionIndex("Constant", n=n)
f2 = f1*mu
R1 = 2 * f1 * (n - 1)
R2 = 2 * f2 * (n - 1)


# make raytracer
RT = ot.Raytracer(outline=[-40, 40, -40, 40, -40, 800], absorb_missing=True)

pt = ot.Point()
# RS0 = ot.RaySource(pt, spectrum=ot.LightSpectrum("Monochromatic", wl=550), pos=[rr, 0, -20])
# RT.add(RS0)
# RS1 = ot.RaySource(pt, spectrum=ot.LightSpectrum("Monochromatic", wl=550), pos=[-rr, 0, -20])
# RT.add(RS1)

s = np.array([-np.sin(np.radians(ang)), 0, np.cos(np.radians(ang))])
RS2 = ot.RaySource(pt, spectrum=ot.LightSpectrum("Monochromatic", wl=550), pos=[rr, 0, 0]-20*s, s=s)
RT.add(RS2)
RS3 = ot.RaySource(pt, spectrum=ot.LightSpectrum("Monochromatic", wl=550), pos=[-rr, 0, 0]-20*s, s=s)
RT.add(RS3)

s = np.array([np.sin(np.radians(ang)), 0, np.cos(np.radians(ang))])
RS2 = ot.RaySource(pt, spectrum=ot.LightSpectrum("Monochromatic", wl=550), pos=[rr, 0, 0]-20*s, s=s)
RT.add(RS2)
RS3 = ot.RaySource(pt, spectrum=ot.LightSpectrum("Monochromatic", wl=550), pos=[-rr, 0, 0]-20*s, s=s)
RT.add(RS3)

# Okular
front = ot.Surface("Sphere", r=r1, R=R1/2)
back = ot.Surface("Circle", r=r1)
L1 = ot.Lens(front, back, n=n_, pos=[0, 0, 0], de=0.5)
RT.add(L1)
L1.move_to([0, 0, L1.pos[2] - L1.tma().vertex_point[0]])

# Objective
front = ot.Surface("Circle", r=r2)
back = ot.Surface("Conic", r=r2, R=-R2/2, k=-0.78)
L2 = ot.Lens(front, back, n=n_, pos=[0, 0, 50], de=0.5)
RT.add(L2)

# move so focal points overlap
L2.move_to([0, 0, L2.pos[2] - (L2.tma().focal_point[0] - L1.tma().focal_point[1])])



RT.trace(4)

# find focus of rays and create a black filter there
aps = ot.Surface("Rectangle", dim=[2*r2, 2*r2])
ap = ot.Aperture(aps, pos=[0, 0, RT.outline[5]])
res, _ = RT.autofocus("Position Variance", z_start=RT.outline[5])
ap.move_to([0, 0, res.x])
RT.add(ap)

# check if all rays reach to this position (and are not absorbed through e.g. not hitting the second lens)
all_rays_hit = np.all(RT.rays.p_list[:, -1, 2] > res.x)


if all_rays_hit:
    RT.trace(4)
    ls = np.sum(RT.rays.optical_lengths(), axis=1)
    ptp = np.ptp(ls)

    print(f"Path lengths (mm): {ls}")
    print(f"Peak-to-Peak difference: {1000*ptp:.5g} µm")
else:
    print("Not all rays hit detector.")
    
# Instantiate the GUI and start it.
sim = TraceGUI(RT, minimalistic_view=True)
sim.run()

