#!/usr/bin/env python3

import numpy as np
import optrace as ot
from optrace.gui import TraceGUI

# init raytracer
RT = ot.Raytracer(outline=[-7, 7, -7, 7, 0, 40])

# add Raysource
RSS = ot.CircularSurface(r=2.5)
RS = ot.RaySource(RSS, divergence="Isotropic", div_angle=8, s=[0, 0, 1], pos=[0, 0, 0], 
                  spectrum=ot.presets.light_spectrum.led_b4)
RT.add(RS)

# custom surface with FunctionSurface2D
surf_func = lambda x, y: -np.tanh(3-2*np.sqrt(x**2 + y**2))/3
front = ot.FunctionSurface2D(func=surf_func, r=3)
back = ot.SphericalSurface(r=3, R=-8)
nL1 = ot.RefractionIndex("Constant", n=1.5)
L1 = ot.Lens(front, back, de=0, pos=[0, 0, 12], n=nL1)
RT.add(L1)

# filter with custom spectrum and mask
mask = lambda x, y: ~((y < 0) & (x**2 + y**2 < 2))
def filter_func(wl):
    T = np.full_like(wl, 0.2)
    T[(wl < 450) | (wl > 630)] = 1
    return T

f_surf = ot.FunctionSurface2D(func=lambda x, y:  np.zeros_like(x), mask_func=mask, r=3)
t_spec = ot.TransmissionSpectrum("Function", func=filter_func)
F = ot.Filter(f_surf, pos=[0, 0, 18], spectrum=t_spec)
RT.add(F)

# custom DataSurface
r = 2.5
Y, X = np.mgrid[-r:r:100j, -r:r:100j]
m = X > 0
Z = np.zeros_like(X)
Z[m] = -(X[m]**2 + Y[m]**2)/40
Z[~m] = -(X[~m]**2 + Y[~m]**2)/40 - X[~m]**2/20
front = ot.DataSurface2D(r=r, data=-Z)
back = ot.DataSurface2D(r=r, data=Z)
nL2 = ot.RefractionIndex("Constant", n=1.6)
L2 = ot.Lens(front, back, de=0, pos=[0, 0, 21], n=nL2)
RT.add(L2)


func3 = lambda x, y: np.abs(y)/3
front = ot.FunctionSurface2D(func=func3, r=5, z_min=0, z_max=5 / 3)
back = ot.CircularSurface(r=5)
L3 = ot.Lens(front, back, n=nL1, pos=[1, 0, 29])
RT.add(L3)

# add Detector
DetS = ot.RectangularSurface(dim=[14, 14])
Det = ot.Detector(DetS, pos=[0, 0, 31.8])
RT.add(Det)

# Instantiate the class and configure its traits.
sim = TraceGUI(RT, ray_opacity=0.03)
sim.run()
