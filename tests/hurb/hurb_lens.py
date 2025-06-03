#!/usr/bin/env python3

import numpy as np

import sys
sys.path.insert(0, "tests")
from hurb_geometry import hurb_lens

import optrace as ot
import optrace.plots as otp
import matplotlib.pyplot as plt

# ambient refractive index
n = 1.33
# n = 1

# pupil radius in mm
ri = 1.5
# ri = 5

# wavelength in nm
wl = 550
# wl = 380
# wl = 780

# detector distance / geometric focal length in mm
zd = 20


r, imgic, imgr = hurb_lens(n, ri, wl, zd, N=2000000, N_px=315, dim_ext_fact=5, use_hurb=True)

# print(np.mean((imgic - rayleigh_curve(r, wl, n, ri, zd))**2))

# plot curves

plt.figure(figsize=(9, 4))
plt.subplot(1, 2, 1)
otp.misc_plots._show_grid()
plt.plot(r*1000, imgic, label="HURB simulation")
plt.plot(r*1000, imgr, label="Theory")

plt.legend()
plt.xlim([r[0]*1e3/5, r[-1]*1e3/5])
plt.xlabel("$r$ in µm")
plt.ylabel("$I$")
plt.suptitle(fr"Pupil and Ideal Lens, pupil r={ri}mm, ambient n={n}, $\lambda_0$={wl}nm, z={zd}mm")

plt.subplot(1, 2, 2)
otp.misc_plots._show_grid()
plt.plot(r*1000, np.log10(1e-5 + imgic), label="HURB simulation")
plt.plot(r*1000, np.log10(1e-5 + imgr), label="Theory")
plt.legend()
plt.xlabel("$r$ in µm")
plt.ylabel("log$_{10}(I)$")

plt.tight_layout()
plt.show()
