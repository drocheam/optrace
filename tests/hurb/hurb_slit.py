#!/usr/bin/env python3

import numpy as np

import sys
sys.path.insert(0, "tests")
from hurb_geometry import hurb_slit

import optrace as ot
import optrace.plots as otp
import matplotlib.pyplot as plt

# detector distance from aperture in mm
zd = 20

# ambient refractive index
n = 1.33
n = 1

# wavelength in nm
wl = 550
# wl = 380
wl = 780

# slit width d1 and height (d2 >> d1) in mm 
d1 = 0.005
d2 = 0.1

r, _, imgic, _, imgr, _ = hurb_slit(n, d1, d2, wl, zd, N=2000000, N_px=315, dim_ext_fact=5, use_hurb=True)

# plot curves

plt.figure(figsize=(9, 4))
plt.subplot(1, 2, 1)
otp.misc_plots._show_grid()
plt.plot(r, imgic, label="HURB simulation")
plt.plot(r, imgr, label="Theory")

plt.legend()
plt.xlim([r[0]/5, r[-1]/5])
plt.xlabel("$r$ in mm")
plt.ylabel("$I$")
plt.suptitle(fr"Slit d={d1*1000}µm, n={n}, $\lambda_0$={wl}nm, z={zd}mm")

plt.subplot(1, 2, 2)
otp.misc_plots._show_grid()
plt.plot(r, np.log10(1e-5 + imgic), label="HURB simulation")
plt.plot(r, np.log10(1e-5 + imgr), label="Theory")
plt.legend()
plt.xlabel("$r$ in mm")
plt.ylabel("log$_{10}(I)$")

plt.tight_layout()
plt.show()
