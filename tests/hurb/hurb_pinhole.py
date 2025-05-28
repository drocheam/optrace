#!/usr/bin/env python3

import numpy as np

import sys
sys.path.insert(0, "tests")
from hurb_geometry import hurb_pinhole

import optrace as ot
import optrace.plots as otp
import matplotlib.pyplot as plt

# detector position in mm
zd = 20

# ambient refractive index
n = 1.33
# n = 1

# pinhole radius in mm
ri = 0.005
# ri = 6

# wavelength in nm
wl = 550
# wl = 380
wl = 780

r, imgic, imgr = hurb_pinhole(n, ri, wl, zd, N=2000000, N_px=315, dim_ext_fact=6, use_hurb=True)

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
plt.suptitle(fr"Pinhole, $r$={ri*1000}µm, ambient $n$={n}, $\lambda_0$={wl}nm, $z$={zd}mm")

plt.subplot(1, 2, 2)
otp.misc_plots._show_grid()
plt.plot(r, np.log10(1e-5 + imgic), label="HURB simulation")
plt.plot(r, np.log10(1e-5 + imgr), label="Theory")
plt.legend()
plt.xlabel("$r$ in mm")
plt.ylabel("log$_{10}(I)$")

plt.tight_layout()
plt.show()

