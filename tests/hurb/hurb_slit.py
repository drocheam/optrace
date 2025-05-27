#!/usr/bin/env python3

import numpy as np
import scipy.special

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


# see https://en.wikipedia.org/wiki/Diffraction#Single-slit_diffraction
def slit_curve(x, wl, n, d, z):
    return np.sinc(d*1e-3*n/(wl*1e-9) * x / z)**2

# make raytracer
RT = ot.Raytracer(outline=[-50, 50, -50, 50, -6, zd+10], use_hurb=True, 
                  n0=ot.RefractionIndex("Constant", n))

# add ray source
rect = ot.RectangularSurface(dim=[d1, d2])
RS = ot.RaySource(rect, s=[0, 0, 1], pos=[0, 0, -5],
                  spectrum=ot.LightSpectrum("Monochromatic", wl=wl))
RT.add(RS)

# Slit
ap_surf = ot.SlitSurface(dim=[d1+2, d2+2], dimi=[d1, d2])
ap = ot.Aperture(ap_surf, pos=[0, 0, 0])
RT.add(ap)

# add Detector
# calculate detector width automatically
dim_ext = 8 / (d1*1e-3*n/(wl*1e-9) / zd) * 5
DetS = ot.RectangularSurface(dim=[dim_ext, dim_ext])
Det = ot.Detector(DetS, pos=[0, 0, zd])
RT.add(Det)

# trace
RT.trace(2000000)

# get irradiance profile along image axis
img = RT.detector_image()
imgi = img.get("Irradiance", 315)
bins, imgic1 = imgi.profile(y=0)
imgic = imgic1[0]
imgic /= np.max(imgic)

# radial vector
r = bins[:-1]+(bins[1]-bins[0])/2

# plot curves

plt.figure(figsize=(9, 4))
plt.subplot(1, 2, 1)
otp.misc_plots._show_grid()
plt.plot(r, imgic, label="HURB simulation")
plt.plot(r, slit_curve(r, wl, n, d1, zd), label="Theory")

plt.legend()
plt.xlim([-dim_ext/13, dim_ext/13])
plt.xlabel("$r$ in mm")
plt.ylabel("$I$")
plt.suptitle(fr"Slit d={d1*1000}µm, n={n}, $\lambda_0$={wl}nm, z={zd}mm")

plt.subplot(1, 2, 2)
otp.misc_plots._show_grid()
plt.plot(r, np.log10(1e-5 + imgic), label="HURB simulation")
plt.plot(r, np.log10(1e-5 + slit_curve(r, wl, n, d1, zd)), label="Theory")
plt.legend()
plt.xlabel("$r$ in mm")
plt.ylabel("log$_{10}(I)$")

plt.tight_layout()
plt.show()
