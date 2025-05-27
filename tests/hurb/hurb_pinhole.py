#!/usr/bin/env python3

import numpy as np
import scipy.special

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


# see https://de.wikipedia.org/wiki/Beugungsscheibchen#Beugung_an_einer_Kreisblende
def rayleigh_curve(x, wl, n, r, z):
    Rnz = 2*np.pi/(wl*1e-9)*n*r/z*x*1e-3
    return (2*scipy.special.j1(Rnz) / Rnz) ** 2

# make raytracer
RT = ot.Raytracer(outline=[-15, 15, -15, 15, -6, zd+10], use_hurb=True, 
                  n0=ot.RefractionIndex("Constant", n))

# add ray source
RS = ot.RaySource(ot.CircularSurface(r=ri), s=[0, 0, 1], pos=[0, 0, -5], 
                  spectrum=ot.LightSpectrum("Monochromatic", wl=wl))
RT.add(RS)


# Pinhole
ap_surf = ot.RingSurface(r=5, ri=ri)
ap = ot.Aperture(ap_surf, pos=[0, 0, 0])
RT.add(ap)

# add Detector
# detector width
dim_ext = 1.22 / (2*np.pi / (wl*1e-9) * n * ri / zd / np.pi) * 1e3 * 6 * 5
DetS = ot.RectangularSurface(dim=[dim_ext, dim_ext])
Det = ot.Detector(DetS, pos=[0, 0, zd])
RT.add(Det)

# trace
RT.trace(2000000)

# get irradiance profile along image axis
img = RT.detector_image()
imgi = img.get("Irradiance", 315)
bins, imgic1 = imgi.profile(x=0)
bins, imgic2 = imgi.profile(y=0)
imgic = 0.5*(imgic1[0] + imgic2[0])  # average of x and y profile for less noise
imgic /= np.max(imgic)

# radial vector
r = bins[:-1]+(bins[1]-bins[0])/2

# plot curves

plt.figure(figsize=(9, 4))
plt.subplot(1, 2, 1)
otp.misc_plots._show_grid()
plt.plot(r, imgic, label="HURB simulation")
plt.plot(r, rayleigh_curve(r, wl, n, ri, zd), label="Theory")

plt.legend()
plt.xlim([-dim_ext/13, dim_ext/13])
plt.xlabel("$r$ in mm")
plt.ylabel("$I$")
plt.suptitle(fr"Pinhole, $r$={ri*1000}µm, ambient $n$={n}, $\lambda_0$={wl}nm, $z$={zd}mm")

plt.subplot(1, 2, 2)
otp.misc_plots._show_grid()
plt.plot(r, np.log10(1e-5 + imgic), label="HURB simulation")
plt.plot(r, np.log10(1e-5 + rayleigh_curve(r, wl, n, ri, zd)), label="Theory")
plt.legend()
plt.xlabel("$r$ in mm")
plt.ylabel("log$_{10}(I)$")

plt.tight_layout()
plt.show()

