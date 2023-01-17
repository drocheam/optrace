#!/usr/bin/env python3

import optrace as ot
from optrace.gui import TraceGUI


D = 30  # desired lens power
achromat = True  # compensate chromatic dispersion

if achromat:
    n1 = ot.presets.refraction_index.LAK8
    n2 = ot.presets.refraction_index.SF10

    n1_ = n1(ot.presets.spectral_lines.e)
    n2_ = n2(ot.presets.spectral_lines.e)
    
    V1 = n1.get_abbe_number(lines=ot.presets.spectral_lines.F_eC_)
    V2 = n2.get_abbe_number(lines=ot.presets.spectral_lines.F_eC_)

    # calculate powers for chromatic dispersion compensation
    # https://en.wikipedia.org/wiki/Achromatic_lens#Design
    D1 = V1/(V1 - V2) *D
    D2 = -V2/(V1 - V2) *D

else:
    # same medium for both lenses
    n1 = n2 = ot.presets.refraction_index.SF10
    n1_ = n2_ = n1(ot.presets.spectral_lines.e)
    V1 = V2 = n1.get_abbe_number(lines=ot.presets.spectral_lines.F_eC_)
    # this choice of D1, D2 leads to a similar R ratio as the achromatic case
    D1 = 2*D
    D2 = -1*D

# calculate curvature from thin-lens lensmaker's equation
# https://en.wikipedia.org/wiki/Lens#Thin_lens_approximation
R2 = (n2_-1)/D2  # plan concave lens
R1 = (n1_-1)/(D1 + (n1_-1)/R2)  # bi-convex lens
R1, R2 = 1000*R1, 1000*R2

print("\nDoublet Properties:")
print(f"n1 = {n1_:.3f},     n2 = {n2_:.3f}")
print(f"V1 = {V1:.2f},     V2 = {V2:.2f}")
print(f"R1 = {R1:.2f} mm,  R2 = {R2:.2f} mm")
print(f"D1 = {D1:.2f} dpt, D2 = {D2:.2f} dpt\n")

# make raytracer
RT = ot.Raytracer(outline=[-5, 5, -5, 5, -15, 60])

# Source 1
RSS1 = ot.CircularSurface(r=0.05)
RS1 = ot.RaySource(RSS1, divergence="None", spectrum=ot.presets.light_spectrum.F_eC_,
                  pos=[3, 0, -10], s=[0, 0, 1])
RT.add(RS1)

# Source 2, same properties, but different position
RS2 = RS1.copy()
RS2.move_to([-3, 0, -10])
RT.add(RS2)

# Lens 1 of doublet
front = ot.SphericalSurface(r=4, R=R1)
back = ot.SphericalSurface(r=4, R=R2)
L1 = ot.Lens(front, back, de=0.2, pos=[0, 0, 0], n=n1)
RT.add(L1)

# Lens 2 of doublet
front = ot.SphericalSurface(r=4, R=R2)
back = ot.CircularSurface(r=4)
# set d1=0, so "pos" is relative to the lens front vertex. In this case d2 is the overall thickness at the optical axis
# Set pos so that lens 2 comes directly after lens 1 with only a tight air gap
L2 = ot.Lens(front, back, d1=0, d2=0.5, pos=[0, 0, L1.extent[5]+0.001], n=n2)
RT.add(L2)

# add Detector
DETS = ot.RectangularSurface(dim=[10, 10])
DET = ot.Detector(DETS, pos=[0, 0, 60])
RT.add(DET)

print("\nDoublet Powers:")
F_, e, C_ = ot.presets.spectral_lines.F_eC_
print(f"{RT.tma(F_).powers[1]:.2f} dpt ({F_:.2f} nm)")
print(f"{RT.tma(e).powers[1]:.2f} dpt ({e:.2f} nm)")
print(f"{RT.tma(C_).powers[1]:.2f} dpt ({C_:.2f} nm)\n")

# run the simulator
sim = TraceGUI(RT, coloring_type="Wavelength", ray_opacity=0.8)
sim.run()
