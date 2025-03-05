#!/usr/bin/env python3

import optrace as ot
import optrace.plots as otp
import numpy as np

# Simulation of vision with an Alcon IQ monfocal intraocular lens after cataract surgery
# Renders a polychromatic pinhole image on the retina for a given pupil and object distance

# simulation parameters
#######################################################################################################################

P = 4.5  # pupil diameter 

N_rays = 3e6  # number of rays (increase for less image noise)
N_px = 189  # pixel side length of the image

oh_angle = 50/1e5  # visual object half angle (e.g. 50mm radius at 100m distance) 

# object distances
g = [100000, 1333, 667]


# create the geometry
#######################################################################################################################

# calculate raytracer geometry
max_g = np.max(g)  # worst case object distance
RS_r_max = oh_angle*max_g  # maximum raysource radius
RT_xy_max = max(RS_r_max, 10)  # lateral maximum size
RT_z0_min = -max(400, max_g)  # largest raysource position to -z direction

# raytracer with worst case size
RT = ot.Raytracer(outline=[-RT_xy_max, RT_xy_max, -RT_xy_max, RT_xy_max, RT_z0_min, 30])

# Arizona eye model preset
eye = ot.presets.geometry.arizona_eye(pupil=P)

# copy ambient refractive index behind eye lens
# and remove eye lens
nE = eye.lenses[1].n2
eye.remove(eye.lenses[1])

# create the Alcon IQ IOL from research data and patent
ACD = 4.15  # selected ACD constant, see publication
# surface geometry is described by https://patents.google.com/patent/US7350916 
front = ot.SphericalSurface(r=3, R=21.557)
back = ot.AsphericSurface(r=3, R=-22, k=-42.1929, coeff=[-2.3318e-04, -2.1144e-05, 8.9923e-06])
# for the n value, see page 6, http://www.okulix.de/okulix-en.pdf
# for the Abbe number, see https://doi.org/10.1371/journal.pone.0228342
n_IOL = ot.RefractionIndex("Abbe", n=1.554, V=37, lines=ot.presets.spectral_lines.FdC)
IOL = ot.Lens(front, back, d1=0, d2=0.593, pos=[0, 0, 0.55+ACD], n=n_IOL, n2=nE, desc="IOL")

# add IOL to eye and eye to raytracer
eye.add(IOL)
RT.add(eye)

# add Detector
DetS = ot.RectangularSurface([4, 4])
Det = ot.Detector(DetS, pos=RT.detectors[0].pos, desc="Retina")
RT.add(Det)


# simulate the image for different object distances
#######################################################################################################################

for gi in g:

    # ever object point emits a cone containing multiple directions
    # this cone is directed towards the eye, as only those are the rays which are relevant for simulation
    RS_r = oh_angle*gi
    RS_sr_angle = np.rad2deg(np.arcsin(3.5/gi))  # 3.5 is maximum pupil size + margin

    # create and add ray source
    RS = ot.RaySource(ot.CircularSurface(r=RS_r), divergence="Isotropic", orientation="Converging",
                      conv_pos=[0, 0, 0], div_angle=RS_sr_angle, pos=[0, 0, -gi], 
                      spectrum=ot.presets.light_spectrum.d65)
    RT.add(RS)

    # iteratively render the retinal image
    # constant extent (tested empirically) so the images are comparable
    # detector_index=1 is the rectangular detector
    # approximate the resolution limit d=4µm for the eye
    det_im = RT.iterative_render(N_rays, detector_index=1, limit=4, extent=[-0.10, 0.10, -0.10, 0.10])

    # calculate and the sRGB image with perceptual RI
    # see publication for the justification of this choice
    im_sRGB = det_im[0].get("sRGB (Perceptual RI)", N_px, L_th=0.01, chroma_scale=0.5)

    # remove raysource
    RT.remove(RS)
    
    # plot the image
    otp.image_plot(im_sRGB, title=f"Alcon IQ, P={P}mm, {1/gi*1e3:.2f}D")

# keep windows open
otp.block()

