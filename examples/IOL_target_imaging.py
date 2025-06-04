#!/usr/bin/env python3

import optrace as ot
import optrace.plots as otp
import numpy as np

# Simulation of vision with an Alcon IQ monfocal intraocular lens after cataract surgery
# Renders a polychromatic target image on the retina for a given pupil and object distance
# Uses Heisenberg uncertainty ray bending (HURB, see the documentation for more details)
# to approximate blurring due to diffraction

# simulation parameters
#######################################################################################################################

# pupil diameter
# P = 4.5
P = 3.0

# image
# img = ot.presets.image.ETDRS_chart_inverted
img = ot.presets.image.ETDRS_chart

N_rays = 1500000  # number of rays (increase for less image noise)
N_px = 189  # pixel side length of the image

# half angle of image
G_angle = np.deg2rad(1.0)

# object distances
g = [100000, 1333, 667]


# create the geometry
#######################################################################################################################

# calculate raytracer geometry
max_g = np.max(g)  # worst case object distance
RS_r_max = G_angle*max_g  # maximum raysource radius
RT_xy_max = max(RS_r_max, 10)  # lateral maximum size
RT_z0_min = -max(400, max_g)  # largest raysource position to -z direction

# raytracer with worst case size and activated HURB
RT = ot.Raytracer(outline=[-RT_xy_max, RT_xy_max, -RT_xy_max, RT_xy_max, RT_z0_min, 30], use_hurb=True)

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
IOL = ot.Lens(front, back, d1=0, d2=0.593, pos=[0, 0, 0.55+ACD], n=n_IOL, n2=nE, desc="Alcon IQ IOL")

# add IOL to eye and the eye to the raytracer
eye.add(IOL)
RT.add(eye)

# add Detector, it will be selectable with detector_index=1
DetS = ot.RectangularSurface([4, 4])
Det = ot.Detector(DetS, pos=RT.detectors[0].pos, desc="Retina")
RT.add(Det)

# simulate the image for different object distances
#######################################################################################################################

for gi in g:

    # divergence angle of the point source to sample the pupil, add some margin
    RS_sr_angle = np.atan(3/gi)/np.pi*180

    # image size from object distance and angle
    G_size = gi*np.tan(G_angle)

    # clear ray sources and add new ray source
    RT.remove(RT.ray_sources)
    RS = ot.RaySource(ot.Point(), divergence="Lambertian", div_angle=RS_sr_angle, pos=[0, 0, -gi],
                      spectrum=ot.presets.light_spectrum.d65)
    RT.add(RS)

    # trace setup
    RT.trace(N_rays)

    # render psf
    psf = RT.detector_image(detector_index=1, extent=[-0.1/1.25, 0.1/1.25, -0.1/1.25, 0.1/1.25])

    # select image and calculate magnification factor
    img1 = img([2*G_size, 2*G_size])
    m = ot.presets.geometry.arizona_eye().tma().image_magnification(RS.pos[2])

    # convolve and output in perceptual RI with fixed chroma scale
    img2 = ot.convolve(img1, psf, m=m, cargs=dict(rendering_intent="Perceptual", L_th=0.01, chroma_scale=0.5),
                       keep_size=True, padding_mode="edge")

    # show the result
    otp.image_plot(img2, flip=True, title=f"{IOL.desc}, {1/gi*1e3:.2f}D, P={P}mm, Perceptual RI")
            
# keep windows open
otp.block()

