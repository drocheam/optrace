#!/usr/bin/env python3

import optrace as ot
import optrace.plots as otp

# same geometry as the 'image_render.py' example,
# but different image and many more rays

RSS = ot.presets.image.tv_testcard2([4, 3])

# make raytracer
RT = ot.Raytracer(outline=[-5, 5, -5, 5, 0, 40])

# add Raysource
RS = ot.RaySource(RSS, divergence="Lambertian", div_angle=8,
                  s=[0, 0, 1], pos=[0, 0, 0])
RT.add(RS)

# add Lens 1
front = ot.SphericalSurface(r=3, R=8)
back = ot.SphericalSurface(r=3, R=-8)
nL1 = ot.RefractionIndex("Constant", n=1.5)
L1 = ot.Lens(front, back, de=0.1, pos=[0, 0, 12], n=nL1)
RT.add(L1)

# add Detector
Det = ot.Detector(ot.RectangularSurface(dim=[10, 10]), pos=[0, 0, 36])
RT.add(Det)

# render and show detector image
pos = [[0, 0, 27.], [0, 0, 30.], [0, 0, 33.], [0, 0, 36.]]
_, Ims = RT.iterative_render(N_rays=15e6, N_px_D=300, pos=pos, no_sources=True)

# show rendered images
for img in Ims:
    otp.r_image_plot(img, "sRGB (Absolute RI)")
otp.block()
