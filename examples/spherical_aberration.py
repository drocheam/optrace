#!/usr/bin/env python3
# ^-- shebang for simple execution on Unix systems

# This quickstart example demonstrates some features of optrace
# The goal is to investigate the spherical aberration of a simple lens

# make sure the optrace library is installed in your system

# first we import the optrace package, as well as the gui
# gui and tracer are seperated so we don't have the overhead of always loading all multiple external libraries
import optrace as ot
from optrace.gui import TraceGUI

# a Raytracer object provides the raytracing functionality and also controls the tracing geometry
# the outline parameter specifies a three dimensional box, in which all rays and geometry are located
# the values are specified as [x0, x1, y0, y1, z0, z1], with x1 > x0, y1 > y0 and z1 > z0
# all coordinates in the tracing geometry are specified in millimeters
RT = ot.Raytracer(outline=[-10, 10, -10, 10, -25, 40])

# all elements in the geometry (sources, lenses, detectors, ...) consist of Surfaces
# a Surface object describes a specific height behaviour depending on 2D x,y-coordinates relative to the surface center
# for the raysource we define a circle surface, which is perpendicular to the z-axis with a radius of 1
RSS0 = ot.CircularSurface(r=1)

# a raysource creates the rays for raytracing, it consists of a surface, a ray divergence behaviour, 
# a specific light spectrum, a specific base orientation of the rays as well as a specific polarization
# in our case we want parallel light (divergence="None") parallel to the z-axis, 
# which is specified by the orientation vector s=[0, 0, 1]
# the light spectrum is chosen as daylight spectrum D65, which can be found in the presets submodule
# the absolute position of its surface is controlled by the parent object, in this case the ray source
RS0 = ot.RaySource(RSS0, divergence="None", spectrum=ot.presets.light_spectrum.d65,
                  pos=[0, 0, -15], s=[0, 0, 1])
# after creation the element needs to be added to the tracing geometry
RT.add(RS0)

# we create a similar ray source, now with a ring surface
# a ring is additionally parametrized by an inner circle with radius ri
RSS1 = ot.RingSurface(r=4.5, ri=1)
RS1 = ot.RaySource(RSS1, divergence="None", spectrum=ot.presets.light_spectrum.d65,
                  pos=[0, 0, -15], s=[0, 0, 1])
RT.add(RS1)

# next we define a lens
# the lens should have a constant refraction index of 1.5,
# this is specified as RefractionIndex with mode "Constant" and a value of 1.5
# generally, a RefractionIndex can also have wavelength dependent behaviour
n = ot.RefractionIndex("Constant", n=1.5)

# a lens consists of a front and back surface
# in the case of spherical lens surfaces, we need to specify a surface radius r and a curvature radius R
# For a biconvex lens the front has a positive curvature and the back a negative one
front = ot.SphericalSurface(r=5, R=15)
back = ot.SphericalSurface(r=5, R=-15)
# the creation of the lens requires both surfaces, as well as a position and a refractive index
# there multiple ways to define the overall lens thickness, or rather the distance between both surfaces
# in our case we provide de, which is a thickness extension between front and back surface
# this means there is a spacing of de=0.2mm between the end of the front surface and the start of the back surface
# the lens position defined by parameter pos is then exactly in the center of these 0.2mm
L = ot.Lens(front, back, de=0.2, pos=[0, 0, 0], n=n)
RT.add(L)

# a detector renders images inside the raytracer and is defined by a single surface
# in our case we want a rectangular detector with side lengths 20mm
# for this a RectangularSurface with a "dim" sides list of [20, 20] is initialized
DETS = ot.RectangularSurface(dim=[20, 20])
# a detector takes a surface and a position as arguments
DET = ot.Detector(DETS, pos=[0, 0, 23.])
RT.add(DET)

# after the geometry definition, the optical setup must be traced next
# this could be done by calling RT.trace(N), with parameter N being the number of rays
# but we also can create a graphical frontend, that automatically traces the scene.

# For this, a TraceGUI object is needed. 
# It takes the raytracer as parameter, while additional parameters can provide graphical settings
# For example, rays will be colored according to their source number by providing color_type="Source"
# and a higher relative ray opacity is set by ray_opacity=0.2
sim = TraceGUI(RT, coloring_mode="Source", ray_opacity=0.2)

# the frontend is now created and needs to be started explicitly
sim.run()

# You can now experiment with different features, e.g.
# 1. Navigate in the three dimensional geometry scene
# 2. Click on ray-surface intersections to display ray properties
# 3. Play around with visual settings in the main tab
# 4. Move the detector and render detector images in the Imaging-Tab
# 5. Focus search for each of the two sources. 
#    In the "Focus" Tab select "Rays From Selected Source Only" and click on "Find Focus" Button.
#    Select the other ray source to find the other focus

