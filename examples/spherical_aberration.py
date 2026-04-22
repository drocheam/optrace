#!/usr/bin/env python3
# ^-- shebang for simple execution on Unix systems

# This quickstart example demonstrates some features of optrace.
# The goal is to investigate the spherical aberration of a simple lens.

# Make sure the optrace library is installed on your system.

# First, we need to import the optrace package and its GUI.
# GUI and raytracer are separated, so we don't have the overhead of always loading all external libraries.
import optrace as ot
from optrace.gui import TraceGUI

# A Raytracer object provides the raytracing functionality and also controls the tracing geometry.
# The outline parameter specifies a three dimensional box, in which the geometry and all rays are located.
# The values are specified as [x0, x1, y0, y1, z0, z1], with x1 > x0, y1 > y0 and z1 > z0.
# All coordinates in the tracing geometry are specified in millimeters.
RT = ot.Raytracer(outline=[-10, 10, -10, 10, -25, 40])

# All elements in the geometry (sources, lenses, detectors, ...) consist of Surfaces.
# A Surface describes a specific height behavior (z) depending on 2D x,y-coordinates relative to the surface center.
# For the RaySource we define a circular surface with radius 1mm, which is perpendicular to the z-axis.
# Its absolute position in the scene is controlled by the parent object.
RSS0 = ot.CircularSurface(r=1)

# A RaySource generates the rays for raytracing. It consists of an emitting surface, a ray divergence behavior, 
# a specific light spectrum, a specific base orientation of the rays as well as a specific polarization.
# Parallel light (divergence="None") parallel to the z-axis is specified by a orientation vector of s=[0, 0, 1].
# The light spectrum is chosen as daylight spectrum D65, which can be found in the presets submodule.
RS0 = ot.RaySource(RSS0, divergence="None", spectrum=ot.presets.light_spectrum.d65,
                  pos=[0, 0, -15], s=[0, 0, 1])
# After its creation the element needs to be added to the tracing geometry.
RT.add(RS0)

# Next, we create a RaySource with a RingSurface.
# A ring is parametrized by an additional inner circle with radius ri.
RSS1 = ot.RingSurface(r=4.5, ri=1)
RS1 = ot.RaySource(RSS1, divergence="None", spectrum=ot.presets.light_spectrum.d65, pos=[0, 0, -15], s=[0, 0, 1])
RT.add(RS1)

# Next, we define a Lens with a constant refraction index of 1.5.
# The index is specified as RefractionIndex object with mode "Constant" and a value of n=1.5.
# Besides constant values, RefractionIndex supports different wavelength-dependent models and even custom behavior.
n = ot.RefractionIndex("Constant", n=1.5)

# A lens consists of a front and back surface.
# In the case of spherical lens surfaces, we need to specify a surface radius r and a curvature radius R.
# For a biconvex lens the front has a positive curvature and the back a negative one.
front = ot.SphericalSurface(r=5, R=15)
back = ot.SphericalSurface(r=5, R=-15)
# The creation of the lens requires both surfaces, as well as a position and a refractive index.
# There are multiple ways to define the overall lens thickness:
# In our case we provide the parameter de, which is a thickness extension between front and back surface.
# This means that there is a spacing of de=0.2mm between the end of the front surface and the start of the back surface.
# The Lens position parameter pos then defines the geometric center of these 0.2mm.
L = ot.Lens(front, back, de=0.2, pos=[0, 0, 0], n=n)
RT.add(L)

# A Detector renders images inside the Raytracer scene and is geometrically defined by a single surface.
# For a rectangular detector with side lengths of 20mm a RectangularSurface with a "dim" of [20, 20] is defined:
DETS = ot.RectangularSurface(dim=[20, 20])
# A detector takes a Surface and a position as arguments
DET = ot.Detector(DETS, pos=[0, 0, 23.])
RT.add(DET)

# After the geometry definition, the optical setup must be simulated next.
# This could be done by either calling RT.trace(N), with N being the number of rays,
# or by creating a graphical frontend, that automatically traces the scene.

# Such a TraceGUI object requires the raytracer as parameter and supports different visualization setting parameters.
# For example, rays will be colored according to their source number by providing color_type="Source",
# while a higher relative ray opacity is set by ray_opacity=0.2.
sim = TraceGUI(RT, coloring_mode="Source", ray_opacity=0.2)

# The GUI is started by calling run().
sim.run()

# You can now experiment with different features:
# 1. Navigate in the three dimensional geometry scene.
# 2. Click on ray-surface intersections to display different ray properties.
# 3. Play around with visual settings in the main tab.
# 4. Move the detector and render detector images in the Imaging-Tab.
# 5. Do a Focus search for each of the two sources. 
#    In the "Focus" Tab select "Rays From Selected Source Only" and click on the "Find Focus" Button.
#    Select the other ray source to find the second focus.

