
.. _usage_elements:

Elements (Lens, Ray Source,  ...)
------------------------------------------

.. |IdealLens| replace:: :class:`IdealLens <optrace.tracer.geometry.ideal_lens.IdealLens>`
.. |Lens| replace:: :class:`Lens <optrace.tracer.geometry.lens.Lens>`
.. |Group| replace:: :class:`Group <optrace.tracer.geometry.group.Group>`
.. |Element| replace:: :class:`Element <optrace.tracer.geometry.element.Element>`
.. |Raytracer| replace:: :class:`Raytracer <optrace.tracer.raytracer.Raytracer>`

.. testsetup:: *

   import optrace as ot
   import numpy as np
   np.set_printoptions(precision=8)

.. role:: python(code)
  :language: python
  :class: highlight


Overview
__________________


Types
##############

In optrace the class |Element| denotes an object which has no, one or two surfaces and belongs to the tracing geometry.

**Tracing Elements**

Tracing elements are Elements with direct ray interaction:

.. list-table::
   :widths: 100 400
   :header-rows: 0
   :align: left

   * - :class:`RaySource <optrace.tracer.geometry.ray_source.RaySource>`
     - An element with a light emitting surface
   * - |Lens|
     - An element with two surfaces on which light is refracted.
   * - |IdealLens|
     - A Lens, except that it has a planar surface and refracts light without aberrations
   * - :class:`Filter <optrace.tracer.geometry.filter.Filter>`
     - Element with a surface on which wavelength-dependent filtering takes place.
   * - :class:`Aperture <optrace.tracer.geometry.aperture.Aperture>`
     - Similar to a Filter, except that incident light is completely absorbed.

**Rendering Elements**

Elements with no ray interaction for tracing, but the possibility to render images of intersecting rays.

.. list-table::
   :widths: 100 400
   :header-rows: 0
   :align: left

   * - :class:`Detector <optrace.tracer.geometry.detector.Detector>`
     - Element with one surface on which images or spectra can be rendered


**Markers**

Markers are Elements for annotations in 3D space.

.. list-table::
   :widths: 100 400
   :header-rows: 0
   :align: left

   * - :class:`PointMarker <optrace.tracer.geometry.marker.point_marker.PointMarker>`
     - Element consisting of a point and a label
   * - :class:`LineMarker <optrace.tracer.geometry.marker.line_marker.LineMarker>`
     - Element consisting of a line and a label


**Volumes**

Objects for plotting volumes in the TraceGUI, for instance an enclosing cylinder or a medium outline.

.. list-table::
   :widths: 100 400
   :header-rows: 0
   :align: left

   * - :class:`BoxVolume <optrace.tracer.geometry.volume.box_volume.BoxVolume>`
     - Volume of a box or cube
   * - :class:`CylinderVolume <optrace.tracer.geometry.volume.cylinder_volume.CylinderVolume>`
     - Cylinder volume with the symmetry axis in direction of the optical axis
   * - :class:`SphereVolume <optrace.tracer.geometry.volume.sphere_volume.SphereVolume>`
     - A spherical volume

.. _element_geometry_props:

Shared Functions/Properties
##############################

All subclasses of |Element| share the following methods and properties.
In the following, an element object :python:`El`, which can be of any the subclasses described above, to demonstrate them.

**Position**

The position of the element is accessed using:

.. code-block:: python

   pos = El.pos

This returns a three element list with x, y, z coordinates.
For an element consisting of a singular surface, line or point, the returned value is identical to the position of this surface/line/point.
For an element consisting of two surfaces, the position depends on how the object was specified, for instance see :ref:`usage_lens`. 

**Extent**

The extent box is the smallest encompassing bounding box that include the element.
The extent property returns a list of coordinate bounds :python:`[x0, x1, y0, y1, z0, z1]`.
It is accessed using:

.. code-block:: python

   extent = El.extent

**Moving**

An object is moved using the :python:`move_to` function.
A single parameter describes the new absolute position as list :python:`[x, y, z]`.

.. code-block:: python

   El.move_to([0.2, -1.25, 3])

The element is moved in such a way that the provided position is its new :python:`El.pos`.

**Rotation**

Using the :python:`rotate` function the objects can be rotated around their center around the z-axis.
The function takes a rotation angle in degrees:

.. code-block:: python

   El.rotate(15)

**Flipping**

Flipping the element rotates it around an axis parallel to the x-axis passing through the position :python:`El.pos`.
Doing so, the order of front and back surface is reversed and each surface also gets flipped.

.. code-block:: python

   El.flip()


RaySource
_______________________

Overview
#############################


A :class:`RaySource <optrace.tracer.geometry.ray_source.RaySource>` defines the properties for rays that it creates, including

* Emitting area/point/line
* light distribution on this area (=image)
* Emitted spectrum and power
* Ray Polarization
* Ray orientation
* Ray divergence
* Source position


Surface/Point/Line Parameter
##################################

A RaySource supports the following base shapes :python:`Point, Line, CircularSurface, RectangularSurface, RingSurface`, which are provided as first parameter to the :python:`RaySource()` constructor.

.. testcode::

   circ = ot.CircularSurface(r=3)
   RS = ot.RaySource(circ)


Position Parameter
##################################

The position in three-dimensional space is provided by the :python:`pos`-parameter.

.. testcode::

   RS = ot.RaySource(circ, pos=[0, 1.2, -3.5])

Power Parameter
##################################

Providing the :python:`power` you can define the cumulative power of all rays. This proves especially useful when working with multiple sources and different power ratios.

.. testcode::

   RS = ot.RaySource(circ, power=0.5)

Orientation Parameter
##################################

The base orientation type of the rays is defined by the :python:`orientation`-parameter.

For :python:`orientation="Constant"` the orientation is independent of the position on the emitting area.
In this case you can provide the orientation vector using the :python:`s`-parameter in cartesian coordinates.

.. testcode::

   RS = ot.RaySource(circ, orientation="Constant", s=[0.7, 0, 0.7])

Or with :python:`s_sph` for spherical coordinates, where the first one is the angle between the orientation and the optical axis and the second the angle inside the lateral plane. Values are provided in degrees, for instance:

.. testcode::

   RS = ot.RaySource(circ, orientation="Constant", s_sph=[20, -30])

If all rays from the source should be converging to a position :python:`conv_pos`, mode :python:`orientation="Converging"` can be used:

.. testcode::

   RS = ot.RaySource(circ, orientation="Converging", conv_pos=[10, 2, -1])

It is also possible to define orientations as a function of the position of the rays. For this we need to set :python:`orientation="Function"` and provide the :python:`or_func` parameter.
This parameter takes two numpy arrays containing the x and y-position and returns a two dimensional array with cartesian vector components in rows.

.. testcode::

   def or_func(x, y, g=5):
       s = np.column_stack((-x, -y, np.ones_like(x)*g))
       ab = (s[:, 0]**2 + s[:, 1]**2 + s[:, 2]**2) ** 0.5
       return s / ab[:, np.newaxis]
   
   RS = ot.RaySource(circ, orientation="Function", or_func=or_func)

As with other functions we can also provide a keyword argument dictionary for the function, in our case this is done by the :python:`or_args` parameter.

.. testcode::

   ... 
   RS = ot.RaySource(circ, orientation="Function", or_func=or_func, or_args=dict(g=10))

Spectrum Parameter
##################################

A :class:`LightSpectrum <optrace.tracer.spectrum.light_spectrum.LightSpectrum>` object is provided with the :python:`spectrum` parameter.
For instance, this can be a predefined spectrum:

.. testcode::

   RS = ot.RaySource(circ, spectrum=ot.presets.light_spectrum.d75)

Or a user defined one:

.. testcode::

   spec = ot.LightSpectrum("Monochromatic", wl=529)
   RS = ot.RaySource(circ, spectrum=spec)


Divergence Parameter
##################################

Divergence defines how rays are distributed relative to their base orientation (:python:`orientation` parameter).

With :python:`divergence="None"` all rays follow their orientation:

.. testcode::

   RS = ot.RaySource(circ, divergence="None", s=[0.7, 0, 0.7])

Paired with :python:`orientation="Constant"` all rays are emitted in parallel.

We can also define lambertian divergence, which follows the cosine law.
:python:`div_angle` defines the half opening angle of the cone volume in which the divergence is generated.

.. testcode::

   RS = ot.RaySource(circ, divergence="Lambertian", div_angle=10)

:python:`divergence="Isotropic"` defines divergence with equal probability in all directions, but again only inside the cone defined by :python:`div_angle`.

.. testcode::

   RS = ot.RaySource(circ, divergence="Isotropic", div_angle=10)

User functions can be defined by :python:`divergence="Function"` and providing the :python:`div_func` parameter.
This function must take angular values in radians up to :python:`div_angle` and return a normalized or unnormalized  probability.

.. testcode::

   RS = ot.RaySource(circ, divergence="Function", div_func=lambda e: np.cos(e)**2, div_angle=10)

For all the combinations above we can also generate a direction distribution inside an circular arc instead of a cone. The correct way to do this is by setting :python:`div_2d=True`. With :python:`div_axis_angle` we can additionally define the orientation of this arc distribution.

.. testcode::

   RS = ot.RaySource(circ, divergence="Function", div_func=lambda e: np.cos(e)**2, div_2d=True, div_axis_angle=20, div_angle=10)


Image Parameter
##################################

Alternatively to a uniformly emitting area there is the possibility to provide light distributions (=images).

For this the emitting surface needs to be a :python:`Image` object.
A Rectangular surface for this image is created automatically.
The size will be equal to the side lengths of the image.

.. testcode::

   image = ot.presets.image.landscape([2, 3])
   RS = ot.RaySource(image)

.. testcode::

   image = ot.RGBImage(np.random.sample((300, 300, 3)), [2, 3])
   RS = ot.RaySource(image)

.. code-block:: python

   image = ot.RGBImage("some_image_path", [2, 3])
   RS = ot.RaySource(image)

Every image color generates a specific physical spectrum matching its color. This spectrum is a linear combination of the sRGB primaries in <>.

With :python:`image` specified the :python:`spectrum` is unused.

Polarization Parameter
##################################

The polarization parameter describes the distribution of the direction of linear light polarizations.

In the default case the directions are random, specified by :python:`polarization="Uniform"`.

.. testcode::

   RS = ot.RaySource(circ, polarization="Uniform")

:python:`polarization="x"` defines polarizations parallel to the x-axis.

.. testcode::

   RS = ot.RaySource(circ, polarization="x")

:python:`polarization="y"` defines polarizations parallel to the y-axis.

.. testcode::

   RS = ot.RaySource(circ, polarization="y")

:python:`polarization="xy"` defines random polarizations of x or y-direction.

.. testcode::

   RS = ot.RaySource(circ, polarization="xy")

The user can also set a user-defined value with :python:`polarization="Constant"` and the :python:`pol_angle` parameter.
The polarization direction is defined by an angle inside the plane perpendicular to the ray direction.

.. testcode::

   RS = ot.RaySource(circ, polarization="Constant", pol_angle=12)

Or alternatively a list with :python:`polarization="List"`, the angular values in :python:`pol_angles` and their probabilities in :python:`pol_probs`.

.. testcode::

   RS = ot.RaySource(circ, polarization="List", pol_angles=[0, 45, 90], pol_probs=[0.5, 0.25, 0.25])

Lastly, a user defined function can be set with  :python:`polarization="Function"` and the :python:`pol_func` parameter.
This parameter takes angles in range :math:`[0, ~2 \pi]` and returns a normalized or unnormalized probability.


Above we talked how for instance for :python:`polarization="x"` the rays are parallel to the x-axis. However, depending on their actual ray orientation this isn't always the case. Read about what the angles mean for rays not parallel to the optical axis in <>.

.. testcode::

   RS = ot.RaySource(circ, polarization="Function", pol_func=lambda ang: np.exp(-(ang - 30)**2/10))


.. _usage_lens:

Lens
________

Overview
##################################


A :class:`Lens <optrace.tracer.geometry.lens.Lens>` consists of two surfaces and a medium with a :class:`RefractionIndex <optrace.tracer.refraction_index.RefractionIndex>` between them.
Additionally we need to provide the position and some thickness parameter, that will be explained later.

Example
##################################


.. testcode:: 

   sph1 = ot.SphericalSurface(r=3, R=10.2)
   sph2 = ot.SphericalSurface(r=3, R=-20)
   n = ot.RefractionIndex("Sellmeier2", coeff=[1.045, 0.266, 0.206, 0, 0])

   L = ot.Lens(sph1, sph2, n=n, pos=[0, 2, 10], de=0.5)

To define a non-standard medium (not the one defined by the raytracing geometry) we can provide the :python:`n2` parameter, that defines the medium after the second lens surface.

.. testcode::

   n2 = ot.RefractionIndex("Constant", n=1.2)
   L = ot.Lens(sph1, sph2, n=n, pos=[0, 2, 10], de=0.5, n2=n2)


.. _usage_lens_thickness:

Lens Thickness
##################################


To allow for simple definitions of lens thickness and positions, there are multiple ways to define the thickness:

* :python:`d`: thickness at the optical axis
* :python:`de`: thickness extension. Distance between largest z-position on front and lowest z-position on back
* :python:`d1`: distance between front surface center z-position and z-position of :python:`pos` of Lens
* :python:`d2`: distance between z-position of :python:`pos` of Lens and z-position of the back surface center


.. figure:: ../images/lens_thickness.svg
   :align: center
   :width: 600
   :class: dark-light

   :math:`d` and :math:`d_\text{e}` for a convex lens, a concave lens and a meniscus lens

While for a convex lens using the :python:`de` is most comfortable, for concave or meniscus lenses the thickness at the optical axis :python:`d` proves more useful.
For instance, a concave lens can be defined like this:

.. testcode::

   L = ot.Lens(sph2, sph1, n=n, pos=[0, 2, 10], d=0.5)

When the lens is defined by :python:`d` or :python:`de` the position :python:`pos[2]` is at the center of the :python:`d` or :python:`de` distance.

With the :python:`d1` and :python:`d2` parameters we can control the position of both surfaces relative to the lens position manually. For instance with :python:`d1=0, d2=...` the lens front starts exactly at the :python:`pos` of the Lens.
On the other hand setting :python:`d1=..., d2=0` leads to the back surface center ending at :python:`pos`.


.. figure:: ../images/lens_thickness_position.svg
   :align: center
   :width: 600
   :class: dark-light

   Defining a convex lens by ``de=...``, by ``d1=0, d2=...`` and by ``d1=..., d2=0``.


All cases in-between are also viable, for instance:

.. testcode::

   L = ot.Lens(sph1, sph2, n=n, pos=[0, 2, 10], d1=0.1, d2=0.6)
   
But only as long as the surfaces don't collide.
With a Lens object you can also access the thickness parameters:

.. doctest::

   >>> L.d
   0.7

.. doctest::
   
   >>> L.de
   0.022566018...

.. doctest::
   
   >>> L.d1
   0.1

.. doctest::
   
   >>> L.d2
   0.6

Or the parameters of its surfaces, like:

.. doctest::

   >>> L.front.ds
   0.45115391...


Paraxial Properties
##################################


As for a setup of many lenses, we can also do paraxial analysis on a simple lens.

To create a ray transfer matrix analysis object (:class:`TMA <optrace.tracer.transfer_matrix_analysis.TMA>` object) we call the member function :python:`tma()`.
From there on we can use it as described in <>.

.. doctest::

   >>> tma = L.tma()
   >>> tma.efl
   12.749973...

As the behavior can differ with the light wavelength, we can also provide a non-default wavelength in nanometers.
Since the lens has no knowledge of the geometry around it, the medium before it is also undefined. By default, a constant refractive index of 1 is assumed, but can be overwritten with the parameter :python:`n0`.

.. doctest::

   >>> tma = L.tma(589.2, n0=ot.RefractionIndex("Constant", n=1.1))
   >>> tma.efl
   17.300045...


Ideal Lens
_____________


An :class:`IdealLens <optrace.tracer.geometry.ideal_lens.IdealLens>` focusses and images light perfectly and without aberrations according to the imaging equation. The geometry is an infinitesimal thin circular area with radius :python:`r`.
Additionally we need to provide the optical power :python:`D` and a position :python:`pos`.

.. testcode::

   IL = ot.IdealLens(r=5, D=12.5, pos=[0, 0, 9.5])

As for a normal Lens a :python:`n2` can be defined. Note that this does not change the optical power or focal length, as they are controlled by the :python:`D` parameter.

.. testcode::

   n2 = ot.RefractionIndex("Constant", n=1.25)
   IL = ot.IdealLens(r=4, D=-8.2, pos=[0, 0, 9.5], n2=n2)


Filter
___________

When light hits a :class:`Filter <optrace.tracer.geometry.filter.Filter>` part of the ray power is transmitted according to the filter's transmittance function.

A Filter is defined by a Surface, a position and the :class:`TransmissionSpectrum <optrace.tracer.spectrum.transmission_spectrum.TransmissionSpectrum>`.

.. testcode::

   spec = ot.TransmissionSpectrum("Rectangle", wl0=400, wl1=500, val=0.5)
   circ = ot.CircularSurface(r=5)
   F = ot.Filter(circ, pos=[0, 0, 23.93], spectrum=spec)


With a filter at hand we can calculate its approximate sRGB color using :python:`F.color()`. The fourth return value is the opacity for visualization. Note that the opacity is more like a visual extra than a simulation of the actual opacity.

Calling the filter with wavelengths returns the transmittance at these wavelengths.

.. doctest::

   >>> wl = np.array([380, 400, 550])
   >>> F(wl)
   array([0. , 0.5, 0. ])


When tracing the raytracer sets all transmission values below a specific threshold :python:`T_TH` to zero. This is done to avoid ghost rays, that are rays that merely contribute to the light distribution or image but are nonetheless calculated and reduce performance. An example could be rays far away from the mean value in normal distribution/ gaussian function.

By default the threshold value is

.. doctest::

   >>> ot.Raytracer.T_TH
   1e-05


Aperture
________________

An :class:`Aperture <optrace.tracer.geometry.aperture.Aperture>` is just a :class:`Filter <optrace.tracer.geometry.filter.Filter>` that absorbs complete. In the most common use cases a :class:`RingSurface <optrace.tracer.geometry.surface.ring_surface.RingSurface>` is applied as Aperture surface. As for other elements, we also need to specify the position :python:`pos`.

.. testcode::

   ring = ot.RingSurface(ri=0.05, r=5)
   AP = ot.Aperture(ring, pos=[0, 2, 10.1])

Detector
__________________

A :class:`Detector <optrace.tracer.geometry.detector.Detector>` enables us to render images and spectra on its geometry. But by itself, it has no effect on raytracing.

It takes a surface parameter and the position parameter as arguments.

.. testcode::

   rect = ot.RectangularSurface(dim=[1.5, 2.3])
   Det = ot.Detector(rect, pos=[0, 0, 15.2])


Markers
_____________

PointMarker
#################

A :class:`PointMarker <optrace.tracer.geometry.marker.point_marker.PointMarker>` is used to annotate positions or elements inside the tracing geometry. While itself having no influence on the tracing process.

In the simplest case a :python:`PointMarker` is defined with a text string and a position for the :class:`Point <optrace.tracer.geometry.point.Point>`.

.. testcode::

   M = ot.PointMarker("Text132", pos=[0.5, 9.1, 0.5])

One can scale the text and marker with :python:`text_factor` or :python:`marker_factor`. The actual size change is handled by the plotting GUI.

.. testcode::

   M = ot.PointMarker("Text132", pos=[0.5, 9.1, 0.5], text_factor=2.3, marker_factor=0.5)

We can also hide the marker point and only display the text with the parameter :python:`label_only=True`.

.. testcode::

   M = ot.PointMarker("Text132", pos=[0.5, 9.1, 0.5], label_only=True)

In contrast, we can hide the text and only plot the marker point by leaving the text empty:

.. testcode::

   M = ot.PointMarker("", pos=[0.5, 9.1, 0.5])


LineMarker
#################


Similarly, a :class:`LineMarker <optrace.tracer.geometry.marker.line_marker.LineMarker>` is a :class:`Line <optrace.tracer.geometry.line.Line>` in the xy-plane with a text annotation.

In the simplest case a :python:`LineMarker` is defined with a text string, radius, angle and a position.

.. testcode::

   M = ot.LineMarker(r=3, desc="Text132", angle=45, pos=[0.5, 9.1, 0.5])

One can scale the text and marker with :python:`text_factor` or :python:`line_factor`. The actual size change is handled by the plotting GUI.

.. testcode::

   M = ot.LineMarker(r=3, desc="Text132", pos=[0.5, 9.1, 0.5], text_factor=2.3, line_factor=0.5)


We can hide the text and only plot the marker line by leaving the text empty:

.. testcode::

   M = ot.LineMarker(r=3, desc="", pos=[0.5, 9.1, 0.5])



Volumes
__________________


BoxVolume
###############

As for a :class:`RectangularSurface <optrace.tracer.geometry.surface.rectangular_surface.RectangularSurface>`, the parameter :python:`dim` defines the x- and y-side lengths in the lateral plane. Parameter :python:`pos` describes the center of this rectangle. For a :class:`BoxVolume <optrace.tracer.geometry.volume.box_volume.BoxVolume>` this surface gets extended by length :python:`length` in positive z-direction, forming a three-dimensional volume.

.. testcode::

   ot.BoxVolume(dim=[10, 20], length=15, pos=[0, 2, 3])

Additionally the plotting opacity and color can be specified:

.. testcode::

   ot.BoxVolume(dim=[10, 20], length=15, pos=[0, 2, 3], opacity=0.8, color=(0, 1, 0))

SphereVolume
#################

A :class:`SphereVolume <optrace.tracer.geometry.volume.sphere_volume.SphereVolume>` is defined by its center position :python:`pos` and the sphere radius :python:`R`:

.. testcode::

   ot.SphereVolume(R=10, pos=[0, 2, 3])

As for the other volumes the plotting opacity and color can be specified:

.. testcode::

   ot.SphereVolume(R=10, pos=[0, 2, 3], opacity=0.8, color=(0, 0, 1))


CylinderVolume
#################

A :class:`CylinderVolume <optrace.tracer.geometry.volume.cylinder_volume.CylinderVolume>` is defined by its front surface center position :python:`pos` and the cylinder radius :python:`r`:

.. testcode::

   ot.CylinderVolume(r=5, length=15, pos=[0, 2, 3])


As for the other volumes the plotting opacity and color can be specified:

.. testcode::

   ot.CylinderVolume(r=5, length=15, pos=[0, 2, 3], opacity=0.8, color=(0.5, 0.1, 0.0))


Custom Volumes
#######################


A custom :class:`Volume <optrace.tracer.geometry.volume.volume.Volume>` can also be defined. It needs a front and back surface as parameter, as well as a position and the thickness distances :python:`d1, d2`. These have the same meaning as for a :class:`Lens <optrace.tracer.geometry.lens.Lens>` in :numref:`usage_lens_thickness`.

We can for instance do this with:

.. testcode::

   front = ot.ConicSurface(r=4, k=2, R=50)
   back = ot.RectangularSurface(dim=[3, 3])
   vol = ot.Volume(front, back, pos=[0, 1, 2], d1=front.ds, d2=back.ds+1)

Here we define a conic front surface and a rectangular surface. :python:`front.ds, back.ds` denotate the total thickness of both surfaces at their center. The overall length for this volumes is then :python:`front.ds + back.ds + 1`, because an additional value of 1 was added.

