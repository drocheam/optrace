Raytracer
------------------------------------------------


.. |IdealLens| replace:: :class:`IdealLens <optrace.tracer.geometry.ideal_lens.IdealLens>`
.. |Lens| replace:: :class:`Lens <optrace.tracer.geometry.lens.Lens>`
.. |Group| replace:: :class:`Group <optrace.tracer.geometry.group.Group>`
.. |Element| replace:: :class:`Element <optrace.tracer.geometry.element.Element>`
.. |Raytracer| replace:: :class:`Raytracer <optrace.tracer.raytracer.Raytracer>`

.. testsetup:: *

   import optrace as ot
   ot.global_options.show_progressbar = False

.. role:: python(code)
  :language: python
  :class: highlight


.. _usage_raytracer:

Raytracer
________________

**Overview**

The |Raytracer| class provides the functionality for tracing, geometry checking, rendering spectra and images, 
and focusing.

Since the |Raytracer| is a subclass of a |Group|, elements can be changed or added in the same way.


.. figure:: ../images/example_legrand1.png
   :width: 700
   :align: center
   :class: dark-light

   Example of a raytracer geometry in the TraceGUI in side view


**Outline**

All objects and rays can only exist in a three-dimensional box, the *outline*.
When initializing the |Raytracer| it is passed as :python:`outline` parameter.

.. testcode::

   RT = ot.Raytracer(outline=[-2, 2, -3, 3, -5, 60])


**Geometry**

Since optrace implements sequential raytracing, the surfaces and objects must be in a well-defined and unique sequence. 
This applies to all elements with interactions of light: :python:`Lens, IdealLens, Filter, Aperture, RaySource`.
The elements :python:`Detector, LineMarker, PointMarker, BoxVolume, SphereVolume, CylinderVolume` 
are excluded from this. All ray source elements must lie before all lenses, filters and apertures. 
And all subsequent lenses, filters, apertures must not collide with each other and be inside the outline.

Rays that hit the outline box will be absorbed.

**Surrounding Media**

In :ref:`usage_lens` we learned that when creating a lens, the :python:`n2` parameter defines the subsequent medium. 
In the case of multiple lenses, the :python:`n2` of the previous lens is the medium prior to the next lens.
In the case of the raytracer, we can define an :python:`n0` which defines the refractive index for all 
undefined :python:`n2=None`, as well as for the region to the first lens.

The following figure shows a setup with lenses :python:`L0, L2` having a :python:`n2` defined and a custom 
:python:`n0` parameter in the raytracer class. The medium before the first lens as well as the medium 
behind :python:`L1` are therefore also :python:`n0`.

.. figure:: ../images/rt_setup_different_ambient_media.svg
   :width: 760
   :align: center
   :class: dark-light

   Schematic figure of a setup with a ray source, three different lenses and three different ambient media


**Absorbing Rays**

optrace ensures that rays not intersecting both lens surfaces get absorbed.

Generally, these rays are error cases.
A ray only hitting one surfaces must enter/leave through the lens side cylinder, 
that is not handled in our sequential simulation.
Rays not hitting the lens at all are typically undesired. 
In real optical systems they would be (hopefully) absorbed by the housing of the system.

**Parameter** :attr:`no_pol <optrace.tracer.raytracer.Raytracer.no_pol>`

The raytracer provides the functionality to trace polarization directions. Thus, not only the polarization vector 
for the ray and ray segment can be calculated, but also the exact transmission at each surface transition.
Unfortunately, the calculation is comparatively computationally intensive.

With the parameter :python:`no_pol=True` no polarizations are calculated and a unpolarized/uniformly polarized light 
is assumed everywhere. Typically this speeds up the tracing by 10-30%.
Whether you can neglect the influence of polarization depends on the exact setup of the geometry.


Tracing
_____________________

**Tracing the system**

Tracing is with the :meth:`Raytracer.trace() <optrace.tracer.raytracer.Raytracer.trace>` 
method of the |Raytracer| class.
It takes the number of rays :python:`N` as parameter.
The method uses the current tracing geometry and stores the ray properties 
internally in a :class:`RayStorage <optrace.tracer.ray_storage.RayStorage>` object.

**Example**

Below you can find an example. A eye preset is loaded and flipped around the x-axis.
A point source is added at the retina and the geometry is traced.

.. testcode::

   import optrace as ot

   # init raytracer 
   RT = ot.Raytracer(outline=[-10, 10, -10, 10, -10, 60])

   # load eye preset
   eye = ot.presets.geometry.arizona_eye(pupil=3)

   # flip, move and add it to the tracer
   eye.flip()
   eye.move_to([0, 0, 0])
   RT.add(eye)

   # create and add divergent point source
   point = ot.Point()
   RS = ot.RaySource(point, spectrum=ot.presets.light_spectrum.d50, 
                     divergence="Isotropic", div_angle=5,
                     pos=[0, 0, 0])
   RT.add(RS)

   # trace
   RT.trace(100000)

   # access ray parameters, render images etc.
   ...


**Accessing the Ray Properties**

Described in :numref:`usage_ray_access`.

**Rendering Images**

Described in :numref:`rimage_rendering`.

**Tracing with many rays**

The number of rays is limited by the RAM usage.
The RAM usage is by default limited by the 
:attr:`Raytracer.MAX_RAY_STORAGE_RAM <optrace.tracer.raytracer.Raytracer.MAX_RAY_STORAGE_RAM>` parameter, 
the actual number of rays results from the complexity of the geometry.
Its default value is 8GB, but it can be set for each |Raytracer| object separately.

To generate images with even more rays, the method 
:meth:`Raytracer.iterative_render <optrace.tracer.raytracer.Raytracer.iterative_render>` can be applied, 
which traces the geometry iteratively without holding all rays in memory. 
More details are available in :numref:`rimage_iterative_render`.


.. _hurb_usage:

Modelling Diffraction
________________________

**Image Blurring**

Image blurring can be applied to rendered images to account for resolution limits.
This process utilizes an Airy disk filter, as detailed in :numref:`image_airy_filter`, 
to approximate the blurring effect. 
It is important to note that this method provides a very generalized approximation.

**Ray Bending**

optrace incorporates experimental support for Heisenberg Uncertainty Ray Bending (HURB). 
Further technical details regarding its implementation are available in :numref:`hurb_details`.
An example for experimentation with HURB is available in :numref:`example_hurb_apertures`.

The current implementation of HURB has the following limitations:

* HURB simulates the blurring associated with edge diffraction. It does not account for interference effects.
* Deviations persist between theoretical and simulated beam profiles. 
  For a detailed comparison, refer to :numref:`hurb_comparison`.
* Ray bending is currently limited to the inner aperture edges 
  of :class:`RingSurface <optrace.tracer.geometry.surface.ring_surface.RingSurface>`
  and :class:`SlitSurface <optrace.tracer.geometry.surface.slit_surface.SlitSurface>` types.
* All apertures are modeled as diffracting elements.
* The aperture stop must be explicitly defined as a surface within the optical setup.

Another issue is that bending leads to large angle rays.
The use of image rendering with automatic extent is discouraged,
as these rays lead to drastically increased automatically set sizes.
Provide the image size manually, see :numref:`rimage_rendering`.

Given these restrictions and the experimental status of the feature, 
HURB requires explicit activation. 
To enable HURB, set :python:`use_hurb=True` during the raytracer initialization:

.. testcode::

    RT = ot.Raytracer(outline=[-2, 2, -3, 3, -5, 60], use_hurb=True)

A custom uncertainty scaling factor can be configured using 
the :attr:`HURB_FACTOR <optrace.tracer.raytracer.Raytracer.HURB_FACTOR>` attribute:

.. testcode::

    RT.HURB_FACTOR = 2.3

Additional information concerning this factor is provided in :numref:`hurb_uncertainty_factor`. 


