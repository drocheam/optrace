Advanced Topics
------------------------------------------------

.. role:: python(code)
  :language: python
  :class: highlight

.. testsetup:: *

   import optrace as ot
   ot.global_options.show_progressbar = False

Raytracing Errors
_________________________

The raytracer emits relevant warnings and errors while simulating the geometry.
These are internally characterized by flags, which are listed in the following table:

.. list-table:: List of raytracing flags and messages
   :widths: 100 600
   :header-rows: 1
   :align: left
   :width: 900px

   * - Flag
     - Description

   * - :obj:`Raytracer.INFOS.ABSORB_MISSING <optrace.tracer.raytracer.Raytracer.INFOS.ABSORB_MISSING>`
     - Rays are absorbed at the current lens surface because they missed it.

   * - :obj:`Raytracer.INFOS.TIR <optrace.tracer.raytracer.Raytracer.INFOS.TIR>`
     - Total inner reflection. As reflections are not simulated, the ray is treated as being absorbed at the refracting surface.

   * - :obj:`Raytracer.INFOS.ILL_COND <optrace.tracer.raytracer.Raytracer.INFOS.ILL_COND>`
     -  Ill-conditioned rays for intersection calculation at a numerical, custom surface. In almost all cases the resulting intersection would be wrong. This can happen if the surface is badly defined numerically or geometrically, there are surface collisions or a ray hits the surface multiple times. Please check the geometry of the raytracer and the surface definition.

   * - :obj:`Raytracer.INFOS.OUTLINE_INTERSECTION <optrace.tracer.raytracer.Raytracer.INFOS.OUTLINE_INTERSECTION>`
     - Rays that would have left the defined geometry outline are absorbed at the affected outline plane.

The raytracer also provides a :attr:`Raytracer.geometry_error <optrace.tracer.raytracer.Raytracer.geometry_error>` 
flag that is set when tracing was aborted due to a critical geometry issue.
Geometry checks are only performed while tracing, 
so :meth:`Raytracer.trace <optrace.tracer.raytracer.Raytracer.trace>` must be called first.

In the case of surface collisions, a subset of collision points will be displayed in the 3D view inside the GUI.


.. _usage_ray_access:

Accessing Ray Properties
_____________________________


Overview
################

**Definitions:**

+-----------------+----------------------------------------------------------------------------------------------------+
| **Ray Section** | part of a ray with a constant direction vector. Sections start and end at surface intersections.   |
+-----------------+----------------------------------------------------------------------------------------------------+
| **Ray**         | entirety of the ray going from the source to its absorption                                        |
+-----------------+----------------------------------------------------------------------------------------------------+
| **N**           | number of rays                                                                                     |
+-----------------+----------------------------------------------------------------------------------------------------+
| **Nt**          | number of sections per ray, equal for all rays                                                     |
+-----------------+----------------------------------------------------------------------------------------------------+

The number of sections is identical for all rays. 
If a ray is absorbed early on, all consecutive sections consist of zero length vectors starting 
at the last position and having their power set to zero. 
Direction and polarization are undefined.

The following table shows an overview of available ray properties.
Some of those are attributes that are stored while tracing, 
while others are functions, which must be calculated from other attributes first.

.. list-table:: List of ray properties
   :widths: 100 100 200 50 400
   :header-rows: 1
   :align: left
   :width: 900px

   * - Quantity
     - Method/Attribute
     - Type
     - Unit
     - Function
   * - Position
     - :attr:`p_list <optrace.tracer.ray_storage.RayStorage.p_list>`
     - :class:`numpy.ndarray` of type :attr:`numpy.float64` of shape (N, Nt, 3)
     - mm
     - 3D starting position for all ray sections 
   * - Direction vectors
     - :meth:`direction_vectors() <optrace.tracer.ray_storage.RayStorage.direction_vectors>`
     - :class:`numpy.ndarray` of type :attr:`numpy.float64` of shape (N, Nt, 3)
     - ``-``
     - normalized (with :python:`normalize=True`) or unnormalized direction vectors for each ray section
   * - Section lengths
     - :meth:`ray_lengths() <optrace.tracer.ray_storage.RayStorage.ray_lengths>`
     - :class:`numpy.ndarray` of type :attr:`numpy.float64` of shape (N, Nt)
     - mm
     - geometrical length of each ray section
   * - Optical section lengths
     - :meth:`optical_lengths() <optrace.tracer.ray_storage.RayStorage.optical_lengths>`
     - :class:`numpy.ndarray` of type :attr:`numpy.float64` of shape (N, Nt)
     - mm
     - optical length of each ray section (geometrical length multiplied by refractive index)
   * - Polarization
     - :attr:`pol_list <optrace.tracer.ray_storage.RayStorage.pol_list>`
     - :class:`numpy.ndarray` of type :attr:`numpy.float32` of shape (N, Nt, 3)
     - ``-``
     - unity 3D polarization vector
   * - Power
     - :attr:`w_list <optrace.tracer.ray_storage.RayStorage.w_list>`
     - :class:`numpy.ndarray` of type :attr:`numpy.float32` of shape (N, Nt)
     - W
     - ray power
   * - Refractive Index
     - :attr:`n_list <optrace.tracer.ray_storage.RayStorage.n_list>`
     - :class:`numpy.ndarray` of type :attr:`numpy.float64` of shape (N, Nt)
     - ``-``
     - section-wise refractive index of the material at the ray's wavelength
   * - Wavelength
     - :attr:`wl_list <optrace.tracer.ray_storage.RayStorage.wl_list>`
     - :class:`numpy.ndarray` of type :attr:`numpy.float32` of shape N
     - nm
     - wavelength of the ray
    

Direct Access
################


After raytracing, the :class:`RayStorage <optrace.tracer.ray_storage.RayStorage>` is accessible 
as :attr:`rays <optrace.tracer.raytracer.Raytracer.rays>` attribute
of the :class:`Raytracer <optrace.tracer.raytracer.Raytracer>`.
Value are accessed by typical numpy array indexing or slicing.
See the table above for the variable names and dimensions.
The number of rays and sections per ray is accessible through 
:attr:`Raytrace.rays.N <optrace.tracer.ray_storage.RayStorage.N>` 
and :attr:`Raytrace.rays.Nt <optrace.tracer.ray_storage.RayStorage.Nt>`.

Let's create an example geometry:

.. testcode::

    # create raytracer
    RT = ot.Raytracer(outline=[-15, 15, -15, 15, -15, 30])

    # add RaySource
    RSS = ot.CircularSurface(r=2)
    RS = ot.RaySource(RSS, pos=[0, 0, -10])
    RT.add(RS)

    # load LeGrand Eye model
    eye = ot.presets.geometry.legrand_eye()
    RT.add(eye)

    # trace
    RT.trace(100000)


To access positions of the third ray section of all rays write:

.. code-block:: python

   RT.rays.p_list[:, 2, :]

To access the wavelength of the tenth ray only:

.. code-block:: python

   RT.rays.wl_list[9]

Access the position z-component of all sections of the twenty-third to twenty-sixth ray by writing:

.. code-block:: python

   RT.rays.p_list[22:25, :, 2]

Access the ray section lengths for the fourth section of each ray:

.. code-block:: python

   RT.rays.ray_lengths()[:, 3]


Masking
################

A masking method of the :class:`RayStorage <optrace.tracer.ray_storage.RayStorage>` class can be applied 
for more control over accessing ray properties.
This method provides an easy interface for most relevant ray properties, while also improving performance, 
as only the desired components are calculated.
A call of :meth:`rays_by_mask <optrace.tracer.ray_storage.RayStorage.rays_by_mask>` without parameters returns 
a tuple of position, direction, polarization, weights, wavelengths, source number, refractive index 
of all rays and ray sections.

.. code-block:: python

   pos, dirs, pols, powers, wls, snum, n = RT.rays.rays_by_mask()


Providing a boolean array as first parameter applies masks to all these elements:

.. code-block:: python

   mask = np.array([0, 1, 0, 1, ...], dtype=bool)
   ... = RT.rays.rays_by_mask(mask)

Providing an additional array of integers also masks the ray sections:

.. code-block:: python

   mask = np.array([0, 1, 0, 1, ...], dtype=bool)
   sec = np.array([3, 0, 5, 1, 1, 2, ...])
   ... = RT.rays.rays_by_mask(mask, sec)

By default, ray direction vectors are normalized.
If this isn't required, you can provide :python:`normalize=False`:

.. code-block:: python

   mask = np.array([0, 1, 0, 1, ...], dtype=bool)
   sec = np.array([3, 0, 5, 1, 1, 2, ...])
   ... = RT.rays.rays_by_mask(mask, sec, normalize=False)

You can restrict the returned properties by setting the :python:`ret` parameter.
All undesired parameters are not calculated, improving the overall performance.
The function still returns a tuple of 7 elements, but undesired elements are set to :python:`None`.
The parameter is a seven element bool list, where all needed properties are marked with :python:`True`:

.. code-block:: python

   ret = [False, True, False, True, True, True, True]
   ... = RT.rays.rays_by_mask(ret=ret)

See the code reference of :func:`rays_by_mask <optrace.tracer.ray_storage.RayStorage.rays_by_mask>` for more details.


Object Descriptions
_____________________________

Child classes of :class:`BaseClass <optrace.tracer.base_class.BaseClass>` supply parameters :python:`desc, long_desc`. 
The former is used as short descriptive string, while the latter is a more verbose one.
These descriptions are used for printing geometry information and labelling elements in the 2D or 3D plots.

Modifying Initialized Objects
____________________________________________

Most objects are `locked` after initialization to avoid issues and hard-to-debug problems.
After locking, object properties can not be changed without specially exposed functions for exactly this purpose.
For instance, changing a lens surface leads to a change of the whole lens geometry, 
which in turn can lead to changes in the lens group or raytracer.
Such changes should only be applicable through specific functions that update everything in a correct and defined way.

Locked objects/properties include:

* all surface types as well as lines and points
* positions of geometrical objects (lens, detector, ...) (specific functions available)
* surface assignment of objects (specific functions available)
* properties of simulated rays
* a calculated ray transfer analysis object (TMA)


.. _usage_color:

Color Conversions
_______________________________


Color conversion are supported via the namespace :mod:`optrace.tracer.color`.
optrace provides conversions for the colorspaces XYZ, sRGB, linear SRGB, CIELUV and xyY 
as well as specific color properties such as saturation and hue in CIELUV.

Check the :ref:`Color Management <color_management>` section for a technical descriptions on the color processing.
API reference section :mod:`optrace.tracer.color` provides information on the usage of the implemented functions.

The sRGB Perceptual Rendering Intent function allows for extra parameters compared to the other rendering intents.
For instance, a fixed chroma scaling factor is set by the :python:`chroma_scale` parameter 
of the :func:`optrace.tracer.color.xyz_to_srgb_linear <optrace.tracer.color.srgb.xyz_to_srgb_linear>` function. 
Such a fixed factor is useful for image comparisons.
A suitable scaling factor can be calculated using :func:`optrace.tracer.color.get_chroma_scale <optrace.tracer.color.srgb.get_chroma_scale>`.
A relative lightness threshold can be set using the :python:`L_th` parameter, 
which excludes darker, mostly invisible colors in the chroma scale calculation.
After chroma scaling, colors remaining outside the gamut are projected onto the gamut edge, 
same as for the absolute rendering intent.
See the docstring of both functions for further information.

