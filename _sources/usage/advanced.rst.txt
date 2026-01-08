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

The raytracer emits warnings and errors while simulating.
Internally these are characterized by flags, which can be found below:

.. list-table:: List of raytracing messages
   :widths: 100 600
   :header-rows: 1
   :align: left
   :width: 900px

   * - Flag
     - Description

   * - :obj:`Raytracer.INFOS.ABSORB_MISSING <optrace.tracer.raytracer.Raytracer.INFOS.ABSORB_MISSING>`
     - Rays are absorbed because they miss a lens surface.

   * - :obj:`Raytracer.INFOS.TIR <optrace.tracer.raytracer.Raytracer.INFOS.TIR>`
     - Total inner reflection. Reflections are not simulated, so the ray is treated as being absorbed at the surface intersection

   * - :obj:`Raytracer.INFOS.ILL_COND <optrace.tracer.raytracer.Raytracer.INFOS.ILL_COND>`
     -  Ill-conditioned rays for hit finding of a numerical, custom surface. In almost all cases the intersection will be wrong. This can happen if the surface is badly defined numerically or geometrically, there are surface collisions or a ray hits the surface multiple times. Please check the geometry of the raytracer and the surface definition.

   * - :obj:`Raytracer.INFOS.OUTLINE_INTERSECTION <optrace.tracer.raytracer.Raytracer.INFOS.OUTLINE_INTERSECTION>`
     - Rays that would leave the outline are absorbed at the outline intersection

The raytracer also provides a :attr:`Raytracer.geometry_error <optrace.tracer.raytracer.Raytracer.geometry_error>` flag that gets set when tracing was aborted due to issues with the geometry.
Geometry checks are only executed while tracing, so :meth:`Raytracer.trace <optrace.tracer.raytracer.Raytracer.trace>` must be called first.

When using the GUI, a subset of surface collision points will also be displayed in the geometry.


.. _usage_ray_access:

Accessing Ray Properties
_____________________________


Overview
################

**Definitions:**

| **Ray Section**: Part of the ray with a constant direction vector. Sections typically start and end at surface intersections.
| **Ray**: Entirety of the ray going from the source to its absorption
| **N**: number of rays
| **Nt**: number of sections per ray, equal for all rays

The number of sections is the same for all rays. 
If a ray gets absorbed early, all consecutive sections consist of zero length vectors starting at the last position and having their power set to zero. 
Direction and polarization are then undefined.

The following table shows an overview of ray properties.
Some of those are attributes that are stored while tracing, while others are functions, as these properties must be calculated from other attributes first.
They are intentionally kept as functions and are not exposed as properties, so an computational overhead is communicated to the user.

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
     - ``-``
     - geometrical length of each ray section
   * - Optical section lengths
     - :meth:`optical_lengths() <optrace.tracer.ray_storage.RayStorage.optical_lengths>`
     - :class:`numpy.ndarray` of type :attr:`numpy.float64` of shape (N, Nt)
     - ``-``
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
     - refractive indices for all ray sections
   * - Wavelength
     - :attr:`wl_list <optrace.tracer.ray_storage.RayStorage.wl_list>`
     - :class:`numpy.ndarray` of type :attr:`numpy.float32` of shape N
     - nm
     - wavelength of the ray
    

Direct Access
################


After raytracing, the ray storage is accessible as attribute of the Raytracer.
Value are accessed by typical numpy array indexing or slicing.
See the table above for the variable names and dimensions.
Number of rays and sections per ray is accessible through :python:`Raytracer.rays.N` and :python:`Raytracer.rays.nt`.

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


To access positions of the third ray section write:

.. code-block:: python

   RT.rays.p_list[:, 2, :]

To access the wavelength of the tenth ray:

.. code-block:: python

   RT.rays.wl_list[9]

Access the position z-component of all sections of the twenty-third to twenty-sixth ray by writing:

.. code-block:: python

   RT.rays.p_list[22:25, :, 2]

Access the ray section lengths for the fourth section:

.. code-block:: python

   RT.rays.ray_lengths()[:, 3]


Masking
################

For more control over accessing ray properties masking methods of the RayStorage class can be applied.
A call of :meth:`rays_by_mask <optrace.tracer.ray_storage.RayStorage.rays_by_mask>` without parameters is:

.. code-block:: python

   RT.rays.rays_by_mask()

... and returns a tuple of position, direction, polarization, weights, wavelengths, source number, refractive index.  

Providing a boolean array as first parameter applies masks to all these elements:

.. code-block:: python

   mask = np.array([0, 1, 0, 1, ...], dtype=bool)
   RT.rays.rays_by_mask(mask)

Providing an additional array of integers also selects the ray sections:

.. code-block:: python

   mask = np.array([0, 1, 0, 1, ...], dtype=bool)
   sec = np.array([3, 0, 5, 1, 1, 2, ...])
   RT.rays.rays_by_mask(mask, sec)

By default, ray direction vectors are normalized, if this isn't needed, you can provide :python:`normalize=False`:

.. code-block:: python

   mask = np.array([0, 1, 0, 1, ...], dtype=bool)
   sec = np.array([3, 0, 5, 1, 1, 2, ...])
   RT.rays.rays_by_mask(mask, sec, normalize=False)

You can restrict the properties by setting the :python:`ret` parameter.
All unneeded parameters are not calculated and set to :python:`None`, speeding things up.
This parameter is a seven element bool list that marks all needed properties:

.. code-block:: python

   ret = [False, True, False, True, True, True, True]
   RT.rays.rays_by_mask(ret=ret)

The function still returns a tuple of 7 elements, but undesired elements have value :python:`None` instead of an array.
See the code reference of :func:`rays_by_mask <optrace.tracer.ray_storage.RayStorage.rays_by_mask>` for more detail.


Object Descriptions
_____________________________

Child classes of :class:`BaseClass <optrace.tracer.base_class.BaseClass>` include parameters :python:`desc, long_desc`. 
The former should be a short descriptive string and the latter a more verbose one.
These descriptions can be user provided and are used in for plotting and text output.

Modifying Initialized Objects
____________________________________________

To avoid issues and hard-to-debug problems, some objects are `locked` after initialization.
This means object properties can not be changed or assigned.
Some objects include specific methods to change their properties after initialization.
For instance, changing a lens surface leads to a change of the lens, which in turn can lead to changes in the lens group or raytracer.
Such changes should only be applicable through specific functions that update everything accordingly in a defined way.

Locked objects/properties include:

* all surface types as well as lines and points
* positions of geometrical objects (lens, detector, ...) (but these are assignable through a function)
* surface assignment (but accessible through specific functions)
* properties of rendered rays
* a calculated ray transfer analysis object (TMA)

The list of traced rays is read-only, since their properties should only be assigned by the simulation itself.

.. _usage_color:

Color Conversions
_______________________________


Color conversion are supported via the namespace :python:`optrace.color`.
optrace provides conversions for the colorspaces XYZ, sRGB, linear SRGB, CIELUV and xyY as well as some color properties like Saturation and Hue in CIELUV.

Check the :ref:`Color Handling <color_management>` section for a technical and fundamental descriptions of color processing and calculation.
Go to the code reference section :mod:`optrace.tracer.color` for information on the usage of implemented functions.

For the sRGB Perceptual Rendering Intent there a extra parameters available.
For instance, a fixed saturation scaling can be set using the :python:`chroma_scale` parameter of the :func:`optrace.tracer.color.xyz_to_srgb_linear <optrace.tracer.color.srgb.xyz_to_srgb_linear>` function.
A suitable scaling factor can be calculated using :func:`optrace.tracer.color.get_saturation_scale <optrace.tracer.color.srgb.get_chroma_scale>`.
This is useful for viable comparison between images, as the saturation scaling factor is the same.
The function :func:`optrace.tracer.color.xyz_to_srgb_linear <optrace.tracer.color.srgb.xyz_to_srgb_linear>` provides the :python:`chroma_scale` parameter to override the best matching one.
Alternatively, a relative lightness threshold can be set using the :python:`L_th` parameter, which excludes colors of darker image regions to calculate/apply the factor in both functions.
This is helpful when the scaling factor is largely affected by color values that are mostly invisible.
If there still colors outside the gamut after the operation (for instance, because they were below :python:`L_th` or the user set :python:`chroma_scale` value was insufficient), they are projected onto the gamut edge as for the absolute rendering intent.
See the docstring of both functions for further information.

