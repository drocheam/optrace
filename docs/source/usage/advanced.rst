Advanced Topics
------------------------------------------------

.. role:: python(code)
  :language: python
  :class: highlight

.. testsetup:: *

   import optrace as ot


Raytracing Errors
_________________________

Doing raytracing there can appear different problems, being for instance caused by the optical or geometrical aspects of the setup.

The raytracer emits warnings and errors while simulating, including affected surface/object and number/ratio of rays.
Internally these are characterized by flags, which can be found below:

.. list-table:: List of raytracing messages
   :widths: 100 600
   :header-rows: 0
   :align: left

   * - **Flag**
     - **Description**
    
   * - :python:`Raytracer.INFOS.ABSORB_MISSING`
     - Rays are absorbed because they miss a lens surface. Occurs in combination with parameter :python:`absorb_missing=True` 

   * - :python:`Raytracer.INFOS.TIR`
     - Total inner reflection. Reflections are not simulated, the ray is treated as being absorbed at the surface intersection

   * - :python:`Raytracer.INFOS.OUTLINE_INTERSECTION`
     - A ray intersects the outline of the raytracer geometry and is therefore absorbed by it

   * - :python:`Raytracer.INFOS.ONLY_HIT_FRONT`
     - A ray hits the front, but misses the back. Must gone through the lens side cylinder, which isn't simulated. Absorb it, since we don't know how to handle it

   * - :python:`Raytracer.INFOS.ONLY_HIT_BACK`
     - A ray hits the back, but misses the front. Must gone through the lens side cylinder, which isn't simulated. Absorb it, since we don't know how to handle it
   
   * - :python:`Raytracer.INFOS.T_BELOW_TTH`
     - A filter transmittivity is below the T_TH threshold of the raytracer and the ray is therefore absorbed. Avoids 'ghost rays' that need to be traced, but don't contribute to anything

   * - :python:`Raytracer.INFOS.ABSORB_MEDIA_TRANS`
     - a ray misses a lens but crosses a medium boundary. Since the medium boundary geometry is not defined (as the medium 1 ends before the lens and medium 2 starts after it) outside the lens, we absorb those rays

   * - :python:`Raytracer.INFOS.ILL_COND`
     -  ill-conditioned rays for hit finding of a numerical, custom surface. In almost all cases the intersection will be wrong. This can happen if the surface is badly defined numerically or geometrically, there are surface collisions or a ray hits the surface multiple times. Please check the geometry of the raytracer and the surface definition.

The raytracer also provides a :attr:`Raytracer.geometry_error <optrace.tracer.raytracer.Raytracer.geometry_error>` flag that gets set when tracing was aborted due to issues with the geometry.
Geometry checks are only executed while tracing, so :meth:`Raytracer.trace <optrace.tracer.raytracer.Raytracer.trace>` must be called to check for those.

When using the GUI, surface collisions will be displayed in the geometry.


.. _usage_ray_access:

Accessing Ray Properties
_____________________________


Overview
################

**Terms:**

| **Ray Section**: Part of the ray, where the direction is constant. Sections typically start and end at surface intersections.
| **Ray**: sum of all its ray sections, entirety of the ray going from the source to the point it is absorbed


**Shapes**:

| **N**: number of rays
| **nt**: number of sections per ray, equal for all rays


The number of sections is the same for all rays. If a ray gets absorbed early, all consecutive sections consist of zero length vectors starting at the last position and having their power set to zero. Direction and polarization are undefined.


.. list-table:: List of ray properties
   :widths: 100 200 50 400
   :header-rows: 0
   :align: left

   * - Name
     - Type
     - Unit
     - Function
   * - :attr:`p_list <optrace.tracer.ray_storage.RayStorage.p_list>`
     - :class:`numpy.ndarray` of type :attr:`numpy.float64` of shape N x nt x 3
     - mm
     - 3D starting position for all ray sections 
   * - :attr:`s0_list <optrace.tracer.ray_storage.RayStorage.s0_list>`
     - :class:`numpy.ndarray` of type :attr:`numpy.float64` of shape N x 3
     - ``-``
     - unity direction vector at the ray source
   * - :attr:`pol_list <optrace.tracer.ray_storage.RayStorage.pol_list>`
     - :class:`numpy.ndarray` of type :attr:`numpy.float32` of shape N x nt x 3
     - ``-``
     - unity 3D polarization vector
   * - :attr:`w_list <optrace.tracer.ray_storage.RayStorage.w_list>`
     - :class:`numpy.ndarray` of type :attr:`numpy.float32` of shape N x nt
     - W
     - ray power
   * - :attr:`n_list <optrace.tracer.ray_storage.RayStorage.n_list>`
     - :class:`numpy.ndarray` of type :attr:`numpy.float64` of shape N x nt
     - ``-``
     - refractive indices for all ray sections
   * - :attr:`wl_list <optrace.tracer.ray_storage.RayStorage.wl_list>`
     - :class:`numpy.ndarray` of type :attr:`numpy.float32` of shape N
     - nm
     - wavelength of the ray
    

Direct Access
################


After tracing the ray storage is accessible as member of the Raytracer.
Value are accessed by typical numpy array indexing or slicing.
See the table above for the variable names and dimensions.
Number of rays and sections per ray is accessible through :python:`Raytracer.rays.N` and :python:`Raytracer.rays.nt`.

Tracing some geometry:

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
    RT.trace(2000)


Access positions of third ray section

.. code-block:: python

   RT.rays.p_list[:, 2, :]

Access wavelength of the tenth ray

.. code-block:: python

   RT.rays.wl_list[9]

Access position z-component of all sections of the twenty-third to twenty-sixth ray

.. code-block:: python

   RT.rays.p_list[22:25, :, 2]


Ray lengths
#################

Ray section lengths, optical lengths as well as direction vectors are not directly accessible, but need to be calculated on demand.

For the former two there are methods available, for the latter one the :meth:`rays_by_mask <optrace.tracer.ray_storage.RayStorage.rays_by_mask>` function from the next section can be used

Functions :meth:`ray_lengths <optrace.tracer.ray_storage.RayStorage.ray_lengths>` and :meth:`optical_lengths <optrace.tracer.ray_storage.RayStorage.optical_lengths>` are available.
Both take an optional boolean mask as argument, that masks the rays to the desired selection:

.. code-block:: python

   mask = np.array([0, 1, 0, 1, ...], dtype=bool)
   RT.rays.optical_lengths(mask)

By default, no rays are masked and all lengths are returned.

Note that these functions return the section lengths, not the overall length.
But the latter can be simply calculated by summing all ray section lengths together for each ray.

Masking
################

For more control over masking and accessing ray properties this can be done with masking methods of the RayStorage class.

A call of :meth:`rays_by_mask <optrace.tracer.ray_storage.RayStorage.rays_by_mask>` without parameters:

.. code-block:: python

   RT.rays.rays_by_mask()

... returns a tuple of position, direction, polarization, weights, wavelengths, source number, refractive index.  

Providing a boolean array as first parameter applies masks to all these elements:

.. code-block:: python

   mask = np.array([0, 1, 0, 1, ...], dtype=bool)
   RT.rays.rays_by_mask(mask)

Providing an additional array of integers also selects the ray sections

.. code-block:: python

   mask = np.array([0, 1, 0, 1, ...], dtype=bool)
   sec = np.array([3, 0, 5, 1, 1, 2, ...])
   RT.rays.rays_by_mask(mask, sec)

By default, ray direction vectors are normalized, if this isn't needed, one can provide :python:`normalize=False`:

.. code-block:: python

   mask = np.array([0, 1, 0, 1, ...], dtype=bool)
   sec = np.array([3, 0, 5, 1, 1, 2, ...])
   RT.rays.rays_by_mask(mask, sec, normalize=False)


Not all properties are always needed.
Undesired ones only lead to decreased performance.
By providing a seven element bool list only the relevant can be selected:

.. code-block:: python

   ret = [False, True, False, True, True, True, True]
   RT.rays.rays_by_mask(ret=ret)

The function still returns a tuple of 7 elements, but undesired elements have value :python:`None` instead of an array.


See the code reference of :func:`rays_by_mask <optrace.tracer.ray_storage.RayStorage.rays_by_mask>` for more detail.

Controlling Threading
_______________________________

All classes in :mod:`optrace.tracer <optrace.tracer>` are derived from the class :class:`BaseClass <optrace.tracer.base_class.BaseClass>`.
Derived object include a boolean :python:`threading` property that is turned on by default.
Turning it off with :python:`threading=False` disabled multithreading and thread creation.
This can be useful when profiling and debugging or if multiple objects are run in parallel anyway.

When creating a raytracer with this option

.. code-block:: python

   RT = ot.Raytracer(..., threading=False)

All methods (tracing, rendering, focussing, ...) of this class as well as all created child objects (image, spectrum, ...) are handled in the main thread and have :python:`threading=False` assigned.


Note that all plotting functions from :mod:`optrace.plots` are run only in the main thread. Furthermore, threading for the module :mod:`optrace.gui` can't be turned off.


Silencing Standard Output
____________________________________________

As mentioned above, all classes in :mod:`optrace.tracer <optrace.tracer>` are derived from the class :class:`BaseClass <optrace.tracer.base_class.BaseClass>`.
This class includes a :python:`silent` parameter that with :python:`silent=True` does not emit any messages to the standard output (terminal), also including the output from the progressbar.
Note that this also goes for warnings and important info messages.
Nevertheless, this option can be useful in automation or minimizing the amount of messages.

The :class:`TraceGUI <optrace.gui.trace_gui.TraceGUI>` and the plotting functions in :mod:`optrace.plots` also support this option.

When providing the parameter in the raytracer class all created child objects share the same property. When :python:`TraceGUI.silent=True` is set, the raytracer is also silenced.



Object Descriptions
_____________________________

Child classes of :class:`BaseClass <optrace.tracer.base_class.BaseClass>` include parameters :python:`desc, long_desc`. The former should be a short descriptive string and the latter a more verbose one.

These descriptions can be user provided and are used in for the plotting in plots or the GUI and for some standard output messages.

Modifying Initialized Objects
____________________________________________

To avoid issues and hard-to-debug problems, some objects are `locked` after initializiation.
This means object properties can not be changed or assigned, or rather only through specific methods.

For instance, changing properties of a surface, like the curvature, would change its extent and the parent object, like a lens, that must also update its properties.
Often it is unclear, what should be adapted in which way. Should the surface be moved? Should the thickness of the lens stay the same or be adapted with the same thickness?
Should the lens center position stay the same?
The procedure is instead to create a new lens including the new surface and to remove the old one.
This is clearly a design decisions to avoid problems and side effects.

The list of traced rays is also read-only, as there is no reason why it should be changeable by the user, as the properties are assigned by the simulation.

Locked objects/properties include:

* all surface types as well as lines and points
* positions of geometrical objects (lens, detector, ...) (but these are assignable through a function)
* surface assignment (but accessible through specific functions)
* properties of rendered rays
* a calculated ray transfer analysis object (TMA)


.. _usage_color:

Color Conversions
_______________________________


Color conversion are supported via the namespace :python:`optrace.color`.
`optrace` provides conversions for the colorspaces XYZ, sRGB, linear SRGB, CIELUV and xyY as well as some color properties like Saturation and Hue in CIELUV.

Check the :ref:`Color Handling <color_management>` section for a technical and fundamental descriptions of color processing and calculation.
Go to the code reference section :mod:`optrace.tracer.color` for information on the usage of implemented functions.


