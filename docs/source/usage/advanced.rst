Advanced Topics
------------------------------------------------

.. role:: python(code)
  :language: python
  :class: highlight

Raytracing Errors
_________________________


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
   * - ``p_list``
     - ``np.ndarray`` of type ``np.float64`` of shape N x nt x 3
     - mm
     - 3D starting position for all ray sections 
   * - ``s0_list``
     - ``np.ndarray`` of type ``np.float64`` of shape N x 3
     - ``-``
     - unity direction vector at the ray source
   * - ``pol_list``
     - ``np.ndarray`` of type ``np.float32`` of shape N x nt x 3
     - ``-``
     - unity 3D polarization vector
   * - ``w_list``
     - ``np.ndarray`` of type ``np.float32`` of shape N x nt
     - W
     - ray power
   * - ``n_list``
     - ``np.ndarray`` of type ``np.float64`` of shape N x nt
     - ``-``
     - refractive indices for all ray sections
   * - ``wl_list``
     - ``np.ndarray`` of type ``np.float32`` of shape N
     - nm
     - wavelength of the ray
    

Direct Access
################


Masking
################


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


Color Conversions
_______________________________



