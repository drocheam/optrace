Overview
------------------------------------------------------------------------

.. testcode:: 
   :hide:

   print("Placeholder:    ")
   import optrace as ot
   from optrace.gui import TraceGUI

.. testoutput::
   :hide:
    
   Placeholder: ...

.. role:: python(code)
  :language: python
  :class: highlight

Structure
___________________


.. list-table:: Overview of `optrace` functionality
   :widths: 100 250 100 125
   :header-rows: 1
   :align: left

   * - Topic
     - Descriptions
     - Classes/Functions
     - Link
   * - Paraxial Analysis
     - Finding the principial planes, focal points, object and image distance, pupils of a setup
     - :class:`optrace.TMA <optrace.tracer.transfer_matrix_analysis.TMA>`
     - :ref:`usage_tma`

   * - Paraxial Imaging
     - Image simulation with the help of a rendered point spread function
     - :func:`optrace.convolve <optrace.tracer.convolve.convolve>`
     - :ref:`usage_convolution`

   * - General Geometrical Optics and Image Simulation
     - Sequential Raytracing and of optical setups. Analysis of ray paths, simulation of detector images and spectra, focus finding.
     - :class:`optrace.Raytracer <optrace.tracer.raytracer.Raytracer>`, :class:`optrace.RImage <optrace.tracer.r_image.RImage>`
     - :ref:`Raytracer <usage_raytracer>`, :ref:`usage_image`, :ref:`usage_focus`, :ref:`usage_ray_access`

   * - Image color conversion and displaying
     - Convert image colors and display rendered or loaded images
     - :mod:`optrace.color <optrace.tracer.color>` and :mod:`optrace.plots <optrace.plots>`
     - :ref:`usage_color`, :ref:`usage_plots` 
   
   * - Graphical Setup Visualization
     - Graphical display of the tracing scene and traced rays as well as some control features for the simulation
     - :class:`optrace.TraceGUI <optrace.gui.trace_gui.TraceGUI>`
     - :ref:`usage_gui`

Namespaces
______________________


The library itself is the primary namespace.
While there is a separete namespace :mod:`optrace.tracer`, all objects are also included in the main one.

.. testcode::

   import optrace as ot

Now objects can be accessed by :python:`ot.Raytracer, ot.CircularSurface, ot.RaySource, ...`.

`optrace` provides plotting functionality for images, spectra, media etc.
These plotting functions are included in the :mod:`optrace.plots` namespace.

.. testcode:: 

   import optrace.plots as otp

The GUI is included in the namespace :mod:`optrace.gui`.
Since the :class:`optrace.gui.TraceGUI <optrace.gui.trace_gui.TraceGUI>` is the only one relevant there, it can be directly imported in the main namespace:

.. testcode::

   from optrace.gui import TraceGUI

