Overview
------------------------------------------------------------------------

.. |TraceGUI| replace:: :class:`TraceGUI <optrace.gui.trace_gui.TraceGUI>`

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


.. list-table:: Overview of optrace functionality
   :widths: 100 250 100 125
   :header-rows: 1
   :align: left

   * - Topic
     - Descriptions
     - Classes/Functions
     - Links

   * - **Paraxial Analysis**
     - Finding principial planes, focal points, object and image distances, exit and entrance pupil
     - :class:`optrace.TMA <optrace.tracer.transfer_matrix_analysis.TMA>`
     - :ref:`usage_tma`

   * - **Paraxial Imaging**
     - Image simulation by convolution with a rendered point spread function
     - :func:`optrace.convolve <optrace.tracer.convolve.convolve>`
     - :ref:`usage_convolution`

   * - **General Geometrical Optics and Image/Spectrum Simulation**
     - Sequential Raytracing of optical setups. Analysis of ray paths, 
       simulation of detector images and spectra, focus finding.
     - :class:`optrace.Raytracer <optrace.tracer.raytracer.Raytracer>`, 
       :class:`optrace.RenderImage <optrace.tracer.image.render_image.RenderImage>`, 
       :class:`optrace.LightSpectrum <optrace.tracer.spectrum.light_spectrum.LightSpectrum>`
     - :ref:`Raytracer <usage_raytracer>`, :ref:`usage_image`, 
       :ref:`usage_spectrum`, :ref:`usage_focus`, :ref:`usage_ray_access`

   * - **Image, Surface, Spectrum and Refractive Index Plotting**
     - Display images, spectra, surfaces and refractive indices graphically
     - :mod:`optrace.plots <optrace.plots>`
     - :ref:`usage_plots` 
   
   * - **Image color conversion**
     - Convert or access image colors
     - :mod:`optrace.color <optrace.tracer.color>` and 
       :class:`optrace.RenderImage <optrace.tracer.image.render_image.RenderImage>`
     - :ref:`usage_color`, :ref:`usage_image` 
   
   * - **Graphical Setup and Visualization**
     - Graphical display of the tracing scene and traced rays as well as some control features for the simulation
     - :class:`optrace.TraceGUI <optrace.gui.trace_gui.TraceGUI>`
     - :ref:`usage_gui`, :ref:`gui_automation`

Namespaces
______________________

:python:`optrace` s the primary namespace.
While there is a separate sub-namespace for the tracer, called :mod:`optrace.tracer`, 
it is automatically included in the main namespace.

.. testcode::

   import optrace as ot

Classes can be now accessed as :python:`ot.Raytracer, ot.CircularSurface, ot.RaySource, ...`.

optrace provides plotting functionality for images, spectra, media etc.
These plotting functions are included in the :mod:`optrace.plots` namespace.

.. testcode:: 

   import optrace.plots as otp

The GUI is included in the namespace :mod:`optrace.gui`.
Only the |TraceGUI| class is relevant there, so it can be directly imported in the main namespace:

.. testcode::

   from optrace.gui import TraceGUI


Global Options
______________________

Global options are controlled through the attributes of the 
class :class:`optrace.global_options <optrace.global_options>`.

Progressbar
###################

For calculation-intensive tasks a progress bar is displayed inside the terminal 
that displays the progress and estimated remaining time.
It can be turned off globally by:

.. testcode::

   ot.global_options.show_progressbar = False

There is also a context manager available for turning it off temporarily:

.. code-block:: python

   with ot.global_options.no_progressbar():
       do_something()

Warnings
###################

optrace outputs warnings of type :exc:`OptraceWarning <optrace.warnings.OptraceWarning>` 
(which are a custom subclass of :exc:`UserWarning`). These can be filtered using the :mod:`warnings` python module.
A simple way to silence them, for example when doing many automated tasks, is by writing:

.. testcode::

   ot.global_options.show_warnings = False

There is also a context manager available for turning it off temporarily:

.. code-block:: python

   with ot.global_options.no_warnings():
       do_something()

Multithreading
###################

By default, multithreading parallelizes tasks like raytracing and image rendering.
However, this is undesired in some cases, especially when debugging or running multiple raytracers in parallel.
Multithreading can be turned off with:

.. testcode::

   ot.global_options.multi_threading = False


Wavelength Range
###################

optrace is optimized for operation for the visible wavelength range of 380 - 780 nm.
The range can be extended by:

.. testcode::

   ot.global_options.wavelength_range = [300, 800]

Note that most presets like refractive indices are not defined for regions outside the default range.
This can lead to issues when using these presets.

Spectral Colormap
######################

Spectral plots (spectrum, refractive index, ray coloring) use a spectral colormap 
that maps wavelength values to their corresponding colors.
For the visible range, this leads to a rainbow-like mapping.

When working in the infrared or ultraviolet region, they would be mapped to a similar hue and a nearly black color.
To make different values discernible, a custom mapping function should be supplied instead.
One example could be:

.. testcode::

   import matplotlib.pyplot as plt
   
   ot.global_options.spectral_colormap = lambda wl: plt.cm.viridis((wl-300)/800)

In this example the colormap is adapted to use the 
`viridis colormap from pyplot <https://matplotlib.org/stable/users/explain/colors/colormaps.html#sequential>`_,
where 300 is mapped to the lowest value of 0 and 800 to the highest value of 1.
The specified function should take a wavelength numpy array (of some length N) as argument 
and return a two dimensional array with RGBA values between 0-1 and shape (N, 4).

The colormap can be reset by setting it to :python:`None`.

.. testcode::
   :hide:

   ot.global_options.spectral_colormap = None
   ot.global_options.wavelength_range = [380, 780]


UI Dark Mode
###################

The UI dark mode is enabled by default.
It can be changed by setting the :python:`ui_dark_mode` parameter.
Changes are applied to all current GUI windows as well as new ones.

To deactivate the mode, use:

.. testcode::

   ot.global_options.ui_dark_mode = False
   
.. figure:: ../images/ui_dark_mode.png
   :align: center
   :width: 800
   :class: dark-light

   With :python:`ui_dark_mode` enabled.

.. figure:: ../images/ui_light_mode.png
   :align: center
   :width: 800
   :class: dark-light

   With :python:`ui_dark_mode` disabled.


Plot Dark Mode
###################

For the content of plotting windows, there is a separate option :python:`plot_dark_mode`.
It is also enabled by default.

To deactivate it, use:

.. testcode::

   ot.global_options.plot_dark_mode = False

Deactivating it is useful for documentation or article output, where image are typically shown on a white background.
Note that changes are only applied to new :obj:`pyplot <matplotlib.pyplot>` windows, not already opened ones.

.. list-table::
   :class: table-borderless

   * - .. figure:: ../images/srgb_spectrum.svg
          :align: center
          :width: 400
          :class: dark-light

          With :python:`plot_dark_mode` enabled.
   
     - .. figure:: ../images/srgb_spectrum_light.svg
          :align: center
          :width: 400
          :class: dark-light

          With :python:`plot_dark_mode` disabled.



.. TODO notes how to run it on wayland

.. TODO notes about specifying number of cores (taskset, PYTHON_CPU_COUNT)
