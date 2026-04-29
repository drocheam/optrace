.. _gui_automation:

GUI Automation
---------------

.. testcode:: 
   :hide:

   import optrace as ot
   ot.global_options.show_progressbar = False
   
   from optrace.gui import TraceGUI

.. role:: python(code)
  :language: python
  :class: highlight

Some feasible automation tasks include:

* changing the geometry
* changing camera views accurately
* displaying specific rays
* picking specific rays
* debugging
* taking automated screenshots
* calling functions with settings not available through the GUI
* ...


The Control Method
________________________

To do automated tasks, use the :meth:`TraceGUI.control <optrace.gui.trace_gui.TraceGUI.control>` method 
instead of :meth:`TraceGUI.run <optrace.gui.trace_gui.TraceGUI.run>`.
It requires an automation function as well as an argument tuple.

.. code-block:: python

   def automated(GUI):

       # do some automation tasks
       ...

   # create the GUI and provide the automation function to TraceGUI.control()
   sim = TraceGUI(RT)
   sim.control(func=automated, args=(sim,))

After the method is executed, the GUI keeps running normally.
This makes it possible to interact with the GUI.
The process of closing the application automatically is described in :numref:`gui_automation_close`

All actions inside the provided automation function run sequentially to avoid race conditions.
During this time, the GUI is mostly unresponsive, 
as the automation function is run in the main thread (as user input would also be).


Applying Properties
________________________


Available TraceGUI properties are discussed in :numref:`gui_tabs`.
All of these can be set programmatically.

While the variables are set, 
the TraceGUI does not execute the functions or actions that react to these changes instantly.
This is in contrast to the standard :meth:`TraceGUI.run <optrace.gui.trace_gui.TraceGUI.run>` method, 
where a setting is applied subsequently 
(for instance, changing the ray color setting updates the color in the scene automatically).

To process all pending events :meth:`TraceGUI.process <optrace.gui.trace_gui.TraceGUI.process>` must be called.
This is not needed after every property change, 
but only if the changes should be visible/executed at this point in time.

.. code-block:: python

   def automated(GUI):

       # change properties
       GUI.minimalistic_view = True
       GUI.hide_labels = True
       GUI.ray_count = 1000

       # GUI properties were set, but the changes need to be processed
       GUI.process()

       # do some other things
       ...

   # create the GUI and provide the automation function to TraceGUI.control()
   sim = TraceGUI(RT)
   sim.control(func=automated, args=(sim,))

Note that some functions, like :meth:`TraceGUI.replot <optrace.gui.trace_gui.TraceGUI.replot>`, 
also call :meth:`TraceGUI.process <optrace.gui.trace_gui.TraceGUI.process>` internally.


Replotting
________________________

While :meth:`TraceGUI.process <optrace.gui.trace_gui.TraceGUI.process>` reacts to changes in the TraceGUI itself, 
it does not handle changes of the raytracer or tracing geometry.

When changing the geometry, the changes are not automatically applied to the scene.
The geometry is also not automatically raytraced.

To force the redrawing and retracing of the full scene, 
you can call :meth:`TraceGUI.replot <optrace.gui.trace_gui.TraceGUI.replot>`.

It is possible to only update changed objects 
with the context manager :meth:`TraceGUI.smart_replot <optrace.gui.trace_gui.TraceGUI.smart_replot>`.
For instance, if a detector is moved, 
there is no need for updating the lenses inside the geometry or retracing the scene.
:meth:`TraceGUI.smart_replot <optrace.gui.trace_gui.TraceGUI.smart_replot>` 
handles the detection of changes and updating automatically.

Here is an example:

.. code-block:: python

   def automated(GUI):

       # replot everything
       GUI.replot()

       # do some actions and at the end replot only changed objects
       # and/or retrace the geometry if needed.
       with GUI.smart_replot():
           some_action_1()

   # create the GUI and provide the automation function to TraceGUI.control()
   sim = TraceGUI(RT)
   sim.control(func=automated, args=(sim,))


.. _gui_camera:

Controlling the Camera
________________________

You can control the camera with the functions :meth:`TraceGUI.set_camera <optrace.gui.trace_gui.TraceGUI.set_camera>` 
and :meth:`TraceGUI.get_camera <optrace.gui.trace_gui.TraceGUI.get_camera>`.
The former sets the properties, while the latter one returns a dictionary of the current settings.

The following settings are available:

.. list-table::
   :header-rows: 1
   :align: left
   :widths: 75 200
   
   * - Property
     - Description
   * - :python:`center`
     - 3D coordinates of center of view in mm
   * - :python:`height`
     - half of vertical visible scene height in mm
   * - :python:`direction`
     - camera view direction vector (direction of vector perpendicular to your monitor and in your viewing direction)
   * - :python:`roll`
     - absolute camera roll angle in degrees 

You can find example code below:

.. code-block:: python

   def automated(sim):

       # store initial camera properties
       cam_props = sim.get_camera()

       # change the center of the view as well as the scaling
       sim.set_camera(center=[1, -0.5, 2], height=2.5)

       # reset to initial view
       sim.set_camera(**cam_props)

   # create the GUI and provide the automation function to TraceGUI.control()
   sim = TraceGUI(RT)
   sim.control(func=automated, args=(sim,))

Applying camera properties at startup is possible using the :python:`initial_camera` parameter of the TraceGUI class.
This parameter is a dictionary that can include all possible parameters of function :meth:`TraceGUI.set_camera <optrace.gui.trace_gui.TraceGUI.set_camera>`.

.. code-block:: python

   sim = TraceGUI(RT, initial_camera=dict(direction=[0, 1, 0], roll=45))


Taking Screenshots
________________________

The :meth:`TraceGUI.screenshot <optrace.gui.trace_gui.TraceGUI.screenshot>` function captures screenshots of the scene.
A path string is required for this function.
The file type is determined automatically from the file name.

Internally, the :obj:`pyvista.Plotter.screenshot` function from `pyvista <https://docs.pyvista.org>`__ is utilized, 
you can therefore use this function's additional parameters.

.. code-block:: python

   def automated(sim):

       # default call
       sim.screenshot("image.png")

       # call with additional parameters
       sim.screenshot("image2.png", magnification=2)

   # create the GUI and provide the automation function to TraceGUI.control()
   sim = TraceGUI(RT)
   sim.control(func=automated, args=(sim,))

Note that the :python:`magnification` parameter leads to a rescaled scene, 
where many annotation elements change their relative size.

.. _usage_gui_selecting_rays:

Selecting Rays
_________________________

By default, a random selection of rays is displayed inside the scene,
where the number is specified by :attr:`TraceGUI.rays_visible <optrace.gui.trace_gui.TraceGUI.rays_visible>`.
A custom selection is set with the function :meth:`TraceGUI.select_rays <optrace.gui.trace_gui.TraceGUI.select_rays>`.
It takes a :python:`mask` parameter, which is a one-dimensional boolean :obj:`numpy.array`, 
and an optional :python:`max_show` parameter, that specified the maximum amount of rays to display.
Parameter :python:`mask` must have the same length as there are rays simulated, 
which is set by :attr:`TraceGUI.ray_count <optrace.gui.trace_gui.TraceGUI.ray_count>`
Note that there is a maximum amount of rays that can be displayed 
(specified by the maximum value of :attr:`TraceGUI.rays_visible <optrace.gui.trace_gui.TraceGUI.rays_visible>`, 
by default :python:`50000`).
If the :python:`mask` includes more values, a random subset is selected.
Accessing :attr:`TraceGUI.ray_selection <optrace.gui.trace_gui.TraceGUI.ray_selection>` 
returns the boolean array for the currently displayed rays.

Typical useful scenarios are debugging or ray analysis.
For instance, only rays from a specific source, region or wavelength range can be selected and displayed.
See :ref:`usage_ray_access` to learn how to access ray properties.
You can find examples for ray selections below.

.. code-block:: python

   def automated(GUI):
       
       # display rays with wavelengths between 400 and 450nm
       mask = (GUI.raytracer.rays.wl_list >= 400) & (GUI.raytracer.rays.wl_list <= 450)
       GUI.select_rays(mask) # no max_show provided, but might be limited by this function

       # display 2000 rays that start at x > 0
       mask = GUI.raytracer.rays.p_list[:, :, 0] > 0
       GUI.select_rays(mask[:, 0], 2000)  # slicing with 0 so mask is 1D
       
       # get the mask for actually displayed selection
       selection = GUI.ray_selection

   # create the GUI and provide the automation function to TraceGUI.control()
   sim = TraceGUI(RT)
   sim.control(func=automated, args=(sim,))


Picking Manually
________________________

The function :meth:`TraceGUI.pick_ray <optrace.gui.trace_gui.TraceGUI.pick_ray>` highlights a full ray.
An integer :python:`index` is required as to select a given ray.
Only currently displayed rays are pickable, 
which are defined by :attr:`TraceGUI.ray_selection <optrace.gui.trace_gui.TraceGUI.ray_selection>`, 
see :ref:`usage_gui_selecting_rays`.
So an :python:`index=50` means that the 51th :python:`True` value 
of :attr:`TraceGUI.ray_selection <optrace.gui.trace_gui.TraceGUI.ray_selection>` is picked.

Function :meth:`TraceGUI.pick_ray_section <optrace.gui.trace_gui.TraceGUI.pick_ray_section>` 
highlights a ray at a given intersection.
The ray is highlighted, a crosshair is shown at the intersection position 
and a ray information text is displayed inside the scene.
Compared to the previous function, an additional integer :python:`section` parameter is required.
An optional parameter :python:`detailed` defines if more detailed information should be shown.
This is equivalent to picking a ray section manually with ``Shift + Click``.

Call :meth:`TraceGUI.reset_picking <optrace.gui.trace_gui.TraceGUI.reset_picking>` 
to deactivate the ray highlighting, information text, and cross hair.

Here is an example:

.. code-block:: python

   def automated(sim):

       # pick the ray with index 100
       sim.pick_ray(index=100)

       # pick ray section 2 of ray 50 with default view
       sim.pick_ray_section(index=50, section=2)

       # pick ray section with detailed view
       sim.pick_ray_section(index=50, section=2, detailed=True)

       # reset (=hide) the picking view
       sim.reset_picking()

   # create the GUI and provide the automation function to TraceGUI.control()
   sim = TraceGUI(RT)
   sim.control(func=automated, args=(sim,))

Showing Plots
________________________

Available plotting functions include :meth:`TraceGUI.source_image <optrace.gui.trace_gui.TraceGUI.source_image>`, 
:meth:`TraceGUI.source_profile <optrace.gui.trace_gui.TraceGUI.source_profile>`,
:meth:`TraceGUI.detector_image <optrace.gui.trace_gui.TraceGUI.detector_image>`, 
:meth:`TraceGUI.detector_profile <optrace.gui.trace_gui.TraceGUI.detector_profile>`,
:meth:`TraceGUI.detector_spectrum <optrace.gui.trace_gui.TraceGUI.detector_spectrum>`, 
:meth:`TraceGUI.source_spectrum <optrace.gui.trace_gui.TraceGUI.source_spectrum>`,
and :meth:`TraceGUI.move_to_focus <optrace.gui.trace_gui.TraceGUI.move_to_focus>`.

There are more parameters available through these direct calls than through the GUI.
For example, it is possible to save an image to the disk.
Additionally, a custom detector/source extent can be specified, a setting not available through the GUI.

.. code-block:: python

   def automated(sim):

       # change plot settings
       sim.image_pixels = 315
       sim.image_mode = "Lightness (CIELUV)"

       # show a source profile with a user-defined extent
       sim.source_profile(extent=[0, 0.1, 0.2, 0.25])

       # save a detector image with higher dpi
       sim.detector_image(path="detector.png", sargs=(dpi=600))

   # create the GUI and provide the automation function to TraceGUI.control()
   sim = TraceGUI(RT)
   sim.control(func=automated, args=(sim,))

.. _gui_automation_custom_UI:

Accessing custom UI elements
_________________________________

Custom elements (see :numref:`custom_elements`) are accessible through a name, 
consisting of their type and a chronological number.
The number corresponds to the order that the element has been created.
Assigning values works analogously to all other parameters.
To button action is called with a specific function.

You can find examples below.

.. code-block:: python

   sim.custom_value_2 = 4.5
   sim.custom_checkbox_1 = False
   sim.custom_selection_3 = "Case 2"

   sim.custom_button_action_1()


.. _gui_automation_close:

Closing Down
________________________

Call the function :meth:`TraceGUI.close <optrace.gui.trace_gui.TraceGUI>` to close the GUI down programmatically.

.. code-block:: python

   def automated(sim):

       # do some things
       ...

       # close everything down
       sim.close()

   # create the GUI and provide the automation function to TraceGUI.control()
   sim = TraceGUI(RT)
   sim.control(func=automated, args=(sim,))

This will not only end the GUI, but all other plots and subwindows. 

