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

This page features information about the GUI automation.
Here are some exemplary tasks that can be achieved with automation:

* changing the geometry and visualizing all changes
* changing camera views accurately
* displaying specific rays
* picking specific rays
* debugging
* taking automated screenshots of the scene
* calling functions with settings not available through the GUI
* ...


The Control Method
________________________


To do automated tasks, the :meth:`TraceGUI.control <optrace.gui.trace_gui.TraceGUI.control>` method is used instead of
It takes an automation function as argument as well as an argument tuple.
Typically, the automated function should take a reference to the TraceGUI, which is supplied by the :python:`args` argument of the control function.

.. code-block:: python

   def automated(GUI):

       # do some things
       ...

   # create the GUI and provide the automation function to TraceGUI.control()
   sim = TraceGUI(RT)
   sim.control(func=automated, args=(sim,))

By default, the control function is executed and the program is kept running.
This makes it possible to interact with the GUI and scene and view the changes and adapt things.
However, closing it is supported with :meth:`TraceGUI.close <optrace.gui.trace_gui.TraceGUI>`, as explained in :numref:`gui_automation_close`

To avoid race conditions, all actions inside the provided automation function are run sequentially.
During this time, the GUI is unresponsive, as the automation function is run in the main thread (as user input would also be) so the interaction with the scene in this time will be limited.


Applying Properties
________________________


Available TraceGUI properties are discussed in :numref:`gui_tabs`.
All these can be set programmatically.

While the variables are set, the TraceGUI does not execute the functions or actions that react to these changes automatically.
This is in contrast to the standard :meth:`TraceGUI.run <optrace.gui.trace_gui.TraceGUI.run>` method, 
where a setting is applied subsequently (for instance, changing the ray color setting updates the color in the scene automatically).

To process all pending events :meth:`TraceGUI.process <optrace.gui.trace_gui.TraceGUI.process>` must be called.
This is not needed after every property change, but only if the changes should be visible.

.. code-block:: python

   def automated(GUI):

       # change properties
       GUI.minimalistic_view = True
       GUI.hide_labels = True
       GUI.ray_count = 1000

       # GUI properties were set, but the changes need to be processed
       GUI.process()

   # create the GUI and provide the automation function to TraceGUI.control()
   sim = TraceGUI(RT)
   sim.control(func=automated, args=(sim,))

Note that some functions, like :meth:`TraceGUI.replot <optrace.gui.trace_gui.TraceGUI.replot>` also call :meth:`TraceGUI.process <optrace.gui.trace_gui.TraceGUI.process>` internally.


Replotting
________________________

While :meth:`TraceGUI.process <optrace.gui.trace_gui.TraceGUI.process>` reacts to changes in the TraceGUI itself, it does not handle changes of the raytracer or tracing geometry.

When changing the geometry, the changes are not automatically applied to the scene.
The geometry is not automatically raytraced as well.

To force the redrawing and retracing of the full scene you can call :meth:`TraceGUI.replot <optrace.gui.trace_gui.TraceGUI.replot>`.

With the context manager :meth:`TraceGUI.smart_replot <optrace.gui.trace_gui.TraceGUI.smart_replot>` it is possible to only update changed objects.
For instance, if a detector is moved, there is no need for updating the lenses inside the geometry or tracing the scene.
:meth:`TraceGUI.smart_replot <optrace.gui.trace_gui.TraceGUI.smart_replot>` handles the detection of changes and updating automatically.
Geometry and raytracing properties are stored when entering into the context manager and compared to when exiting.


Here is an example:

.. code-block:: python

   def automated(GUI):

       # replot everything
       GUI.replot()

       # do some actions and at the end replot only changed objects
       # and/or retrace the geometry if needed.
       with GUI.smart_replot():
           ...

   # create the GUI and provide the automation function to TraceGUI.control()
   sim = TraceGUI(RT)
   sim.control(func=automated, args=(sim,))

When controlling the TraceGUI through the CommandWindow of the GUI, there is also the option to replot all objects automatically.
The implementation is done internally in the same way by using :meth:`TraceGUI.smart_replot <optrace.gui.trace_gui.TraceGUI.smart_replot>`.


.. _gui_camera:

Controlling the Camera
________________________


Controlling the camera is done using the functions :meth:`TraceGUI.set_camera <optrace.gui.trace_gui.TraceGUI.set_camera>` and :meth:`TraceGUI.get_camera <optrace.gui.trace_gui.TraceGUI.get_camera>`.
The former sets properties, while the latter one returns a dictionary of the current settings.

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

Below you can find example code:

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


Applying camera properties can also be done initially using the :python:`initial_camera` parameter of the TraceGUI class.
This parameter is a dictionary that can include all possible parameters of function :meth:`TraceGUI.set_camera <optrace.gui.trace_gui.TraceGUI.set_camera>`.

.. code-block:: python

   sim = TraceGUI(RT, initial_camera=dict(direction=[0, 1, 0], roll=45))


Taking Screenshots
________________________

Screenshots of the scene can be taken with the :meth:`TraceGUI.screenshot <optrace.gui.trace_gui.TraceGUI.screenshot>` function.
A path string is required.
The file type is determined automatically.

Internally, the :obj:`mayavi.mlab.savefig` function from `mayavi <https://docs.enthought.com/mayavi/mayavi>`__ is used.
Therefore it also supports its additional parameters.

Below you can find examples for a call:

.. code-block:: python

   def automated(sim):

       # default call
       sim.screenshot("image.png")

       # call with additional parameters
       sim.screenshot("image2.png", magnification=2)

   # create the GUI and provide the automation function to TraceGUI.control()
   sim = TraceGUI(RT)
   sim.control(func=automated, args=(sim,))

Note that using the :python:`magnification` parameter leads to a rescaled scene, where some elements change their relative size.

.. _usage_gui_selecting_rays:

Selecting Rays
_________________________


By default, a random selection of rays is displayed inside the scene where the number is specified by :attr:`TraceGUI.rays_visible <optrace.gui.trace_gui.TraceGUI.rays_visible>`.
A custom selection can be set using function :meth:`TraceGUI.select_rays <optrace.gui.trace_gui.TraceGUI.select_rays>`.
It takes a :python:`mask` parameter, which is a one-dimensional boolean :obj:`numpy.array`, and an optional :python:`max_show` parameter, that specified the maximum amount of rays to display.
Parameter :python:`mask` must have the same length as there are rays simulated, which is set by :attr:`TraceGUI.ray_count <optrace.gui.trace_gui.TraceGUI.ray_count>`
Note that there is a maximum amount of rays that can be displayed (specified by the maximum value of :attr:`TraceGUI.rays_visible <optrace.gui.trace_gui.TraceGUI.rays_visible>`, by default :python:`50000`).
If the :python:`mask` includes more values, a random subset is selected.
Accessing :attr:`TraceGUI.ray_selection <optrace.gui.trace_gui.TraceGUI.ray_selection>` returns the boolean array for the currently displayed rays.

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
       
       # get mask for actually displayed selection
       selection = GUI.ray_selection

   # create the GUI and provide the automation function to TraceGUI.control()
   sim = TraceGUI(RT)
   sim.control(func=automated, args=(sim,))


Picking Manually
________________________


The function :meth:`TraceGUI.pick_ray <optrace.gui.trace_gui.TraceGUI.pick_ray>` highlights a full ray.
A :python:`index` parameter is required as integer to select a given ray.
Only currently displayed rays can be picked, which are defined by :attr:`TraceGUI.ray_selection <optrace.gui.trace_gui.TraceGUI.ray_selection>`, see :ref:`usage_gui_selecting_rays`.
So an :python:`index=50` means that the 50th :python:`True` value of :attr:`TraceGUI.ray_selection <optrace.gui.trace_gui.TraceGUI.ray_selection>` is picked.

Function :meth:`TraceGUI.pick_ray_section <optrace.gui.trace_gui.TraceGUI.pick_ray_section>` highlights a ray at a given intersection.
The ray is highlighted, a crosshair is shown at the intersection position and a ray information text is shown inside the scene.
Compared to the previous function, an additional integer :python:`section` parameter is needed.
An optional parameter :python:`detailed` defines if more detailed information should be shown.
This would be equivalent to picking a section manually in the scene with the Shift key hold.

To deactivate the ray highlight, information text and cross hair, :meth:`TraceGUI.pick_ray <optrace.gui.trace_gui.TraceGUI.reset_picking>` should be called.

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

Available plotting functions include :meth:`TraceGUI.source_image <optrace.gui.trace_gui.TraceGUI.source_image>` , :meth:`TraceGUI.source_cut <optrace.gui.trace_gui.TraceGUI.source_cut>`, 
:meth:`TraceGUI.detector_image <optrace.gui.trace_gui.TraceGUI.detector_image>`, :meth:`TraceGUI.detector_cut <optrace.gui.trace_gui.TraceGUI.detector_cut>`,
:meth:`TraceGUI.detector_spectrum <optrace.gui.trace_gui.TraceGUI.detector_spectrum>`, :meth:`TraceGUI.source_spectrum <optrace.gui.trace_gui.TraceGUI.source_spectrum>`,
:meth:`TraceGUI.move_to_focus <optrace.gui.trace_gui.TraceGUI.move_to_focus>`.

There are more settings available than through the GUI.
For example, it is possible to save a image instead to show it.
Additionally a custom detector/source extent can be specified, which is not available through the GUI.

.. code-block:: python

   def automated(sim):

       # change plot settings
       sim.image_pixels = 315
       sim.image_mode = "Lightness (CIELUV)"

       # show a source cut with a user-defined extent
       sim.source_cut(extent=[0, 0.1, 0.2, 0.25])

       # save a detector image with higher dpi
       sim.detector_image(path="detector.png", sargs=(dpi=600))

       # example for an automated focus plots
       sim.detector_index = 1
       sim.source_index = 0
       sim.cost_function_plot = True
       sim.move_to_focus()

   # create the GUI and provide the automation function to TraceGUI.control()
   sim = TraceGUI(RT)
   sim.control(func=automated, args=(sim,))

.. _gui_automation_close:

Closing Down
________________________

To close the GUI down programmatically, the function :meth:`TraceGUI.close <optrace.gui.trace_gui.TraceGUI>` can be called:

.. code-block:: python

   def automated(sim):

       # do some things
       ...

       # close everything down
       sim.close()

   # create the GUI and provide the automation function to TraceGUI.control()
   sim = TraceGUI(RT)
   sim.control(func=automated, args=(sim,))

This will close all GUI windows, as well as all plots and exit all background tasks.

