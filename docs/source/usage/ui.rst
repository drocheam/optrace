Using the GUI
---------------

.. testsetup:: *

   import optrace as ot

.. role:: python(code)
  :language: python
  :class: highlight



Loading the GUI
____________________


**Example**

As always, import the main optrace namespace:

.. testcode::

   import optrace as ot

To import the TraceGUI into the current namespace write:

.. testcode::

   from optrace.gui.trace_gui import TraceGUI

Let's create some exemplary geometry:

.. testcode::

   RT = ot.Raytracer(outline=[-10, 10, -10, 10, -10, 60])

   disc = ot.CircularSurface(r=3)
   RS = ot.RaySource(disc, pos=[0, 0, -5])
   RT.add(RS)

   eye = ot.presets.geometry.legrand_eye()
   RT.add(eye)

A TraceGUI takes the raytracer as argument:

.. testcode::

   sim = TraceGUI(RT)

The GUI is now assigned to the :python:`sim` variable, but has not started yet.
For running the GUI you need to write:

.. code-block:: python

   sim.run()

This loads the main window and also raytraces the geometry.

**Parameters**

When creating the GUI, additional properties can be assigned.
For instance, setting the scene to high contrast mode and increasing the amount of rays traced, we can write instead:

.. testcode::

   sim = TraceGUI(RT, high_contrast=True, ray_count=2000000)

Available properties are discussed in :numref:`gui_tabs`.


**UI Theme**

The TraceGUI uses Qt5 as UI backend. Qt5 supports different themes that can be controlled with the :python:`ui_theme` parameter on the TraceGUI initialization.

.. testcode::

   sim = TraceGUI(RT, ui_theme="Windows")

Details on styles can be found in the `Qt documentation <https://doc.qt.io/qt-5/qstyle.html#details>`__.
Available themes depend on your system and Qt installation, but can be extended using plugins.
Normally, at least styles :python:`"Windows"` and :python:`"Fusion"` should be available on all systems.
Most notably, dark themes like in :numref:`ui_dark_theme` prove especially useful in low light environments.


UI Overview
_________________

Full UI
######################

.. figure:: ../images/ui_full.png
   :align: center
   :width: 800


Scene
######################


Details on the scene navigation are found in the mayavi documentation :ref:`here <mayavi:interaction-with-the-scene>` under "Mouse Interaction".
There are also keyboard shortcuts available that are discussed in :numref:`gui_keyboard_shortcuts`.


Toolbar
######################

The mayavi scene toolbar is positioned above the scene. It includes buttons for the pipeline view window, different perspectives, fullscreen, screenshot saving and scene settings. Details are found in the mayavi documentation :ref:`here <mayavi:interaction-with-the-scene>`.

Sidebar
######################

The sidebar is positioned at the right hand side of the scene and consists of multiple tabs:

.. list-table::
   :align: left
   :stub-columns: 1
   :widths: 150 350

   * - Main Tab
     - Includes settings for raytracing, scene visualization and buttons for opening additional windows
   * - Image Tab
     - Features options for rendering source and detector images
   * - Spectrum Tab
     - Settings for the rendering of source or detector light spectrum histograms
   * - Focus Tab
     - Option View and result output for finding the focus in the optical setup
   * - Debug Tab
     - Advanced options, especially for development of optrace

The following figure shows all tabs except the debug tab. 
The UI elements will be discussed in the following sections.

.. list-table::
   :align: center

   * - .. figure:: ../images/ui_main_tab.png
          :align: center
          :width: 200

     - .. figure:: ../images/ui_image_tab.png
          :align: center
          :width: 200

     - .. figure:: ../images/ui_spectrum_tab.png
          :align: center
          :width: 200

     - .. figure:: ../images/ui_focus_tab.png
          :align: center
          :width: 200


Additional Windows
#######################


Beside the main window there are additional windows in the interface. These will be discussed in :numref:`gui_windows`, but a quick overview is given here:

.. list-table::
   :align: left
   :header-rows: 1
   :stub-columns: 0
   :widths: 100 250 350

   * - Window
     - Access
     - Function
   * - Pipeline View
     - Leftmost button in the toolbar
     - Access to viewing and editing the mayavi graphical elements
   * - Scene Settings
     - Rightmost button in the toolbar
     - mayavi settings, including lighting and scene properties
   * - Command Window
     - button at the bottom of the main tab in the sidebar
     - command execution and history for controlling the GUI and raytracer
   * - Property Browser
     - button at the bottom of the main tab in the sidebar
     - overview of raytracer, scene and ray properties as well as cardinal points

The Scene
____________________


.. _gui_tabs:

Sidebar Tabs
____________________


Main Tab
#######################


.. list-table::
   :header-rows: 1
   :align: left
   
   * - Property
     - Variable Name / Method
     - Values
     - Description
   * - Rays
     - :attr:`ray_count <optrace.gui.trace_gui.TraceGUI.ray_count>`
     - integer, 0 - 6000000
     -
   * - Absorb Rays Missing Lens
     - :attr:`absorb_missing <optrace.gui.trace_gui.TraceGUI.absorb_missing>`
     - :python:`True` or :python:`False`
     -
   * - Plotting
     - :attr:`plotting_type <optrace.gui.trace_gui.TraceGUI.plotting_type>`
     - :python:`'Rays'` or :python:`'Points'`
     -
   * - Coloring
     - :attr:`coloring_type <optrace.gui.trace_gui.TraceGUI.coloring_type>`
     - :python:`'Plain', 'Power', 'Wavelength', 'Source', 'Polarization xz', 'Polarization yz', 'Refractive Index'`
     -
   * - Count
     - :attr:`rays_visible <optrace.gui.trace_gui.TraceGUI.rays_visible>`
     - integer, 1 - 1000
     -
   * - Opacity
     - :attr:`ray_opacity <optrace.gui.trace_gui.TraceGUI.ray_opacity>`
     - float, 1e-05 - 1
     -
   * - Width
     - :attr:`ray_width <optrace.gui.trace_gui.TraceGUI.ray_width>`
     - float, 1 - 20
     -
   * - More Minimalistic Scene
     - :attr:`minimalistic_view <optrace.gui.trace_gui.TraceGUI.minimalistic_view>`
     - :python:`True` or :python:`False`
     -
   * - Maximize Scene
     - :attr:`maximize_scene <optrace.gui.trace_gui.TraceGUI.maximize_scene>`     
     - :python:`True` or :python:`False`
     -
   * - High Contrast Mode
     - :attr:`high_contrast <optrace.gui.trace_gui.TraceGUI.high_contrast>`
     - :python:`True` or :python:`False`
     -
   * - Vertical Labels
     - :attr:`vertical_labels <optrace.gui.trace_gui.TraceGUI.vertical_labels>`
     - :python:`True` or :python:`False`
     -
   * - Open Property Browser
     - :meth:`open_property_browser() <optrace.gui.trace_gui.TraceGUI.open_property_browser>`
     -
     -
   * - Open Command Window
     - :meth:`open_command_window() <optrace.gui.trace_gui.TraceGUI.open_command_window>`
     -
     -

Image Tab
#######################


.. list-table::
   :header-rows: 1
   :align: left
   
   * - Property
     - Variable Name / Method
     - Values
     - Description
   * - Source 
     - :attr:`source_selection <optrace.gui.trace_gui.TraceGUI.source_selection>`
     - string
     -
   * - Detector
     - :attr:`detector_selection <optrace.gui.trace_gui.TraceGUI.detector_selection>` 
     - string
     - 
   * - z_det
     - :attr:`z_det <optrace.gui.trace_gui.TraceGUI.z_det>`
     - float
     - 
   * - Image Mode
     - :attr:`image_type <optrace.gui.trace_gui.TraceGUI.image_type>`
     - string, one of :attr:`RImage.display_modes <optrace.tracer.r_image.RImage.display_modes>`
     -
   * - Projection Method
     - :attr:`projection_method <optrace.gui.trace_gui.TraceGUI.projection_method>`
     - string, one of :attr:`SphericalSurface.sphere_projection_methods <optrace.tracer.geometry.surface.spherical_surface.SphericalSurface.sphere_projection_methods>`
     - 
   * - Pixels_xy
     - :attr:`image_pixels <optrace.gui.trace_gui.TraceGUI.image_pixels>`
     - integer, one of :attr:`RImage.SIZES <optrace.tracer.r_image.RImage.SIZES>`
     - 
   * - Logarithmic Scaling 
     - :attr:`log_image <optrace.gui.trace_gui.TraceGUI.log_image>`
     - :python:`True` or :python:`False`
     -
   * - Flip Detector Image
     - :attr:`flip_det_image <optrace.gui.trace_gui.TraceGUI.flip_det_image>`
     - :python:`True` or :python:`False`
     -
   * - Rays from Selected Source Only
     - :attr:`det_image_one_source <optrace.gui.trace_gui.TraceGUI.det_image_one_source>`
     - :python:`True` or :python:`False`
     -
   * - Source Image
     - :meth:`show_source_image() <optrace.gui.trace_gui.TraceGUI.show_source_image>`
     -
     - 
   * - Detector Image 
     - :meth:`show_detector_image() <optrace.gui.trace_gui.TraceGUI.show_detector_image>`
     -
     - 
   * - Cut at
     - :attr:`cut_dimension <optrace.gui.trace_gui.TraceGUI.cut_dimension>`
     - :python:`'x', 'y'`
     - 
   * - Cut Value
     - :attr:`cut_value <optrace.gui.trace_gui.TraceGUI.cut_value>`
     - float
     - 
   * - Source Image Cut
     - :meth:`show_source_cut() <optrace.gui.trace_gui.TraceGUI.show_source_cut>`
     -
     - 
   * - Detector Image Cut
     - :meth:`show_detector_cut() <optrace.gui.trace_gui.TraceGUI.show_detector_cut>`
     -
     - 
   * - Activate Filter 
     - :attr:`activate_filter <optrace.gui.trace_gui.TraceGUI.activate_filter>`
     - :python:`True` or :python:`False`
     - 
   * - Resolution Limit 
     - :attr:`filter_constant <optrace.gui.trace_gui.TraceGUI.filter_constant>`
     -  float, 0.3 - 40
     -

Spectrum Tab
#######################

.. list-table::
   :header-rows: 1
   :align: left
   
   * - Property
     - Variable Name / Method
     - Values
     - Description
   * - Source 
     - :attr:`source_selection <optrace.gui.trace_gui.TraceGUI.source_selection>`
     - string
     -
   * - Detector
     - :attr:`detector_selection <optrace.gui.trace_gui.TraceGUI.detector_selection>` 
     - string
     - 
   * - z_det
     - :attr:`z_det <optrace.gui.trace_gui.TraceGUI.z_det>`
     - float
     - 
   * -  Source Spectrum
     - :meth:`show_source_spectrum() <optrace.gui.trace_gui.TraceGUI.show_source_spectrum>`
     - 
     -
   * - Rays from Selected Source Only 
     - :attr:`det_spectrum_one_source <optrace.gui.trace_gui.TraceGUI.det_spectrum_one_source>` 
     - :python:`True` or :python:`False`
     -
   * -  Detector Spectrum
     - :meth:`show_detector_spectrum() <optrace.gui.trace_gui.TraceGUI.show_detector_spectrum>`
     - 
     -
   * -  Spectrum Properties
     - 
     - string
     -

Focus Tab
#######################

.. list-table::
   :header-rows: 1
   :align: left
   
   * - Property
     - Variable Name / Method
     - Values
     - Description
   * - Source 
     - :attr:`source_selection <optrace.gui.trace_gui.TraceGUI.source_selection>`
     - string
     -
   * - Detector
     - :attr:`detector_selection <optrace.gui.trace_gui.TraceGUI.detector_selection>` 
     - string
     - 
   * - z_det
     - :attr:`z_det <optrace.gui.trace_gui.TraceGUI.z_det>`
     - float
     - 
   * -  Focus Mode     
     - :attr:`focus_type <optrace.gui.trace_gui.TraceGUI.focus_type>`
     - string, one of :attr:`Raytracer.autofocus_methods <optrace.tracer.raytracer.Raytracer.autofocus_methods>`
     -
   * -  Rays From Selected Source Only
     - :attr:`af_one_source <optrace.gui.trace_gui.TraceGUI.af_one_source>`
     - :python:`True` or :python:`False`
     -
   * -  Plot Cost Function
     - :attr:`focus_cost_plot <optrace.gui.trace_gui.TraceGUI.focus_cost_plot>`
     - :python:`True` or :python:`False`
     -
   * -  Find Focus
     - :meth:`move_to_focus() <optrace.gui.trace_gui.TraceGUI.move_to_focus>`
     - 
     -
   * -  Optimization  Output
     - 
     - string
     -

.. _gui_windows:

Additional Windows
____________________

Pipeline View
#######################


`<https://docs.enthought.com/mayavi/mayavi/pipeline.html>`__

`<https://docs.enthought.com/mayavi/mayavi/mayavi_objects.html>`__


.. figure:: ../images/ui_pipeline.png
   :align: center
   :width: 600


Property Viewer
#######################

.. figure:: ../images/ui_property_browser.png
   :align: center
   :width: 600

Command Window
#######################


.. TODO single elements from the history can be copied by selecting them and pressing ctrl+c

.. figure:: ../images/ui_command_window.png
   :align: center
   :width: 600

Tips and Tricks
____________________


**Keyboard Shortcuts**

The following keyboard shortcuts are available inside the scene:

.. _gui_keyboard_shortcuts:

.. list-table:: Available keyboards shortcuts
   :header-rows: 1
   :align: center
   :widths: 100 300

   * - Shortcut
     - Function
   * - ``y``
     - set scene view to default y view
   * - ``h``
     - maximize scene (hide toolbar and sidebar)
   * - ``v``
     - toggle minimalistic view option
   * - ``c``
     - toggle high contrast mode
   * - ``r``
     - toggle plotting type of rays (points or beams)
   * - ``d``
     - render detector image with the current settings
   * - ``n``
     - randomly re-chose the plotted rays
   * - ``s``
     - save a screenshot of the scene
   * - ``f``
     - | set the camera focal point to the position of the mouse. 
       | Useful for scene rotations, since the geometry is rotated around this point.
   * - ``l``
     - change lighting properties
   * - ``3``
     - anaglyph view (view for red-cyan 3D glasses)

**Changing the UI Theme Externally**

UI themes can also be set externally, however any theme set inside the script overwrites the global style.

From outside the theme can either be provided by setting an environment variable:

.. code-block:: bash

   env QT_STYLE_OVERRIDE=kvantum-dark python ./examples/microscope.py

...Or by providing a ``style`` parameter when calling the script/intepreter.

.. code-block:: bash

   python ./examples/microscope.py -style kvantum-dark

Note that the mentioned style needs to be supported by your Qt installation. The above syntax is that for an Unix system and can differ for other systems.

.. _ui_dark_theme:

.. figure:: ../images/ui_kvantum_theme.png
   :align: center
   :width: 600

   UI with the dark theme.

**Passing Properties to the GUI object**

Under some circumstances it is useful to provide additional parameters like properties or functions to the GUI so they can be accessed in the control window.
For instance, we implemented a function that changes the geometry in some specific way or steps through different source or lens constellations.

As example, the user can define some function :python:`func` inside his script and pass it to the TraceGUI:

.. testcode::

   def func(a, b, c):
        # do some complicated things inside here
        ...

   sim = TraceGUI(RT, important_function=func)

:python:`func` get assigned to the TraceGUI under the name :python:`important_function`. Therefore it can be used inside the command window as :python:`self.important_function`.

This is not limited to functions but works for arbitrary objects, however note that the assigned name must not collide with any variable or method name already implemented in the TraceGUI class.

