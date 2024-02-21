
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



The Control Method
________________________

.. TODO explain how control() is used, what is available and that things are run sequentially


Applying Properties
________________________

.. TODO refer to available properties and emphasize that process_events() might often be needed so things happen

Controlling the Camera
________________________

.. TODO explain set_camera, get_camera and what the parameters mean
.. TODO also explain initial_camera

Taking Screenshots
________________________

.. TODO explain screenshot, refer to the parameters from the mayavi documentation
.. TODO note that magnification rescales some elements in the geometry

Picking Manually
________________________

.. TODO explain pick_ray, pick_ray_section and reset_picking

Showing Plots
________________________

.. TODO explain plotting call (properties are set by TraceGUI variables) and what parameters can be manually set (extent etc.)

Closing Down
________________________

.. TODO explain how the GUI is closed and that it closes all windows

