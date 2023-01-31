Plotting
--------------


Namespace
_____________


The module ``optrace.plots`` features some plots and visualizations for different kinds of objects.
You can for example import it as ``otp``

.. testcode::

   import optrace.plots as otp

..And call a ``plotting_function`` (only an example) by writing:

.. code-block::

   otp.plotting_function(...)

Functions
_____________

Detailed descriptions and examples are found in other sections, such as:
Surface plots (:numref:`surface_plotting`), spectrum plots (:numref:`spectrum_plots`), refraction index and abbe plots (:numref:`index_plots`), image plots (:numref:`image_plots`), focus cost plot (:numref:`focus_cost_plot`), chromatcity plots (:numref:`chromaticity_plots`).



Parameters
______________

All methods work with a ``block`` parameter that gets passed down to the ``plt.show()`` with ``plt`` being ``matplotlib.pyplot`` from the ``matplotlib`` plotting library. With ``block=True`` the rest of the program gets halted.

.. code-block:: python

   some_plotting_function(..., block=True)

Most methods also include a ``title`` argument that lets the user define a different plot title.

.. code-block:: python

   some_plotting_function(..., title="Name of plot")

Functions outputting messages to the standard output (usually the terminal) can be muted with ``silent=True``.

.. code-block:: python

   some_loud_plotting_function(..., silent=True)

Legends and labels inside the figures are generated from descriptions from the objects. Make sure the create your objects with a ``desc=".."`` or ``long_desc="..."`` parameter so they feature some expressive name.

.. code-block:: python

   obj = Object(..., desc="Abc378")
   obj2 = Object(..., long_desc="Some long description")

   some_plotting_function([obj, obj2], ...)
