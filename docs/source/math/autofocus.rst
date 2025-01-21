
.. _autofocus:

*******************
Autofocus Methods
*******************

.. role:: python(code)
  :language: python
  :class: highlight


Procedure
=============================

The autofocus procedure is illustrated in the following figure.

Position Variance differs in a way from all other methods, that the cost function is smooth, has a distinct minimum and can therefore be simply minimized.
All other methods first sample the search region for good starting points and then minimize relative to this point.
This ensures some robustness against local minima and a non-smooth cost function.

This interval sampling of cost function values is also helpful to create the autofocus debug plot mentioned in :numref:`focus_cost_plot`.
To ensure that even with method :python:`"Position Variance"` this sampling takes place, the boolean :python:`return_cost` parameter is available.
With it set to :python:`True` it also takes place to enable the plotting of the cost function.

.. figure:: ../images/FocusPAP.svg
   :width: 400
   :align: center
   :class: dark-light
   
   Autofocus process flowchart.


------------

**References**

.. footbibliography::

