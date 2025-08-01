
.. _autofocus:

*******************
Focus Methods
*******************

.. role:: python(code)
  :language: python
  :class: highlight


Procedure
=============================

The focus search procedure is illustrated in the subsequent flowchart.

.. figure:: ../images/FocusPAP.svg
   :width: 400
   :align: center
   :class: dark-light
   
   Autofocus process flowchart.


RMS Spot Size
=============================

**Standard RMS**

An analytical solution exists for the RMS spot size method. 
The following derivation closely follows :footcite:`Boussemaere_2023_19`.
For ray sections that start at the coordinates :math:`x_i, y_i, z_0`, 
their positions at an additional axial distance :math:`t` are given by:

.. math::
   x_i^*&=x_i+\frac{s_{x, i}}{s_{z, i}} t \\
   y_i^*&=y_i+\frac{s_{y, i}}{s_{z, i}} t
   :label: eq_rms_spot_size_propagation

We can define the RMS spot size relative to the center coordinates :math:`x_c^*, y_c^*`. 
This average position also propagates, originating from :math:`x_c, y_c` and possessing a direction vector :math:`s_c`.
The cost function is:

.. math::
   R_v(d) &=\sqrt{\frac{1}{N} \sum_{i=1}^N\left(\left(x_i^*-x_c^*\right)^2+\left(y_i^*-y_c^*\right)^2\right)} \\
   &=\sqrt{\frac{1}{N} \sum_{i=1}^N\left(\left(\Delta x_i+d \Delta \theta_{x, i}\right)^2
   +\left(\Delta y_i+d \Delta \theta_{y, i}\right)^2\right)}
   :label: eq_rms_spot_size_cost_function

Introducing the new relative coordinates :math:`\Delta x_i, \Delta y_i` 
and the relative direction :math:`\Delta \theta_{x,i}, \Delta \theta_{y,i}`:

.. math::
   \Delta x_i & =x_i-x_c \\
   \Delta y_i & =y_i-y_c \\
   \Delta \theta_{x, i} & =\frac{s_{x, i}}{s_{z, i}}-\frac{s_{x, c}}{s_{z, c}} \\
   \Delta \theta_{y, i} & =\frac{s_{y, i}}{s_{z, i}}-\frac{s_{y, c}}{s_{z, c}}
   :label: eq_rms_spot_size_relative_coordinates

Applying variational calculus results in :footcite:`Boussemaere_2023_19`:

.. math::
   t = -\frac{\sum_{i=1}^N \left(\Delta \theta_{x, i} \Delta x_i+\Delta \theta_{y, i} \Delta y_i \right)}
   {\sum_{i=1}^N \left(\Delta \theta_{x, i}^2+\Delta \theta_{y, i}^2 \right)}
   :label: eq_rms_spot_size_solution

Thus, the focal position is located at :math:`z_0 + t`.

**Ray Weighted RMS**

It is possible to include additional weights :math:`w_i` for each ray, which can represent attributes such as ray power.

.. math::
   R_v(d) =\sqrt{\frac{1}{N} \sum_{i=1}^N\left(\left(w_i\left(x_i^*-x_c^*\right)\right)^2
   +\left(w_i\left(y_i^*-y_c^*\right)\right)^2\right)} \\
   :label: eq_weighted_rms_spot_size_cost_function

The weights :math:`w_i` can be isolated from the rest of the expression, 
resulting in a factor of :math:`w_i^2` for all terms. This yields a solution of:

.. math::
   t = -\frac{\sum_{i=1}^N w_i^2 \left(\Delta \theta_{x, i} \Delta x_i+\Delta \theta_{y, i} \Delta y_i \right)}
   {\sum_{i=1}^N w_i^2 \left(\Delta \theta_{x, i}^2 + \Delta \theta_{y, i}^2 \right)}
   :label: eq_weighted_rms_spot_size_solution

**Position Weighted RMS**

Utilizing other strictly monotonically increasing functions that depend on 
:math:`r^2 = \left(x_i^* - x_c^*\right)^2 + \left(y_i^* - y_c^*\right)^2` does not yield additional benefits. 
These functions all share the same position for the minimum 
but might present numerical challenges or be more complex to compute.
Due to its simplicity and convexity optimizing just :math:`r^2` should be preferred.


Optimization Methods
====================================

The :func:`scipy.optimize.minimize` function, along with its optimization methods, 
is utilized under the hood for various optimization tasks. 

For the Irradiance Variance method, the :external:ref:`Nelder-Mead <optimize.minimize-neldermead>` 
solver is employed directly. This choice is based on the observation that the cost function is typically 
smooth and straightforward to minimize in most situations.

In contrast, for the methods Image Sharpness and Image Center Sharpness, 
the cost function tends to be much noisier and may include numerous local minima. 
To address this challenge, the search region is initially sampled at multiple points. 
Subsequently, minimization is initiated relative to the smallest found cost. 
The :external:ref:`COBYLA <optimize.minimize-cobyla>` solver yields good results in these cases.

Pixel Dimensions for Rendering Methods
==================================================

Methods such as Irradiance Variance, Image Sharpness, and Image Center Sharpness render multiple images
denoted as :math:`P_z`, each with a pixel count of :math:`N_\text{px} \cdot N_\text{px}`.

The side length in pixels, :math:`N_\text{px}`, is influenced by the number of rays used for focus determination. 
When working with a small number of rays, it is advantageous to keep :math:`N_\text{px}` low to minimize noise effects. 
Conversely, as the number of rays increases, :math:`N_\text{px}` can be gradually increased to resolve finer details. 
Distributing :math:`N` rays over a square area necessitates increasing :math:`N_\text{px}` 
proportionally to :math:`\sqrt{N}` to maintain a relatively consistent Signal-to-Noise Ratio (SNR). 
The implemented formula follows the form :math:`N_\text{px} = \text{offset} + \text{factor} \cdot \sqrt{N}`.

For simplicity, the same pixel count is used for both image dimensions.

------------

**References**

.. footbibliography::

