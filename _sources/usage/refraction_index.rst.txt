Refractive Indices
-----------------------

.. role:: python(code)
  :language: python
  :class: highlight


.. testsetup:: *

   import optrace as ot
   import numpy as np


Defining the Model
________________________

**Constant**

The simplest case, a constant (wavelength-independent) 
:class:`RefractionIndex <optrace.tracer.refraction_index.RefractionIndex>`, is defined as:

.. testcode::

   n = ot.RefractionIndex("Constant", n=1.54)

**Center Index and Abbe Number**

For most materials only a single refractive index :math:`n_c` and an Abbe number :math:`V` are provided, 
but not a full :math:`n(\lambda)`-curve.
Such a material is modelled by:

.. testcode::

   n = ot.RefractionIndex("Abbe", n=1.5, V=32)

Note that materials with the same :math:`n_c`, :math:`V` can still differ slightly, 
as many dispersion curves produce these two values.
Details on how the model is estimated are located in :numref:`index_from_abbe`.

You can also specify the wavelength combination, for which :python:`n` and :python:`V` are specified:

.. testcode::

   n = ot.RefractionIndex("Abbe", n=1.5, V=32, lines=ot.presets.spectral_lines.FeC)


**Common Index Models**
   
The subsequent equations describe common refractive index models used in typical simulation software.
They are taken from :footcite:`ComsolDispersion` and :footcite:`ZemaxHagen`.
A comprehensive list of different index models is found in :footcite:`palmer2005`.

Generally, all coefficients must be expressed in powers of µm, which is also true for the wavelength input values.

.. list-table::
   :widths: 300 900
   :header-rows: 1

   * - Name
     - Equation

   * - Cauchy

     - .. math::
          n = c_0 + \frac{c_1}{\lambda^2} + \frac{c_2}{\lambda^4} + \frac{c_3}{\lambda^6}
          :label: n_cauchy

   * - Conrady

     - .. math::
          n = c_0+ \frac{c_1} {\lambda} + \frac{c_2} {\lambda^{3.5}}
          :label: n_conrady

   * - Extended

     - .. math::
          n^2 = c_0+c_1 \lambda^2+ \frac{c_2} {\lambda^{2}}+ \frac{c_3} {\lambda^{4}}+ 
          \frac{c_4} {\lambda^{6}}+ \frac{c_5} {\lambda^{8}}+ \frac{c_6} {\lambda^{10}}+\frac{c_7} {\lambda^{12}}
          :label: n_extended

   * - Extended2

     - .. math::
          n^2 = c_0+c_1 \lambda^2+ \frac{c_2} {\lambda^{2}}+ \frac{c_3} {\lambda^{4}}+\frac{c_4} 
          {\lambda^{6}}+\frac{c_5} {\lambda^{8}}+c_6 \lambda^4+c_7 \lambda^6
          :label: n_extended2

   * - Handbook of Optics 1

     - .. math::
          n^2 = c_0+\frac{c_1}{\lambda^2-c_2}-c_3 \lambda^2
          :label: n_optics1

   * - Handbook of Optics 2

     - .. math::
          n^2 = c_0+\frac{c_1 \lambda^2}{\lambda^2-c_2}-c_3 \lambda^2
          :label: n_optics2

   * - Herzberger

     - .. math::
          \begin{align}
          n =&~ c_0+c_1 L+c_2 L^2+c_3 \lambda^2+c_4 \lambda^4+c_5 \lambda^6 \\
          &\text{ with   } L= \frac{1} {\lambda^2-0.028 \mathrm{\,µm}^2}
          \end{align}
          :label: n_herzberger

   * - Sellmeier1

     - .. math::
          n^2 = 1+\frac{c_0 \lambda^2}{\lambda^2-c_1}+\frac{c_2 \lambda^2}
          {\lambda^2-c_3}+\frac{c_4 \lambda^2}{\lambda^2-c_5}
          :label: n_sellmeier1 

   * - Sellmeier2

     - .. math::
          n^2 = 1+c_0+\frac{c_1 \lambda^2}{\lambda^2-c_2^2}+\frac{c_3}{\lambda^2-c_4^2}
          :label: n_sellmeier2 

   * - Sellmeier3

     - .. math::
          n^2 = 1+\frac{c_0 \lambda^2}{\lambda^2-c_1}+\frac{c_2 \lambda^2}{\lambda^2-c_3}+
          \frac{c_4 \lambda^2}{\lambda^2-c_5}+\frac{c_6 \lambda^2}{\lambda^2-c_7}
          :label: n_sellmeier3 

   * - Sellmeier4

     - .. math::
          n^2 = c_0+\frac{c_1 \lambda^2}{\lambda^2-c_2}+\frac{c_3 \lambda^2}{\lambda^2-c_4}
          :label: n_sellmeier4 

   * - Sellmeier5

     - .. math::
          n^2 = 1+\frac{c_0 \lambda^2}{\lambda^2-c_1}+\frac{c_2 \lambda^2}{\lambda^2-c_3}+
          \frac{c_4 \lambda^2}{\lambda^2-c_5}+\frac{c_6 \lambda^2}{\lambda^2-c_7}+\frac{c_8 \lambda^2}{\lambda^2-c_9}
          :label: n_sellmeier5 

   * - Schott

     - .. math::
          n^2 = c_0+c_1 \lambda^2+\frac{c_2}{ \lambda^{2}}+\frac{c_3} {\lambda^{4}}+\frac{c_4}
          {\lambda^{6}}+\frac{c_5} {\lambda^{8}}
          :label: n_schott 


An example of a Schott model material is initialized in the following manner:

.. testcode::

   n = ot.RefractionIndex("Schott", coeff=[2.13e-06, 1.65e-08, -6.98e-11, 1.02e-06, 6.56e-10, 0.208])

**User Data**

The :python:`"Data"` type allows a model definition from a wavelength in nanometers (380.0 - 780.0) and index list.
Intermediary values are interpolated linearly.

.. testcode::

   wls = np.linspace(380, 780, 10)
   vals = np.array([1.6, 1.58, 1.55, 1.54, 1.535, 1.532, 1.531, 1.53, 1.529, 1.528])
   n = ot.RefractionIndex("Data", wls=wls, vals=vals)

**User Function**

optrace supports custom user functions for the refractive index with the :python:`func`-parameter: 

.. testcode::

   n = ot.RefractionIndex("Function", func=lambda wl: 1.6 - 1e-4*wl)

The first parameter must be the wavelength in nanometers, 
while additional parameters are provided by the :python:`func_args` parameter.

.. testcode::

   n = ot.RefractionIndex("Function", func=lambda wl, n0: n0 - 1e-4*wl, func_args=dict(n0=1.6))


Calculating the Index Values
______________________________

Index values are calculated by calling the object with a wavelength vector.
The return value is a vector of the same shape as the input.

.. doctest::

   >>> n = ot.RefractionIndex("Abbe", n=1.543, V=62.1)
   >>> wl = np.linspace(380, 780, 5)
   >>> n(wl)
   array([1.56237795, 1.54967655, 1.54334454, 1.5397121 , 1.53742915])

Abbe Number
__________________

Every :class:`RefractionIndex <optrace.tracer.refraction_index.RefractionIndex>` object provides a 
:meth:`abbe_number <optrace.tracer.refraction_index.RefractionIndex.abbe_number>`-method:

.. doctest::

   >>> n = ot.presets.refraction_index.LAF2
   >>> n.abbe_number()
   44.850483919254984

The function can be called with a different spectral line combination 
from :mod:`ot.presets.spectral_lines <optrace.tracer.presets.spectral_lines>`:

.. doctest::

   >>> n.abbe_number(ot.presets.spectral_lines.F_eC_)
   44.57150709341499

A list of predefined lines can be found in :numref:`spectral_lines`.
You can also specify a user defined list of three wavelengths:

.. doctest::

   >>> n.abbe_number([450, 580, 680])
   30.59379412865849

To check if a medium is dispersive, call:

.. doctest::

   >>> print(n.is_dispersive())
   True

.. _agf_load:

Loading material catalogues (.agf)
_________________________________________

optrace supports importing ``.agf`` catalogue files that contain different material definitions.
The function :func:`ot.load_agf <optrace.tracer.load.load_agf>` requires a file path and 
returns a dictionary of media, with the key being the name and the value being the refractive index object.

For instance, a way to load the Schott catalogue and accessing the material ``N-LAF21`` is shown below.

.. code-block:: python

   n_schott = ot.load_agf("schott.agf")
   n_laf21 = n_schott["N-LAF21"]


Different ``.agf`` files are located in
`this repository <https://github.com/nzhagen/zemaxglass/tree/master/AGF_files>`__ 
or `this one <https://github.com/edeforas/Astree/tree/master/glass>`__.

Information on the file format are available `here <https://neurophysics.ucsd.edu/Manuals/Zemax/ZemaxManual.pdf>`__ and
and `here <https://github.com/nzhagen/zemaxglass/blob/master/ZemaxGlass_user_manual.pdf>`__.

Plotting
________________

See :ref:`index_plots`.


.. _refraction_index_presets:

Presets
_________________

optrace provides many material presets, which can be accessed using ``ot.presets.refractive_index.<name>``, 
where ``<name>`` is the material name.
The materials are also grouped into lists 
:python:`ot.presets.refractive_index.glasses, ot.presets.refractive_index.plastics, ot.presets.refractive_index.misc`. 

The following plots visualize the index curves and Abbe plots group-wise.

.. list-table::
   :widths: 500 500
   :class: table-borderless

   * - .. figure:: ../images/glass_presets_n.svg
          :width: 500
          :align: center
          :class: dark-light

          Refraction index curves for the glass presets.

     - .. figure:: ../images/glass_presets_V.svg
          :width: 500
          :align: center
          :class: dark-light
       
          Abbe diagram for the glass presets.
   
   * - .. figure:: ../images/plastics_presets_n.svg
          :width: 500
          :align: center
          :class: dark-light
          
          Refraction index curves for the plastic presets.
       
     - .. figure:: ../images/plastics_presets_V.svg
          :width: 500
          :align: center 
          :class: dark-light
          
          Abbe diagram for the plastic presets.


   * - .. figure:: ../images/misc_presets_n.svg
          :width: 500
          :align: center
          :class: dark-light
           
          Refraction index curves for miscellaneous presets.
     
     - .. figure:: ../images/misc_presets_V.svg
          :width: 500
          :align: center
          :class: dark-light
          
          Abbe diagram for the miscellaneous presets. `Air` and `Vacuum` are modelled non-dispersive and missing in this plot.


------------

**References**

.. footbibliography::


