Refractive Indices
-----------------------

.. role:: python(code)
  :language: python
  :class: highlight


.. testsetup:: *

   import optrace as ot
   import numpy as np


Defining the Index
________________________

**Constant**

In the simplest case a constant (wavelength-independent) :class:`refractive index <optrace.tracer.refraction_index.RefractionIndex>` is defined as:

.. testcode::

   n = ot.RefractionIndex("Constant", n=1.54)

**By Abbe Number**

In many cases materials are characterized by the index at a center wavelength and the Abbe number only.
However, materials with these same quantities can still differ slightly.

.. testcode::

   n = ot.RefractionIndex("Abbe", n=1.5, V=32)

Details on how the model is estimated are found in :numref:`index_from_abbe`.

You can also specify the wavelength combination, for which :python:`n` and :python:`V` are specified:

.. testcode::

   n = ot.RefractionIndex("Abbe", n=1.5, V=32, lines=ot.presets.spectral_lines.FeC)


**Common Index Models**
   
The subsequent equations describe common refractive index models used in simulation software.
They are taken from :footcite:`ComsolDispersion` and :footcite:`ZemaxHagen`.
A comprehensive list of different index models is found in :footcite:`palmer2005`.

Generally, all coefficients must be given in powers of µm, while the same is true for the wavelength input.

.. list-table::
   :widths: 300 900

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
          n^2 = c_0+c_1 \lambda^2+ \frac{c_2} {\lambda^{2}}+ \frac{c_3} {\lambda^{4}}+ \frac{c_4} {\lambda^{6}}+ \frac{c_5} {\lambda^{8}}+ \frac{c_6} {\lambda^{10}}+\frac{c_7} {\lambda^{12}}
          :label: n_extended

   * - Extended2

     - .. math::
          n^2 = c_0+c_1 \lambda^2+ \frac{c_2} {\lambda^{2}}+ \frac{c_3} {\lambda^{4}}+\frac{c_4} {\lambda^{6}}+\frac{c_5} {\lambda^{8}}+c_6 \lambda^4+c_7 \lambda^6
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
          &\text{ with   } L= \frac{1} {\lambda^2-0.028 {\mu m^2}}
          \end{align}
          :label: n_herzberger

   * - Sellmeier1

     - .. math::
          n^2 = 1+\frac{c_0 \lambda^2}{\lambda^2-c_1}+\frac{c_2 \lambda^2}{\lambda^2-c_3}+\frac{c_4 \lambda^2}{\lambda^2-c_5}
          :label: n_sellmeier1 

   * - Sellmeier2

     - .. math::
          n^2 = 1+c_0+\frac{c_1 \lambda^2}{\lambda^2-c_2^2}+\frac{c_3}{\lambda^2-c_4^2}
          :label: n_sellmeier2 

   * - Sellmeier3

     - .. math::
          n^2 = 1+\frac{c_0 \lambda^2}{\lambda^2-c_1}+\frac{c_2 \lambda^2}{\lambda^2-c_3}+\frac{c_4 \lambda^2}{\lambda^2-c_5}+\frac{c_6 \lambda^2}{\lambda^2-c_7}
          :label: n_sellmeier3 

   * - Sellmeier4

     - .. math::
          n^2 = c_0+\frac{c_1 \lambda^2}{\lambda^2-c_2}+\frac{c_3 \lambda^2}{\lambda^2-c_4}
          :label: n_sellmeier4 

   * - Sellmeier5

     - .. math::
          n^2 = 1+\frac{c_0 \lambda^2}{\lambda^2-c_1}+\frac{c_2 \lambda^2}{\lambda^2-c_3}+\frac{c_4 \lambda^2}{\lambda^2-c_5}+\frac{c_6 \lambda^2}{\lambda^2-c_7}+\frac{c_8 \lambda^2}{\lambda^2-c_9}
          :label: n_sellmeier5 

   * - Schott

     - .. math::
          n^2 = c_0+c_1 \lambda^2+\frac{c_2}{ \lambda^{2}}+\frac{c_3} {\lambda^{4}}+\frac{c_4} {\lambda^{6}}+\frac{c_5} {\lambda^{8}}
          :label: n_schott 


In the case of the Schott model, the initialization looks as follows:

.. testcode::

   n = ot.RefractionIndex("Schott", coeff=[2.13e-06, 1.65e-08, -6.98e-11, 1.02e-06, 6.56e-10, 0.208])

**User Data**

With type :python:`"Data"` a wavelength and index vector should be provided.
Values in-between are interpolated linearly.

.. testcode::

   wls = np.linspace(380, 780, 10)
   vals = np.array([1.6, 1.58, 1.55, 1.54, 1.535, 1.532, 1.531, 1.53, 1.529, 1.528])
   n = ot.RefractionIndex("Data", wls=wls, vals=vals)

**User Function**

optrace supports custom user functions for the refractive index. The function takes one parameter, which is a wavelength numpy array with wavelengths in nanometers.

.. testcode::

   n = ot.RefractionIndex("Function", func=lambda wl: 1.6 - 1e-4*wl)

When providing a function with multiple parameters, you can use the :python:`func_args` parameter.

.. testcode::

   n = ot.RefractionIndex("Function", func=lambda wl, n0: n0 - 1e-4*wl, func_args=dict(n0=1.6))


Getting the Index Values
___________________________

The refractive index values are calculated when calling the refractive index object with a wavelength vector.
The call returns a vector of the same shape as the input.

.. doctest::

   >>> n = ot.RefractionIndex("Abbe", n=1.543, V=62.1)
   >>> wl = np.linspace(380, 780, 5)
   >>> n(wl)
   array([1.56237795, 1.54967655, 1.54334454, 1.5397121 , 1.53742915])

Abbe Number
__________________

With a refractive index object at hand the Abbe number can be calculated with

.. doctest::

   >>> n = ot.presets.refraction_index.LAF2
   >>> n.abbe_number()
   44.850483919254984

Alternatively the function can be called with a different spectral line combination from :mod:`ot.presets.spectral_lines <optrace.tracer.presets.spectral_lines>`:

.. doctest::

   >>> n.abbe_number(ot.presets.spectral_lines.F_eC_)
   44.57150709341499

Or specify a user defined list of three wavelengths:

.. doctest::

   >>> n.abbe_number([450, 580, 680])
   30.59379412865849


You can also check if a medium is dispersive by calling

.. doctest::

   >>> print(n.is_dispersive())
   True


A list of predefined lines can be found in :numref:`spectral_lines`.

.. _agf_load:

Loading material catalogues (.agf)
_________________________________________

optrace can also load .agf catalogue files containing different materials.
The function :func:`ot.load_agf <optrace.tracer.load.load_agf>` requires a file path and returns a dictionary of media, with the key being the name and the value being the refractive index object.

For instance, loading the Schott catalogue and accessing the material ``N-LAF21`` can be done as follows:

.. code-block:: python

   n_schott = ot.load_agf("schott.agf")
   n_laf21 = n_schott["N-LAF21"]


Different ``.agf`` files are found in `this repository <https://github.com/nzhagen/zemaxglass/tree/master/AGF_files>`__ or `this one <https://github.com/edeforas/Astree/tree/master/glass>`__.

Information on the file format can be found `here <https://neurophysics.ucsd.edu/Manuals/Zemax/ZemaxManual.pdf>`__ and
and `here <https://github.com/nzhagen/zemaxglass/blob/master/ZemaxGlass_user_manual.pdf>`__.

Plotting
________________

See :ref:`index_plots`.


.. _refraction_index_presets:

Presets
_________________

optrace comes with multiple material presets, which can be accessed using ``ot.presets.refractive_index.<name>``, where ``<name>`` is the material name.
The materials are also grouped into multiple lists :python:`ot.presets.refractive_index.glasses, ot.presets.refractive_index.plastics, ot.presets.refractive_index.misc`. 

These groups are plotted below in an index and an Abbe plot.


.. list-table::
   :widths: 500 500
   :class: table-borderless

   * - .. figure:: ../images/glass_presets_n.svg
          :width: 500
          :align: center
          :class: dark-light

          Refraction index curves for different glass presets.

     - .. figure:: ../images/glass_presets_V.svg
          :width: 500
          :align: center
          :class: dark-light
       
          Abbe diagram for different glass presets.
   
   * - .. figure:: ../images/plastics_presets_n.svg
          :width: 500
          :align: center
          :class: dark-light
          
          Refraction index curves for different plastic presets.
       
     - .. figure:: ../images/plastics_presets_V.svg
          :width: 500
          :align: center 
          :class: dark-light
          
          Abbe diagram for different plastic presets.


   * - .. figure:: ../images/misc_presets_n.svg
          :width: 500
          :align: center
          :class: dark-light
           
          Refraction index curves for miscellaneous presets.
     
     - .. figure:: ../images/misc_presets_V.svg
          :width: 500
          :align: center
          :class: dark-light
          
          Abbe diagram for miscellaneous presets. *Air* and *Vacuum* are missing here, because they are modelled without dispersion.


------------

**References**

.. footbibliography::


