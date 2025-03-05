.. _usage_spectrum:

Spectrum Classes
--------------------------------------------------

.. role:: python(code)
  :language: python
  :class: highlight

.. testsetup:: *

   import optrace as ot
   import numpy as np


.. |LightSpectrum| replace:: :class:`LightSpectrum <optrace.tracer.spectrum.light_spectrum.LightSpectrum>`
.. |Spectrum| replace:: :class:`Spectrum <optrace.tracer.spectrum.spectrum.Spectrum>`
.. |TransmissionSpectrum| replace:: :class:`TransmissionSpectrum <optrace.tracer.spectrum.transmission_spectrum.TransmissionSpectrum>`
.. |RaySource| replace:: :class:`RaySource <optrace.tracer.geometry.ray_source.RaySource>`
   
LightSpectrum
______________________

A |LightSpectrum| defines wavelength-dependent emittance of light, with all spectral values being :math:`geq 0`.
|LightSpectrum| objects are needed when creating a |RaySource| or to render spectral distribution of detectors.

Creating the Spectrum
#########################


**Units**

For line spectra (modes :python:`"Monochromatic"` and :python:`"Lines"`) the spectral unit are Watts, while for all other modes the spectral power density is given as unit W/nm.
Distribution and shape parameters (:python:`val, line_vals, ...`) are given in the same units.
For spectra used in raytracing, the absolute scaling is unimportant, as the values will be rescaled by the parent ray source, which specifies the overall power.

**Constant**

This mode defines a constant spectrum with value :python:`val`:

.. testcode::
    
   spec = ot.LightSpectrum("Constant", val=12.3)

**Monochromatic**

This mode implements a monochromatic source with wavelength :python:`wl`.

.. testcode::
    
   spec = ot.LightSpectrum("Monochromatic", wl=423.56, val=3)

**Lines**

A line spectrum consist of multiple monochromatic sources.
The argument :python:`lines` is a list of wavelengths, while :python:`line_vals` is a list with the same number of elements describing the height/power of each wavelength.

.. testcode::

   spec = ot.LightSpectrum("Lines", lines=[458, 523, 729.6], line_vals=[0.5, 0.2, 0.1])


**Rectangle**

The following equations defines a spectrum with a rectangular function with bounds :python:`wl0`, :python:`wl1` and a scaling factor :python:`val`:

.. testcode::
    
   spec = ot.LightSpectrum("Rectangle", wl0=520, wl1=689, val=0.15)


**Gaussian**

A Gaussian spectrum is modelled mathematically with a scaling factor :math:`S_0`, a center wavelength :math:`\lambda_0` and a standard deviation :math:`\sigma_\lambda`:

.. math::
   S(\lambda) = S_0 \exp \left( -\frac{\left(\lambda - \lambda_0\right)^2}{2 \sigma_\lambda^2}\right)
   :label: eq_spectrum_gauss

The spectrum object is created with mode :python:`"Gaussian"`, a mean value :python:`mu` and standard deviation :python:`sig`, all given in nanometers.
Note that the Gaussian function will be truncated to the visible range [380nm, 780nm].

.. testcode::
    
   spec = ot.LightSpectrum("Gaussian", mu=478, sig=23.5, val=0.89)


**Blackbody Radiator**


The spectral radiance of a blackbody according to Planck's Law is given as: :footcite:`PlanckWiki`

.. math::
   B_\lambda (\lambda, ~T) = \frac{2 h c^2}{\lambda^5} \frac{1}{\exp\left(\frac{h  c } {\lambda k_\text{B}  T}\right) - 1}
   :label: planck_radiator

The equation contains the speed of light :math:`c`, the Planck constant :math:`h` and the Boltzmann constant :math:`k_\text{B}`:

.. math::
   c =&~ 299792458 ~\text{m/s}\\
   h =&~ 6.62607015\cdot 10^{-34} ~\text{J s}\\
   k_\text{B} =&~ 1.380649 \cdot 10^{-23} ~\text{J/K}\\

Note that :math:`\lambda` must be specified in meters in the above equation.


.. topic:: Note

   The spectral radiance :math:`B_\lambda` (Power per solid angle, source area and wavelength) is given in units :math:`\text{W}/(\text{m}^3~\text{sr})`, whereas the units in this class should be :math:`\text{W/nm}` (Power per wavelength). Since :math:`B_\lambda` is constant over the source area and angle independent, converting it corresponds to a simple rescaling. 


There is an option to normalize the spectrum, so the peak value equals one.
This can prove useful for plotting the spectrum.
If the peak wavelength is inside the visible range, then the Stefan–Boltzmann law can be applied to calculate the normalization factor.
Otherwise the maximum value will lie at one of the edges of the visible range.

A blackbody radiator, following Planck's law, with a specific temperature of :python:`T` in Kelvin, is initialized as:

.. testcode::
    
   spec = ot.LightSpectrum("Blackbody", T=3890, val=2)

The :python:`val` parameter defines the peak value in W/nm.

**User Function/Data**

With the Data/Function mode, the spectrum is modelled by a user function/ data set. 
With a dataset, the data will be linearly interpolated.

This function requires a wavelength array in nm as input and returns a numpy array of the same shape.

.. testcode::
    
   spec = ot.LightSpectrum("Function", func=lambda wl: np.arctan(wl - 520)**2)


If a function with multiple parameters is utilized, additional arguments can be provided in the :python:`func_args` parameter dictionary.

.. testcode::
    
   spec = ot.LightSpectrum("Function", func=lambda wl, c: np.arctan(wl - c)**2, func_args=dict(c=489))

For discrete datasets, the :python:`"Data"` mode proves useful. 
In this case the |LightSpectrum| constructor takes a wavelength array :python:`wls` and a value array :python:`vals` as arguments, where both must be of the exact same one-dimensional shape.

.. testcode::
    
   wls = np.linspace(450, 600, 100)
   vals = np.cos(wls/500)

   spec = ot.LightSpectrum("Data", wls=wls, vals=vals)

Note that :python:`wls` needs to be monotonically increasing with the same step size and needs to be inside the visible range [380nm, 780nm].


**Histogram**

This spectrum type is not user created, but is rendered on a detector or source. 
It consists of a list of bins and bin values.

Calculating Spectral Values
##############################

The |LightSpectrum| object can be called with a wavelength array to calculate the spectral values:

.. doctest::

   >>> wl = np.linspace(400, 500, 5)
   >>> spec(wl)
   array([0.        , 0.        , 0.62160997, 0.58168242, 0.54030231])


Wavelength Characteristics
###############################


.. list-table:: Wavelength characteristics functions
   :widths: 120 50 250
   :header-rows: 1
   :align: center
   
   * - Function
     - Unit
     - Meaning
   * - :meth:`peak_wavelength <optrace.tracer.spectrum.light_spectrum.LightSpectrum.peak_wavelength>`
     - nm
     - wavelength for the spectral peak
   * - :meth:`centroid_wavelength <optrace.tracer.spectrum.light_spectrum.LightSpectrum.centroid_wavelength>`
     - nm
     - power-weighted average wavelength, see `Centroid Wavelength <https://en.wikipedia.org/wiki/Spectral_centroid>`__
   * - :meth:`fwhm <optrace.tracer.spectrum.light_spectrum.LightSpectrum.fwhm>`
     - nm
     - full-width-at-half-maximum wavelength range
   * - :meth:`dominant_wavelength <optrace.tracer.spectrum.light_spectrum.LightSpectrum.dominant_wavelength>`
     - nm
     - | same hue wavelength, see `Dominant Wavelength <https://en.wikipedia.org/wiki/Dominant_wavelength>`__
       | :python:`np.nan` if not existent
   * - :meth:`complementary_wavelength <optrace.tracer.spectrum.light_spectrum.LightSpectrum.complementary_wavelength>`
     - nm
     - | opposite hue wavelength, see `Dominant Wavelength <https://en.wikipedia.org/wiki/Dominant_wavelength>`__
       | :python:`np.nan` if non-existent

For instance, we can calculate the peak wavelength of the LED B1 standard illuminant by doing:

.. doctest::

   >>> spec = ot.presets.light_spectrum.led_b1
   >>> spec.peak_wavelength()
   605.00225...

Note that with multiple same height peaks or a broad constant peak region the first peak value will be returned.

The centroid wavelength for the spectrum is:

.. doctest::

   >>> spec.centroid_wavelength()
   592.39585...

The dominant wavelength is calculated using:

.. doctest::

   >>> spec.dominant_wavelength()
   584.75088...

When dominant or complementary are not existent (e.g. magenta can't be described by a wavelength), the values are set to NaN (not a number).
You can find a visualization on both dominant and complementary wavelengths `on this Wiki page <https://en.wikipedia.org/wiki/Dominant_wavelength>`__.

The FWHM (full width at half maximum) is calculated usin:

.. doctest::

   >>> spec.fwhm()
   129.18529...


The method calculates the smallest FWHM around the highest peak.
While it is possible to calculate this value for all spectral shapes, it is only meaningful as width characterization for functions with a distinctive peak and an outward fall-off.
For instance this metric does not make sense for a spectrum consisting of multiple separated bell-shaped curves.


Power
#############

The spectral power can be calculated with:

.. doctest::

   >>> spec.power()
   3206.9749...

And the luminous power in lumen units with:

.. doctest::

   >>> spec.luminous_power()
   999886.86...


Rendering a LightSpectrum
#################################

Read section :ref:`rimage_rendering` for details on rendering images, rendering spectra is done in a similar way.
Analogously to rendering a source image, we can render a spectrum with :meth:`source_spectrum <optrace.tracer.raytracer.Raytracer.source_spectrum>` and by providing a :python:`source_index` parameter (default to zero).
With a raytracer object :python:`RT`, a source spectrum from source 1 is rendered with:

.. code-block:: python

   spec = RT.source_spectrum(source_index=1)

For a detector spectrum the :meth:`detector_spectrum <optrace.tracer.raytracer.Raytracer.detector_spectrum>` function is applied. 
It takes a :python:`detector_index` argument, that defaults to zero.

.. code-block:: python

   spec = RT.detector_spectrum(detector_index=0)

Additionally we can render only a specific source by providing a :python:`source_index` or limit the detector area by providing the :python:`extent` parameter, as we did for the :meth:`detector_image <optrace.tracer.raytracer.Raytracer.detector_image>` method.

.. code-block:: python

   spec = RT.detector_spectrum(detector_index=0, source_index=1, extent=[0, 1, 0, 1])

The above methods return a :class:`LightSpectrum <optrace.tracer.spectrum.light_spectrum.LightSpectrum>` object with type :python:`spectrum_type="Histogram"`.


TransmissionSpectrum
______________________

The :class:`Filter <optrace.tracer.geometry.filter.Filter>` class requires a |TransmissionSpectrum|. 
All relative transmission values need to be inside the [0, 1] range.
The |TransmissionSpectrum| provides less modes than the |LightSpectrum| class. 
Compared to the latter, the scaling factor :python:`vall` now becomes important.
This class defines a new :python:`inverse` parameter, that subtracts the defined function from a value of one. 
This has the effect of turning the transmittance behavior into absorptance. 
A Gaussian bandpass becomes a notch filter, a rectangular bandpass a rectangular blocking filter.

**Constant**

A neutral density filter is defined with mode :python:`"Constant"` and the linear transmittance value.

.. testcode::
    
   spec = ot.TransmissionSpectrum("Constant", val=0.5)

**Gaussian**

Colored filters (most commonly bandpass filters) can be created with a Gaussian function.

.. testcode::
    
   spec = ot.TransmissionSpectrum("Gaussian", mu=550, sig=30, val=1)

A Gaussian notch filter is easily defined with parameter :python:`inverse=True`.

.. testcode::
    
   spec = ot.TransmissionSpectrum("Gaussian", mu=550, sig=30, val=1, inverse=True)

**Rectangle**

A rectangular pass filter is modelled by a rectangular function.

.. testcode::
    
   spec = ot.TransmissionSpectrum("Rectangle", wl0=500, wl1=650, val=0.1)

A rectangular blocking filter can be defined with :python:`inverse=True`.

.. testcode::
    
   spec = ot.TransmissionSpectrum("Rectangle", wl0=500, wl1=650, inverse=True)

Creating an edgepass filter becomes easy by setting the bound to the edge of the visible range.

.. testcode::
    
   spec = ot.TransmissionSpectrum("Rectangle", wl0=500, wl1=780)


**User Data/Function**

Creating a |TransmissionSpectrum| with discrete data is equivalent to a |LightSpectrum|.
However, all function/data values need to be inside the range [0, 1].

**Getting Spectral Values**

As for the |LightSpectrum| object, we can get the spectral values with:

.. doctest::

   >>> wl = np.linspace(400, 550, 5)
   >>> spec(wl)
   array([0., 0., 0., 1., 1.])

Spectrum
______________________

|Spectrum| is the parent class of both |LightSpectrum| and |TransmissionSpectrum|. 
It defines the following modes: :python:`"Monochromatic", "Rectangle", "List", "Function", "Data", "Gaussian", "Constant"`. 
Compared to |LightSpectrum|, only modes :python:`"Histogram"` and :python:`"Blackbody"` are missing.
Generally the |Spectrum| class is not exposed to the user. 
But, for instance, the color matching functions :python:`ot.presets.spectrum.x, ot.presets.spectrum.y, ot.presets.spectrum.z` are objects of this type.


Plotting
________________

See :ref:`spectrum_plots`.


.. _spectral_lines:

Spectral Lines Presets
______________________

optrace provides some spectral wavelength lines in its presets.

.. list-table:: Fraunhofer lines commonly used for Abbe number determination :footcite:`AbbeWiki`
   :widths: 70 70 70 70
   :header-rows: 1
   :align: center
   
   * - Name
     - | Wavelength 
       | in nm
     - Element
     - Color
   * - h
     - 404.6561
     - Hg
     - violet
   * - g
     - 435.8343
     - Hg
     - blue
   * - F'
     - 479.9914
     - Cd
     - blue
   * - F
     - 486.1327
     - H
     - blue
   * - e
     - 546.0740
     - Hg
     - green
   * - d
     - 587.5618
     - He
     - yellow
   * - D
     - 589.2938
     - Na
     - yellow
   * - C'
     - 643.8469
     - Cd
     - red
   * - C
     - 656.272
     - H
     - red
   * - r
     - 706.5188
     - He
     - red
   * - A'
     - 768.2
     - K
     - IR-A

Due to limitations in python variable names, presets with a trailing apostrophe are named with an trailing underscore.
For instance, F' is named :python:`F_`.

.. doctest::
    
   >>> ot.presets.spectral_lines.F_
   479.9914

The most common wavelength combinations for Abbe numbers are FdC, FDC, FeC and F'eC'.

.. doctest::
    
   >>> ot.presets.spectral_lines.F_eC_
   [479.9914, 546.074, 643.8469]

The following table provides the dominant wavelengths of the sRGB primaries (ITU-R BT.709). 
Dimensioning the scaling factors in the provided way produces D65 sRGB-white for equal R, G, B mixing ratios.

.. list-table:: Dominant wavelengths of sRGB primaries. Derived by optimization. 
   :widths: 70 70 70
   :header-rows: 1
   :align: center
   
   * - Name
     - | Wavelength 
       | in nm
     - Scaling Factor
   * - R
     - 611.2826
     - 0.5745000
   * - G
     - 549.1321
     - 0.5985758
   * - B
     - 464.3118
     - 0.3895581

These wavelengths are useful for simulating color mixing.

.. doctest::
    
   >>> ot.presets.spectral_lines.rgb
   [464.3118, 549.1321, 611.2826]

Spectrum Presets
______________________

The following figures demonstrate the predefined presets for |Spectrum| and |LightSpectrum|.

.. list-table::
   :widths: 500 500
   :class: table-borderless

   * - .. figure:: ../images/Standard_illuminants.svg
          :width: 500
          :align: center
          :class: dark-light
          
          CIE standard illuminants. Available as :python:`ot.presets.light_spectrum.<name>` with :python:`a, d50, ...` as :python:`<name>`

     - .. figure:: ../images/LED_illuminants.svg
          :width: 500
          :align: center
          :class: dark-light
           
          CIE standard illuminants LED series. Available as :python:`ot.presets.light_spectrum.<name>` with :python:`led_b1, led_b2, ...` as :python:`<name>`
           
   * - .. _fig_led_illuminants:

       .. figure:: ../images/Fluor_illuminants.svg
          :width: 500
          :align: center
          :class: dark-light
         
          CIE standard illuminants Fluorescent series. Available as :python:`ot.presets.light_spectrum.<name>` with :python:`fl2, fl7, ...` as :python:`<name>`

     - .. figure:: ../images/srgb_spectrum.svg
          :width: 500
          :align: center
          :class: dark-light
         
          Possible sRGB primary spectra.
          Available as :python:`ot.presets.light_spectrum.<name>` with :python:`srgb_r, srgb_g, ...` as :python:`<name>`

   * - .. figure:: ../images/cie_cmf.svg
          :width: 500
          :align: center
          :class: dark-light
         
          CIE color matching functions.
          Available as :python:`ot.presets.spectrum.<name>` with :python:`x, y, z` as :python:`<name>`
     
     - 
  

Other presets include spectra from spectral lines combination in :numref:`spectral_lines`. Namely :python:`ot.presets.light_spectrum.<name>` with :python:`FdC, FDC, FeC, F_eC_, rgb` as :python:`<name>`.


------------

**References**

.. footbibliography::


