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

A |LightSpectrum| defines emittance or a similar quantity for light output depending on the wavelength. All spectral values need be greater or equal to zero.
|LightSpectrum| objects are used when creating a |RaySource| or when we want to render the spectral distribution of light hitting a detector.

Creating the Spectrum
#########################


**Units**

For line spectra (modes :python:`"Monochromatic"` and :python:`"Lines"`) the spectral unit is W, while for all other modes the spectral power density with unit W/nm.
The actual values and height parameters (:python:`val, line_vals, ...`) are therefore given in the same unit.
For spectra used in raytracing the absolute height is unimportant, as the function gets rescaled to the correct power by the parent ray source.

**Constant**

This defines a constant spectrum with value :math:`S_0`:

.. math::
   S(\lambda) = S_0
   :label: eq_spectrum_const

The |LightSpectrum| is initialized in the following, with :python:`val` as :math:`S_0`:

.. testcode::
    
   spec = ot.LightSpectrum("Constant", val=12.3)

**Monochromatic**

This implements a monochromatic source with wavelength :math:`\lambda_0`.

.. math::
   S(\lambda) = S_0\, \delta(\lambda - \lambda_0)
   :label: eq_spectrum_mono

.. testcode::
    
   spec = ot.LightSpectrum("Monochromatic", wl=423.56, val=3)

**Lines**

A line spectrum consist of multiple monochromatic sources, parameters are a set of power and wavelength combinations :math:`L=\left\{(P_1,~\lambda_1),~(P_2,~\lambda_2),~\dots\right\}`.

.. math::
   S(\lambda) = \sum_{(S_i, ~\lambda_i) \in L}  S_i \, \delta(\lambda - \lambda_i)
   :label: eq_spectrum_lines

Argument :python:`lines` is a list of wavelengths, while :python:`line_vals` is a list with the same number of elements describing the height/power of each wavelength.

.. testcode::

   spec = ot.LightSpectrum("Lines", lines=[458, 523, 729.6], line_vals=[0.5, 0.2, 0.1])


**Rectangle**

Spectrum with a rectangular function :math:`\Pi(\lambda)` with bounds :math:`\lambda_0,~\lambda_1` and a scaling factor :math:`S_0`.

.. math::
   S(\lambda) = S_0\, \Pi_{[\lambda_0,~\lambda_1]}(\lambda)
   :label: eq_spectrum_rect

.. testcode::
    
   spec = ot.LightSpectrum("Rectangle", wl0=520, wl1=689, val=0.15)


**Gaussian**

A gaussian spectrum is modelled with a scaling factor :math:`S_0`, a center wavelength :math:`\lambda_0` and a standard deviation :math:`\lambda_\sigma`.

.. math::
   S(\lambda) = S_0 \exp \left( -\frac{\left(\lambda - \lambda_0\right)^2}{2 \lambda^2_\sigma}\right)
   :label: eq_spectrum_gauss

When programming, the gaussian function is created with :python:`"Gaussian"`, a mean value :python:`mu` and standard deviation :python:`sig`, all given in nanometers.
Note that the gaussian function will be truncated to the visible range [380nm, 780nm].

.. testcode::
    
   spec = ot.LightSpectrum("Gaussian", mu=478, sig=23.5, val=0.89)


**Blackbody Radiator**


The spectral radiance for a blackbody according to Planck's Law is: :footcite:`PlanckWiki`

.. math::
   B_\lambda (\lambda, ~T) = \frac{2 h c^2}{\lambda^5} \frac{1}{\exp\left(\frac{h  c } {\lambda k_\text{B}  T}\right) - 1}
   :label: planck_radiator

The equation contains the speed of light :math:`c`, the Planck constant :math:`h` and the Boltzmann constant :math:`k_\text{B}`:

.. math::
   c =&~ 299792458 ~\text{m/s}\\
   h =&~ 6.62607015\cdot 10^{-34} ~\text{J s}\\
   k_\text{B} =&~ 1.380649 \cdot 10^{-23} ~\text{J/K}\\

Note that :math:`\lambda` must be specified in meters.


.. topic:: Note

   The spectral radiance :math:`B_\lambda` (Power per solid angle, source area and wavelength) is given in units :math:`\text{W}/(\text{m}^3~\text{sr})`, whereas the units in this class should be :math:`\text{W/nm}` (Power per wavelength). Since :math:`B_\lambda` is constant over the source area and angle independent, converting it corresponds to a simple rescaling. 
   This is done while raytracing, where a specfic desired power is matched.


There is an option to normalize the spectrum, such that the peak value inside the visible range is equal to one.
If the peak wavelength is inside the visible range then to Stefan–Boltzmann law can be applied to calculate the normalization factor, otherwise the maximum value should lie at one of the edges of the visible range.

A blackbody radiator, following Planck's law, with a specific temperature :python:`T` in Kelvin is initialized with:

.. testcode::
    
   spec = ot.LightSpectrum("Blackbody", T=3890, val=2)

The :python:`val` parameter defines the peak value in W/nm.

**User Function/Data**

With Data/Function mode the spectrum is simply modelled by a user function/ data set. With a data set data is linearly interpolated.

.. math::
   S(\lambda) = S_F(\lambda)
   :label: eq_spectrum_user

This function must take wavelength array in nm as input and also return a numpy array with the same shape.

.. testcode::
    
   spec = ot.LightSpectrum("Function", func=lambda wl: np.arctan(wl - 520)**2)


If a function with multiple parameters is utilized, additional arguments can be put in the :python:`func_args` parameter dictionary.

.. testcode::
    
   spec = ot.LightSpectrum("Function", func=lambda wl, c: np.arctan(wl - c)**2, func_args=dict(c=489))

For discrete datasets the :python:`"Data"` mode proves useful. In this case the |LightSpectrum| constructor takes a wavelength array :python:`wls` and a value array :python:`vals`, both being the same shape and one dimensional numpy arrays.

.. testcode::
    
   wls = np.linspace(450, 600, 100)
   vals = np.cos(wls/500)

   spec = ot.LightSpectrum("Data", wls=wls, vals=vals)

Note that :python:`wls` needs to be monotonically increasing with the same step size and needs to be inside the visible range [380nm, 780nm].


**Histogram**

This spectrum type generally is not user created, but is rendered on a detector or source. It consists of a list of bins and bin values.

Getting Spectral Values
#########################

The |LightSpectrum| object can be called with wavelengths to get the spectral values:

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
     - Wavelength with the spectrum peak
   * - :meth:`centroid_wavelength <optrace.tracer.spectrum.light_spectrum.LightSpectrum.centroid_wavelength>`
     - nm
     - power-weighted average wavelength, see `Centroid Wavelength <https://en.wikipedia.org/wiki/Spectral_centroid>`__
   * - :meth:`fwhm <optrace.tracer.spectrum.light_spectrum.LightSpectrum.fwhm>`
     - nm
     - full width half maximum wavelength range
   * - :meth:`dominant_wavelength <optrace.tracer.spectrum.light_spectrum.LightSpectrum.dominant_wavelength>`
     - nm
     - | wavelength with the same hue as the spectrum, see `Dominant Wavelength <https://en.wikipedia.org/wiki/Dominant_wavelength>`__
       | NaN if not existent
   * - :meth:`complementary_wavelength <optrace.tracer.spectrum.light_spectrum.LightSpectrum.complementary_wavelength>`
     - nm
     - | wavelength with the opposite hue as the spectrum, see `Dominant Wavelength <https://en.wikipedia.org/wiki/Dominant_wavelength>`__
       | NaN if not existent

As an example we can load the LED B1 standard illuminant, that can also be seen in :numref:`fig_led_illuminants`.
Then the peak wavelength is calculated with:

.. doctest::

   >>> spec = ot.presets.light_spectrum.led_b1
   >>> spec.peak_wavelength()
   605.00225...

Note that with multiple same height peaks or a broad constant peak region the first peak value is returned. However, due to numerical precision this is not always the case.
In our example the power-weighted average wavelength (centroid) is different from this:

.. doctest::

   >>> spec.centroid_wavelength()
   592.39585...

The dominant wavelength is calculated using:

.. doctest::

   >>> spec.dominant_wavelength()
   584.75088...

When dominant or complementary are not existent, as for instance magenta can't be described by a single wavelength, the values are set to NaN (not a number).
You can find a visual explanation on both dominant and complementary wavelength `on this Wiki page <https://en.wikipedia.org/wiki/Dominant_wavelength>`__.

The FWHM (full width at half maximum) can be calculated with:

.. doctest::

   >>> spec.fwhm()
   129.18529...


The function calculates the smallest FWHM around the highest peak. Note that for some spectral distributions, for instance multiple gaussians, this function is not suitable, as the FWHM is not meaningful here.


Power
#############

The spectral power in W can be calculated with:

.. doctest::

   >>> spec.power()
   3206.9749...

And the luminous power in lumens with:

.. doctest::

   >>> spec.luminous_power()
   999886.86...


Rendering a LightSpectrum
#################################

Rendering a light spectrum is  done on the ray source or detector surface.
Read section :ref:`rimage_rendering` for details on rendering images, rendering spectra is done in a similar way.
Analogously to rendering a source image, we can render a spectrum with :meth:`source_spectrum <optrace.tracer.raytracer.Raytracer.source_spectrum>` and by providing a :python:`source_index` parameter (default to zero).
With a raytracer object called :python:`RT` a source spectrum is rendered with:

.. code-block:: python

   spec = RT.source_spectrum(source_index=1)

For a detector spectrum the :meth:`detector_spectrum <optrace.tracer.raytracer.Raytracer.detector_spectrum>` function is applied. It takes a :python:`detector_index` argument, that also defaults to zero.

.. code-block:: python

   spec = RT.detector_spectrum(detector_index=0)

Additionally we can limit the rendering to a source by providing a :python:`source_index` or limit the detector area by providing the :python:`extent` parameter, as we did for the :meth:`detector_image <optrace.tracer.raytracer.Raytracer.detector_image>`.

.. code-block:: python

   spec = RT.detector_spectrum(detector_index=0, source_index=1, extent=[0, 1, 0, 1])

The above methods return an object of type :class:`LightSpectrum <optrace.tracer.spectrum.light_spectrum.LightSpectrum>` with :python:`spectrum_type="Histogram"`.

TransmissionSpectrum
______________________

A TransmissionSpectrum is applied as filter function for a Filter element. All transmission values need to be inside the [0, 1] range.
The TransmissionSpectrum provides less modes than the LightSpectrum class. Note that now the scaling factor :python:`vall` becomes important.
This class defines a new :python:`inverse` parameter, that subtracts the defined function from a value of one. This has the effect that the function instead does not define the transmittance behavior, but the absorption one. A gaussian bandpass becomes a notch filter, a rectangular bandpass a rectangular blocking one.

**Constant**

A neutral density filter is defined with mode :python:`"Constant"` and the linear transmittance value.

.. testcode::
    
   spec = ot.TransmissionSpectrum("Constant", val=0.5)

**Gaussian**

Colored filters (or bandpass filters) are often similar to a Gaussian function.

.. testcode::
    
   spec = ot.TransmissionSpectrum("Gaussian", mu=550, sig=30, val=1)

A gaussian notch filter can be defined with :python:`inverse=True`.

.. testcode::
    
   spec = ot.TransmissionSpectrum("Gaussian", mu=550, sig=30, val=1, inverse=True)

**Rectangle**

A rectangular pass filter can be modelled by a rectangular function.

.. testcode::
    
   spec = ot.TransmissionSpectrum("Rectangle", wl0=500, wl1=650, val=0.1)

A rectangular blocking filter can be defined with :python:`inverse=True`.

.. testcode::
    
   spec = ot.TransmissionSpectrum("Rectangle", wl0=500, wl1=650, inverse=True)

An edgepass filter can be created by simply setting one of the bounds to the bound of the visible range.

.. testcode::
    
   spec = ot.TransmissionSpectrum("Rectangle", wl0=500, wl1=780)


**User Data/Function**

Creating a |TransmissionSpectrum| with discrete data or a user function works exactly like for the |LightSpectrum|, however all function/data values need to be inside range [0, 1].

**Getting Spectral Values**

As for the |LightSpectrum| object we can get the spectral values with:

.. doctest::

   >>> wl = np.linspace(400, 550, 5)
   >>> spec(wl)
   array([0., 0., 0., 1., 1.])

Spectrum
______________________

|Spectrum| is the parent class of both |LightSpectrum|, |TransmissionSpectrum|. It defines the following modes: :python:`"Monochromatic", "Rectangle", "List", "Function", "Data", "Gaussian", "Constant"`. Compared to |LightSpectrum| only modes :python:`"Histogram"` and :python:`"Blackbody"` are missing.
Generally the |Spectrum| class is not used by the user. But for instance the color matching functions  :python:`ot.presets.spectrum.x, ot.presets.spectrum.y, ot.presets.spectrum.z` are objects of this class.


Plotting
________________

See :ref:`spectrum_plots`.


.. _spectral_lines:

Spectral Lines Presets
______________________

`optrace` has some spectral wavelength lines defined in its presets.
While there are many such lines, only those relevant for the calculation of the Abbe number are built-in.

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

Due to limitations in python variable names, the presets with a trailing apostrophe are instead named with an trailing underscore, for instance F' is named :python:`F_`.

.. doctest::
    
   >>> ot.presets.spectral_lines.F_
   479.9914

The most common wavelength combinations for Abbe numbers are FdC, FDC, FeC and F'eC'.

.. doctest::
    
   >>> ot.presets.spectral_lines.F_eC_
   [479.9914, 546.074, 643.8469]

In the next table the dominant wavelengths of the sRGB primaries can be found. The dominant wavelength is the wavelength that produces a color with the same hue as the reference color.
The scaling factors are dimensioned such that the sum of these three monochromatic light sources produces sRGB-white.

.. list-table:: Dominant wavelengths of sRGB primaries. Own work. 
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

These wavelengths prove useful when trying to simulate color mixing.

.. doctest::
    
   >>> ot.presets.spectral_lines.rgb
   [464.3118, 549.1321, 611.2826]

Spectrum Presets
______________________


Below you can find some predefined presets for |Spectrum| and |LightSpectrum|.

.. list-table::
   :widths: 500 500
   :class: table-borderless

   * - .. figure:: ../images/Standard_illuminants.svg
          :width: 500
          :align: center
          :class: dark-light
          
          CIE standard illuminants. Available as ``ot.presets.light_spectrum.<name>`` with ``a, d50, ...`` as ``<name>``

     - .. figure:: ../images/LED_illuminants.svg
          :width: 500
          :align: center
          :class: dark-light
           
          CIE standard illuminants LED series. Available as ``ot.presets.light_spectrum.<name>`` with ``led_b1, led_b2, ...`` as ``<name>``
           
   * - .. _fig_led_illuminants:

       .. figure:: ../images/Fluor_illuminants.svg
          :width: 500
          :align: center
          :class: dark-light
         
          CIE standard illuminants Fluorescent series. Available as ``ot.presets.light_spectrum.<name>`` with ``fl2, fl7, ...`` as ``<name>``

     - .. figure:: ../images/srgb_spectrum.svg
          :width: 500
          :align: center
          :class: dark-light
         
          Possible sRGB primary spectra.
          Available as ``ot.presets.light_spectrum.<name>`` with ``srgb_r, srgb_g, ...`` as ``<name>``

   * - .. figure:: ../images/cie_cmf.svg
          :width: 500
          :align: center
          :class: dark-light
         
          CIE color matching functions.
          Available as ``ot.presets.spectrum.<name>`` with ``x, y, z`` as ``<name>``
     
     - 
  

Other presets include spectra from spectral lines combination in :numref:`spectral_lines`. Namely :python:`ot.presets.light_spectrum.<name>` with :python:`FdC, FDC, FeC, F_eC_, rgb` as :python:`<name>`.


------------

**References**

.. footbibliography::


