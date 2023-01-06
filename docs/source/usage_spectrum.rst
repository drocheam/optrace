Spectrum (LightSpectrum, TransmissionSpectrum)
--------------------------------------------------

.. testsetup:: *

   import optrace as ot
   import numpy as np


LightSpectrum
______________________

A LightSpectrum defines emittance or a similar quantity for light output depending on the wavelength. All spectral valuues need be greater or equal to zero.

LightSpectrum objects are used when creating RaySources or when we want to render the spectral distribution of light hitting a detector.

**Constant**

A constant (or uniform) LightSpectrum is defined using:

.. testcode::
    
   spec = ot.LightSpectrum("Constant")

**Monochromatic**

We can also define a spectrum with only a single wavelength:

.. testcode::
    
   spec = ot.LightSpectrum("Monochromatic", wl=423.56)

**Lines**

Multiple spectral lines are created with mode ``"Lines"``.
Argument ``lines`` is a list of wavelengths, while ``line_vals`` is a list with the same number of elements describing the height/power of each wavelength.

.. testcode::

   spec = ot.LightSpectrum("Lines", lines=[458, 523, 729.6], line_vals=[0.5, 0.2, 0.1])


**Rectangle**

A rectangular window is defined with ``"Rectangle"`` and lower and upper wavelength bounds.

.. testcode::
    
   spec = ot.LightSpectrum("Rectangle", wl0=520, wl1=689)


**Gaussian**

A gaussian function can be created with ``"Gaussian"``, a mean value ``mu`` and standard deviation ``sig``, all given in nanometers.
Note that the gaussian function will be truncated to the visible range [380nm, 780nm].

.. testcode::
    
   spec = ot.LightSpectrum("Gaussian", mu=478, sig=23.5)


**Blackbody Radiator**

A blackbody radiator, following Planck's law, with a specific temperature ``T`` in Kelvin is initialized with:

.. testcode::
    
   spec = ot.LightSpectrum("Blackbody", T=3890)

For the user it is also possible to create an own function with the ``func`` parameter. This function must take wavelength array in nm as input and also return a numpy array with the same shape.

**User Function**

.. testcode::
    
   spec = ot.LightSpectrum("Function", func=lambda wl: np.arctan(wl - 520)**2)


If a function with multiple parameters is utilized, additional arguments can be put in the ``func_args`` parameter dictionary.

.. testcode::
    
   spec = ot.LightSpectrum("Function", func=lambda wl, c: np.arctan(wl - c)**2, func_args=dict(c=489))

For discrete datasets the ``"Data"`` mode proves useful. In this case the ``LightSpectrum()`` constructor takes a wavelength array ``wls`` and a value array ``vals``, both being the same shape and one dimensional numpy arrays.

.. testcode::
    
   wls = np.linspace(450, 600, 100)
   vals = np.cos(wls/500)

   spec = ot.LightSpectrum("Data", wls=wls, vals=vals)

Note that ``wls`` needs to be monotonically increasing with the same step size and needs to be inside the visible range [380nm, 780nm].


**Histogram**

This spectrum type generally is not user created, but is rendered on a detector or source. It consists of a list of bins and bin values.

**Rescaling Factor**

Modes ``Constant, Rectangle, Gaussian`` also have a ``vals`` parameter that is a rescaling factor for the function. For tracing it is irrelevant, as the function is rescaled automatically to match the power specified in the RaySource.
However, for plotting the spectrum or for the TransmissionSpectrum the parameter will prove useful.

**Getting Spectral Values**

The LightSpectrum object can be called with wavelengths to get the spectral values:

.. doctest::

   >>> wl = np.linspace(400, 500, 5)
   >>> spec(wl)
   array([0.        , 0.        , 0.62160997, 0.58168242, 0.54030231])


TransmissionSpectrum
______________________

A TransmissionSpectrum is applied as filter function for a Filter element. All transmission values need to be inside the [0, 1] range.

The TransmissionSpectrum provides less modes than the LightSpectrum class. Note that now the scaling factor ``vall`` becomes important.

This class defines a new ``inverse`` parameter, that subtracts the defined function from a value of one. This has the effect that the function instead does not define the transmittance behavior, but the absorption one. A gaussian bandpass becomes a notch filter, a rectangular bandpass a rectangular blocking one.

**Constant**

A neutral density filter is defined with mode ``"Constant"`` and the linear transmittance value.

.. testcode::
    
   spec = ot.TransmissionSpectrum("Constant", val=0.5)

**Gaussian**

Colored filters (or bandpass filters) are often similar to a Gaussian function.

.. testcode::
    
   spec = ot.TransmissionSpectrum("Gaussian", mu=550, sig=30, val=1)

A gaussian notch filter can be defined with ``inverse=True``.

.. testcode::
    
   spec = ot.TransmissionSpectrum("Gaussian", mu=550, sig=30, val=1, inverse=True)

**Rectangle**

A rectangular pass filter can be modelled by a rectangular function.

.. testcode::
    
   spec = ot.TransmissionSpectrum("Rectangle", wl0=500, wl1=650, val=0.1)

A rectangular blocking filter can be defined with ``inverse=True``.

.. testcode::
    
   spec = ot.TransmissionSpectrum("Rectangle", wl0=500, wl1=650, inverse=True)

An edgepass filter can be created by simply setting one of the bounds to the bound of the visible range.

.. testcode::
    
   spec = ot.TransmissionSpectrum("Rectangle", wl0=500, wl1=780)


**User Data/Function**

Creating a TransmissionSpectrum with discrete data or a user function works exactly like for the LightSpectrum, however all function/data values need to be inside range [0, 1].

**Getting Spectral Values**

As for the LightSpectrum object we can get the spectral values with:

.. doctest::

   >>> wl = np.linspace(400, 550, 5)
   >>> spec(wl)
   array([0. , 0. , 0. , 1., 1.])

Spectrum
______________________

This is the parent class of both ``LightSpectrum, TransmissionSpectrum``. It defines the following modes: ``Monochromatic, Rectangle, List, Function, Data, Gaussian, Constant``. Compared to ``LightSpectrum`` modes only ``"Histogram"`` and ``"Blackbody"`` are missing.
Generally the ``Spectrum`` class is not used by the user. But for instance the color matching functions ``ot.presets.spectrum.x, ot.presets.spectrum.y, ot.presets.spectrum.z`` are objects of this class.


Presets
______________________


Below you can find some predefined presets for ``Spectrum, LightSpectrum``.

.. figure:: images/Standard_illuminants.svg
   :width: 600
   :align: center
  
   CIE standard illuminants. Available as ``ot.presets.light_spectrum.<name>`` with ``a, d50, ...`` as ``<name>``

.. figure:: images/LED_illuminants.svg
   :width: 600
   :align: center
   
   CIE standard illuminants LED series. Available as ``ot.presets.light_spectrum.<name>`` with ``led_b1, led_b2, ...`` as ``<name>``
   
.. figure:: images/Fluor_illuminants.svg
   :width: 600
   :align: center
  
   CIE standard illuminants Fluorescent series. Available as ``ot.presets.light_spectrum.<name>`` with ``fl2, fl7, ...`` as ``<name>``

.. figure:: images/srgb_spectrum.svg
   :width: 600
   :align: center
  
   Possible sRGB primary spectra.
   Available as ``ot.presets.light_spectrum.<name>`` with ``srgb_r, srgb_g, ...`` as ``<name>``

.. figure:: images/cie_cmf.svg
   :width: 600
   :align: center
  
   CIE color matching functions.
   Available as ``ot.presets.spectrum.<name>`` with ``x, y, z`` as ``<name>``
  
Other presets include spectra from spectral lines combination in :numref:`spectral_lines`. Namely ``ot.presets.light_spectrum.<name>`` with ``FdC, FDC, FeC, F_eC_`` as ``<name>``.
