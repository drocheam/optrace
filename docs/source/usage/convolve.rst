.. _usage_convolution: 

PSF Convolution
------------------------------------------------------------------------

.. |RenderImage| replace:: :class:`RenderImage <optrace.tracer.image.render_image.RenderImage>`
.. |LinearImage| replace:: :class:`LinearImage <optrace.tracer.image.linear_image.LinearImage>`
.. |RGBImage| replace:: :class:`RGBImage <optrace.tracer.image.rgb_image.RGBImage>`

.. role:: python(code)
  :language: python
  :class: highlight


.. testsetup:: *

   import optrace as ot
   ot.global_options.show_progressbar = False


Overview
_______________

Convolution can speed up image formation significantly by parallelizing the effect of a PSF (point spread function) for every object point.
This results in a nearly noise-free image in a small fraction of the time normally required.
A PSF is nothing different than the impulse response of the optical system for the given object and image distance combination.
It changes spatially, but for a small angular range it can be assumed constant.
However this also implies that many aberration can't be simulated this way (astigmatism, coma, vignette, distortion, ...).
In cases where convolution would be viable a possible approach would be to render a PSF and then apply the PSF to an object using the convolution functionality in optrace.

The convolution approach requires:

* the PSF
* the object image to convolve
* PSF size
* object size
* the system's magnification factor

Both colored objects and PSFs are supported.
Note however, that without knowing the correct spectral distribution on both (instead only the values in a color space) the convolution only simulates one of many possible solutions. 
See the details in section :numref:`psf_color_handling`.

Usage
____________

**Simple Calls**

A simple call to :func:`convolve <optrace.tracer.convolve.convolve>` required the object and PSF, as well as the magnification factor. 

The object should be either of type |RGBImage| or |LinearImage|, while the PSF should be either |LinearImage| or |RenderImage|.
For more details on image classes see :numref:`image_classes`.
Colored PSFs can be only used with class |RenderImage|, as all human visible colors are required for convolution.
The image objects include the information about their size and position.

This magnification factor is equivalent the magnification factor known from geometrical optics.
Therefore :math:`\lvert m \rvert > 1` means a magnification and :math:`\lvert m\rvert < 1` an image shrinking.
:math:`m > 0` is an upright image, while :math:`m < 0` correspond to a flipped image.

Let's define a predefined a image and PSF:

.. testcode::

   img = ot.presets.image.ETDRS_chart_inverted([0.5, 0.5])
   psf = ot.presets.psf.halo()

You can then call :func:`convolve <optrace.tracer.convolve.convolve>` in the following way:

.. testcode::

   img2 = ot.convolve(img, psf, m=0.5)

The function returns the convolved image object :python:`img2`.
When :python:`img` and :python:`psf` are of type |LinearImage|, :python:`img2` is also a |LinearImage|.
For all other cases color information is generated and :python:`img2` is a |RGBImage|.

**Slicing and Padding**

While doing a convolution, the output image grows in size by half the PSF size in each direction.
By providing :python:`keep_size=True` the padded data can be neglected for the resulting image.

.. testcode::

   img2 = ot.convolve(img, psf, m=0.5, keep_size=True)

The convolution operation requires the data outside of the image.
By default, the image is padded with zeros before convolution.

Other modes are also available.
For instance, padding with white is done in the following fashion:

.. testcode::

   img2 = ot.convolve(img, psf, m=0.5, keep_size=True, padding_mode="constant", padding_value=[1, 1, 1])

:python:`padding_value` specifies the values used for constant padding for each channel.
Depending on type of :python:`img`, it needs to have three or only one element.

To reduce boundary effects, edge padding is a viable choice:

.. testcode::

   img2 = ot.convolve(img, psf, m=0.5, keep_size=True, padding_mode="edge")

**Color conversion**

The convolution of colored images can produce colors outside of the sRGB gamut.
To allow for a correct mapping into the gamut, conversion arguments can be provided by the :python:`cargs` argument.
By default it is set to :python:`dict(rendering_intent="Absolute", normalize=True, clip=True, L_th=0, chroma_scale=None)`.

Provide a :python:`cargs` dictionary to override this setting.

.. testcode::

   img2 = ot.convolve(img, psf, m=0.5, keep_size=True, padding_mode="edge", cargs=dict(rendering_intent="Perceptual"))

The above command overrides the :python:`rendering_intent` while leaving the other default options unchanged.

**Normalization**

When convolving two |LinearImage| objects it is recommended to normalize the PSF integral to 1.
Doing so, the overall power of the image is preserved.

Restrictions
_______________________

* it is not possible to convolve two |RGBImage| or two |RenderImage|
* image and PSF resolutions must be between 50x50 pixels and 4 megapixels
* the PSF needs to be twice as large as the image scaled with the magnification factor
* when convolving two colored images, the resulting image is only one possible solution of many
* :func:`scipy.signal.fftconvolve` is involved, so small numerical errors in dark image regions can appear
* convolution of sphere projected images (see :numref:`image_sphere_projections`) is prohibited, as distances are non-linear

Examples
__________________________

**Image Example**


.. list-table:: Image convolution from the :ref:`example_psf_imaging` example
   :class: table-borderless

   * - .. figure:: ../images/example_psf1.svg
          :align: center
          :width: 400
          :class: dark-light

   
     - .. figure:: ../images/example_psf2.svg
          :align: center
          :width: 400
          :class: dark-light


.. figure:: ../images/example_psf3.svg
   :align: center
   :width: 400
   :class: dark-light



**Code Example**


The following example loads an image preset and convolves it with a square PSF created as a numpy array.

.. testcode::
  
   import numpy as np

   # load image preset
   img = ot.presets.image.ETDRS_chart_inverted([0.9, 0.9])

   # square psf
   psf_data = np.zeros((200, 200))
   psf_data[50:150, 50:150] = 1

   psf = ot.LinearImage(psf_data, [0.1, 0.08])

   # convolution
   img2 = ot.convolve(img, psf, m=-1.75)


Presets
_____________________

The are multiple PSF presets available.

All presets are normalized such that the integral image sum equals 1.

**Circle**

A circular PSF is defined with the :python:`d` circle parameter.

.. testcode::

   psf = ot.presets.psf.circle(d=3.5) 

**Gaussian**

A simple Gaussian intensity distribution is described as:

.. math::

   I_{\sigma}(x, y) = \exp \left(  \frac{-x^2 - y^2}{2 \sigma^2}\right)

The shape parameter :python:`sig` defines the Gaussian's standard deviation:

.. testcode::

   psf = ot.presets.psf.gaussian(sig=2.0) 

**Airy**

The Airy function is:

.. math::

   I_{d}(x, y) = \left( \frac{2 J_1(r_d)}{r_d} \right)^2

.. math::

   r_d = 3.8317 \frac{\sqrt{x^2 + y^2}}{r}

Where :math:`J_1` is the Bessel function of the first kind of order 1.
The resolution limit :python:`r` is described as distance from the center to the first root.

.. testcode::

   psf = ot.presets.psf.airy(r=2.0) 

**Glare**

A glare is modelled as two different Gaussians, a broad and a narrow one
Parameter :math:`a` describes the relative intensity of the larger one.

.. math::

  I_{\sigma_1,\sigma_2}(x, y) = \left(1-a\right)\exp \left(  \frac{-x^2 - y^2}{2 \sigma_1^2}\right) + a\exp \left(  \frac{-x^2 - y^2}{2 \sigma_2^2}\right)

.. testcode::

   psf = ot.presets.psf.glare(sig1=2.0, sig2=3.5, a=0.05) 


**Halo**

A halo is modelled as a central Gaussian and annular Gaussian function around :math:`r`.
:math:`\sigma_1, \sigma_2` describe the standard deviations of both.
:math:`a` describes the intensity of the ring.

.. math::

   I_{\sigma_1, \sigma_2, d}(x, y) = \exp \left(  \frac{-x^2 - y^2}{2 \sigma_1^2}\right) +  a \exp \left(  \frac{-\left(\sqrt{x^2 + y^2} - r\right)^2}{2 \sigma_2^2}\right) 

.. testcode::

   psf = ot.presets.psf.halo(sig1=0.5, sig2=0.25, r=3.5, a=0.05) 


.. _psf_preset_gallery:

Preset Gallery
_____________________


.. list-table:: PSF presets with default parameters. Plotted with human brightness perception. 
   :class: table-borderless

   * - .. figure:: ../images/psf_circle.svg
          :align: center
          :width: 400
          :class: dark-light

          Exemplary Circle PSF.
   
     - .. figure:: ../images/psf_gaussian.svg
          :align: center
          :width: 400
          :class: dark-light

          Exemplary Gaussian PSF.

   * - .. figure:: ../images/psf_airy.svg
          :align: center
          :width: 400
          :class: dark-light

          Exemplary Airy PSF.
   
     - .. figure:: ../images/psf_halo.svg
          :align: center
          :width: 400
          :class: dark-light

          Exemplary Halo PSF.
   
   * - .. figure:: ../images/psf_glare.svg
          :align: center
          :width: 400
          :class: dark-light

          Exemplary Glare PSF.

     - 


