.. _usage_convolution: 

PSF Convolution
------------------------------------------------------------------------


.. role:: python(code)
  :language: python
  :class: highlight


.. testsetup:: *

   import optrace as ot
   ot.global_options.show_progressbar = False

Convolving
_______________

**Overview**

Instead of waiting a long time for high quality, noise-free detector image renders in some cases the image can also be created by a PSF (point spread function) convolution. 
A PSF is nothing different than the impulse response of the optical system for the given object and image distance combination.
It has to be noted, that the PSF actually changes spatially, but for paraxial imaging it can be assumed constant.
However this also means, that many aberration can't be simulated this way (astigmatism, coma, vignette, distortion, ...).

In cases where convolution would be viable a possible approach would be to render a PSF and then apply the PSF to an object using the convolution functionality in ``optrace``.

For the convolution to work the object and PSF size are needed besides the PSF and objects itself. 
Most optical systems also have a magnification factor that rescaled the object size by a specific amount.

Both colored objects and PSFs are supported.

Note however, that without knowing the correct spectral distribution on both (instead only the values in a color space) the convolution only simulates one of many possible cases. 
Also see the section :numref:`psf_color_handling` on this.


**Usage**

The image should be either of type RGBImage or LinearImage.
The PSF should be either LinearImage or RenderImage.
For more details on image classes see <>.

Colored PSFs can be only used with class RenderImage, as all human visible colors are needed for convolution.

Let's use predefined images and psf:

.. testcode::

   img = ot.presets.image.ETDRS_chart_inverted([0.5, 0.5])

   psf = ot.presets.psf.halo()


Typically the imaging system has a magnification factor :python:`m` that is also needed to scale the input object size.
This factor is equivalent the magnification factor known from geometrical optics.
Therefore ``abs(m) > 1`` means a magnification and ``abs(m) < 1`` an image shrinking.
``m > 0`` is an upright image, while ``m < 0`` correspond to a flipped image.

You can then call :python:`convolve` like this:

.. testcode::

   img2 = ot.convolve(img, psf, m=0.5)

The function returns the convolved image object :python:``img2``.
When ``img`` and ``psf`` are of type LinearImage, ``img2`` is also a LinearImage.
In all other cases color information are generated and ``img2`` is a RGBImage.


**Slicing and padding**

While doing a convolution the output image grows in size by half the PSF size in that direction.
So the output image has a higher pixel count as well as larger side lengths.
If it is required to leave both properties the same, ``slice_=True`` can be provided so the image is sliced back to its initial size:

.. testcode::

   img2 = ot.convolve(img, psf, m=0.5, slice_=True)

When doing the convolution, the operation `guesses` what lies behind the image edges, as it also must use this data near the boundary of the image.
By default it pads the data outside with zeros, but other modes can also be set.

For instance, padding with white is done in the following fashion:

.. testcode::

   img2 = ot.convolve(img, psf, m=0.5, slice_=True, padding_mode="constant", padding_value=[1, 1, 1])

``padding_value`` must have the same number of elements as ``img`` has channels, so one for a LinearImage and three for an RGBImage.

Edge padding is done as follows:

.. testcode::

   img2 = ot.convolve(img, psf, m=0.5, slice_=True, padding_mode="edge")

**Color conversion**

When convolving with a PSF of type RenderImage, colors of the resulting image may lie outside the sRGB gamut.
Using a rendering intent conversion they are projected/clipped them into the gamut.
This is done by the ``cargs`` argument (conversion arguments).

By default it is set to :python:`dict(rendering_intent="Absolute", normalize=True, clip=True, L_th=0, sat_scale=None)`.

You can provide a ``cargs`` dictionary that overrides this setting.

.. testcode::

   img2 = ot.convolve(img, psf, m=0.5, slice_=True, padding_mode="edge", cargs=dict(rendering_intent="sRGB (Perceptual RI)"))

The above command overrides the ``rendering_intent`` while leaving the other default options unchanged.

**Normalization**

When convolving two LinearImages it is recommended to normalize the PSF sum to 1, 
so the sum of the input image and output image is preserved (with the sum for instance corresponding to the power).

Restrictions
_______________________

* two RGBImage or two RenderImage can't be convolved
* resolutions for both image and PSF must be between 50x50 pixels and 4 megapixels
* the size of the PSF can't be twice the size than the image scaled by the magnification factor
* when convolving two colored images, the resulting image is only one possible solution of many
* the convolution is done using :func:`scipy.signal.fftconvolve`, so due to numerical errors small values in dark image regions can appear
* convolution of images that have been sphere projected (see <>) is prohibited, as it doesn't make sense geometrically.
  In the projection always one of distance, area or angle is non-linear.

Examples
__________________________

**Image Example**


.. list-table:: Image convolution from ``./examples/psf_imaging.py``

   * - .. figure:: ../images/example_psf1.svg
          :align: center
          :width: 400

   
     - .. figure:: ../images/example_psf2.svg
          :align: center
          :width: 400


.. figure:: ../images/example_psf3.svg
   :align: center
   :width: 400


**Code Example**


The following example loads an image preset and convolves it with a square PSF that was created as a numpy array.

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

`optrace` features presets for different PSF shapes.
In the next section a gallery of point spread function presets can be found.
Alternatively a more mathematical description is featured in section :numref:`math_psf_presets`.

All presets are normalized such that the image sum is 1.

**Circle**

A circle PSF is defined using the :python:`d` parameter that defines the circle diameter.

.. testcode::

   psf = ot.presets.psf.circle(d=3.5) 

**Gaussian**

A gaussian function can model the zeroth order shape of an airy disc.
The shape parameter `sig` defines the gaussian's standard deviation.

.. testcode::

   psf = ot.presets.psf.gaussian(sig=2.0) 

**Airy**

An Airy PSF also include higher order diffraction and is also characterized by the resolution limit which is the first zero crossing position relative to its core.

.. testcode::

   psf = ot.presets.psf.airy(r=2.0) 

**Glare**

The glare consists of two gaussians, the first with parameter :python:`sig1`, the other with larger :python:`sig2` and relative intensity :python:`a`.

.. testcode::

   psf = ot.presets.psf.glare(sig1=2.0, sig2=3.5, a=0.05) 


**Halo**

A halo consists of a center gaussian with :python:`sig1` and intensity 1, as well as a ring at :math:`r` with standard deviation :python:`sig2` with intensity :math:`a`.

.. testcode::

   psf = ot.presets.psf.halo(sig1=0.5, sig2=0.25, r=3.5, a=0.05) 



.. _psf_preset_gallery:

Preset Gallery
_____________________


.. list-table:: PSF presets

   * - .. figure:: ../images/psf_circle.svg
          :align: center
          :width: 400

          Exemplary Circle PSF.
   
     - .. figure:: ../images/psf_gaussian.svg
          :align: center
          :width: 400

          Exemplary Gaussian PSF.

   * - .. figure:: ../images/psf_airy.svg
          :align: center
          :width: 400

          Exemplary Airy PSF.
   
     - .. figure:: ../images/psf_halo.svg
          :align: center
          :width: 400

          Exemplary Halo PSF.
   
   * - .. figure:: ../images/psf_glare.svg
          :align: center
          :width: 400

          Exemplary Glare PSF.

     - 


