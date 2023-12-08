.. _usage_convolution: 

PSF Convolution
------------------------------------------------------------------------


.. role:: python(code)
  :language: python
  :class: highlight

.. mock progressbar library, so we have no stdout output for it

.. testsetup:: *

   import sys 
   import mock
   sys.modules['progressbar'] = mock.MagicMock()

   import optrace as ot


Convolving
_______________

**Overview**

Instead of waiting a long time for high quality, noise-free detector image renders in some cases the image can also be created by a PSF (point spread function= convolution. A PSF is nothing different than the impulse response of the optical system for the given object and image distance combination.
It has to be noted, that the PSF changes spatially, but for paraxial imaging it can be assumed constant.
However this also means, that many aberration can't be simulated this way (astigmatism, coma, vignette, distortion, ...).

In cases where convolution would be viable a possible approach would be to render a PSF and then apply the PSF to an object using the convolution functionality in ``optrace``.

For the convolution to work the object and PSF size are needed besides the PSF and objects itself. Most optical systems also have a magnification factor that rescaled the object size by a specific amount.

Both colored objects and PSFs are supported. 
Note however, that without knowing the correct spectral distribution on both (instead only the values in a color space) the convolution only simulates one of many possible cases. 
Also see the section :numref:`psf_color_handling` on this.


**Usage**


The image is a sRGB numpy array with dimensions (Ny, Nx, 3) with value range 0-1. Alternatively a filepath to an image file can be provided.
In contrast, the PSF holds linear power/intensity information and is there a simpler (Ny2, Nx2) shape. Alternatively an :class:`RImage <optrace.tracer.r_image.RImage>` object can be used to apply a colored PSF.

Parameters :python:`s_img` and :python:`s_psf` describe the side lengths of both images in millimeters, specified as list of two floats.


.. testcode::

   img = ot.presets.image.ETDRS_chart_inverted
   s_img = [0.5, 0.5]

   psf, s_psf = ot.presets.psf.halo()


Typically the imaging system has a magnification factor :python:`m` that is also needed to scale the input object size.
This factor is equivalent the magnification factor known from geometrical optics.

You can call :python:`convolve` like this:

.. testcode::

   img2, s2 = ot.convolve(img, s_img, psf, s_psf, m=0.5)

The function returns the convolved sRGB image :python:`img2`, the new image side lengths :python:`s2`.

The additional parameter :python:`rendering_intent` defines the used intent for the sRGB conversion.

.. testcode::

   img2, s2 = ot.convolve(img, s_img, psf, s_psf, rendering_intent="Perceptual")


**Restrictions**

* object image is a (Ny, Nx, 3) sRGB array or filepath to a sRGB image
* PSF is either an intensity array or an RImage object
* resolutions must be between 50x50 pixels and 4 megapixels
* the size of the PSF can't be much larger than the image scaled by the magnification factor
* as side lengths of PSF and object can be otherwise arbitrary the pixels are generally non-square


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
   img = ot.presets.image.ETDRS_chart_inverted

   # image size
   s_img = [0.9, 0.9]
  
   # square psf
   psf = np.zeros((200, 200))
   psf[50:150, 50:150] = 1

   # psf size
   s_psf = [0.1, 0.08]

   # convolution
   img2, s2 = ot.convolve(img, s_img, psf, s_psf, m=-1.75)


Image Plotting
________________

Images, whether they are numpy arrays or paths to image files, are plotted with the :func:`image_plot <optrace.plots.misc_plots.image_plot>` function.
Additionally a tuple of image side lengths is required.

Import the plotting functionality:

.. testcode::

   import optrace.plots as otp

.. testcode::
   :hide:

   import matplotlib.pyplot as plt
   plt.close("all")

Then call the plot with:

.. testcode::

   otp.image_plot(img, s_img)

A user title is provided with the :python:`title` parameter, additionally the image can be flipped (rotated 180 degrees) with :python:`flip=True`.
Like all other plotting function the window can block the execution of the rest of the program with :python:`block=True`.

.. testcode::

   otp.image_plot(img, s_img, title="Input Image", flip=True, block=False)



Presets
_____________________

`optrace` features presets for different PSF shapes.
In the next section a gallery of point spread function presets can be found.
Alternatively a more mathematical description is featured in section :numref:`math_psf_presets`.

**Circle**

A circle PSF is defined using the :python:`d` parameter that defines the circle diameter.

.. testcode::

   psf, s_psf = ot.presets.psf.circle(d=3.5) 

**Gaussian**

A gaussian function can model the zeroth order shape of an airy disc.
The shape parameter `sig` defines the gaussian's standard deviation.

.. testcode::

   psf, s_psf = ot.presets.psf.gaussian(sig=2.0) 

**Airy**

An Airy PSF also include higher order diffraction and is also characterized by the resolution limit which is the first zero crossing position relative to its core.

.. testcode::

   psf, s_psf = ot.presets.psf.airy(r=2.0) 

**Glare**

The glare consists of two gaussians, the first with parameter :python:`sig1`, the other with larger :python:`sig2` and relative intensity :python:`a`.

.. testcode::

   psf, s_psf = ot.presets.psf.glare(sig1=2.0, sig2=3.5, a=0.05) 


**Halo**

A halo consists of a center gaussian with :python:`sig1` and intensity 1, as well as a ring at :math:`r` with standard deviation :python:`sig2` with intensity :math:`a`.

.. testcode::

   psf, s_psf = ot.presets.psf.halo(sig1=0.5, sig2=0.25, r=3.5, a=0.05) 



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


