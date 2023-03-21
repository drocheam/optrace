PSF Convolution
------------------------------------------------------------------------

.. TODO

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


**Usage**

.. TODO explain: image side lengths, non-square pixels, different resolutions, image and psf can be file paths, limitations on color images, images must be sRGB

The image is a sRGB array with dimensions (Ny, Nx, 3) with value range 0-1. Alternatively a filepath to an image file can be provided.
In contrast, the PSF holds linear power/intensity information and is there a simpler (Ny2, Nx2) shape. Alternatively an RImage object can be used to apply a colored PSF.

Parameters :python:`s_img` and :python:`s_psf` describe the side lengths of both images in millimeters, specified as list of two floats.


.. testcode::

   img = ot.presets.image.ETDRS_chart_inverted
   s_img = [0.5, 0.5]

   psf, s_psf = ot.presets.psf.halo()


The call to :python:`convolve` looks like this:

.. testcode::

   img2, s2 = ot.convolve(img, s_img, psf, s_psf)

The function returns the convolved image :python:`img2`, the new image side lengths :python:`s2`.

Additional parameters for this function include :python:`silent`, which omits all text output like a progressbar and informational messages. :python:`threading=False` disables multithreading and :python:`rendering_intent` defines the used intent for the sRGB conversion.

.. testcode::

   img2, s2 = ot.convolve(img, s_img, psf, s_psf, silent=True, threading=False, rendering_intent="Perceptual")


**Restrictions**

* image is a (Ny, Nx, 3) sRGB array or filepath to a sRGB image
* PSF is either an intensity array or an RImage object
* the value range should be inside 0-1
* resolutions must be between 50x50 pixels and 4 megapixels
* at most one image or PSF has color information
* the size of the PSF can't be much larger than the image scaled by the magnification factor


**Example for Intensity Images**

.. TODO explain that these are not linear intensities

.. testcode::
   
   import numpy as np

   # intensity function
   X, Y = np.mgrid[-1:1:200j, -1:1:200j]  # data grid
   img = np.sin(30*X**2)**2 + Y**2  # data function
   
   # make a sRGB array
   img = np.repeat(img[:, :, np.newaxis], 3, axis=2)  # repeat so we have three channels
   img = ot.color.srgb_linear_to_srgb(img)  # convert intensities to sRGB (gamma correction)

   # image size
   s_img = [0.9, 0.9]
  
   # square psf
   psf = np.zeros((200, 200))
   psf[50:150, 50:150] = 1

   # psf size
   s_psf = [0.1, 0.08]

   # convolution
   img2, s2 = ot.convolve(img, s_img, psf, s_psf)


Image Plotting
________________


.. testcode::

   import optrace.plots as otp

.. testcode::
   :hide:

   import matplotlib.pyplot as plt
   plt.close("all")


.. testcode::

   otp.image_plot(img, s_img)

.. testcode::

   otp.image_plot(img, s_img, title="Input Image", flip=True, block=False)



Presets
_____________________


**Circle**

.. testcode::

   psf, s_psf = ot.presets.psf.circle(d=3.5) 

**Gaussian**

.. testcode::

   psf, s_psf = ot.presets.psf.gaussian(d=2.0) 

**Airy**

.. testcode::

   psf, s_psf = ot.presets.psf.airy(d=2.0) 

**Glare**

.. testcode::

   psf, s_psf = ot.presets.psf.glare(d1=2.0, d2=3.5, a=0.05) 


**Halo**

.. testcode::

   psf, s_psf = ot.presets.psf.halo(d1=2.0, d2=3.5, a=0.05, w=0.1) 



Preset Gallery
_____________________


.. list-table:: PSF presets

   * - .. figure:: ../images/psf_circle.svg
          :align: center
          :width: 400

          Circle PSF with standard parameters.
   
     - .. figure:: ../images/psf_gaussian.svg
          :align: center
          :width: 400

          Gaussian PSF with standard parameters.

   * - .. figure:: ../images/psf_airy.svg
          :align: center
          :width: 400

          Airy PSF with standard parameters.
   
     - .. figure:: ../images/psf_halo.svg
          :align: center
          :width: 400

          Halo PSF with standard parameters.
   
   * - .. figure:: ../images/psf_glare.svg
          :align: center
          :width: 400

          Glare PSF with standard parameters.

     - 


