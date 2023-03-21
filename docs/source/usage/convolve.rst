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


.. testcode::

   img = ot.presets.image.ETDRS_chart_inverted
   s_img = [0.5, 0.5]

   psf, s_psf = ot.presets.psf.halo()

When provided as numpy arrays,
:python:`img` and :python:`psf` are interpreted as sRGB image data. They therefore must be three dimensional with three elements in the third dimension.
Alternatively each can be path string to an image file that will be loaded inside :python:`convolve`.

The call to :python:`convolve` looks like this:

.. testcode::

   img2, s2, dbg = ot.convolve(img, s_img, psf, s_psf)

The function returns the convolved image :python:`img2`, the new image side lengths :python:`s2` as well as a debugging information dictionary :python:`dbg`.

Additional parameters for this function include :python:`silent`, which omits all text output like a progressbar and informational messages. :python:`threading=False` disables multithreading and :python:`k` defines an interpolation parameter for :obj:`scipy.interpolate.RectBivariateSpline` that is used inside the function.

.. testcode::

   img2, s2, dbg = ot.convolve(img, s_img, psf, s_psf, silent=True, threading=False, k=5)


**Restrictions**


* PSF and image must be path strings or numpy arrays with shape (Ny, Nx, 3)
* the value range should be inside 0-1
* array values are interpreted as sRGB values, not linear intensities
* resolutions must be between 50x50 pixels and 4 megapixels
* at most one image has color information
* the size of the PSF must be smaller than that of the image

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
   psf = np.zeros((200, 200, 3))
   psf[50:150, 50:150] = 1
   # conversion to sRGB not needed, as we have binary values (0, 1)

   # psf size
   s_psf = [0.1, 0.08]

   # convolution
   img2, s2, dbg = ot.convolve(img, s_img, psf, s_psf)


Plotting
________________

**Simple Image Plot**

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


**Debugging Information**



.. testcode::

   otp.convolve_debug_plots(img2, s2, dbg)


.. testcode::

   otp.convolve_debug_plots(img2, s2, dbg, log=True, log_exp=5, block=False)



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


