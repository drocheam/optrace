.. _usage_image:

Image Classes
---------------------------------

.. |RenderImage| replace:: :class:`RenderImage <optrace.tracer.image.render_image.RenderImage>`
.. |ScalarImage| replace:: :class:`ScalarImage <optrace.tracer.image.scalar_image.ScalarImage>`
.. |GrayscaleImage| replace:: :class:`GrayscaleImage <optrace.tracer.image.grayscale_image.GrayscaleImage>`
.. |RGBImage| replace:: :class:`RGBImage <optrace.tracer.image.rgb_image.RGBImage>`

.. role:: python(code)
  :language: python
  :class: highlight


.. testsetup:: *

   import optrace as ot
   ot.global_options.show_progressbar = False


.. _image_classes:

Overview
______________


.. list-table::
   :widths: 300 900
   :header-rows: 1
   :align: left

   * - Name
     - Description

   * - |RGBImage|
     - Image object containing three channel sRGB data as well as geometry information. 
   
   * - |GrayscaleImage|
     - Grayscale version of |RGBImage|, useful for black and white images. 
       Indicates image intensities with sRGB gamma correction.

   * - |ScalarImage|
     - Image object with only a single channel modeling a physical or physiological property.

   * - |RenderImage|
     - | Raytracer created image that holds raw XYZ colorspace and power data. 
       | This object allows for the creation of |RGBImage| and |ScalarImage| objects.

For both |RGBImage| and |GrayscaleImage| the pixel values don't correspond the physical intensities,
but non-linearly scaled values for human perception.


Creation of ScalarImage, GrayscaleImage and RGBImage
_________________________________________________________


|ScalarImage|, |GrayscaleImage| and |RGBImage| require a data argument and a geometry argument.
The latter can be either provided as side length list :python:`s` or a positional :python:`extent` parameter.

:python:`s` is a two element list describing the side lengths of the image. 
The first element gives the length in x-direction, the second in y-direction.
The image is automatically centered at :python:`x=0` and :python:`y=0`

Alternatively the edge positions are described using the :python:`extent` parameter.
It defines the x- and y- position of the edges as four element list.
For instance, :python:`extent=[-1, 2, 3, 5]` describes that the geometry of the image reaches from :python:`x=-1`
to :python:`x=2` and :python:`y=3` to :python:`y=5`.

The data argument must be a numpy array with either two dimensions (|ScalarImage| and |GrayscaleImage|) 
or three dimensions (|RGBImage|). In both cases, the data should be non-negative and in the case of the |RGBImage| 
lie inside the value range of :python:`[0, 1]`.

The following example creates a random |GrayscaleImage| using a numpy array and the :python:`s` argument:

.. testcode::
  
   import numpy as np

   img_data = np.random.uniform(0, 0.5, (200, 200))

   img = ot.GrayscaleImage(img_data, s=[0.1, 0.08])


While a random, spatially offset |RGBImage| is created with:

.. testcode::
  
   import numpy as np

   img_data = np.random.uniform(0, 1, (200, 200, 3))

   img = ot.RGBImage(img_data, extent=[-0.2, 0.3, 0.08, 0.15])


It is also possible to load image files.
For this, the data is specified as relative or absolute path string:

.. code-block:: python

   img = ot.RGBImage("image_file.png", extent=[-0.2, 0.3, 0.08, 0.15])


Loading a |GrayscaleImage| or |ScalarImage| is also possible.
However, in the case of a three channel image file, there can't be any significant coloring.
An exception gets thrown in that case.
If this is the case, either remove color information or convert it to an achromatic color space.


|RGBImage| and |GrayscaleImage| presets are available in :numref:`image_presets`. 
For convolution there are multiple PSF |GrayscaleImage| presets, see :numref:`psf_preset_gallery`.

.. _rimage_rendering:


Rendering a RenderImage
_____________________________________

**Example Geometry**

The below snippet generates a geometry with multiple sources and detectors. 

.. testcode::

    # make raytracer
    RT = ot.Raytracer(outline=[-5, 5, -5, 5, -5, 60])

    # add Raysources
    RSS = ot.CircularSurface(r=1)
    RS = ot.RaySource(RSS, divergence="None", spectrum=ot.presets.light_spectrum.FDC,
                      pos=[0, 0, 0], s=[0, 0, 1], polarization="y")
    RT.add(RS)

    RSS2 = ot.CircularSurface(r=1)
    RS2 = ot.RaySource(RSS2, divergence="None", s=[0, 0, 1], spectrum=ot.presets.light_spectrum.d65,
                       pos=[0, 1, -3], polarization="Constant", pol_angle=25, power=2)
    RT.add(RS2)

    # add Lens 1
    front = ot.ConicSurface(r=3, R=10, k=-0.444)
    back = ot.ConicSurface(r=3, R=-10, k=-7.25)
    nL1 = ot.RefractionIndex("Cauchy", coeff=[1.49, 0.00354, 0, 0])
    L1 = ot.Lens(front, back, de=0.1, pos=[0, 0, 10], n=nL1)
    RT.add(L1)

    # add Detector 1
    Det = ot.Detector(ot.RectangularSurface(dim=[2, 2]), pos=[0, 0, 0])
    RT.add(Det)

    # add Detector 2
    Det2 = ot.Detector(ot.SphericalSurface(R=-1.1, r=1), pos=[0, 0, 40])
    RT.add(Det2)

    # trace the geometry
    RT.trace(1000000)

**Source Image**

Rendering a source image is done with the :meth:`source_image <optrace.tracer.raytracer.Raytracer.source_image>` 
method of the :class:`Raytracer <optrace.tracer.raytracer.Raytracer>` class. 
Note that the scene must be traced before.

Example:

.. testcode::

   simg = RT.source_image()

This renders an |RenderImage| for the first source.
The following code renders it for the second source (since index counting starts at zero) and additionally provides 
the resolution limit :python:`limit` parameter of 3 µm.

.. testcode::

   simg = RT.source_image(source_index=0, limit=3)


**Detector Image**

Calculating a :meth:`detector_image <optrace.tracer.raytracer.Raytracer.detector_image>` is done in a similar fashion:

.. testcode::

   dimg = RT.detector_image()

Compared to :meth:`source_image <optrace.tracer.raytracer.Raytracer.source_image>`, you can not only provide a 
:python:`detector_index`, but also a :python:`source_index`, which limits the rendering to the light from this source. 
By default all sources are used for image generation.

.. testcode::

   dimg = RT.detector_image(detector_index=0, source_index=1)

For spherical surface detectors a :python:`projection_method` can be chosen. 
Moreover, the extent of the detector can be limited with the :python:`extent` parameter, that is provided as 
:python:`[x0, x1, y0, y1]` with :math:`x_0 < x_1, ~ y_0 < y_1`. 
By default, the extent gets adjusted automatically to contain all rays hitting the detector.
The :python:`limit` parameter can also be provided, 
as for :meth:`source_image <optrace.tracer.raytracer.Raytracer.source_image>`.

.. testcode::

   dimg = RT.detector_image(detector_index=0, source_index=1, extent=[0, 1, 0, 1], limit=3,
                            projection_method="Orthographic")


.. _rimage_iterative_render:

Iterative Render
_______________________

When tracing, the amount of rays is limited by the system's available RAM. 
Many million rays would not fit in the finite working memory.

The function :meth:`iterative_render <optrace.tracer.raytracer.Raytracer.iterative_render>` exists 
to allow the usage of even more rays.
It does multiple traces and iteratively adds up the image components to a summed image. 
In this way there is no upper bound on the ray count. 
With enough available user time, images can be rendered with many billion rays.

Parameter :python:`N` provides the overall number of rays for raytracing.
The returned value of :meth:`iterative_render <optrace.tracer.raytracer.Raytracer.iterative_render>`
is a list of rendered detector images.

If the detector position parameter :python:`pos` is not provided, 
a single detector image is rendered at the position of the detector specified by :python:`detector_index`.

.. testcode::

   rimg_list = RT.iterative_render(N=1000000, detector_index=1) 

If :python:`pos` is provided as coordinate, the detector is moved before tracing.

.. testcode::

   rimg_list = RT.iterative_render(N=10000, pos=[0, 1, 0], detector_index=1) 

If :python:`pos` is a list, :python:`len(pos)` detector images are rendered. 
All other parameters are either automatically repeated :python:`len(pos)` times or can be specified 
as list with the same length as :python:`pos`.

Exemplary calls:

.. testcode::

   rimg_list = RT.iterative_render(N=10000, pos=[[0, 1, 0], [2, 2, 10]], detector_index=1) 
   rimg_list = RT.iterative_render(N=10000, pos=[[0, 1, 0], [2, 2, 10]], detector_index=[0, 0], 
                                   limit=[None, 2], extent=[None, [-2, 2, -2, 2]]) 


**Tips for Faster Rendering**

With large rendering times, even small speed-up amounts add up significantly:

* Setting the raytracer option :python:`RT.no_pol` skips the calculation of the light polarization. 
  Note that depending on the geometry the polarization direction can have an influence of the amount 
  of light transmission at different surfaces. It is advised to experiment beforehand, 
  if the parameter seems to have any effect on the image.
  Depending on the geometry, :python:`no_pol=True` can lead to a speed-up of 10-40%.
* Prefer inbuilt surface types to data or function surfaces
* try to limit the light through the geometry to rays hitting all lenses. For instance:
    - Moving the color filters to the front of the system avoids the calculation of ray refractions 
      that get absorbed at a later stage.
    - Orienting the ray direction cone of the source towards the setup, therefore maximizing rays hitting all lenses.
      See the :ref:`example_arizona_eye_model` example on how this could be done. 

Saving and Loading a RenderImage
___________________________________________


**Saving**

A |RenderImage| can be saved on the disk for later use in optrace. 
This is done with the following command, that takes a file path as argument:

.. code-block:: python

   dimg.save("RImage_12345")

The file ending should be ``.npz``, but gets added automatically otherwise. 
This function **overrides files** and throws an exception when saving failed.


**Loading**

The static method :meth:`load <optrace.tracer.image.render_image.RenderImage.load>` 
from the |RenderImage| loads the saved file. 
It requires a path and returns the |RenderImage| object arguments.

.. code-block:: python

   dimg = ot.RenderImage.load("RImage_12345")


.. _image_sphere_projections:

Sphere Projections
___________________________


With a spherical detector surface, there are multiple ways to project it down to a rectangular surface. 
Note that there is no possibility to correctly represents angles, distances and areas at the same time. 

Below you can find the projection methods implemented in optrace and links to a more detailed explanation.
Details on the math applied are found in the math section in :numref:`sphere_projections`.

The available methods are:

.. list-table::
   :widths: 150 300 
   :align: left
   :stub-columns: 1

   * - :python:`"Orthographic"`
     - Perspective projection, sphere surface seen from far away :footcite:`OrthographicProjWiki`

   * - :python:`"Stereographic"`
     - Conformal projection (preserving local angles and shapes) :footcite:`SteographicProjWiki`

   * - :python:`"Equidistant"`
     - Projection keeping the radial direction from a center point equal :footcite:`EquidistantProjWiki`

   * - :python:`"Equal-Area"`
     - Area preserving projection :footcite:`EqualAreaProjWiki`

.. list-table::
    `Tissot's indicatrices <https://en.wikipedia.org/wiki/Tissot%27s_indicatrix>`__ for different projection methods. 
   All circles should have the same size, shape and brightness. Taken from the :ref:`example_sphere_projections` example
   :class: table-borderless


   * - .. figure:: ../images/indicatrix_equidistant.webp
          :align: center
          :width: 450
          :class: dark-light

     - .. figure:: ../images/indicatrix_equal_area.webp
          :align: center
          :width: 450
          :class: dark-light

   * - .. figure:: ../images/indicatrix_stereographic.webp
          :align: center
          :width: 450
          :class: dark-light

     - .. figure:: ../images/indicatrix_orthographic.webp
          :align: center
          :width: 450
          :class: dark-light


.. _image_airy_filter:

Resolution Limit Filter
___________________________

Unfortunately, optrace does not take wave optics into account when simulating. 
To estimate the effect of a resolution limit the :class:`RenderImage <optrace.tracer.image.render_image.RenderImage>` 
class provides a limit parameter. 
For a given limit value a corresponding Airy disc is created, that is convolved with the image.
This parameter describes the Rayleigh limit, being half the size of the Airy disc core (zeroth order), 
known from the equation:

.. math::
   :label: eq_rayleigh

   r = 0.61 \frac{\lambda}{\text{NA}}

Where :math:`\lambda` is the wavelength and :math:`\text{NA}` is the numerical aperture.
While the limit is wavelength dependent, one fixed value is applied to all wavelengths for simplicity.
Only the first two diffraction orders (core + 2 rings) are used, higher orders should have a negligible effect.

.. note::

   | The limit parameter is only an estimation of how large the impact of a resolution limit on the image is.
   | The simulation neither knows the actual limit nor takes into interference and diffraction.


.. list-table:: Images of the focus in the :ref:`example_achromat` example. From left to right: 
   No filter, filter with 1 µm size, filter with 5 µm size. 
   For a setup with a resolution limit of 5 µm we are clearly inside the limit, 
   but even for 1 µm we are diffraction limited.   
   :class: table-borderless

   * - .. figure:: ../images/rimage_limit_off.webp
          :align: center
          :height: 300
          :class: dark-light
   
     - .. figure:: ../images/rimage_limit_on.webp
          :align: center
          :height: 300
          :class: dark-light
     
     - .. figure:: ../images/rimage_limit_on2.webp
          :align: center
          :height: 300
          :class: dark-light

The limit parameter can be applied while either creating the |RenderImage| (:python:`ot.RenderImage(..., limit=5)`) 
or by providing it to methods the create an |RenderImage| (:python:`Raytracer.detector_image(..., limit=1)`, 
:python:`Raytracer.iterative_render(..., limit=2.5)`.


Generating Images from RenderImage
_____________________________________

**Usage**

From a |RenderImage| multiple image modes can be generated with the 
:meth:`get <optrace.tracer.image.render_image.RenderImage.get>` method.
The function takes an optional pixel size parameter, that determines the pixel count for the smaller image size.
Internally the :class:`RenderImage <optrace.tracer.image.render_image.RenderImage>` stores its data with a 
pixel count of 945 for the smaller side, while the larger side is either 1, 3 or 5 times this size, 
depending on the side length ratio. Therefore no interpolation takes place that would falsify the results.
To only join full bins, the available sizes are reduced to:

.. doctest::

   >>> ot.RenderImage.SIZES
   [1, 3, 5, 7, 9, 15, 21, 27, 35, 45, 63, 105, 135, 189, 315, 945]

As can be seen, all sizes are integer factors of 945.
All sizes are odd, so there is always a pixel/line/row for the image center.
Without a center pixel/line/row the center position would be badly defined, either being offset 
or jumping around depending on numerical errors.

In the function :meth:`get <optrace.tracer.image.render_image.RenderImage.get>` the nearest value from 
:attr:`RenderImage.SIZES <optrace.tracer.image.render_image.RenderImage.SIZES>` to the user selected value is chosen.
Let us assume the :python:`dimg` has a side length of :python:`s=[1, 2.63]`, 
so it was rendered in a resolution of 945x2835. This is the case because the nearest side factor to 2.63 is 3 and
because 945 is the size for all internally rendered images.
From this resolution the image can be scaled to 315x945 189x567 135x405 105x315 63x189 45x135 35x105 27x81 21x63 15x45
9x27 7x21 5x15 3x9 1x3.
The user image is then scaled into size 315x945, as it is the nearest to a size of 500.

These restricted pixel sizes lead to typically non-square pixels.
But these are handled correctly by plotting and processing functions.
They will only become relevant when exporting the image to an image file, where the pixels must be square. 
More details are available in section :numref:`image_saving`.

To get a Illuminance image with 315 pixels we can write:

.. testcode::

   img = dimg.get("Illuminance", 500)

Only for image modes :python:`"sRGB (Perceptual RI)"` and :python:`"sRGB (Absolute RI)"` the returned object type 
is :class:`RGBImage <optrace.tracer.image.rgb_image.RGBImage>` .
For all other modes it is of type :class:`ScalarImage <optrace.tracer.image.scalar_image.ScalarImage>`.

For mode :python:`"sRGB (Perceptual RI)"` there are two optional additional parameters :python:`L_th` 
and :python:`chroma_scale`. See :numref:`usage_color` for more details.


**Image Modes**


.. list-table::
   :widths: 150 500 
   :align: left
   :stub-columns: 1

   * - :python:`"Irradiance"`
     - Image of power per area
   * - :python:`"Illuminance"`
     - Image of luminous power per area
   * - :python:`"sRGB (Absolute RI)"`
     - A human vision approximation of the image. Colors outside the gamut are chroma-clipped. 
       Preferred sRGB-Mode for "natural"/"everyday" scenes.
   * - :python:`"sRGB (Perceptual RI)"`
     - Similar to sRGB (Absolute RI), but with uniform chroma-scaling. 
       Preferred mode for scenes with monochromatic sources or highly dispersive optics.
   * - :python:`"Outside sRGB Gamut"`
     - Pixels outside the sRGB gamut are shown in white
   * - :python:`"Lightness (CIELUV)"`
     - Human vision approximation in greyscale colors. Similar to Illuminance, but with non-linear brightness function.
   * - :python:`"Hue (CIELUV)"`
     - Hue image from the CIELUV colorspace
   * - :python:`"Chroma (CIELUV)"`
     - Chroma image from the CIELUV colorspace. Depicts how colorful an area seems compared
       to a similar illuminated grey area.
   * - :python:`"Saturation (CIELUV)"`
     - Saturation image from the CIELUV colorspace. How colorful an area seems compared to its brightness. 
       Quotient of Chroma and Lightness. 

The difference between chroma and saturation is more thoroughly explained in :footcite:`BriggsChroma`. 
An example for the difference of both sRGB modes is seen in :numref:`color_dispersive1`. 


.. list-table:: Renderes images from the :ref:`example_image_render` example. From left to right, 
   top to bottom: sRGB (Absolute RI), sRGB (Perceptual RI), Outside sRGB Gamut, 
   Lightness, Irradiance, Illuminance, Hue, Chroma, Saturation.
   :class: table-borderless

   * - .. figure:: ../images/rgb_render_srgb1.webp
          :align: center
          :width: 330
          :class: dark-light

          sRGB Absolute RI

     - .. figure:: ../images/rgb_render_srgb2.webp
          :align: center
          :width: 330
          :class: dark-light

          sRGB Perceptual RI
     
     - .. figure:: ../images/rgb_render_srgb3.webp
          :align: center
          :width: 330
          :class: dark-light

          Values outside of sRGB
   
   * - .. figure:: ../images/rgb_render_lightness.webp
          :align: center
          :width: 330
          :class: dark-light

          Lightness (CIELUV)
    
     - .. figure:: ../images/rgb_render_irradiance.webp
          :align: center
          :width: 330
          :class: dark-light

          Irradiance

     - .. figure:: ../images/rgb_render_illuminance.webp
          :align: center
          :width: 330
          :class: dark-light
     
          Illuminance

   * - .. figure:: ../images/rgb_render_hue.webp
          :align: center
          :width: 330
          :class: dark-light

          Hue (CIELUV)

     - .. figure:: ../images/rgb_render_chroma.webp
          :align: center
          :width: 330
          :class: dark-light

          Chroma (CIELUV)
     
     - .. figure:: ../images/rgb_render_saturation.webp
          :align: center
          :width: 330
          :class: dark-light

          Saturation (CIELUV)


Converting between GrayscaleImage and RGBImage
___________________________________________________

Use :meth:`RGBImage.to_grayscale_image() <optrace.tracer.image.rgb_image.RGBImage.to_grayscale_image>` to convert a
colored |RGBImage| to a grayscale image. 
The channels are weighted according to their luminance, see question 9 of :footcite:`Poynton_1997`.
Use :meth:`GrayscaleImage.to_rgb_image() <optrace.tracer.image.grayscale_image.GrayscaleImage.to_rgb_image>` 
to convert a |GrayscaleImage| to an RGB image. All grayscale values are repeated for the R, G, B channels.
Both methods require no parameters and return the other image object type.

Image Profile
_____________________________________

An image profile is a line profile of a generated image in x- or y-direction.
It is created by the :meth:`profile() <optrace.tracer.image.base_image.BaseImage.profile>` method.
The parameters :python:`x` and :python:`y` define the positional value for the profile.

The following example generates an image profile in y-direction at :python:`x=0`:

.. testcode::

   bins, vals = img.profile(x=0)

For a profile in x-direction we can write:

.. testcode::

   bins, vals = img.profile(y=0.25)

The function returns a tuple of the histogram bin edges and the histogram values, both one dimensional numpy arrays.
Note that the bin array is larger by one element.


.. _image_saving:

Saving Images
___________________________________________


|ScalarImage| and |RGBImage| can be saved to disk in the following way:

.. code-block:: python

   img.save("image_render_srgb.jpg")

The file type is automatically determined from the file ending in the path string.

Often times the image is flipped, but it can be flipped using :python:`flip=True`. 
This rotates the image by 180 degrees.

.. code-block:: python

   img.save("image_render_srgb.jpg", flip=True)


Depending on the file type ,there can be additional saving parameters provided, for instance compression settings:

.. code-block:: python

   import cv2
   img.save("image_render_srgb.jpg", params=[cv2.IMWRITE_PNG_COMPRESSION, 1], flip=True)


See 
`cv2.ImwriteFlags <https://docs.opencv.org/4.x/d8/d6a/group__imgcodecs__flags.html#ga292d81be8d76901bff7988d18d2b42ac>`_ 
for more information.
The image is automatically interpolated so the exported image has the same side length ratio
as the |RGBImage| or |ScalarImage| object.

.. note::

   While the Image has arbitrary, generally non-square pixels, for the export the image is 
   rescaled to have square pixels. However, in many cases there is no exact ratio that matches the side ratio with 
   integer pixel counts. For instance, an image with sides 12.532 x 3.159 mm and a desired export size of 105 pixels 
   for the smaller side leads to an image of 417 x 105 pixels. This matches the ratio approximately, 
   but is still off by -0.46 pixels (around -13.7 µm). This error gets larger the smaller the resolution is.

Plotting Images
_________________

See :ref:`image_plots`.

Image Properties
________________________


**Overview**

Classes |ScalarImage|, |RenderImage|, |RGBImage| share property methods.
These include geometry information and metadata.
When a |ScalarImage| or |RGBImage| is created from a |RenderImage|, the metadata and geometry
is automatically propagated into the new object.

**Size Properties**

.. doctest::

   >>> dimg.extent
   array([-0.0081,  1.0081, -0.0081,  1.0081])

.. doctest::

   >>> dimg.s[1]
   1.0162

The data shape:

.. doctest::

   >>> dimg.shape
   (945, 945, 4)


:python:`Apx` is the area per pixel in mm²:

.. doctest::

   >>> dimg.Apx
   1.1563645362671817e-06

**Metadata**

.. doctest::

   >>> dimg.limit
   3.0
   
   >>> dimg.projection is None
   True

**Data Access**

Access the underlying array data using:

.. code-block:: python

   dimg.data

**Image Powers (RenderImage only)**

Power in W and luminous power in lm:

.. testcode::

   dimg.power()
   dimg.luminous_power()

**Image Mode (RGBImage/GrayscaleImage/ScalarImage only)**

.. doctest::
   
   >>> img.quantity
   'Illuminance'


.. _image_presets:

Image Presets
____________________


Below you can find different images presets.
As for the image classes, a specification of either the :python:`s` or :python:`extent` geometry parameter is
mandatory.
One possible call could be:

.. testcode::

   img = ot.presets.image.cell(s=[0.2, 0.3])

.. list-table:: Photos of natural scenes or objects
   :class: table-borderless

   * - .. figure:: ../../../optrace/resources/images/cell.webp
          :align: center
          :height: 300

          Cell image for microscope examples 
          (`Source <https://lexica.art/prompt/960d8351-f474-4cc0-b84b-4e9521754064>`__). 
          Usable as :obj:`ot.presets.image.cell <optrace.tracer.presets.image.cell>`.
   
     - .. figure:: ../../../optrace/resources/images/fruits.webp
          :align: center
          :width: 400
        
          Photo of different fruits on a tray 
          (`Source <https://www.pexels.com/photo/sliced-fruits-on-tray-1132047/>`__).
          Usable as :obj:`ot.presets.image.fruits <optrace.tracer.presets.image.fruits>`.
   
   * - .. figure:: ../../../optrace/resources/images/interior.webp
          :align: center
          :width: 400

          Green sofa in an interior room (`Source <https://www.pexels.com/photo/green-2-seat-sofa-1918291/>`__).
          Usable as :obj:`ot.presets.image.interior <optrace.tracer.presets.image.interior>`
   
     - .. figure:: ../../../optrace/resources/images/landscape.webp
          :align: center
          :width: 400
          
          Landscape image of a mountain and water scene 
          (`Source <https://www.pexels.com/photo/green-island-in-the-middle-of-the-lake-during-daytime-724963/>`__).
          Usable as :obj:`ot.presets.image.landscape  <optrace.tracer.presets.image.landscape>`
   
   * - .. figure:: ../../../optrace/resources/images/documents.webp
          :align: center
          :width: 400
          
          Photo of a keyboard and documents on a desk 
          (`Source <https://www.pexels.com/photo/documents-on-wooden-surface-95916/>`__).
          Usable as :obj:`ot.presets.image.documents <optrace.tracer.presets.image.documents>`.
     
     - .. figure:: ../../../optrace/resources/images/group_photo.webp
          :align: center
          :width: 400
          
          Photo of a group of people in front of a blackboard 
          (`Source <https://www.pexels.com/photo/photo-of-people-standing-near-blackboard-3184393/>`__).
          Usable as :obj:`ot.presets.image.group_photo <optrace.tracer.presets.image.group_photo>`
   
   * - .. figure:: ../../../optrace/resources/images/hong_kong.webp
          :align: center
          :width: 350

          Photo of a Hong Kong street at night 
          (`Source <https://www.pexels.com/photo/cars-on-street-during-night-time-3158562/>`__).
          Usable as :obj:`ot.presets.image.hong_kong <optrace.tracer.presets.image.hong_kong>`.
   
     -  

.. _table_image_presets_aberrations:

.. list-table:: Test images for color, resolution or distortion. The ETDRS chart images, Siemens star and grid methods
   return |GrayscaleImage|, all other images |RGBImage|.
   :class: table-borderless
   
   * - .. figure:: ../../../optrace/resources/images/ETDRS_chart.png
          :align: center
          :width: 300

          ETDRS Chart standard (`Source <https://commons.wikimedia.org/wiki/File:ETDRS_Chart_2.svg>`__).
          Usage with :obj:`ot.presets.image.ETDRS_chart <optrace.tracer.presets.image.ETDRS_chart>`.
          
     - .. figure:: ../../../optrace/resources/images/ETDRS_chart_inverted.png
          :align: center
          :width: 300
          
          ETDRS Chart standard. Edited version of the ETDRS image.
          Usage with :obj:`ot.presets.image.ETDRS_chart_inverted <optrace.tracer.presets.image.ETDRS_chart_inverted>`

   * - .. figure:: ../../../optrace/resources/images/tv_testcard1.png
          :align: center
          :width: 300

          TV test card #1 (`Source <https://commons.wikimedia.org/wiki/File:TestScreen_square_more_colors.svg>`__).
          Usage with :obj:`ot.presets.image.tv_testcard1 <optrace.tracer.presets.image.tv_testcard1>`
   
     - .. figure:: ../../../optrace/resources/images/tv_testcard2.png
          :align: center
          :width: 400

          TV test card #2 (`Source <https://commons.wikimedia.org/wiki/File:Bulgarian_colour_testcard.png>`__).
          Usage with :obj:`ot.presets.image.tv_testcard2 <optrace.tracer.presets.image.tv_testcard2>`
   
   * - .. figure:: ../../../optrace/resources/images/color_checker.webp
          :align: center
          :width: 400

          Color checker chart 
          (`Source <https://commons.wikimedia.org/wiki/File:X-rite_color_checker,_SahiFa_Braunschweig,_AP3Q0026_edit.jpg>`__).
          Usage with :obj:`ot.presets.image.color_checker <optrace.tracer.presets.image.color_checker>`
     
     - .. figure:: ../../../optrace/resources/images/eye_test_vintage.webp
          :align: center
          :width: 400

          Photo of a vintage eye test chart. 
          `Image Source <https://www.publicdomainpictures.net/en/view-image.php?image=284944&picture=eye-test-chart-vintage>`__
          Usage with :obj:`ot.presets.image.eye_test_vintage <optrace.tracer.presets.image.eye_test_vintage>`.

   * - .. figure:: ../images/grid.png
          :align: center
          :width: 300
          
          White grid on black background with 10x10 cells. Useful for distortion characterization.
          Usage with :obj:`ot.presets.image.grid <optrace.tracer.presets.image.grid>`

     - .. figure:: ../../../optrace/resources/images/siemens_star.png
          :align: center
          :width: 300

          Siemens star image. 
          Own creation.
          Usage with :obj:`ot.presets.image.siemens_star <optrace.tracer.presets.image.siemens_star>`
   


------------

**References**

.. footbibliography::

