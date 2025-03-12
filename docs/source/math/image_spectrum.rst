********************************
Image and Spectrum Rendering
********************************

.. role:: python(code)
  :language: python
  :class: highlight



Image Rendering
====================

Image rendering consists of two stages: 

 1. Hit detection with the detector
 2. Image calculation from ray position and weights

Hit finding for a detector is more complicated as a normal surface, as there are no requirements for the detector position.
It is therefore also possible for the detector to be inside other surfaces.

Instead of calculation a surface hit once, is is calculated for all sections of a ray that are inside the z-range of the detector.
After knowing the hit coordinates, one can check if they fall inside the region where the ray section is defined and if it is therefore is a valid hit.
In the other case only the ray section extension hits the surface, but the ray already changed direction due to a adjacent surface.

To speed things up, calculations are done in threads, while each thread gets a subset of rays. 
Rays not reaching the detector at all, starting before it or getting absorbed before hitting the detector are sorted out as early as possible.

At the end of the procedure it is known for each rays if it hits the detector and where.
If an extent option was provided by the user, only the rays inside this extent are selected.
Otherwise the rectangular extent gets calculated from the outermost rays.

If the detector is not planar (e.g. a sphere section) the coordinates are first mapped with a projection method that are described in :numref:`sphere_projections`.
For all hitting rays a two-dimensional histogram is generated, with a beforehand defined pixel size.
The pixel count is higher as requested, as each image is rendered in a higher resolution to allow for resolution changes after rendering, see :numref:`rimage_rendering`.

Image rendering is also done in threads.
The created RImage object hold images for the three tristimulus values X, Y, Z, that can encompass all human-visible colors.
The illuminance image can be directly calculated from the Y component and the pixel size, so an explicit rendering of this image is not required.
The fourth image is an irradiance image, which is calculated from the ray powers and the pixel sizes.
The aforementioned threads each get one of the four images.

After image rendering the image is optionally filtered with an Airy-disk resolution filter and then rescaled to the desired resolution.

.. figure:: ../images/DetectorPAP.svg
   :width: 400
   :align: center
   :class: dark-light
   
   Detector intersection and image rendering flowchart.

.. _sphere_projections:

Sphere Projections
=========================

The relative distance to center and the z-position of the other sphere end are

.. math::
   r &= \sqrt{(x-x_0)^2  + (y - y_0)^2}\\
   z_m &= z_0 + R
   :label: sph_projections_pars

**Equidistant**

Adapted version of :footcite:`EquidistantProjWiki`.

.. math::
   \theta &= \arctan\left(\frac{r}{z-z_m}\right)\\
   \phi &= \text{arctan2}(y-y_0, ~x-x_0)\\
   :label: equidistant_proj_pars

The projected coordinates are then

.. math::
   x_p &= -\theta \cdot \text{sgn}(R) \cos(\phi)\\
   y_p &= -\theta \cdot \text{sgn}(R) \sin(\phi)\\
   :label: equidistant_proj_eq

**Orthographic**

The hit coordinates :math:`x` and :math:`y` are kept as is.
Related: :footcite:`OrthographicProjWiki`.

**Stereographic**

Adapted version of :footcite:`SteographicProjWiki`.

.. math::
   \theta &= \frac{\pi}{2} - \arctan\left(\frac{r}{z-z_m}\right)\\
   \phi &= \text{arctan2}(y-y_0, ~x-x_0)\\
   r &= 2 \tan\left(\frac{\pi}{4} - \frac{\theta}{2}\right)\\
   :label: stereographic_proj_pars
   
The projected coordinates are then

.. math::
   x_p &= -r \cdot  \text{sgn}(R) \cos(\phi)\\
   y_p &= -r \cdot \text{sgn}(R) \sin(\phi)\\
   :label: stereographic_proj_eq

**Equal-Area**

Adapted version of :footcite:`EqualAreaProjWiki`.

.. math::
   x_r = \frac{x - x_0} {\lvert R \rvert}\\
   y_r = \frac{y - y_0} {\lvert R \rvert}\\
   z_r = \frac{z - z_m} {R}\\
   :label: equal_area_proj_pars

The projected coordinates are then

.. math::
   x_p = \sqrt{\frac{2}{1-z_r} x_r}\\
   y_p = \sqrt{\frac{2}{1-z_r} y_r}\\
   :label: equal_area_proj_eq


Spectrum Rendering
====================

Spectrum rendering works in a similar way to image rendering.
Ray intersections are calculated, only hitting rays are selected and a histogram is rendered.
But compared to an image, this is a spectral histogram within a wavelength range resulting from the rays wavelengths and powers.
Instead of an :class:`RenderImage <optrace.tracer.image.render_image.RenderImage>` a :class:`LightSpectrum <optrace.tracer.spectrum.light_spectrum.LightSpectrum>` object is created with type :python:`"Histogram"`.

The number of bins for the histogram is:

.. math::
   N_\text{b} = 1 + 2 \; \text{floor} \left(\frac{ \text{max}\left( 51, \frac{\sqrt{N}}{2}\right)} {2}\right)

This formula ensures :math:`N_\text{b}` is odd, so the center is well-defined.
Independent of the number of rays :math:`N` the minimum of bins is set to 51 and scales with the square root of this number above a specific value.
The latter is due to the SNR of the mean also increasing with :math:`\sqrt{N}` for normal-distributed noise.
So the number of bins is adapted so that the SNR stays the same, but the spectrum resolution increases.

Spectrum Color
=================

Analogously to :numref:`xyz_color_space` the tristimulus values for the light spectrum :math:`S(\lambda)` can be calculated with:

.. math::
   X &=\int_{\lambda} S(\lambda) x(\lambda) ~d \lambda \\
   Y &=\int_{\lambda} S(\lambda) y(\lambda) ~d \lambda \\
   Z &=\int_{\lambda} S(\lambda) z(\lambda) ~d \lambda
   :label: XYZ_Calc_Spectrum

From there on, typical color model conversions can be applied.

------------

**References**

.. footbibliography::

