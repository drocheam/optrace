********************************
Image and Spectrum Rendering
********************************

.. role:: python(code)
  :language: python
  :class: highlight



Image Rendering
====================

Image rendering consists of two main stages: 

1. Hit detection with the detector
2. Image calculation from ray position and weights

Determining hits for a detector is more complex than for a standard surface,
as there are no specific constraints on the detector's position. 
Consequently, the detector can potentially be located within other surfaces.

Rather than calculating a single surface hit, intersections are computed for all sections of a ray 
that lie within the detector's z-range. 
Once the coordinates of a potential hit are determined,
it is necessary to verify if these coordinates fall within the defined region of the ray section, 
thereby confirming a valid hit. 
In some cases, only the virtual extension of the ray section may intersect with the surface, 
while the ray itself may have altered its direction due to the refraction on an adjacent surface.

To enhance efficiency, these calculations are executed in parallel threads, 
with each thread processing a subset of rays. 
Rays that do not reach the detector, whether because they begin ahead of it or are absorbed prior to reaching it, 
are filtered out as early as possible in the process.

Upon completing the procedure, it becomes clear whether each ray impacts the detector 
and the specific location of this impact. 
If the user specifies an extent option, only rays falling within this extent are selected. 
Otherwise, an automatic rectangular extent is calculated based on the outermost ray intersections.

If the detector is non-planar (e.g., a section of a sphere), the coordinates are initially mapped using 
a projection method as described in :numref:`sphere_projections`. 
For all intersecting rays, a two-dimensional histogram is generated based on a predefined pixel size. 
The pixel count exceeds the requested amount because each image is rendered at a higher resolution, 
allowing for resolution adjustments post-rendering, as detailed in :numref:`rimage_rendering`.

Image rendering is performed using parallel threads. 
The generated `RenderImage` object comprises images for the three tristimulus values X, Y, and Z, 
which can represent the full spectrum of human-visible colors. 
An illuminance image is directly derivable from the Y component and the pixel size, 
negating the need for explicit rendering of this image. 
The fourth image is an irradiance image, calculated from the ray powers and pixel sizes. 
Each thread is assigned one of these four images to process.

Following image rendering, the final image may be optionally filtered with an Airy-disk resolution filter 
(see :numref:`image_airy_filter`) and then rescaled to the desired resolution.


.. figure:: ../images/DetectorPAP.svg
   :width: 400
   :align: center
   :class: dark-light
   
   Detector intersection and image rendering flowchart.

.. _sphere_projections:

Sphere Projections
=========================

The use of sphere projections and example images are illustrated in :numref:`image_sphere_projections`. 
The relative distance to the center :math:`r` 
and the z-position of the opposite end of the sphere :math:`z_m` are calculated as:

.. math::
   r &= \sqrt{(x-x_0)^2  + (y - y_0)^2}\\
   z_m &= z_0 + R
   :label: sph_projections_pars

**Equidistant**

The following equation is an adaptation from :footcite:`EquidistantProjWiki`:

.. math::
   \theta &= \arctan\left(\frac{r}{z-z_m}\right)\\
   \phi &= \text{arctan2}(y-y_0, ~x-x_0)\\
   :label: equidistant_proj_pars

The projected coordinates are given by:

.. math::
   x_p &= -\theta \cdot \text{sgn}(R) \cos(\phi)\\
   y_p &= -\theta \cdot \text{sgn}(R) \sin(\phi)\\
   :label: equidistant_proj_eq

**Orthographic**

The hit coordinates :math:`x` and :math:`y` remain unchanged. 
For further reference, see :footcite:`OrthographicProjWiki`.

**Stereographic**

The following formulation is adapted from :footcite:`SteographicProjWiki`:

.. math::
   \theta &= \frac{\pi}{2} - \arctan\left(\frac{r}{z-z_m}\right)\\
   \phi &= \text{arctan2}(y-y_0, ~x-x_0)\\
   r &= 2 \tan\left(\frac{\pi}{4} - \frac{\theta}{2}\right)\\
   :label: stereographic_proj_pars
   
The projected coordinates are given by:

.. math::
   x_p &= -r \cdot  \text{sgn}(R) \cos(\phi)\\
   y_p &= -r \cdot \text{sgn}(R) \sin(\phi)\\
   :label: stereographic_proj_eq

**Equal-Area**

This equation, adapted from :footcite:`EqualAreaProjWiki`, is as follows:

.. math::
   x_r = \frac{x - x_0} {\lvert R \rvert}\\
   y_r = \frac{y - y_0} {\lvert R \rvert}\\
   z_r = \frac{z - z_m} {R}\\
   :label: equal_area_proj_pars

The projected coordinates are given by:

.. math::
   x_p = \sqrt{\frac{2}{1-z_r} x_r}\\
   y_p = \sqrt{\frac{2}{1-z_r} y_r}\\
   :label: equal_area_proj_eq


Spectrum Rendering
====================

Spectrum rendering operates in a similar way to image rendering. 
Ray intersections are computed, and only rays that successfully intersect are selected for rendering into a histogram. 
Unlike a conventional image, this process generates a spectral histogram within a specified wavelength range, 
derived from the wavelengths and powers of the rays.

In place of a :class:`RenderImage <optrace.tracer.image.render_image.RenderImage>`, 
a :class:`LightSpectrum <optrace.tracer.spectrum.light_spectrum.LightSpectrum>` object is instantiated, 
with its spectral type set to :python:`"Histogram"`.

The number of bins for the histogram is determined by the equation:

.. math::
   N_\text{b} = 1 + 2 \; \text{floor} \left(\frac{ \text{max}\left( 51, \frac{\sqrt{N}}{2}\right)} {2}\right)

This formula ensures that :math:`N_\text{b}` is odd, thereby providing a well-defined center. 
Regardless of the number of rays :math:`N`, the minimum number of bins is fixed at 51, 
with the count scaling according to the square root of :math:`N` beyond a certain threshold.
This scaling is necessary because the Signal-to-Noise Ratio (SNR) of the mean increases proportionally 
with :math:`\sqrt{N}` in the presence of normally distributed noise. 
Consequently, the number of bins is adjusted to maintain a consistent SNR while enhancing the spectrum's resolution.

Spectrum Color
=================

Analogous to :numref:`xyz_color_space`, the tristimulus values for the light spectrum :math:`S(\lambda)` 
can be calculated using the following integrals:

.. math::
   X &=\int_{\lambda} S(\lambda) x(\lambda) ~d \lambda \\
   Y &=\int_{\lambda} S(\lambda) y(\lambda) ~d \lambda \\
   Z &=\int_{\lambda} S(\lambda) z(\lambda) ~d \lambda
   :label: XYZ_Calc_Spectrum

Subsequent to this calculation, typical color model conversions can be carried out.

------------

**References**

.. footbibliography::

