.. _usage_focus:

Focus Search
-----------------------

.. role:: python(code)
  :language: python
  :class: highlight

.. testsetup:: *

   import optrace as ot
   ot.global_options.show_progressbar = False

   RT = ot.Raytracer(outline=[-1, 1, -1, 1, 0, 60])
   RS = ot.RaySource(ot.Point(), pos=[0, 0, 1])
   RT.add(RS)
   RS = ot.RaySource(ot.Point(), pos=[0, 0, 1])
   RT.add(RS)
   RT.trace(1000)

Focus Modes
____________________

The following focus methods are available:

.. list-table::
   :widths: 300 500
   :align: left

   * - **RMS Spot Size**
     - minimal variance of the lateral ray position
   * - **Irradiance Variance**
     - highest irradiance variance
   * - **Image Sharpness**
     - sharpest edges for the whole image
   * - **Image Center Sharpness**
     - sharpest edges in the center region of the image

The methods use all available rays, for better results the scene should have been traced with much rays as possible.
The methods are explained in more detail down below.

There are multiple applications for focus search, below you can find method recommendations.

**Case 1**: Perfect, nearly ideal focal point
 * **Examples:** Focus of an ideal lens. Paraxial illumination of a real lens
 * **Preferred methods:** RMS Spot Size. Irradiance Variance is also suitable, but has worse performance.

Below you can find an example.
Both RMS Spot Size and Irradiance Variance find a similar focal position, differing only in 70 µm.
Note the different scaling of the images.

.. list-table::
   Comparison between "RMS Spot Size" (left) and "Irradiance Variance" (right) in linear (top) and logarithmic
   lightness values (bottom).
   Example :ref:`example_double_gauss`. Focus search for ray source 0 only. 2 million rays.
   :class: table-borderless

   * - .. figure:: ../images/focus_gauss_rms.webp
          :align: center
          :width: 450
          :class: dark-light

     - .. figure:: ../images/focus_gauss_var.webp
          :align: center
          :width: 450
          :class: dark-light

**Case 2:**  Strong aberrations or no distinct focal point
 * **Examples:** Lens with large spherical aberration, multifocal lens
 * **Preferred methods:** Irradiance Variance.

In the following example there are noticeable amounts of spherical aberration.
RMS Spot Size tries the minimize the radial distance of the outer rays, sacrificing a sharp core.
Irradiance Variance correctly finds a suitable focal position.
Note the logarithmic plots, that show the outer rays.
In the right case they merely contribute to the image, but they have large impact on the RMS, why the RMS method fails.

.. list-table::
   Comparison between "RMS Spot Size" (left) and "Irradiance Variance" (right) in linear (top) and logarithmic 
   lightness values (bottom).
   Example :ref:`example_spherical_aberration`.
   :class: table-borderless

   * - .. figure:: ../images/focus_sphere_rms.webp
          :align: center
          :width: 450
          :class: dark-light

     - .. figure:: ../images/focus_sphere_var.webp
          :align: center
          :width: 450
          :class: dark-light
   
   * - .. figure:: ../images/focus_sphere_rms_log.webp
          :align: center
          :width: 450
          :class: dark-light

     - .. figure:: ../images/focus_sphere_var_log.webp
          :align: center
          :width: 450
          :class: dark-light

**Case 3:** Finding the optimal image distance
 * **Example:** Actual image position (not just the paraxial) in a multi-lens setup.
 * **Preferred methods:** Image Sharpness. 
   With large amounts of curvature of field Image Center Sharpness should be selected, 
   to find a best-fit focus for the image center region.

For the image sharpness methods to work best, a source image with high contrast and sharp edges should be used.
For instance, the grid or Siemens star presets, depicted in table :numref:`table_image_presets_aberrations`.

In the following figure you can find an example for image sharpness focussing for a setup with 
large amounts of field of curvature. While in the left case more image regions are somewhat sharp, in the right case 
the sharpness is optimized for the center region.

.. list-table::
   Comparison between "Image Sharpness" (left) and "Image Center Sharpness" (right) for a setup with 
   large amounts of field of curvature. Example :ref:`example_image_render`, Grid image, pupil of 1mm, 5 million rays.
   :class: table-borderless

   * - .. figure:: ../images/focus_image_sharpness_grid.webp
          :align: center
          :width: 450
          :class: dark-light

     - .. figure:: ../images/focus_image_center_sharpness_grid.webp
          :align: center
          :width: 450
          :class: dark-light


Limitations
__________________

Limitations include:

* due to restrictions of the search region the search can't find a focus that lies between the maximum and minimum
  z-value of a surface
* rays absorbed in the search region by the raytracer outline are handled as not absorbed
* in more complex cases only a local minimum is found
* see the limitations of each method below. 

Usage
______________


For focus search you will need to trace the :class:`Raytracer <optrace.tracer.raytracer.Raytracer>` geometry.
The :meth:`focus_search <optrace.tracer.raytracer.Raytracer.focus_search>` function is then called by 
passing the focus mode and a starting position. 
The search takes place around the starting point, with the search region between the largest z-position of the last
aperture, filter, lens or ray source and the smallest z-position of the next aperture, filter, lens or outline.


.. testcode::

   res, fsdict = RT.focus_search("RMS Spot Size", 12.09)

:python:`focus_search` returns two results, where the first one is a :class:`scipy.optimize.OptimizeResult` 
object with information on the root finding. The found z-position is accessed with :python:`res.x`.
The second return value includes some additional information, for instance needed for the cost plot, 
see :ref:`focus_cost_plot`.

By default, rays from all sources are used to focus_search.
Optionally a :python:`source_index` parameter can be provided to limit the search to a specific ray source.

.. testcode::

   res, fsdict = RT.focus_search("RMS Spot Size", 12.09, source_index=1)

If the output dictionary :python:`fsdict` should include sampled cost function values, 
the parameter :python:`return_cost` must be set to :python:`True`:

.. testcode::

   res, fsdict = RT.focus_search("RMS Spot Size", 12.09, return_cost=True)

This is required when plotting the cost function using 
:meth:`focus_search_cost_plot <optrace.plots.misc_plots.focus_search_cost_plot>`, see :ref:`focus_cost_plot`.
It is deactivated by default to increase the performance of methods :python:`"RMS Spot Size", "Irradiance Variance"` 

Cost Plot
_________________

.. note::

   Generally it is recommended to plot the cost function of the optimization so one can see 
   if there are multiple minima and how distinct the found value is.
   The TraceGUI has an option for plotting the cost function.

See :ref:`focus_cost_plot`.

.. _focus_positional_methods:


Mathematical Formulation of the Methods
___________________________________________

RMS Spot Size
=========================================

Minimizing the position variance :math:`\sigma^2` of lateral ray positions :math:`x` 
and :math:`y` at axial position :math:`z`.  All positions are weighted with their power :math:`P` 
when calculating the weighted variance :math:`\sigma^2_P`. 
The Pythagorean sum is applied using both variances to get a simple quantity :math:`R_\text{v}` for optimization.

.. math::
   \text{minimize}~~ R_\text{v}(z) := \sqrt{\sigma^2_{x,P}(z) + \sigma^2_{y,P}(z)}
   :label: autofocus_position

This procedure is simple and performant. 
However, the disadvantage of this method is that it minimizes the position variance of all beams. 
For example, if there is a strong outlying halo, the method also tries to keep it as small as possible, 
which can lead to a compromise between the halo and the size of the actual focus.


Irradiance Variance
=====================

Renders a power histogram for rays at position :math:`z`. 
This histogram is divided by pixel area to get an irradiance image :math:`E(z)`
The approach then calculates the variance of the pixel values and finds the :math:`z` with the largest variance.

The most outside rays define the image dimensions, the absolute image size therefore varies along the beam path. 
This can be an issue when few rays are far away from the optical axis, 
since the resolution suffers because of these marginal rays.

The variance is large when there are bright areas in the image (with much power per area)
or if there is a large variance between pixels, which should be the case if unblurred structures are present.
For a minimization, the variance is inverted.
For a more smooth cost function and a better data range the square root of the variance is used.

.. math::
   \text{minimize}~~ I_\text{v}(z) := \frac{1}{\sqrt[4]{\sigma_E^2(z)}}
   :label: autofocus_image

Image Sharpness
==================

The power image :math:`P(x, y, z)` is transformed into the Fourier domain, creating a Fourier power image :math:`p_f` 
with image frequencies :math:`f_x` and :math:`f_y`.
Using the Pythagorean theorem we can join the frequency components into a radial frequency.
The radial frequency of each pixel is scaled with the corresponding pixel power.
We want to maximize this product, which is large when there are many high frequency components 
in the original image :math:`P_z` or when high frequency components have a high power.

.. math::
   P_f(f_x, f_y, z) = \mathcal{F}\left\{ P(x, y, z)\right\}
   :label: autofocus_image_sharpness_fourier

.. math::
   \text{minimize}~~ F_\text{p}(z) := \frac{1}{ \sqrt{\sum_{x,y} P_f(f_x, f_y, z) \left( f^2_x + f_y^2 \right)}}
   :label: autofocus_image_sharpness

For a minimization, the term is normalized by the pixel count :math:`N` and inverted.
This method is independent of the image size, as only the power image and not the irradiance map is employed.

A disadvantage of this method is that it tries to maximize the sharpness of the *whole* image.
Only a compromise solution is found for images with spatial varying blur.

Image Center Sharpness
========================

To put more emphasis on the image center, the following weighing function is applied:

.. math::
   w_r = \begin{cases} (1 - r^2)^2 &~~\text{for}~ r \leq 1\\ 1 & ~~\text{for}~ r > 1\\ \end{cases}
   :label: autofocus_image_center_sharpness_weighting

Here, :math:`r` is the normalized image radius with values between 0 and 1, 
describing the radial position on the pixel grid.
The Fourier transform is then:

.. math::
   P_{f,w}(f_x, f_y, z) = \mathcal{F}\left\{ P(x, y, z)~ w_r \right\}
   :label: autofocus_image_center_sharpness_fourier

From here on, the steps are the same as for the method Image Sharpness.
