
*****************
Autofocus
*****************

Goal
==================

Focus finding can be categorized into two different goals

1. finding a focal point
2. finding the position of an image plane in an imaging system


Application Cases
=======================

Below you can find multiple application cases an preferred autofocus methods.

**Case 1**: perfect, ideal focal point
 * **examples:** focus of an ideal lens. Small, local illumination of a real lens
 * **preferred methods:** all methods find the focus correctly, for performance reason "Position Variance" should be used

**Case 2:**  broad or no distinct focal point
 * **examples:** lens with large spherical aberration, multifocal lens
 * **preferred methods:** None, largely different behavior depending on method choice
 * **behaviour known from experience**
    * Position Variance: finds a compromise between multiple foci, often inbetween their position
    * Airy Disc Weighting: Ignores glares, halos and rays with large distance from airy disc
    * Irradiance Maximum: finds the focus with the largest irradiance
    * Image Sharpness: Not suited, since its searches for sharp structures
    * Irradiance Variance: similar behavior to Image Sharpness and Irradiance Maximum

**Case 3:** finding the image distance
 * **example:** lens setup with multiple lenses, we want to find the distance where the image has the highest sharpness
 * **preferred methods:** Image Sharpness, in some specific edge cases Irradiance Variance/Maximum might work.


.. topic:: Note

   Generally it is recommended to plot the cost function of the optimization so one can see if there are multiple minima and how distinct the found value is.
   The TraceGUI has an option for plotting the cost function.


Position Methods
====================

The two following methods use a weighting of the spatial ray positions.
Above a few ten thousand rays the results are consistent enough that the result isn't dependent on the number of rays anymore.
This is why internally the methods only use a subset if there are more rays available than needed.


**Position Variance**

Minimizing the position variance :math:`\sigma^2` of lateral ray positions :math:`X_z` and :math:`Y_z` at axial position :math:`z`. All positions are weighted with their power :math:`P` when calculating the weighted variance :math:`\sigma^2_P`. The pythagorean sum is applied using both variances to get a simple quantity :math:`R_\text{v}` for optimization.

.. math::
   \text{minimize}~~ R_\text{v}(z) := \sqrt{\sigma^2_P(X_z) + \sigma^2_P(Y_z)}
   :label: autofocus_position

This procedure is simple and performant. However, the disadvantage of this method is that it minimizes the position variance of all beams. For example, if there is a strong outlying halo, the method also tries to keep it as small as possible, which can lead to a compromise between the halo and the size of the actual focus.

**Airy Disc Weighting**

A virtual detector with roughly the spatial sensitivity :math:`S` of an airy disc.
The numerical aperture :math:`\text{NA}` needed is estimated using the ray angles.
Only the zeroth order of the airy disc is used, there it can be approximated using a gaussian, see :footcite:`AiryWiki`.

The stimulus for one ray is the product of gauss function value at radial position :math:`r_i(z)` from the disc center and ray power :math:`P_i(z)`. Summing up all ray stimuli and dividing by the overall power we get the cost function value for position :math:`z`.

.. math::
   \text{maximize}~~ S(z) := \frac{\displaystyle\sum_{i}^{} P_i(z) \cdot \exp \left( {-0.5\left(\frac{r_i(z)}{0.42\,r_0}\right)^2} \right)}{\displaystyle\sum_{i}^{} P_i(z)}
   :label: autofocus_airy

with

.. math::
   r_0 = \frac{\lambda}{\text{NA}}
   :label: autofocus_airy_r


:math:`S(z)` is bound to :math:`[0, 1]` with :math:`S=0` being completely defocused light and :math:`S=1` being an ideal focus where all the light is exactly in the center of the gaussian function.

It can be shown, that when the light is also distributed according to a gaussian pattern with the same center as the airy disc approximation, with a standard deviation being factor :math:`k` times larger, the stimulus :math:`S_k` is then:

.. math::
   S_k = \frac{1}{\sqrt{k^2 + 1}}

Exemplary values for :math:`S_k` can be found in the following table

.. list-table:: Values for :math:`S_k` 
   :widths: 50 50 50 50 50 50 50 50 50
   :header-rows: 1
   :align: center

   * - :math:`k`
     - 0
     - 1
     - 2
     - 3
     - 5
     - 10
     - 20
     - :math:`\infty`
   * - :math:`S_k`
     - 1.00
     - 0.71
     - 0.45
     - 0.32
     - 0.20
     - 0.10
     - 0.05
     - 0.00


In physical reality we can't get a higher value for :math:`S_k` than that for :math:`k=1` since this is equivalent to the resolution limit. Since the simulation does not factor in wave-optical properties, they can nevertheless appear in the raytracer.
Another disadvantage of the method is that it ignores all behavior of the beams far outside the sensitive range of the virtual receiver. 

Image Methods
==============

The next three methods render multiple images (actually being power histograms) :math:`P_z` with pixel number :math:`N_\text{px} \cdot N_\text{px}`.
:math:`N_\text{px}` is dependent on the number of rays used for focus finding, for few rays we want to keep the number also low to minimize effects of noise. For a larger amount of rays we can increase the number step by step. This is actually even implicitly needed to resolve small structures.
:math:`N` rays being distributed on a square area means we need to increase :math:`N_\text{px}` proportionally to :math:`\sqrt{N}` to achieve a somehow constant SNR. The formula implemented has the form :math:`N_\text{px} = \text{offset} + \text{factor} \cdot \sqrt{N}`.

The most outside rays define the image dimensions, the absolute image size therefore varies along the beam path. This can be an issue when few rays are far away from the optical axis, since the resolution suffers because of these marginal rays.

In contrast to the methods above the following methods always use all rays available after tracing to achieve satisfying results. However, this can lead to large processing times for many million rays.


**Irradiance Variance**

Render a power histogram for rays at position :math:`z`. Divide by pixel area to get an irradiance image :math:`E_z`
Calculate the variance of the pixel values. Find the :math:`z` with the largest variance.

The variance is large when there are bright areas in the image (much power per area) or if their is large value variance between the pixels, which is typically the case if structures are present.


.. math::
   \text{maximize}~~ I_\text{v}(z) := \sigma^2(E_z)
   :label: autofocus_image


**Irradiance Maximum**

Similar to Irradiance Variance, but instead the maximum value in :math:`E_z` is maximized.

.. math::
   \text{maximize}~~ I_\text{p}(z) := \text{max}(E_z)
   :label: autofocus_maximum

**Image Sharpness**

We are using the power image :math:`P_z` and transform it into the fourier domain.
This creates an fourier power image :math:`p_f` with image frequencies :math:`f_x` and :math:`f_y`.
Using the pythagorean theorem we can join the frequency components into a radial frequency.
The radial frequency of each pixel is scaled with the corresponding pixel power.
We want to maximize this product, which is large when there are many high frequency components in the original image :math:`P_z` or when high frequency components have a high power.

.. math::
   \text{maximize}~~ F_\text{p}(z) := p_\text{f} \cdot \sqrt{f^2_x + f^2_y}
   :label: autofocus_image_sharpness

This method is independent of the image size, since we used the power image instead of a irradiance image.

Additional Notes
======================

.. topic:: Notes

   * As the name suggests, minimization methods in scipy try to find the minimum of a cost function. Some methods above however require a maximization. In these cases the cost function was simply inverted or subtracted from a reference value.
   * For the methods Irradiance Variance and Maximum the root of the cost function is taken, so the value range and value changes are more smooth.
   * focus finding always searches in the region between two lenses or a lens and the outline. 
   * focus finding ignores filters, apertures and the outline while finding the focus. So if a ray exists inside the search region but is absorbed or filtered in the region it is assumed as not being so.
   * if any rays in this region intersect with the tracing outline, this is not handled.




------------

**Sources**

.. footbibliography::

