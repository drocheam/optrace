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

Focus search can be categorized into two different categories:

1. finding a focal point
2. finding the position of an image plane


The following focus methods are available:

.. list-table::
   :widths: 200 400
   :align: left

   * - **Position Variance**
     - minimizes the variance of the lateral ray position
   * - **Airy Disc Weighting**
     - weights the ray positions with a spatial sensitivity of the zeroth order of an airy disc
   * - **Irradiance Maximum**
     - finds the position with the highest irradiance
   * - **Irradiance Variance**
     - finds the positions with the highest irradiance variance
   * - **Image Sharpness**
     - finds the position with the sharpest edges

The methods are explained in more detail in Sections :numref:`focus_positional_methods` and :numref:`focus_rendering_methods`.

Usage
______________


For focus search you will need to trace the :class:`Raytracer <optrace.tracer.raytracer.Raytracer>` geometry.
The :meth:`focus_search <optrace.tracer.raytracer.Raytracer.focus_search>` function is then called by passing the focus mode and a starting position.
Focus Search then tries to find the focus in a search region between the last lens (or the outline) and the next lens (or the outline).

.. testcode::

   res, fsdict = RT.focus_search("Position Variance", 12.09)

:python:`focus_search` returns two results, where the first one is a :class:`scipy.optimize.OptimizeResult` object with information on the root finding.
The found z-position is accessed with :python:`res.x`.
The second return value includes some additional information, for instance needed for the cost plot, see :ref:`focus_cost_plot`.

By default, rays from all sources are used to focus_search.
Optionally a :python:`source_index` parameter can be provided to limit the search to a specific ray source.

.. testcode::

   res, fsdict = RT.focus_search("Position Variance", 12.09, source_index=1)

For modes :python:`"Position Variance", "Airy Disc Weighting"` the ray number is limited, as above a certain number the quality of results hardly increases anymore.
Should it be needed, you can increase or lower this number with the parameter :python:`N`.
It is set to 100000 by default. 

Mode :python:`"Position Variance"` uses a slightly different approach for root finding, which leads to some parameters missing in the second return parameter :python:`fsdict`.
To include the results needed for a :ref:`focus_cost_plot`, set the parameter :python:`return_cost=True`.

.. testcode::

   res, fsdict = RT.focus_search("Position Variance", 12.09, N=10000, return_cost=True)


Limitations
__________________

Below you can find some limitations of the focus search:

* search only possible between lenses or a lens and the outline
* the behavior of filters and apertures in the search region is ignored
* rays absorbed in the search region by the raytracer outline are handled as unabsorbed
* in more complex cases only a local minimum is found
* see the limitations of each method below. 

Application Cases
____________________

There are multiple applications for focus search, below you can find method recommendations.

**Case 1**: Perfect, ideal focal point
 * **Examples:** Focus of an ideal lens. Paraxial illumination of a real lens
 * **Preferred methods:** All methods should find the focus correctly, for performance reason "Position Variance" should be preferred

**Case 2:**  Broad or no distinct focal point
 * **Examples:** Lens with large spherical aberration, multifocal lens
 * **Preferred methods:** None, largely different behavior depending on the method
 * **Behaviour known from experience**
    * Position Variance: Finds a compromise between multiple foci, often inbetween their position
    * Airy Disc Weighting: Ignores glares, halos and rays with large distance from airy disc
    * Irradiance Maximum: Finds the focus with the largest irradiance
    * Image Sharpness: Not suited, since its searches for sharp structures
    * Irradiance Variance: similar behavior to Image Sharpness and Irradiance Maximum

**Case 3:** Finding the image distance
 * **Example:** Lens setup with multiple lenses, we want to find the distance where the image has the highest sharpness
 * **Preferred methods:** Image Sharpness, in some specific edge cases Irradiance Variance/Maximum might work.


Cost Plot
_________________

.. topic:: Note

   Generally it is recommended to plot the cost function of the optimization so one can see if there are multiple minima and how distinct the found value is.
   The TraceGUI has an option for plotting the cost function.

See :ref:`focus_cost_plot`.

.. _focus_positional_methods:

Positional Methods
====================

The two following methods use a weighting of the spatial ray positions.
Above a few ten thousand rays the results are consistent enough that the result isn't dependent on the number of rays anymore.
This is why internally the methods use only a random ray subset if there are more rays available than really needed.


**Position Variance**

Minimizing the position variance :math:`\sigma^2` of lateral ray positions :math:`X_z` and :math:`Y_z` at axial position :math:`z`. 
All positions are weighted with their power :math:`P` when calculating the weighted variance :math:`\sigma^2_P`. 
The Pythagorean sum is applied using both variances to get a simple quantity :math:`R_\text{v}` for optimization.

.. math::
   \text{minimize}~~ R_\text{v}(z) := \sqrt{\sigma^2_P(X_z) + \sigma^2_P(Y_z)}
   :label: autofocus_position

This procedure is simple and performant. 
However, the disadvantage of this method is that it minimizes the position variance of all beams. 
For example, if there is a strong outlying halo, the method also tries to keep it as small as possible, which can lead to a compromise between the halo and the size of the actual focus.

**Airy Disc Weighting**

A virtual detector with roughly the spatial sensitivity :math:`S` of an Airy disc.
The numerical aperture :math:`\text{NA}` is estimated using the ray angles.
Only the zeroth order of the airy disc is used, which can be approximated using a Gaussian curve, see :footcite:`AiryWiki`.

The *stimulus* for one ray is the product of the function value at radial position :math:`r_i(z)` from the disc center and ray power :math:`P_i(z)`. 
Summing up all ray stimuli and dividing by the overall power we get the cost function value for position :math:`z`.
To turn this equation into a minimization, the term is subtracted from 1.

.. math::
   \text{minimize}~~ S(z) := 1 - \frac{\displaystyle\sum_{i}^{} P_i(z) \cdot \exp \left( {-0.5\left(\frac{r_i(z)}{0.42\,r_0}\right)^2} \right)}{\displaystyle\sum_{i}^{} P_i(z)}
   :label: autofocus_airy

with

.. math::
   r_0 = \frac{\lambda}{\text{NA}}
   :label: autofocus_airy_r


:math:`S(z)` is bound to :math:`[0, 1]` with :math:`S=0` being completely defocused light and :math:`S=1` being an ideal focus where all the light is exactly in the center of the Gaussian function.
The center of the disc is determined by the average x- and y-coordinate of all rays.

The stimulus :math:`S_k` of a :math:`k`-times larger standard deviation is:

.. math::
   S_k = \frac{1}{\sqrt{k^2 + 1}}

Exemplary values for :math:`S_k`:

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

In the physical reality we can't get a higher value for :math:`S_k` than that for :math:`k=1` since this is equivalent to the resolution limit. 
Since the simulation does not factor in wave-optical properties, higher values can nevertheless appear in the raytracer.
Another disadvantage of the method is that it ignores all behavior of the beams far outside the sensitive range of the virtual receiver. 

.. _focus_rendering_methods:

Rendering Methods
==================

The next three methods render multiple images :math:`P_z` with pixel number :math:`N_\text{px} \cdot N_\text{px}`.
The side length pixel number :math:`N_\text{px}` is dependent on the number of rays used for focus finding. 
For few rays we want to keep the number low to minimize the effects of noise.
For a larger amount of rays we can increase the number step by step. 
This is needed to resolve small structures.
:math:`N` rays being distributed on a square area means we need to increase :math:`N_\text{px}` proportionally to :math:`\sqrt{N}` to achieve a somehow constant SNR. 
The formula implemented has the form :math:`N_\text{px} = \text{offset} + \text{factor} \cdot \sqrt{N}`.

The most outside rays define the image dimensions, the absolute image size therefore varies along the beam path. This can be an issue when few rays are far away from the optical axis, since the resolution suffers because of these marginal rays.

In contrast to the methods above, the following methods always use all rays available to achieve satisfying results. 
However, this can lead to long processing times for many million rays.


**Irradiance Variance**

Renders a power histogram for rays at position :math:`z`. 
This histogram is divided by pixel area to get an irradiance image :math:`E_z`
The approach then calculates the variance of the pixel values and finds the :math:`z` with the largest variance.

The variance is large when there are bright areas in the image (with much power per area) or if there is a large variance between pixels, which should be the case if unblurred structures are present.
For a minimization, the variance is inverted.
For a more smooth cost function and a better data range the square root of the variance is used.

.. math::
   \text{minimize}~~ I_\text{v}(z) := \frac{1}{\sqrt{\sigma^2(E_z)}}
   :label: autofocus_image


**Irradiance Maximum**

Similar to Irradiance Variance, but instead the maximum value in :math:`E_z` is maximized.

.. math::
   \text{minimize}~~ I_\text{p}(z) := \frac{1}{\sqrt{\text{max}(E_z)}}
   :label: autofocus_maximum

**Image Sharpness**

The power image :math:`P_z` is transformed into the Fourier domain, creating a Fourier power image :math:`p_f` with image frequencies :math:`f_x` and :math:`f_y`.
Using the Pythagorean theorem we can join the frequency components into a radial frequency.
The radial frequency of each pixel is scaled with the corresponding pixel power.
We want to maximize this product, which is large when there are many high frequency components in the original image :math:`P_z` or when high frequency components have a high power.

.. math::
   \text{minimize}~~ F_\text{p}(z) := \frac{N}{p_\text{f} \cdot \sqrt{f^2_x + f^2_y}}
   :label: autofocus_image_sharpness

For a minimization, the term is normalized by the pixel count :math:`N` and inverted.
This method is independent of the image size, as only the power image and not the irradiance map is employed.

A disadvantage of this method is that it tries to maximize the sharpness of the *whole* image.
Only a compromise solution is found for images with spatial varying blur.

