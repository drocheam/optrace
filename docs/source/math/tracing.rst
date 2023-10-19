
*********************************
Tracing Procedure
*********************************


Tracing Process
========================

Main steps of the tracing process are surface hit detection, refraction index and direction calculation as well as handling of non-hitting rays.
For ideal lenses, filters and apertures only one surface hit detection takes place, while a normal lens requires two.
Doing almost all calculations in threads speeds up the computations.
Every thread gets some rays assigned which are only processed within the same thread.
This includes the ray creation and propagation through the whole system.
While not explicitly mentioned, new ray polarization and weights are calculated in the refraction calculation function.

.. figure:: ../images/TracePAP.svg
   :width: 400
   :align: center
   
   Tracing process flowchart.


Refraction
====================


Commonly found forms of the law of refraction are composed of an input and output angle. For the tracing process a form having only vectors as parameters, as well as circumventing the calculation of any angles, would be more convenient. 

The following figure shows the refraction of a ray on a curved surface.

.. figure:: ../images/refraction_interface.svg
   :width: 300
   :align: center
   
   Refraction on a curved interface.

:math:`n_1, n_2` are the refractive indices of the media, :math:`s,s'` input and output propagation vectors. Both these vectors, as well as the normal vector :math:`n`, need to be normalized for subsequent calculations. Note that :math:`s` and :math:`n` need to point in the same half space direction, meaning :math:`s \cdot n \geq 0`.

An equation for such a form of the refraction law can be found in :footcite:`OptikHaferkorn` or :footcite:`Greve_2006`:

.. math::
   s^{\prime}=\frac{n_1}{n_2} s-n\left\{\frac{n_1}{n_2}(n s)-\sqrt{1-\left(\frac{n_1}{n_2}\right)^{2}\left[1-(n s)^{2}\right]}\right\}
   :label: refraction

In case of total internal reflection (TIR) the root argument becomes negative. In `optrace` TIR rays get absorbed, as reflections are not modelled, and the user is notified with a message.


.. _tracing_pol:

Polarization
====================

The following calculations are similar to :footcite:`Yun:11`.

The polarization vector :math:`E` can be decomposed into a :math:`E_\text{p}` -component, lying in the surface normal - incidence vector plane, and a :math:`E_\text{s}` -component lying perpendicular to this plane. With refraction on an interface the component :math:`E_\text{s}` is equal for both ray vectors :math:`s, s'`, while :math:`E_\text{p}` is rotated around :math:`E_\text{s}` towards :math:`s'` creating the new component :math:`E_\text{p}'`.
Note that for our calculations all vectors are unity vectors, while length information of the polarization components is contained in the scaling factors :math:`A_\text{tp}, A_\text{ts}`.

.. figure:: ../images/refraction_interface_polarization.svg
   :width: 620
   :align: center

   Ray polarization components before and after refraction.


**Case 1**:

For :math:`s \parallel s'` the new polarization vector is equal to the old one.

**Case 2**

For :math:`s \nparallel s'` the new polarization vector differs from the old one.

According to optics the polarization and polarization components need to be orthogonal to the propagation direction. 
Additionally, both polarization components are perpendicular to each other. Assuming all mentioned vectors are unity vectors, we can calculate:

.. math::
    \begin{align}
    E_\text{s} &= \frac{s' \times s}{|| s' \times s ||}\\
    E_\text{p} &= E_\text{s} \times s\\
    E_\text{p}' &= E_\text{s} \times s'\\
    \end{align}
    :label: pol_E

Since :math:`||E_\text{p}|| = ||E_\text{s}|| = ||E|| = 1` the amplitude components are then:

.. math::
   \begin{align}
        A_\text{tp} &= E_\text{p} \cdot E\\
        A_\text{ts} &= E_\text{s} \cdot E\\
   \end{align}
   :label: pol_A

For the new polarization unity vector, which also composed of two components, we finally get

.. math::
   E' = A_\text{ps} E_\text{s} + A_\text{tp} E_\text{p}'
   :label: pol_E2

Transmission
====================

The new ray powers are calculated from the transmission which in turn can be calculated from already derived properties in refraction calculation.

According to the Fresnel equations the transmission of light is dependent on the polarization direction.
The subsequent equations describe this behavior :footcite:`FresnelWiki`.

.. math::
   t_{\mathrm{s}}=\frac{2\, n_{1} \cos \varepsilon}{n_{1} \cos \varepsilon+n_{2} \cos \varepsilon'}
   :label: ts_coeff

.. math::
   t_{\mathrm{p}}=\frac{2\, n_{1} \cos \varepsilon}{n_{2} \cos \varepsilon+n_{1} \cos \varepsilon'}
   :label: tp_coeff

.. math::
   T=\frac{n_{2} \cos \varepsilon'}{n_{1} \cos \varepsilon} \left( (A_\text{ts} t_\text{s})^2  + (A_\text{tp} t_\text{p})^2 \right)
   :label: T

:math:`A_\text{ts}` and :math:`A_\text{tp}` are the polarization components from equations :math:numref:`pol_A`. Occurring cosine terms are calculated from the direction and normal vectors as :math:`\cos \varepsilon = n \cdot s` and :math:`\cos \varepsilon' = n \cdot s'`.


For light hitting the surface perpendicular this yields an expression independent of the polarization: :footcite:`Kaschke2014`

.. math::
   T_{\varepsilon=0} = \frac{4 n_1 n_2 }{(n_1 + n_2)^2}
   :label: T_special


Refraction at an Ideal Lens
===========================


Ray with unnormalized direction vector :math:`s_0` and intersection :math:`P = (x_0, y_0, 0)` on the lens with focal length :math:`f` and the corresponding point on the focal plane :math:`P_f = (x_f, y_f, f)`.
Optics tells us that ideally parallel rays meet in the same position in the focal plane. Therefore a ray with the same direction, but hitting the lens at the optical axis, can used to determine position :math:`P_f`.

.. _image_ideal_refraction:

.. figure:: ../images/ideal_refraction.svg
   :width: 500
   :align: center

   Geometry for refraction on an ideal lens.

**Cartesian Representation**

Calculating positions :math:`x_f,~y_f` is simply done calculating the linear ray equations :math:`x(z), y(z)` at :math:`z=f`.
For :math:`x_f` we get:

.. math::   
   x_f = \frac{s_{0x}}{s_{0z}} f
   :label: refraction_ideal_xf

Similarly for :math:`y_f`

.. math::
   y_f = \frac{s_{0y}}{s_{0z}} f
   :label: refraction_ideal_yf

:math:`s_{0z} = 0` is prohibited by forcing all rays to have a positive z-direction component.

Knowing point :math:`P_f` the outgoing propagation vector :math:`s_0'` is calculated.

.. math::
   s_0' = P_f - P = \begin{pmatrix} \frac{s_{0x}}{s_{0z}}f - x_0 \\ \frac{s_{0y}}{s_{0z}}f - y_0 \\ f \end{pmatrix}
   :label: refraction_ideal_s0


Normalizing gets us:

.. math::
   s' = \frac{s_0'}{||s_0'||}
   :label: refraction_ideal_s0_normalized



**Angular Representation**

Taking the x-component of the propagation vector

.. math::
   s_{0x}' = \frac{s_{0x}}{s_{0z}}f - x_0

and dividing it by :math:`f` gives us

.. math::
   \frac{s_{0x}'}{f} = \frac{s_{0x}}{s_{0z}} - \frac{x_0}{f}

From :numref:`image_ideal_refraction` follows :math:`\tan \varepsilon_x' = \frac{s_{0x}}{f}` and :math:`\tan \varepsilon_x = \frac{s_{0x}}{s_{0z}}` and therefore

.. math::
   \tan \varepsilon_x' = \tan \varepsilon_x - \frac{x_0}{f}

Analogously in y-direction we get

.. math::
   \tan \varepsilon_y' = \tan \varepsilon_y - \frac{y_0}{f}

This angular representation is a formulation also found in :footcite:`BRULS2015659`.


Filtering
==================

When passing through a filter a ray with power :math:`P_i` and wavelength :math:`\lambda_i` gets attenuated according to the filter's transmission function :math:`T_\text{F}(\lambda)`:

.. math::
   P_{i+1} = 
   \begin{cases}
        P_{i}~ T_\text{F}(\lambda_i) & \text{for}~~ T_\text{F}(\lambda_i) > T_\text{th}\\
        0  & \text{else}\\
   \end{cases}
   :label: eq_filtering


Additionally, ray powers get set to zero if the transmission falls below a specific threshold :math:`T_\text{th}`. By doing so, *ghost rays* are avoided, these are rays that still need to be propagated while raytracing, but hold only little power. Because their contribution to image forming is negligible, they should be absorbed as soon as possible to speed up tracing.

As a side note, apertures are also implemnted as filters, but with :math:`T_\text{F}(\lambda) = 0` for all wavelengths.

Geometry Checks
==========================

Geometry checks before tracing include:

 * all tracing revelant elements must be inside the outline
 * no object collisions
 * defined, sequential order
 * raysources available
 * all raysources come before all other kinds of elements

Collision checks are done by first sorting the elements and then comparing positions on adjacent surfaces.
After randomly sampling many points it needs to be checked if the position order in z-direction is equal.
While this doesn't guarantee no collisions, while raytracing the sequentiality is checked for each ray and warnings are emitted.

Outline Intersection
========================

After each surface hit calculation at the current surface the rays from the last surface are checked for a collision with the outline. 
This is done by calculating an intersection of the ray with the six faces of the outline box.
Since the intersection of a line and plane is straightforward, the calculations are quite simple.
Only the nearest hit in positive direction of all six intersections is used.
If the outline is hit before the collision with the current non-outline surface, the rays are absorbed at the outline.

Abnormal Lens Rays
==========================

If rays don't hit both front and back surface of a lens, they either 

 1. miss the lens completely 
 2. hit only one of these surfaces while passing through the lens cylinder, which is the surface connecting both front and back
 3. hit the lens cylinder twice

Case 1 is valid behavior that doesn't need to be adressed.
The cylinder surface behavior is not modelled, so we are forced to absorb these rays and output a warning message in case 2.
Currently there is no differentiation between cases 1 and 3, which for the latter is inconvient, as it is treated as passing through the lens without interaction.
This is due to missing collision detection with the cylinder surface, clearly a bug and maybe will be fixed in the future.
However, setting the raytracer options to ``absorb_missing=True`` absorbs both rays from cases 1 and 3.


Hit Finding
=============================

Hit finding for analytical surfaces is described in :numref:`analytical_hit_find` and for numerical/user function surfaces in :numref:`numerical_hit_find`.


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
The pixel count is higher as requested, as each image is rendered in a higher resolution to allow for resolution changes after rendering, see :numref:`rimage_overview`.

Image rendering is also done in threads.
The created RImage object hold images for the three tristimulus values X, Y, Z, that can encompass all human-visible colors.
The illuminance image can be directly calculated from the Y component and the pixel size, so an explicit rendering of this image is not required.
The fourth image is an irradiance image, which is calculated from the ray powers and the pixel sizes.
The aforementioned threads each get one of the four images.

After image rendering the image is optionally filtered with an Airy-disk resolution filter and then rescaled to the desired resolution.

.. figure:: ../images/DetectorPAP.svg
   :width: 400
   :align: center
   
   Detector intersection and image rendering flowchart.


Spectrum Rendering
====================

Spectrum rendering works in a similar way to image rendering.
Ray intersections are calculated, only hitting rays are selected and a histogram is rendered.
But compared to an image, this is a spectral histogram within a wavelength range resulting from the rays wavelengths and powers.
Instead of an ``RImage`` a ``LightSpectrum`` object is created with type ``"Histogram"``.


The number of bins for the histogram is:

.. math::
   N_\text{b} = 1 + 2 \; \text{floor} \left(\frac{ \text{max}\left( 51, \frac{\sqrt{N}}{2}\right)} {2}\right)

This formula ensures :math:`N_\text{b}` is odd, so the center is well-defined.
Independent of the number of rays :math:`N` the minimum of bins is set to 51 and scales with the square root of this number above a specific value.
The latter is due to the SNR of the mean also increasing with :math:`\sqrt{N}` for normal-distributed noise.
So the number of bins is adapted so that the SNR stays the same, but the spectrum resolution increases.

------------

**References**

.. footbibliography::

