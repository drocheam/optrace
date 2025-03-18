
*********************************
Tracing Procedure
*********************************

.. role:: python(code)
  :language: python
  :class: highlight

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
   :class: dark-light
   
   Tracing process flowchart.


Refraction
====================


Commonly found forms of the law of refraction are composed of an input and output angle. For the tracing process a form having only vectors as parameters, as well as circumventing the calculation of any angles, would be more convenient. 

The following figure shows the refraction of a ray on a curved surface.

.. figure:: ../images/refraction_interface.svg
   :width: 400
   :align: center
   :class: dark-light
   
   Refraction on a curved interface.

:math:`n_1, n_2` are the refractive indices of the media, :math:`s,s'` input and output propagation vectors. Both these vectors, as well as the normal vector :math:`n`, need to be normalized for subsequent calculations. Note that :math:`s` and :math:`n` need to point in the same half space direction, meaning :math:`s \cdot n \geq 0`.

An equation for such a form of the refraction law can be found in :footcite:`OptikHaferkorn` or :footcite:`Greve_2006`:

.. math::
   s^{\prime}=\frac{n_1}{n_2} s-n\left\{\frac{n_1}{n_2}(n s)-\sqrt{1-\left(\frac{n_1}{n_2}\right)^{2}\left[1-(n s)^{2}\right]}\right\}
   :label: refraction

In case of total internal reflection (TIR) the root argument becomes negative. In optrace TIR rays get absorbed, as reflections are not modelled, and the user is notified with a message.


.. _tracing_pol:

Polarization
====================

The following calculations are similar to :footcite:`Yun:11`.

The polarization vector :math:`E` can be decomposed into a :math:`E_\text{p}` -component, lying in the surface normal - incidence vector plane, and a :math:`E_\text{s}` -component lying perpendicular to this plane. With refraction on an interface the component :math:`E_\text{s}` is equal for both ray vectors :math:`s, s'`, while :math:`E_\text{p}` is rotated around :math:`E_\text{s}` towards :math:`s'` creating the new component :math:`E_\text{p}'`.
Note that for our calculations all vectors are unity vectors, while length information of the polarization components is contained in the scaling factors :math:`A_\text{tp}, A_\text{ts}`.

.. figure:: ../images/refraction_interface_polarization.svg
   :width: 700
   :align: center
   :class: dark-light

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
   :class: dark-light

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

Intersection Calculation
============================

Surface Extension
--------------------

To simplify the handling of non-intersecting rays and enforce the same number of ray sections for each ray, the surface is extended so all rays intersect a surface.
Gaps on the surface are filled and the surface edge is extended radially towards infinity.
A ray intersection is now calculated for the extended surface, but afterwards the ray is marked as hitting or non-hitting depending on a surface mask.

.. figure:: ../images/surface_extension.svg
   :width: 900
   :align: center
   :class: dark-light

   Surface Extension


Intersection of a Ray with a Plane
-----------------------------------

The intersection of all planar surface types (**CircularSurface**, **RectangularSurface**, **RingSurface**) as well as the **TiltedSurface** can be computed analytically.
The following definitions hold:

.. math::
   \text{surface normal vector:}~~~~   \vec{n} &= (n_x, n_y, n_z)\\
   \text{surface center vector:}~~~~ \vec{q} &= (x_0, y_0, z_0)\\
   \text{point on ray or surface:}~~~~ \vec{p} &= (x, y, z)\\
   \text{ray support vector:}~~~~ \vec{p_0} &= (x_{0p}, y_{0p}, z_{0p})\\

The surface point normal equation is:

.. math::
   (\vec{p} - \vec{q})\cdot \vec{n} = 0
   :label: plane_normal_eq_intersection

The ray equation in dependence of ray paramter :math:`t` is as follows:

.. math::
   \vec{p} = \vec{p_0} + \vec{s} \cdot t
   :label: line_equation_common

Inserting these equations into each other leads to

.. math::
    (\vec{p_0} + \vec{s}\cdot t_\text{h} - \vec{q}) \cdot \vec{n} = 0
   :label: plane_intersection_formula0

Rearranging gives us the ray parameter for the hit point :math:`t_\text{h}` for case :math:`\vec{s} \cdot \vec{n} \neq 0`:

.. math::
   t_\text{h} = \frac{(\vec{q} - \vec{p_0})\cdot \vec{n}}{\vec{s} \cdot \vec{n}}
   :label: plane_intersection_formula

which can be inserted into the ray equation to get the intersection position.
If :math:`\vec{s} \cdot \vec{n} \neq 0`, surface normal and ray direction vector are perpendicular.
If the ray lies inside the plane, there are infinite solutions, if it doesn't there is none.

Intersection of a Ray with a Conic Surface
--------------------------------------------

Intersections with a **ConicSurface** can also be calculated numerically.

.. math::
   \text{Ray support vector:}~~~~   \vec{p} &= (p_x, p_y, p_z)\\
   \text{Ray direction vector:}~~~~ \vec{s} &= (s_x, s_y, s_z)\\
   \text{Center of surface:}~~~~    \vec{q} &= (x_0, y_0, z_0)
   :label: IntersectionAsphere0

.. math::
   p_z + s_z t = z_0 + \frac{\rho r^2}{1 + \sqrt{1-(k+1)\rho^2 r^2}}
   :label: IntersectionAsphere1

with

.. math::
   r^2 = (p_x + s_x t - x_0)^2 + (p_y+s_y t - y_0)^2
   :label: IntersectionAsphere2

Some work in rearranging leads to the following equation, valid for :math:`\rho \neq 0` (conic section is not a plane):

.. math::
   A t^2 + 2 B t + C = 0
   :label: IntersectionAsphere3

with

.. math:: 
   A &= 1 + k s_z^2\\
   B &= o_x s_x + o_y s_y - \frac{s_z}{\rho} + (k+1) o_z s_z\\
   C &= o_x^2 + o_y^2 - 2\frac{o_z}{\rho} + (k+1) o_z^2\\
   \vec{o} &= \vec{p} - \vec{q} = (o_x, o_y, o_z)
   :label: IntersectionAsphere4

The solutions for :math:`t` are (:math:`\rho \neq 0`):

.. math::
   t = 
   \begin{cases}
       \frac{-B \pm \sqrt{B^2 -CA}}{A} & \text{for}~~ A \neq 0, ~~ B^2 - CA \geq 0 \\
       -\frac{C}{2B} & \text{for}~~ A = 0, ~~B \neq 0\\
       \{\mathbb{R}\} & \text{for}~~ A = 0, ~~B = 0, ~~C = 0\\
       \emptyset & \text{else}
   \end{cases}
   :label: IntersectionAsphere5

The first case produces two intersections, while in the second the ray touches the surface tangentially.
In the fourth case there is no intersection.
In cases one and two it has to be checked if the intersection lies inside the valid radial range of the surface, as the conic equation describes the surface for all possible :math:`r`.
If there are still two valid intersections in the first case, the one with the lower axial position is used.
The third case is only relevant for rays lying along a linear section of a surface, for instance a plane or a cone surface.
This is further explained below.
Equation :math:numref:`IntersectionAsphere5` produces the same results as :footcite:`Boussemaere_2021_8`, although in his derivation the constants :math:`A, B, C` include an additional curvature scaling and the factor 2 included in :math:`B`.

In the case of :math:`\rho = 0` (plane) equation :math:numref:`IntersectionAsphere1` simplifies to :math:`p_z + s_z t = z_0`, leading to:

.. math::
   t = 
   \begin{cases}
       - \frac{p_z - z_0}{s_z} & \text{for}~~ s_z \neq 0 \\
       \{\mathbb{R}\} & \text{for}~~ s_z = 0, p_z = z_0\\
       \emptyset & \text{for}~~ s_z = 0,~p_z\neq z_0\\
   \end{cases}
   :label: IntersectionAsphere6

The first case produces exactly one intersection and corresponds to the special case of equation :math:numref:`plane_intersection_formula` for :math:`\vec{n} = (0, 0, 1)`.
In the second case the ray lies inside the plane, producing infinite intersection positions.
In the third case the ray is parallel, but offset to the plane, producing no hits.
We enforce :math:`s_z > 0` in our simulator, so only the first case is relevant.

For :math:`\rho \rightarrow \infty, r \in \mathbb{R}` the conic section becomes a cone for :math:`k < -1`, a line from :math:`z = z_0` to :math:`z \rightarrow \infty` for :math:`k = -1` or a point at :math:`(x_0, y_0, z_0)` for :math:`k > -1`.
In the first two cases the conic sections consists of long linear segments.
Depending on the cases of equation :math:numref:`IntersectionAsphere5`, a ray can intersect once, twice, don't hit the surface or lie inside of it.
For a point (:math:`k > -1`) there can be no hit or a single one.
For the latter case :math:`B^2 - CA = 0` holds, producing a single intersection in the first case of equation :math:numref:`IntersectionAsphere5`.


.. _numerical_hit_find:

Numerical Hit Search
-----------------------

For the asphere equation or arbitrary analytical functions there is no analytical solution for a ray intersection.
In such cases the solution needs be calculated iteratively.

.. math::
   \text{Ray support vector:}~~~~   \vec{p_0} &= (p_x, p_y, p_z)\\
   \text{Ray direction vector:}~~~~ \vec{s} &= (s_x, s_y, s_z)\\
   \text{Point on Ray:}~~~~ \vec{p_t} &= (x_t, y_t, z_t)\\

Ray line equation depending on ray parameter :math:`t`:

.. math::
   \vec{p_t}(t)=\vec{p}_{0}+t \cdot \vec{s}
   :label: pt

Cost function :math:`G` with surface function :math:`f`:

.. math::
   G(t)=z_{t}-f\left(x_{t}, y_{t}\right)
   :label: G


The parameters :math:`x_t,y_t,z_t` can be determined from equation :math:numref:`pt`. 
For the position determination of the hit, the root of this scalar function :math:`G` must now be found. 
Typical optimization algorithms are suitable for this purpose. 

However, these have the disadvantage that they do not have a guaranteed convergence. 
Therefore, the ray tracer uses the Regula-Falsi method. 
This is a simple iterative method, which is guaranteed to converge superlinearly with a slight modification. 
The prerequisite for the procedure is, however, that an interval with a root is known.
Since the minimum and maximum extent of the surface in the z-direction in the raytracer is known, this criterion is given, because a hit can only occur within this range. The method basically works by trying to shrink the interval including the function root in every iteration. 
A well-written explanation can be found in :footcite:`RegulaFalsiWiki`.

Now, in some cases the interval may hardly decrease in size from one iteration to the next. 
To prevent slow convergence the procedure is therefore extended to the so called Illinois algorithm, which is explained in :footcite:`IllinoisAlgoWiki`.

The implementation in optrace differs only in parallelizing the optimization and only iterating the next step with not already converged rays.

Normal Calculation
====================

General
--------------------

Surface normals are required to calculate the refraction, polarization and transmittance of a ray.
Equation for an analytical, unnormalized normal vector: :footcite:`NormalWiki`

.. math::
   \vec{n} = 
   \begin{pmatrix}
        -\frac{\partial z}{\partial x}\\
        -\frac{\partial z}{\partial y}\\
        1\\
   \end{pmatrix}
   :label: normal_general

The vector needs to be normalized:

.. math::
   \vec{n}_0 = \frac{\vec{n}}{|| \vec{n} ||}
   :label: normal_numerical_norm


Planar Surfaces
--------------------

For surface types **Rectangle**, **Circle** and **Ring** :math:`n` is simply :math:`n = (0, 0, 1)`.
Surface type **TiltedSurface** provides a user specified value for :math:`n`, which is normalized internally.

Asphere 
---------------------------

The radial derivative of an asphere equation <> is

.. math::
   m_r = \frac{\partial z}{\partial r} = \frac{\rho r}{\sqrt{1 - (k+1)\rho^2 r^2}} + \sum_{i=1}^{m}  2i \cdot  a_i \cdot r^{2i - 1} = \tan \alpha
   :label: asphere_deriv

This radial component needs to be rotated around the vector :math:`(0, 0, 1)` by the angle :math:`\phi`.
According to the general normal calculation <> the x- and y- components of the unnormalized vector are:

.. math::
    \frac{\partial z}{\partial x} &= m_r \cos \phi\\
    \frac{\partial z}{\partial y} &= m_r \sin \phi\\
   :label: normal_general_asph

Which can be applied to equation <> above.

Conic Surface
--------------------

The derivative of the conic function is a simplifcation of the asphere case, where the polynomials are missing:

.. math::
   m_r = \frac{\rho r}{\sqrt{1 - (k+1)\rho^2 r^2}}
   :label: conic_derivative

:math:`m_r` is equivalent to :math:`\tan \alpha`, the tangens of angle between surface normal and vector :math:`(0, 0, 1)`.
By using geometrical relations, :math:`\sin \alpha` can be calculated, which provides the radial component :math:`n_r` of the normalized normal vector:

.. math::
   n_r = -\sin{\alpha} = -\frac{m_r}{\sqrt{m_r^2+1}} = -\frac{\rho r}{\sqrt{1- k\rho^2 r^2}}
   :label: conic_nr

The normalized vector :math:`n` for the **ConicSurface** is then:

.. math::
   \vec{n} = 
   \begin{pmatrix}
      n_r \cos \phi\\
      n_r \sin \phi\\
      \sqrt{1- n_r^2}
   \end{pmatrix}
   :label: conic_n


For a **SphericalSurface** this simplifies further to:

.. math::
   \vec{n} = 
   \begin{pmatrix}
        -\rho r \cos \phi \\
        -\rho {}r \sin \phi\\
        \sqrt{ 1 - \rho^2 r^2}\\
   \end{pmatrix}
   :label: sphere_n


FunctionSurface
--------------------

For types **FunctionSurface1D**, **FunctionSurface2D** the derivatives needed for the normal vector are calculated numerically.
One exception is the case, where a derivative function was provided to the object.

The central first derivative has the form:

.. math::
   f'(x) = \frac{f(x+\varepsilon) - f(x-\varepsilon)}{2 \varepsilon}
   :label: central_first_deric
    
.. math::
    \frac{\partial z}{\partial x} &\approx \frac{z(x + \varepsilon, ~y) - z(x - \varepsilon, ~y)}{2\varepsilon}\\
    \frac{\partial z}{\partial y} &\approx \frac{z(x, ~y + \varepsilon) - z(x, ~y - \varepsilon)}{2\varepsilon}\\

In the case of a **FunctionSurface1D** both derivatives are equal and only one value needs to be calculated.
The step width is chosen as maximum of optimal step width :math:`\varepsilon_\text{o}` and minimal positional step width :math:`\varepsilon_\text{p}`:

.. math::
   \varepsilon = \max (\varepsilon_\text{o}, ~\varepsilon_\text{n})
   :label: eps_selection


The optimal step width is :footcite:`DiffIntMorken`:

.. math::
   \varepsilon_\text{o} = \sqrt[3]{3 \varepsilon_\text{f} \left| \frac{f(x)}{f^{(3)}(x)} \right|} 
   :label: optimal_step_width

with :math:`\varepsilon_\text{f}` being the machine precision for the used floating type.
Expecting mostly spherical surfaces, the main function component is :math:`x^2`.
Higher polynomial orders are less prominent, so one valid assumption can be :math:`\left| \frac{f(x)}{f^{(3)}(x)} \right| = 50`. 
While this might be different for every function, due to the forth root a quotient being 1000 times larger only leads to a change of around factor :math:`10` in :math:`\varepsilon_\text{0}`.
With :math:`\varepsilon_\text{f} \approx 2.22\cdot 10^{-16}` for a 64bit floating point number, we get :math:`\varepsilon_\text{o} \approx 3.22 \cdot 10^{-5}`.
Optrace units are given in millimeters, so this is equivalent to a value of :math:`32.2\,` nm.

Not only differences in :math:`f` need to be representable, but :math:`x+\varepsilon` needs to be different from :math:`x`.
For this it must be ensured, that :math:`x+ \varepsilon > x (1 + \varepsilon_\text{f})` for every coordinate :math:`x` on the surface.
With :math:`r_\text{max}` being the largest absolute distance on the surface the minimal bound is

.. math::
   \varepsilon_\text{p} = r_\text{max} ~\varepsilon_\text{f}
   :label: machine_eps_scaling

It is recommended to center the surface at :math:`x=0` before differentiation so :math:`r_\text{max}` is minimal. 


DataSurface
----------------------------

lorem ipsum

------------

**References**

.. footbibliography::

