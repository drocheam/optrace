********************************
Miscellaneous
********************************

.. _index_from_abbe:

Curve from Abbe Number
============================

In many cases only refractive index and the Abbe number are known or provided. 
To simulate such materials a wavelength dependent model must be generated first.
While there are infinite possible curves that produce the same parameters, it is expected that real materials with the same index and Abbe number differ only slightly in the visible region, where these parameters are provided for.

We assume a model in the form of:

.. math::
   n(\lambda) = A + \frac{B}{\lambda^2 - d}
   :label: n_from_abbe_base

With :math:`d=0.014\, \mu\text{m}^2`, which is a compromise between the Cauchy (:math:`d=0`) and the Herzberger (:math:`d=0.028\,\mu\text{m}^2`) model.

With :math:`n_\text{s}:=n(\lambda_\text{s}),~n_\text{c}:=n(\lambda_\text{c}),~n_\text{l}:=n(\lambda_\text{l})` and the Abbe number equation in :math:numref:`n_from_abbe_base` one can solve for :math:`A,~B`:

.. math::
   B =&~ \frac{1}{V}\frac{n_\text{c}-1}{\frac{1}{\lambda^2_\text{s} - d} - \frac{1}{\lambda^2_\text{l}-d}}\\
   A =&~ n_\text{c} - \frac{B}{\lambda^2_\text{c}-d}
   :label: n_from_abbe_solution

Parameters :math:`V`, :math:`n_\text{c}` and the spectral lines :math:`\lambda_\text{s},~\lambda_\text{c},~\lambda_\text{l}` are provided by the user.

TiltedSurface Equation
============================

.. math::
   \text{normal vector:}~~~~   \vec{n} &= (n_x, n_y, n_z)\\
   \text{surface center vector:}~~~~ \vec{q} &= (x_0, y_0, z_0)\\
   \text{point on surface:}~~~~ \vec{p} &= (x, y, z)\\

point normal equation for a plane:

.. math::
   (\vec{p} - \vec{q})\cdot \vec{n} = 0
   :label: plane_normal_eq_tilted_surface

being equivalent to

.. math::
   (x - x_0) \cdot n_x + (y- y_0) \cdot n_y + (z-z_0)\cdot n_z = 0
   :label: tilted_surface0

can be rearranged to the surface function for :math:`n_z \neq 0`:

.. math::
   z(x, y) = z_0 - (x - x_0) \cdot \frac{n_x}{n_z} - (y- y_0) \cdot \frac{n_y}{n_z}
   :label: tilted_surface

Flipping and Rotation
=======================

Flipping a surface is implemented as a 180 degree rotation around the x-axis.

This is equivalent to negating its relative shape :math:`z_r` to an offset :math:`z_0` and mirroring the y-component: :math:`z_0 + z_r(x, y) \Rightarrow z_0 - z(x, -y)`.
For a surface with rotational symmetry this simplifies to :math:`z_0 + z_r(r) \Rightarrow z_0 - z(r)`

Rotation is done by changing the accessing coordinates through a rotation of the coordinate system:

:math:`z(x, y) \Rightarrow z(x_0 + r \cos \alpha, y_0 + r \sin \alpha)` with :math:`r = \sqrt{(x-x_0)^2 + (y - y_0)^2}`.
Here :math:`x_0,\; y_0` are the rotation center coordinates and :math:`\alpha` is the rotation angle.

By simply changing the value of the rotation angle the surface values can be rotated without actually rotating the surface.

.. _circle_sampling:

.. _ring_sampling:

Equal Area Radial Sampling of Ring and Circle
==================================================

An area element of a circle in polar coordinates can be represented as:

.. math::
   \text{d}A = \text{d}r  ~\text{d}\phi
   :label: ring_sampling_area_element

:math:`\text{d}\phi` can be rewritten as circle segment

.. math::
   \text{d}A = \text{d}r  ~\frac{2 \pi}{N} r
   :label: ring_sampling_area_element2

with :math:`N` being the number of segments.
Let us define a function :math:`r(u)` which gives us radial values and its derivative outputs radial spacing values.

.. math::
   \text{d}A = r'(u)  ~\frac{2 \pi}{N} r(u)
   :label: ring_sampling_area_element_diff_eq

For uniformly sampled data, :math:`\text{d}A` needs to be kept constant in regards to a uniform variable :math:`u`. This is equivalent to the condition :math:`\frac{\text{d}A}{\text{d}u} = 0`.

.. math::
   \frac{\text{d}A}{\text{d}u} = \frac{2\pi}{N} \frac{\text{d}}{\text{d}u} r'(u)  r(u) = r''(u) r(u) + (r'(u))^2 = 0
   :label: ring_sampling_area_element_diff_eq2

Solutions of this non linear differential equation of second order are in the form of

.. math::
   r(u) = \sqrt{c_1 + c_2 u}
   :label: ring_sampling_area_element_diff_eq_solution

For convenience we set the constants to :math:`c_1 = 0, ~c_2=1`. For output values in :math:`[r_i, ~R]` the corresponding input values are then :math:`[r^2_i, ~R^2]`. Rewriting :math:`r` and :math:`u` as random variables gives us:

.. math::
   \mathcal{R} = \sqrt{\mathcal{U}_{[r^2_\text{i}, R^2]}}
   :label: ring_sampling_R

The polar angle is uniformly spaced

.. math::
   \Phi = \mathcal{U}_{[0, 2\pi]}
   :label: ring_sampling_Phi

Resulting 3D positions are then

.. math::
   x =&~ x_0 + \mathcal{R} \cos \Phi\\ 
   y =&~ y_0 + \mathcal{R} \sin \Phi\\ 
   z =&~ z_0
   :label: ring_sampling_xyz


Circle sampling is implemented as ring with :math:`r_\text{i} = 0`.
Related: :footcite:`WolframDiskPicking`.

------------

**References**

.. footbibliography::
