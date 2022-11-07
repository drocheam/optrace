
*****************
Geometry
*****************

Point
===============

Defined by its coordinate :math:`x_0,~y_0,~z_0` and is only defined there.

Line
===============

Lwt :math:`R` be half the length of the line and :math:`\alpha` its angle relative to the x-axis.

The random cartesian coordinates are then

The line is then defined for :math:`r \in [-R, ~R]`

.. math::
   x =&~ x_0 + r \cos \alpha\\ 
   y =&~ y_0 + r \sin \alpha\\ 
   z =&~ z_0

Surfaces
===============

Conic
--------------------

.. math::
   :label: conic

   z(x,~y)= z_0 + \frac{\rho r^{2}}{1+\sqrt{1-(1+k)(\rho r)^{2}}}


with

.. math::
   r^2 = (x-x_0)^2 + (y-y_0)^2
   :label: asphere_r

and the surface center

.. math::
   \vec{q} = (x_0, y_0, z_0)


Sphere
--------------------

Handled as conic with :math:`k=0`


Rectangle
--------------------

Surface with function :math:`z(x,~y)=z_0` and rectangular extent in the xy-plane,
while its sides are parallel to the x and y-axis.
Its center lies at :math:`\vec{q} = (x_0, y_0, z_0)`.

Ring
-------------

Surface with function :math:`z(x,~y)=z_0` and being defined by a circular area around
:math:`\vec{q} = (x_0, y_0, z_0)` with radius :math:`r` and inner radius :math:`0 < r_i < r`


Circle
-------------

Ring with :math:`r_i = 0`.


Tilted Surface
--------------------

.. math::
   \text{normal vector:}~~~~   \vec{n} &= (n_x, n_y, n_z)\\
   \text{surface center vector:}~~~~ \vec{q} &= (x_0, y_0, z_0)\\
   \text{point on surface:}~~~~ \vec{p} &= (x, y, z)\\

point normal equation for a plane:

.. math::
   (\vec{p} - \vec{q})\cdot \vec{n} = 0

being equivalent to

.. math::
   (x - x_0) \cdot n_x + (y- y_0) \cdot n_y + (z-z_0)\cdot n_z = 0

can be rearranged to the surface function for :math:`n_z \neq 0`:

.. math::
   z(x, y) = z_0 - (x - x_0) \cdot \frac{n_x}{n_z} - (y- y_0) \cdot \frac{n_y}{n_z}


Hit Detection
===============

A numerical method can be found in {}

Intersection Ray with xy-Plane
-----------------------------------

.. math::
   \text{surface normal vector:}~~~~   \vec{n} &= (n_x, n_y, n_z)\\
   \text{surface center vector:}~~~~ \vec{q} &= (x_0, y_0, z_0)\\
   \text{point on ray or surface:}~~~~ \vec{p} &= (x, y, z)\\
   \text{ray support vector:}~~~~ \vec{p_0} &= (x_{0p}, y_{0p}, z_{0p})\\

surface point normal equation:

.. math::
   (\vec{p} - \vec{q})\cdot \vec{n} = 0

ray equation in depencence of ray paramter :math:`t`:

.. math::
   \vec{p} = \vec{p_0} + \vec{s} \cdot t

inserting these equations into each other leads to

.. math::
    (\vec{p_0} + \vec{s}\cdot t_\text{h} - \vec{q}) \cdot \vec{n} = 0

rearranging gives us the ray parameter for the hit point :math:`t_\text{h}`:

.. math::
   t_\text{h} = \frac{(\vec{q} - \vec{p_0})\cdot \vec{n}}{\vec{s} \cdot \vec{n}}

which can be inserted into the ray equation to get the hit point

Intersection Ray with a Conic
--------------------------------------

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

Some work in rearanging leads to

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

The solutions for :math:`t` are

.. math::
   t = 
   \begin{cases}
       \frac{-B \pm \sqrt{B^2 -CA}}{A} & \text{for}~~ A \neq 0, ~~ B^2 - CA \geq 0 \\
       -\frac{C}{2B} & \text{for}~~ A = 0, ~~B \neq 0\\
       \{\mathbb{R}\} & \text{for}~~ A = 0, ~~B = 0, ~~C = 0\\
       \emptyset & \text{else}
   \end{cases}
   :label: IntersectionAsphere5

Surface Extension
--------------------


.. figure:: images/Oberflächen_Erweiterung.svg
   :width: 700
   :align: center

   Surface Extension


Normal Calculation
====================

General
--------------------


.. math::
   \vec{n_0} = 
   \begin{pmatrix}
        -\frac{\partial z}{\partial x}\\
        -\frac{\partial z}{\partial y}\\
        1\\
   \end{pmatrix}
   :label: normal_general

Needs to be normalized using

.. math::
   \vec{n} = \frac{\vec{n_0}}{\lvert n_0 \rvert}
   :label: normal_general_norm

Numerical
--------------------

.. math::
   \vec{n_0} = 
   \begin{pmatrix}
        z(x - \varepsilon, ~y) - z(x + \varepsilon, ~y)\\
        z(x, ~y - \varepsilon) - z(x, ~y + \varepsilon)\\
        \varepsilon\\
   \end{pmatrix}
   :label: normal_numerical

Needs to be normalized using

.. math::
   \vec{n} = \frac{\vec{n_0}}{\lvert n_0 \rvert}
   :label: normal_numerical_norm


See {} for notes on choosing :math:`\varepsilon`.


Plane
--------------------

.. math::
   \vec{n} = 
   \begin{pmatrix}
        0\\
        0\\
        1\\
   \end{pmatrix}
   :label: normal_plane

Conic
--------------------

The derivative of the conic function is

.. math::
   m = \tan{\alpha} = \frac{\text{d}z(r)}{\text{d}r} = \frac{\rho r}{\sqrt{1 - (k+1)\rho^2 r^2}}
   :label: conic_derivative

.. math::
   n_r = -\sin{\alpha} = -\frac{m}{\sqrt{m^2+1}} = -\frac{\rho r}{\sqrt{1- k\rho^2 r^2}}
   :label: conic_nr

.. math::
   n_x &= n_r \cos \phi\\
   n_y &= n_r \sin \phi\\
   n_z &= \sqrt{1- n_r^2}
   :label: conic_nxyz

.. math::
   \vec{n} = 
   \begin{pmatrix}
        n_x\\
        n_y\\
        n_z\\
   \end{pmatrix}
   :label: conic_n


Sphere
--------------------

With :math:`k=0` and :math:`\rho := \frac{1}{R}` the conic normal simplifies to

.. math::
   \vec{n} = 
   \begin{pmatrix}
        -\rho r \cos \phi \\
        -\rho {}r \sin \phi\\
        \sqrt{ 1 - \rho^2 r^2}\\
   \end{pmatrix}
   :label: sphere_n


Numerical Differentiation
=============================

Source step width equations: http://www.uio.no/studier/emner/matnat/math/MAT-INF1100/h08/kompendiet/diffint.pdf

the C++ boost library does it in a similar way, but also ensuring x+9h is representable:
https://github.com/boostorg/math/blob/1ce6dda2fb9b8d3cd2f54c76fe5a8cc3d0e430f9/include/boost/math/differentiation/finite_difference.hpp


Central first derivative
--------------------------

For a first derivative of the form

.. math::
   f'(x) = \frac{f(x+\varepsilon) - f(x-\varepsilon)}{2 \varepsilon}

the optimal step width is

.. math::
   \varepsilon_\text{o} = \sqrt[3]{3 \varepsilon_\text{f} \left| \frac{f(x)}{f^{(3)}(x)} \right|} 

with :math:`\varepsilon_\text{f}` being the machine precision for the used floating type.
Expecting mostly spherical surfaces, the main function component is :math:`x^2`.
Higher polynomial orders are less prominent, so one valid assumption can be :math:`\left| \frac{f(x)}{f^{(3)}(x)} \right| = 50`. 
While this might be different for every function, due to the forth root a quotient being 1000 times larger only leads to a change of around factor :math:`10` in :math:`\varepsilon_\text{0}`.

With :math:`\varepsilon_\text{f} \approx 2.22\cdot 10^{-16}` for a 64bit floating point number, we get :math:`\varepsilon_\text{o} \approx 3.22 \cdot 10^{-5}`.
Optrace units are given in millimeters, so this is equivalent to a value of :math:`32.2\,` nm.


Step Width Selection
---------------------------------

Not only differences in :math:`f` need to be representable, but :math:`x+\varepsilon` needs to be different from :math:`x`.
For this it must be ensured, that :math:`x+ \varepsilon > x (1 + \varepsilon_\text{f})` for every coordinate :math:`x` on the surface.
With :math:`R` being the largest absolute distance on the surface the minimal bound is

.. math::
   \varepsilon_\text{n} = R ~\varepsilon_\text{f}

It is recommended to center the surface at :math:`x=0` so :math:`R` is minimal. This only works if the surface is centered beforehand, shifting afterwards also ruins numerical precision.

The finally chosen step width is the higher one:

.. math::
   \varepsilon = \max (\varepsilon_\text{o}, ~\varepsilon_\text{n})


Curvature Circle
=======================

TODO

Sphere Projections
=========================

The relative distance to center and the z-position of the other sphere end are

.. math::
   r &= \sqrt{(x-x_0)^2  + (y - y_0)^2}\\
   z_m &= z_0 + R

**Equidistant**

https://en.wikipedia.org/wiki/Azimuthal_equidistant_projection

.. math::
   \theta &= \arctan\left(\frac{r}{z-z_m}\right)\\
   \phi &= \text{arctan2}(y-y_0, ~x-x_0)\\

The projected coordinates are then

.. math::
   x_p &= -\theta \cdot \text{sgn}(R) \cos(\phi)\\
   y_p &= -\theta \cdot \text{sgn}(R) \sin(\phi)\\

**Stereographic**

https://en.wikipedia.org/wiki/Stereographic_map_projection

.. math::
   \theta &= \frac{\pi}{2} - \arctan\left(\frac{r}{z-z_m}\right)\\
   \phi &= \text{arctan2}(y-y_0, ~x-x_0)\\
   r &= 2 \tan\left(\frac{\pi}{4} - \frac{\theta}{2}\right)\\
   
The projected coordinates are then

.. math::
   x_p &= -r \cdot  \text{sgn}(R) \cos(\phi)\\
   y_p &= -r \cdot \text{sgn}(R) \sin(\phi)\\

**Equal-Area**

https://en.wikipedia.org/wiki/Lambert_azimuthal_equal-area_projection

.. math::
   x_r = \frac{x - x_0} {\lvert R \rvert}\\
   y_r = \frac{y - y_0} {\lvert R \rvert}\\
   z_r = \frac{z - z_m} {R}\\

The projected coordinates are then

.. math::
   x_p = \sqrt{\frac{2}{1-z_r} x_r}\\
   y_p = \sqrt{\frac{2}{1-z_r} y_r}\\


Random Sampling
=======================

Point
--------------

Since a point only has one position :math:`x_0,~y_0,~z_0`, all random values have these coordinates.

Line
-------------

Generate a uniform variable :math:`\mathcal{U}_\text{[-R,R]}` with :math:`R` being half the length of the line.

The random cartesian coordinates are then

.. math::
   x =&~ x_0 + \mathcal{U}_{[-R,R]} \cos \alpha\\ 
   y =&~ y_0 + \mathcal{U}_{[-R,R]} \sin \alpha\\ 
   z =&~ z_0

:math:`x_0,~y_0,~z_0` are the central coordinates of the line and :math:`\alpha` is its angle relative to the x-axis.

Rectangle
---------------

For uniform random positions on a rectangular surface we need to generate two independent random uniform variables, where each range is the extent :math:`[x_0,~x_1,~y_0,~y_1]` of the rectangle.

The random cartesian coordinates are then

.. math::
   x =&~ \mathcal{U}_{[x_0,x_1]}\\
   y =&~ \mathcal{U}_{[y_0,y_1]}\\
   z =&~ z_0

Ring
--------------

An area element of a circle in polar coordinates can be represented as:

.. math::
   \text{d}A = \text{d}r  ~\text{d}\phi

:math:`\text{d}\phi` can be rewritten as circle segment

.. math::
   \text{d}A = \text{d}r  ~\frac{2 \pi}{N} r

with :math:`N` being the number of segments.
Let us define a function :math:`r(u)` which gives us radial values and its derivative outputs radial spacing values.

.. math::
   \text{d}A = r'(u)  ~\frac{2 \pi}{N} r(u)

For uniformly sampled data, :math:`\text{d}A` needs to be kept constant in regards to a uniform variable :math:`u`. This is equivalent to the condition :math:`\frac{\text{d}A}{\text{d}u} = 0`.

.. math::
   \frac{\text{d}A}{\text{d}u} = \frac{2\pi}{N} \frac{\text{d}}{\text{d}u} r'(u)  r(u) = r''(u) r(u) + (r'(u))^2 = 0

Solutions of this non linear differential equation of second order are in the form of

.. math::
   r(u) = \sqrt{c_1 + c_2 u}

For convenience we set the constants to :math:`c_1 = 0, ~c_2=1`. For output values in :math:`[r_i, ~R]` the corresponding input values are then :math:`[r^2_i, ~R^2]`. Rewriting :math:`r` and :math:`u` as random variables gives us:

.. math::
   \mathcal{R} = \sqrt{\mathcal{U}_{[r^2_\text{i}, R^2]}}

The polar angle is uniformly spaced

.. math::
   \Phi = \mathcal{U}_{[0, 2\pi]}

Resulting 3D positions are then

.. math::
   x =&~ x_0 + \mathcal{R} \cos \Phi\\ 
   y =&~ y_0 + \mathcal{R} \sin \Phi\\ 
   z =&~ z_0

Circle
------------

Implemented as ring with :math:`r_\text{i} = 0`


