
*************************************
Heisenberg Uncertainty Ray Bending
*************************************

Overview
====================

.. math::
   \Delta x \Delta p_x \geq \frac{\hbar}{2}

.. math::
   p = \hbar k = \hbar \frac{2 \pi}{\lambda} = \frac{n(\lambda_0) h}{\lambda_0}

.. math::
   \tan \sigma_x = \frac{\Delta p_x}{p}

.. math::
   \tan \sigma_y = \frac{\Delta p_y}{p}

.. math::
   \tan \sigma_x = \frac{1}{2\Delta x k}

.. math::
   \tan \sigma_y = \frac{1}{2\Delta y k}

.. math::
   p\left(\tan \theta_x, \tan \theta_y\right)=\frac{1}{2 \pi \tan \sigma_x \tan \sigma_y}
   \exp\left(-\frac{\tan^2 \theta_x}{2 \tan^2 \sigma_x} -\frac{\tan^2 \theta_y}{2 \tan^2 \sigma_y}\right)


Distances
================================

**Overview**

Instead of using uncertainties :math:`\Delta x, \Delta y` for axes :math:`x,y`, 
uncertainties :math:`\Delta a, \Delta b` for orthogonal axes :math:`a,b` inside the aperture plane are more useful. 

**Rectangle**

Rectangle with width :math:`B` and height :math:`A`, centered at :math:`(x_0, y_0)`.
The distance of a ray passing through the aperture at :math:`(x, y)` to both edges is then:

.. math::
   \Delta a &= \lvert y - y_0 \rvert - A\\
   \Delta b &= \lvert x - x_0 \rvert - B

With the direction vectors to the edges being equivalent to unity vectors in x and y direction:

.. math::
   b = \Delta b \left(\begin{array}{c}
   1 \\
   0 \\
   0
   \end{array}\right)

.. math::
   a = \Delta a \left(\begin{array}{c}
   0 \\
   1 \\
   0
   \end{array}\right)

For a rotated rectangle the ray positions need to be converted 
into the rotated rectangle reference coordinate system by rotation,
while the direction vectors :math:`a, b` need to be converted from the rotated system to absolute coordinates.

**Circle**

A circle with radius :math:`R` is centered at :math:`(x_0, y_0)`.
The polar coordinates in the aperture coordinate system are then:

.. math::
   r &= \sqrt{(x-x_0)^2 + (y-y_0)^2}\\
   \phi &= \text{atan2}(y-y_0, x-x_0)\\

The minor axis of the ellipse is:

.. math::
   \Delta b = R - r

The major axis length :math:`a` is calculated by matching the curvature of the ellipse and the aperture,
this results in:

.. math::
   \Delta a = \sqrt{\Delta b R}

The distance vectors are then:

.. math::
   b = \Delta b \left(\begin{array}{c}
   \cos \phi \\
   \sin \phi \\
   0
   \end{array}\right)

.. math::
   a = \Delta a \left(\begin{array}{c}
   \sin \phi \\
   -\cos \phi \\
   0
   \end{array}\right)


Direction
================================

.. math::
   b'_0 &= \frac{a_0 \times s}{\lvert a_0 \times s\rvert}\\
   a'_0 &= \frac{s \times b'_0}{\lvert s \times b'_0\rvert}

.. math::
   \Delta a' &= \Delta a \cos \psi_a\\
   \Delta b' &= \Delta b \cos \psi_b\\

.. math::
   s' = \frac{s + b'_0 \lvert s \rvert  \tan \theta_{b'} + a'_0 \lvert s \rvert  \tan \theta_{a'}}
   {\Big\lvert s + b'_0 \lvert s \rvert  \tan \theta_{b'} + a'_0 \lvert s \rvert  \tan \theta_{a'} \Big\rvert}

Assuming :math:`s` as unity vector and simplifying using the orthogonality of :math:`s, a'_0, b'_0`:

.. math::
   s' = \frac{s + b'_0 \tan \theta_{b'} + a'_0 \tan \theta_{a'}}
   {\sqrt{ 1 + \tan^2 \theta_{b'} + \tan^2 \theta_{a'}}}

------------

**References**

.. footbibliography::

