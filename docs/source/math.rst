Algorithms
----------------------------


Surfaces
______________________________________

Asphere
*********************************

.. math::
   :label: asphere

   z(x,~y)= z_0 + \frac{\rho r^{2}}{1+\sqrt{1-(1+k)(\rho r)^{2}}}


with

.. math::
   r = \sqrt{(x-x_0)^2 + (y-y_0)^2}
   :label: asphere_r

Sphere
*********************************
Handled as asphere with :math:`k=0`


Planar Surfaces
*********************************

Includes **Circle**, **Ring** and **Rectangle**.

Surface with center at :math:`(x_0, y_0, z_0)` and surface function :math:`z(x,~y)=z_0`.


Sampling
______________________________________

.. figure:: images/Random_Sampling_Comparison.svg
   :width: 600
   :align: center

   Sampling Comparison


Refraction
______________________________________

Direction
*********************************

.. figure:: images/Vektorielle_Brechung2-fs8.png
   :width: 200
   :align: center
.. figure:: images/Vektorielle_Brechung-fs8.png
   :width: 300
   :align: center

   Images and Equation: :cite:`OptikHaferkorn`

.. math::
   s^{\prime}=\frac{n_1}{n_2} s-n\left\{\frac{n_1}{n_2}(n s)-\sqrt{1-\left(\frac{n_1}{n_2}\right)^{2}\left[1-(n s)^{2}\right]}\right\}
   :label: refraction

Polarization
*********************************

Transmission
*********************************

Source: :cite:`FresnelWiki`

.. math::
   t_{\mathrm{s}}=\frac{2\, n_{1} \cos \varepsilon}{n_{1} \cos \varepsilon+n_{2} \cos \varepsilon'}
   :label: ts

.. math::
   t_{\mathrm{p}}=\frac{2\, n_{1} \cos \varepsilon}{n_{2} \cos \varepsilon+n_{1} \cos \varepsilon'}
   :label: tp

.. math::
   T=\frac{n_{2} \cos \varepsilon'}{n_{1} \cos \varepsilon}t^{2}
   :label: T

with

.. math::
   t^2 = \left(t_\text{p}^2 + t_\text{s}^2\right)/\,2
   :label: t

Hit Detection
______________________________________

Numerical
*********************************

.. math::
   \vec{p}(t)=\vec{p}_{0}+t \cdot \vec{s}
   :label: pt

.. math::
   G(t)=z_{t}-f\left(x_{t}, y_{t}\right)
   :label: G

.. figure:: images/Illinois.png
   :width: 500
   :align: center

   Comparison of the standard Regula-Falsi-algorithm (left) and the Illinois-algorithm :cite:`DiscontinuitiesSlides`.

Intersection Ray with xy-Plane
*********************************

Intersection Ray with Asphere
*********************************


.. math::
   \text{Ray support vector:}~~~~   p &= (p_x, p_y, p_z)\\
   \text{Ray direction vector:}~~~~ s &= (s_x, s_y, s_z)\\
   \text{Center of surface:}~~~~    q &= (x_0, y_0, z_0)

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
   o_x &= p_x - x_0\\
   o_y &= p_y - y_0\\
   o_z &= p_z - z_0
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
*********************************

.. figure:: images/Oberflächen_Erweiterung.svg
   :width: 700
   :align: center

   Surface Extension


Normal Calculation
______________________________________

General
*********************************

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
   \vec{n} = \frac{1}{\sqrt{n_x^2 + n_y^2 + 1}} \vec{n_0}
   :label: normal_general_norm

Numerical
*********************************

.. math::
   \vec{n_0} = 
   \begin{pmatrix}
        z(x - \epsilon / 2, ~y) - z(x + \epsilon/2, ~y)\\
        z(x, ~y - \epsilon / 2) - z(x, ~y + \epsilon/2)\\
        \epsilon\\
   \end{pmatrix}
   :label: normal_numerical

Needs to be normalized using

.. math::
   \vec{n} = \frac{1}{\sqrt{n_x^2 + n_y^2 + n_z^2}} \vec{n_0}
   :label: normal_numerical_norm

Plane
*********************************

.. math::
   \vec{n} = 
   \begin{pmatrix}
        0\\
        0\\
        1\\
   \end{pmatrix}
   :label: normal_plane

Asphere
*********************************

The derivative of the asphere function is

.. math::
   m = \frac{\text{d}z(r)}{\text{d}r} = \frac{\rho r}{\sqrt{1 - (k+1)\rho^2 r^2}}
   :label: asphere_derivative

.. math::
   n_r = -\frac{\rho r}{\sqrt{1- k\rho^2 r^2}}
   :label: asphere_nr

.. math::
   n_x &= n_r \cos \phi\\
   n_y &= n_r \sin \phi\\
   n_z &= \sqrt{1- n_r^2}
   :label: asphere_nxyz

.. math::
   \vec{n} = 
   \begin{pmatrix}
        n_x\\
        n_y\\
        n_z\\
   \end{pmatrix}
   :label: asphere_n


Sphere
*********************************

With :math:`k=0` the asphere normal simplifies to

.. math::
   \vec{n} = 
   \begin{pmatrix}
        -\rho r \cos \phi \\
        -\rho r \sin \phi\\
        \sqrt{ 1 - \rho^2 r^2}\\
   \end{pmatrix}
   :label: sphere_n


Autofocus
______________________________________

**Position Variance Autofocus**

.. math::
   \text{minimize}~~ R_\text{v}(z) := \sqrt{\sigma^2_P(X_z) + \sigma^2_P(Y_z)}
   :label: autofocus_position


**Image Variance Autofocus**

.. math::
   \text{maximize}~~ I_\text{v}(z) := \sigma^2(E_z)
   :label: autofocus_image

**Airy Disc Weighting**

.. math::
   \text{maximize}~~ S(z) := \frac{\displaystyle\sum_{i}^{} P_i(z) \cdot \exp \left( {-0.5\left(\frac{r_i(z)}{0.42\,r_0}\right)^2} \right)}{\displaystyle\sum_{i}^{} P_i(z)}
   :label: autofocus_airy

with

.. math::
   r_0 = 0.61 \frac{\lambda}{\text{NA}}
   :label: autofocus_airy_r

Color Conversion
______________________________________

**Matrizen und Quellen in diesem Abschnitt stimmen nicht, siehe Quellcode**

**Conversion XYZ to RGB_Linear**

.. figure:: images/CIE_1931_XYZ_Color_Matching_Functions.svg
   :width: 400
   :align: center

   :cite:`TristimulusAcdx`

.. math::
   X &=\int_{\lambda} P(\lambda) \cdot \bar{x}(\lambda) ~d \lambda \\
   Y &=\int_{\lambda} P(\lambda) \cdot \bar{y}(\lambda) ~d \lambda \\
   Z &=\int_{\lambda} P(\lambda) \cdot \bar{z}(\lambda) ~d \lambda
   :label: XYZ_Calc

Source Matrix:  http://brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html

Source gamma correction: https://en.wikipedia.org/wiki/SRGB#Specification_of_the_transformation


.. math::
   	\left[\begin{array}{l}
		R_{\text {linear}} \\
		G_{\text {linear}} \\
		B_{\text {linear}}
	\end{array}\right]=\left[\begin{array}{ccc}
        +3.2404542 & -1.5371385 & -0.4985314 \\
        -0.9692660 & +1.8760108 & +0.0415560 \\
        +0.0556434 & -0.2040259 & +1.0572252
	\end{array}\right]\left[\begin{array}{c}
		X_\text{D65} \\
		Y_\text{D65} \\
		Z_\text{D65}
	\end{array}\right]
    :label: XYZ2RGB

.. math::
   C_{\text {sRGB}}= \begin{cases}12.92\cdot C_{\text {linear}}, & C_{\text {linear}} \leq 0.0031308 \\[1.5ex] 
   1.055\cdot C_{\text {linear}}^{1 / 2.4}-0.055, & C_{\text {linear}}>0.0031308\end{cases}
   :label: Gamma_Correction

**sRGB to XYZ**


Source Matrix: http://brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html

Source gamma correction: https://en.wikipedia.org/wiki/SRGB#Specification_of_the_transformation


.. math::
   	C_{\text {linear }}= \begin{cases}\displaystyle\frac{C_{\text {sRGB}}}{12.92}, & C_{\text {sRGB}} \leq 0.04045 \\[1.5ex]
	\displaystyle\left(\frac{C_{\text {sRGB}}+0.55}{1.055}\right)^{2.4}, & C_{\text {sRGB}}>0.04045\end{cases}
    :label: Gamma_Correction_Reverse

.. math::
	\left[\begin{array}{l}
   			X_{\text {D65}} \\
			Y_{\text {D65}} \\
			Z_{\text {D65}}
		\end{array}\right]=\left[\begin{array}{ccc}
            0.4124564 & 0.3575761 & 0.1804375\\
            0.2126729 & 0.7151522 & 0.0721750\\
            0.0193339 & 0.1191920 & 0.9503041
		\end{array}\right]\left[\begin{array}{c}
			R_{\text{linear}} \\
			G_{\text{linear}} \\
			B_{\text{linear}}
	\end{array}\right]
    :label: RGB2XYZ


**Random Wavelengths from sRGB**

.. list-table:: sRGB primaries
   :widths: 50 50 50 50 50
   :header-rows: 1
   :align: center

   * - Color value
     - Red
     - Green
     - Blue
     - D65   
   * - :math:`x` 
     - 0.6400
     - 0.3000 
     - 0.1500 
     - 0.3127
   * - :math:`y` 
     - 0.3300
     - 0.6000 
     - 0.0600 
     - 0.3290
   * - :math:`z` 
     - 0.0300 
     - 0.0100 
     - 0.7900 
     - 0.3583
   * - :math:`Y` 
     - 0.2127 
     - 0.7152 
     - 0.0722 
     - 1.0000
   * - sRGB 
     - [1, 0, 0] 
     - [0, 1, 0] 
     - [0, 0, 1] 
     - [1, 1, 1]

The gaussian function is defined as

.. math::
   S(\lambda, \mu, \sigma)=\frac{1}{\sqrt{2 \pi \sigma^{2}}} \exp \left(-\frac{(\lambda-\mu)^{2}}{2 \sigma^{2}}\right)
   :label: Gauss_Opt

Curves with same color stimulus as primaries

.. math::
    r_0(\lambda) =&~  88.4033043 \cdot \Big[ S(\lambda, 660.255528, 35.6986569)\\
                & + 0.0665761658 \cdot S(\lambda, 552.077348, 150.000000)\Big]\\
    g_0(\lambda) =&~  83.4999030 \cdot  S(\lambda, 539.131090, 33.3116417)\\
    b_0(\lambda) =&~ 118.345477  \cdot  S(\lambda, 415.035902, 47.2130145)\\

.. _rgb_curve1:
.. figure:: images/rgb_curves1.svg
   :width: 500
   :align: center

Rescale for same Y stimulus as primaries

.. math::
    r(\lambda) =&~ 1.24573718 \cdot r_0(\lambda)\\
    g(\lambda) =&~ 1.00000000 \cdot g_0(\lambda)\\
    b(\lambda) =&~ 1.12354883 \cdot b_0(\lambda)\\

.. _rgb_curve2:
.. figure:: images/rgb_curves2.svg
   :width: 500
   :align: center

The RGB channel probabilities are the sRGBLinear values of the pixel (= mixing ratio), scaled with the area ratio (= overall probability ratio) of the curve.
Rescaling factors:

.. math::
    r_\text{P} = 1.38950586\\
    g_\text{P} = 1.00000000\\
    b_\text{P} = 1.22823756\\

Bibliography
______________________________________
.. bibliography::

