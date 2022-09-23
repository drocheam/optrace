***********************
Color Conversion
***********************

**Matrizen und Quellen in diesem Abschnitt stimmen nicht, siehe Quellcode**

Conversion XYZ to RGB_Linear 
=================================================

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

sRGB to XYZ 
=================================================


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


Random Wavelengths from sRGB 
=================================================

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
   :label: r0g0b0_curves

.. _rgb_curve1:
.. figure:: images/rgb_curves1.svg
   :width: 500
   :align: center

Rescale for same Y stimulus as primaries

.. math::
    r(\lambda) =&~ 1.24573718 \cdot r_0(\lambda)\\
    g(\lambda) =&~ 1.00000000 \cdot g_0(\lambda)\\
    b(\lambda) =&~ 1.12354883 \cdot b_0(\lambda)\\
    :label: rgb_curves

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
   :label: r_g_b_factors


