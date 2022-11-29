***********************
Color Conversion
***********************


Spectrum to XYZ
=================================================

.. figure:: images/cie_cmf.svg
   :width: 650
   :align: center

With a light power spectrum :math:`P(\lambda)` and the three  curves from the figure above we can calculate the tristimulus values :math:`X, Y, Z`.

.. math::
   X &=\int_{\lambda} P(\lambda) \cdot x(\lambda) ~d \lambda \\
   Y &=\int_{\lambda} P(\lambda) \cdot y(\lambda) ~d \lambda \\
   Z &=\int_{\lambda} P(\lambda) \cdot z(\lambda) ~d \lambda
   :label: XYZ_Calc


xyY
============

**XYZ to xyY**

The following formulas are valid for :math:`X, Y, Z > 0`, otherwise we set :math:`x=x_r,~y=y_r,~Y=0`, where :math:`x_r,y_r` are the whitepoint coordinates. Typically the whitepoint D65 is used with :math:`x_r=0.31272,~y_r=0.32903`, see CIE Colorimetry, 3. Edition, 2004, table 11.3.

.. math::
   \begin{aligned}
   x &= \frac{X}{X + Y + Z} \\
   y &= \frac{Y}{X + Y + Z} \\
   Y &= Y 
   \end{aligned}

**xyY to XYZ**

.. math::
   \begin{aligned}
   X &= x \cdot \frac{Y}{y} \\
   Y &= Y\\ 
   Z &= (1 - x - y) \cdot \frac{Y}{y} \\
   \end{aligned}

sRGB
==============

sRGB uses the D65 whitepoint with coordinates :math:`X=0.95047,~Y=1,~Z=1.08883`, see https://en.wikipedia.org/wiki/Illuminant_D65.


**Conversion XYZ to sRGB**

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


**Conversion sRGB to XYZ**

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


CIELUV
==============


**XYZ to CIELUV**

Source: http://www.brucelindbloom.com/index.html?Eqn_XYZ_to_Luv.html

The following equations are valid for :math:`X, Y, Z > 0`, otherwise we set :math:`L = 0, ~u=0,~v=0`.

.. math::
   \begin{aligned}
   &L= \begin{cases}116 \sqrt[3]{y_r}-16 & \text { if } y_r>\epsilon \\
   \kappa y_r & \text { otherwise }\end{cases} \\
   &u=13 L\left(u^{\prime}-u_r^{\prime}\right) \\
   &v=13 L\left(v^{\prime}-v_r^{\prime}\right)
   \end{aligned}

With 

.. math::
   \begin{aligned}
   \epsilon &= 0.008856\\
   \kappa &= 903.3\\
   y_r &=\frac{Y}{Y_r} \\
   u^{\prime} &=\frac{4 X}{X+15 Y+3 Z} \\
   v^{\prime} &=\frac{9 Y}{X+15 Y+3 Z}
   \end{aligned}

:math:`Y_r` is taken from the white point coordinates :math:`(X_r,~Y_r,~Z_r)`, typically those of the standard illuminant D65. On the other hand :math:`u'_r` and :math:`v'_r` are the :math:`u', ~v'` values for these whitepoint coordinates.

**CIELUV to XYZ**

Changed version of formulas in http://www.brucelindbloom.com/Eqn_Luv_to_XYZ.html

The following equations are valid for :math:`L > 0`, for :math:`L = 0` all values are set as :math:`X=Y=Z=0`.

.. math::
   Y= \begin{cases}\left(\frac{L+16} {116}\right)^3 & \text { if } L>\kappa \epsilon \\ L / \kappa & \text { otherwise }\end{cases}

.. math::
   \begin{aligned}
   X &= \frac{9}{4} \cdot \frac{u + 13 L u'_r}{v + 13 L v'_r}\\
   Z &= 3 Y \cdot \left(\frac{13 L}{v + 13 L v'_r}  - \frac{5}{3}\right) - \frac{X}{3}\\
   \end{aligned}


**CIELUV to u'v'L**

The following equations are valid for :math:`L > 0`, for :math:`L = 0` we set :math:`u' = u'_r, ~v' = v'_r`.

.. math::
   \begin{aligned}
   L &= L\\
   u' &= u'_r + \frac{u}{13 L}\\
   v' &= v'_r + \frac{v}{13 L}\\
   \end{aligned}

**CIELUV Chroma**

https://en.wikipedia.org/wiki/Colorfulness#Chroma

.. math::
   C = \sqrt{u^2 + v^2}

**CIELUV Hue**

https://en.wikipedia.org/wiki/Colorfulness#Chroma

.. math::
   H = \text{arctan2}(v, u)

**CIELUV Saturation**

https://en.wikipedia.org/wiki/Colorfulness#Saturation

The following equations are valid for :math:`L > 0`, for :math:`L = 0` we set :math:`S=0`.

.. math::
   S = \frac{C}{L}

sRGB to Spectrum 
=================================================

This process is commonly referred to as *Spectral Upsampling*.

Requirements:
 1. illuminants with same color coordinates as the sRGB primaries
 2. same luminance ratios as sRGB primaries
 3. simple, smooth spectral functions
 4. broad spectrum
 5. relatively few light in non-visible regions (infrared and ultraviolet)

Points 1 and 2 simplify the upsampling process, since the mixing ratio of the linear sRGB values can be used directly. In principle we could create a new color space and gamut, that includes the sRGB gamut. But with this we would need to add additional color space conversions.

Points 3 and 4 are needed to approximate natural illuminants close to reality. Adding all sRGB primaries together for a white spectrum should lead to no missing regions in the spectral range. Such gaps would lower the color rendering index (CRI) of the illuminant, which is basically the measure to quantify faithfully rendering object colors when illuminated with this light. For instance, a light spectrum with a yellow gap fails to render purely yellow colors.

Point 5 ensures most of the traced light actually contributes to a rendered image. A color image in sRGB, which is a color space for human vision, should lead to an image with colors in human vision. Rays with colors far outside the visible spectrum would be a waste of rendering time.

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
    r_0(\lambda) =&~  75.1660756583 \cdot \Big[ S(\lambda, 639.854491, 30.0)\\
                & + 0.0500907584 \cdot S(\lambda, 418.905848, 80.6220465)\Big]\\
    g_0(\lambda) =&~  83.4999222966 \cdot  S(\lambda, 539.13108974, 33.31164968)\\
    b_0(\lambda) =&~  47.99521746361 \cdot \Big[ S(\lambda, 454.833119, 20.1460206)\\
                & + 0.184484176 \cdot S(\lambda, 459.658190, 71.0927568)\Big]\\
   :label: r0g0b0_curves

.. _rgb_curve1:
.. figure:: images/rgb_curves1.svg
   :width: 600
   :align: center

Rescale for same Y stimulus as primaries

.. math::
    r(\lambda) =&~ 0.951190393 \cdot r_0(\lambda)\\
    g(\lambda) =&~ 1.000000000 \cdot g_0(\lambda)\\
    b(\lambda) =&~ 1.163645855 \cdot b_0(\lambda)\\
    :label: rgb_curves

.. _rgb_curve2:
.. figure:: images/rgb_curves2.svg
   :width: 600
   :align: center

The RGB channel probabilities are the sRGBLinear values of the pixel (= mixing ratio), scaled with the area ratio (= overall probability ratio) of the curve.
Rescaling factors:

.. math::
    r_\text{P} = 0.885651229244\\
    g_\text{P} = 1.000000000000\\
    b_\text{P} = 0.775993481741\\
   :label: r_g_b_factors

The resulting spectrum for sRGB white (coordinates :math:`[1.0, 1.0, 1.0]`) looks as follows:

.. _rgb_white:
.. figure:: images/rgb_white.svg
   :width: 600
   :align: center

