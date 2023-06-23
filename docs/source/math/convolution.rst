***********************
PSF Convolution
***********************



Procedure
=================================================

The point spread function (PSF) of a optical setup can be seen as impulse response of the same.
A convolution with this PSF is equivalent to applying the transfer function of the system to an input.
Note that this procedure can only simulate some optical effects, as it assumes the PSF is spatially constant.
This ignores aberrations like coma, off-axis astigmatism, field curvature, vignetting, distortion and others.

Image convolution is done by applying the convolution theorem of the Fourier transformation:

.. math::
   g \ast h=\mathcal{F}^{-1}\{G \cdot H\}
   :label: eq_conv_theorem
    
Where :math:`g` and :math:`h` are the original functions. A convolution in the original domain is therefore a multiplication in the Fourier domain.

In the case of two dimensional images the convolution also becomes a two dimensional one.
optrace

Initially, the pixel sizes, counts and ratios and overall sizes of the PSF and image differ.
The processing steps consist of the following ones:

1. convert both image and PSF to linear sRGB values
2. interpolate the PSF so that the grid locations match
3. pad the PSF
4. convolve channel-wise using :func:`scipy.signal.convolve`
5. convert the resulting image back to sRGB linear while doing gamut clipping


.. _psf_color_handling:

Limitations on Color
=================================================

**Overview**

Ravikumar et al described the convolution of polychromatic point spread functions in detail :footcite:`Ravikumar_2008`. A physically correct approach would be the convolution on a per-wavelength-basis, therefore needing a spectral distribution for every location on object and PSF. With the restriction of a spatially constant distribution on the object, scaled only with an intensity factor, a PSF in RGB channels is also sufficient. This can be described as the spectrum being homogeneous over the object. In the case of a sRGB monitor image, the emission of each pixel can be described as linear combination of the channel emission spectra. In this case the whole object is heterogeneous, but homogeneous on a per-channel-basis. Therefore convolution on a per-channel basis is also viable for a RGB colored PSF and object.
"Natural scenes" can have largely spatially varying spectral distributions, that would lead to different results. It is important to note that the result of the above a approach is only one possible solution with the assumption of such an man-made RGB object.

Proofs of this concept are shown by Ravikumar et. al., while building on the results of Barnden :footcite:`Ravikumar_2008,Barnden_1974`.


Let's define two terms, that will be useful later:

* single-colored: an image having the same hue and saturation for all pixels, but different lightness/brightness/intensity values at different locations. This also includes an image without colors.
* multicolored: image with arbitrary hue, saturation and brightness pixels


To put it short, the convolution approach produces correct results if

* both image and PSF are single-colored
* the image is single-colored and the PSF multicolored, or vice versa
* if both image and PSF are multicolored, but under the assumption that the object emits a superposition of the same three RGB spectra everywhere

For physically correct results the PSF should have a color space with all human visible colors and the color values should be linear to physical intensities/powers.


**Proof**

This sections presents an alternative proof of this concept.

The convoluted image :math:`\text{im}_2` is calculated by a two dimensional spatial convolution between the image :math:`\text{im}` and the point spread function :math:`\text{psf}`.
When done correctly, all three not only depend on the position :math:`x, y` inside the image but also the wavelength :math:`\lambda` as the image and PSF can have different spectral distributions depending on the location.

.. math::
   \text{im}_2(x, y, \lambda) &= \text{im}(x, y, \lambda) \ast\ast\; \text{psf}(x, y, \lambda)\\
   &= \iint \text{im}(\tau_x, \tau_y, \lambda) \cdot \text{psf}(x-\tau_x, y-\tau_y, \lambda)  \;\text{d} \tau_x \,\text{d}\tau_y\\
   :label: eq_conv_double_conv

Converting a spatial and spectral image into a color channel is done by multiplying it with a color matching function :math:`r(\lambda)` and integrating over all wavelengths.

.. math::
   \text{im}_r(x, y) = \int \text{im}(x, y, \lambda) \cdot r(\lambda) \, \text{d}\lambda
   :label: eq_conv_channel

The following proposition is applied in a later derivation:

.. math::
   \int f(x) \,\text{d}x \cdot \int g(x) \,\text{d}x = \iint f(x) \cdot g(y) \;\text{d}x\,\text{d}y
   :label: eq_conv_int_sep

In the next step we want to proof that convolving the image channels is the same as calculating the image with equation :math:numref:`eq_conv_double_conv` and then converting it to a color channel.

.. math::
   \text{im}_{2,r} = \int   \text{im}_2(x, y, \lambda) \cdot r(\lambda) \;\text{d}\lambda \stackrel{!}{=} \text{im}_{r}(x, y) \ast\ast\; \text{psf}_r(x, y) 
   :label: eq_conv_desired

This is done by expanding all integrals:

.. math::
   \text{im}_{2,r}(x, y) 
   &= \text{im}_{r}(x, y) \ast\ast\; \text{psf}_r(x, y)\\
   &= \iint \text{im}_r(\tau_x, \tau_y) \cdot \text{psf}_r(x-\tau_x, y-\tau_y)  \;\text{d} \tau_x \,\text{d}\tau_y\\
   &= \iint \left( \int \text{im}(\tau_x, \tau_y, \lambda) \cdot r(\lambda) \, \text{d}\lambda \cdot \int \text{psf}(x-\tau_x, y-\tau_y, \lambda) \cdot r(\lambda) \,\text{d}\lambda \right) \;\text{d} \tau_x \,\text{d}\tau_y\\
   &= \iint \left( \int \text{im}(\tau_x, \tau_y, \lambda_1) \cdot r(\lambda_1) \, \text{d}\lambda_1 \cdot \int \text{psf}(x-\tau_x, y-\tau_y, \lambda_2) \cdot r(\lambda_2) \,\text{d}\lambda_2 \right) \;\text{d} \tau_x \,\text{d}\tau_y\\
   &= \iiiint \text{im}(\tau_x, \tau_y, \lambda_1) \cdot \text{psf}(x-\tau_x, y-\tau_y, \lambda_2) \cdot r(\lambda_1) \cdot r(\lambda_2) \;\text{d}\lambda_1 \, \text{d}\lambda_2  \,\text{d} \tau_x \,\text{d}\tau_y\\
   &= \iint \left(  \iint \text{im}(\tau_x, \tau_y, \lambda_1) \cdot \text{psf}(x-\tau_x, y-\tau_y, \lambda_2) \,\text{d} \tau_x \,\text{d}\tau_y \right) \cdot r(\lambda_1) \cdot r(\lambda_2) \;\text{d}\lambda_1 \, \text{d}\lambda_2  \\
   &= \iint \Bigl[  \text{im}(x, y, \lambda_1) \ast\ast\; \text{psf}(x, y, \lambda_2)\Bigr] \cdot r(\lambda_1) \cdot r(\lambda_2) \;\text{d}\lambda_1 \, \text{d}\lambda_2\\
   :label: eq_conv_proof


Unfortunately, the above form can't be led to that of :math:numref:`eq_conv_desired` without further restrictions.

One such restrictions could be that the image pixels are composed of a linear combination of spectral distributions :math:`S_\text{im,r}, S_\text{im,g}, S_\text{im,b}`. While the factors :math:`\text{im}_\text{r},\text{im}_\text{g},\text{im}_\text{b}` vary for each pixel, the spectral distributions don't vary locally.

.. math::
   \text{im}(x, y, \lambda_1) = \text{im}_\text{r}(x, y) S_\text{im,r}(\lambda_1) + \text{im}_\text{g}(x, y) S_\text{im,g}(\lambda_1) +\text{im}_\text{b}(x, y) S_\text{im,b}(\lambda_1)
   :label: eq_srgb_comp


The spectral distributions have their corresponding color matching functions (CMF) :math:`r, g, b`

An important criterion is that all three spectral distributions are orthogonal to the other channels color matching functions, but are non-orthogonal to their own CMF. What this means is that for instance the red spectrum :math:`S_\text{im,r}` only gets registered by the :math:`r` color matching function according to :math:numref:`eq_conv_channel` but not the :math:`g,b` ones, leading to an exclusively red signal in the color space.
This criterion is equivalent to the spectral distributions leading to color values on all three corners of the triangle sRGB color gamut that is indirectly defined by the CMF.

Plugging :math:numref:`eq_srgb_comp` into :math:numref:`eq_conv_proof` we can rewrite:

.. math::
   \text{im}_{2,r}(x, y) 
   &= \iint \Bigl[  \text{im}(x, y, \lambda_1) \ast\ast\; \text{psf}(x, y, \lambda_2)\Bigr] \cdot r(\lambda_1) \cdot r(\lambda_2) \;\text{d}\lambda_1 \, \text{d}\lambda_2\\
   &= \int \Biggl[  \left( \int\text{im}(x, y, \lambda_1) \cdot r(\lambda_1)\;\text{d}\lambda_1 \right) \ast\ast\; \text{psf}(x, y, \lambda_2)\Biggr]  \cdot r(\lambda_2) \, \text{d}\lambda_2\\
   &= \int \Biggl[  \left( \int \Bigl\{ \text{im}_\text{r}(x, y) S_\text{im,r}(\lambda_1) + \text{im}_\text{g}(x, y) S_\text{im,g}(\lambda_1) +\text{im}_\text{b}(x, y) S_\text{im,b}(\lambda_1) \Bigr\} \cdot r(\lambda_1)\;\text{d}\lambda_1 \right) \ast\ast\; \text{psf}(x, y, \lambda_2)\Biggr]  \cdot r(\lambda_2) \, \text{d}\lambda_2\\
   &= \int \Biggl[  \left( \int \text{im}_\text{r}(x, y) S_\text{im,r}(\lambda_1) \cdot r(\lambda_1)\;\text{d}\lambda_1 \right) \ast\ast\; \text{psf}(x, y, \lambda_2)\Biggr]  \cdot r(\lambda_2) \, \text{d}\lambda_2\\
   &= \int S_\text{im,r}(\lambda_1) \cdot r(\lambda_1) \, \text{d}\lambda_1 \cdot \int \Bigl[\text{im}(x, y) \ast\ast\; \text{psf}(x, y, \lambda_2)\Bigr] \cdot r(\lambda_2) \;\text{d}\lambda_2\\
   &= R_\text{im} \cdot \int   \Bigl[\text{im}(x, y) \ast\ast\; \text{psf}(x, y, \lambda_2)\Bigr] \cdot r(\lambda_2) \;\text{d}\lambda_2\\
   &= R_\text{im} \cdot \int   \text{im}_2(x, y, \lambda_2) \cdot r(\lambda_2) \;\text{d}\lambda_2\\
   :label: eq_conv_img_independent
    
This works because the spectral components for the image become independent of the other channel signals. Furthermore, the image convolution becomes independent of the wavelength :math:`\lambda_1` and this part can be integrated separately, leading to a constant factor of  :math:`R_\text{im}`.

For all this to work the convolution needs to take place in a linear color space system with the orthogonality criterion from before.
In our case the linear sRGB colorspace is applied, while also negative values are used to contain all human-visible colors, which wouldn't be the case for the typical positive-value gamut. Linearity would also be lost because of gamut clipping.
Color matching functions :math:`r, g, b` were chosen according to sRGB specifications and spectral distributions according to the procedure in :numref:`random_srgb`.


.. _math_psf_presets:

PSF Presets
=================================================


`optrace` includes multiple PSF presets.

.. TODO Quellen und Dokumentation

**Gaussian**

A simple gaussian intensity distribution is described as:

.. math::

   I_{d}(x, y) = \exp \left(  \frac{-x^2 - y^2}{2 \sigma^2}\right)


**Airy**

The Airy function is:

.. math::

   I_{d}(x, y) = \left( \frac{2 J_1(r_d)}{r_d} \right)^2

.. math::

   r_d = 3.8317 \frac{\sqrt{x^2 + y^2}}{r}

The resolution limit is described as distance from the center to the first function zero, so the diameter describes the distance between the zero on one and the other side.

**Glare**

A glare consists of two different gaussians. Parameter :math:`a` describes the relative intensity of the larger one.

.. math::

  I_{\sigma_1,\sigma_2,d}(x, y) = \left(1-a\right)\exp \left(  \frac{-x^2 - y^2}{2 \sigma_1^2}\right) + a\exp \left(  \frac{-x^2 - y^2}{2 \sigma_2^2}\right)


**Halo**

A halo consists of a central gaussian and annular gaussian function around :math:`r`.
:math:`sig_1, sig_2` describe the standard deviations of the gaussians.
:math:`a` describes the intensity of the ring.

.. math::

   I_{d_1,d_2,a,w}(x, y) = \exp \left(  \frac{-x^2 - y^2}{2 \sigma_1^2}\right) +  a \exp \left(  \frac{-\left(\sqrt{x^2 + y^2} - r\right)^2}{2 \sigma_2^2}\right) 





------------

**References**

.. footbibliography::

