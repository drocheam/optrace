***********************
PSF Convolution
***********************

.. role:: python(code)
  :language: python
  :class: highlight


.. _psf_color_handling:

Convolution of Colored Images
================================

Polychromatic Convolution
--------------------------------------

In general, image convolution must be performed for each wavelength, with both the image and the PSF 
typically being wavelength-dependent:

.. math::
   \text{im}_2(x, y, \lambda) = \text{im}(x, y, \lambda) \otimes \text{psf}(x, y, \lambda)
   :label: eq_conv_per_wavelength

Special Case Spectral Homogenity
--------------------------------------

In the case of so-called spectral homogeneity, the spectrum :math:`s` is consistent across the entire image 
but is scaled by a location-dependent intensity factor :math:`\text{im}_s(x, y)`. 
The concept of spectral homogeneity goes back to :footcite:`Barnden_1974`,
who also describes this mathematical simplification. 
:footcite:`Ravikumar_2008` also uses this to generate color images.

For an example :math:`\text{im}_s(\lambda)`, it holds that:

.. math::
   \text{im}_2(x, y, \lambda) 
   &= \left(\text{im}_r(\lambda) \text{im}_s(x, y)\right) \otimes \text{psf}(x, y, \lambda)\\
   &= \text{im}_s(x, y) \otimes \left(\text{im}_s(\lambda) \text{psf}(x, y, \lambda)\right)\\
   &= \text{im}_s(x, y) \otimes \text{psf}_s(x, y, \lambda)\\
   :label: eq_conv_special_case_spectral_homogenity

The spectrum is a constant with respect to the convolution expression and can therefore also be multiplied with the PSF. 
We define the new spectrally weighted PSF :math:`\text{psf}_s(x, y, \lambda)`. 

Convolution of RGB Images
--------------------------------------

Typically, the spectral profile of the image result is not of interest, 
but rather the color coordinates for a color representation on the monitor.
For instance, to calculate the red stimulus produced by the convolved image,
it can be multiplied by the sRGB red color matching function :math:`r(\lambda)` and integrated over all wavelengths:

.. math::
   \text{im}_{2,r\rightarrow r}(x, y) 
   &= \int \Big[\text{im}_r(x, y) \otimes \text{psf}_r(x, y, \lambda)\Big] r(\lambda)\;\text{d}\lambda\\
   &= \text{im}_r(x, y) \otimes \int\text{psf}_r(x, y, \lambda) r(\lambda)\;\text{d}\lambda\\
   &= \text{im}_r(x, y) \otimes \text{psf}_{r\rightarrow r}(x, y)\\
   :label: eq_conv_rgb_r_to_r

But only the PSF has a wavelength dependency, so the image can be factored out of the integral expression. 
Since the convolution is independent of wavelength, the integration can also be performed before the convolution. 
We obtain the red-to-red PSF :math:`\text{psf}_{r \rightarrow r}(x, y)`.

If the light spectra :math:`\text{im}_r(\lambda), \text{im}_g(\lambda), \text{im}_b(\lambda)` are chosen such that 
they have the same color coordinates as the primary spectra of sRGB, they are linearly independent color channels, 
with which all colors within the sRGB color space can be composed as a linear combination.
The PSF can also be decomposed into three individual channel PSFs that are likewise linearly independent.

The RGB color image from the red image :math:`\text{im}_r(x, y)` with spatially homogeneous :math:`\text{im}_r(\lambda)` 
for convolution with the PSF can then be simplified and summarized as:

.. math::
   \text{im}_{2,r\rightarrow rgb}(x, y) =
   \left[\begin{array}\,
   \text{im}_r(x, y) \otimes \text{psf}_{r\rightarrow r}(x, y)\\
   \text{im}_r(x, y) \otimes \text{psf}_{r\rightarrow g}(x, y)\\
   \text{im}_r(x, y) \otimes \text{psf}_{r\rightarrow b}(x, y)\\
   \end{array}\right]
   :label: eq_conv_rgb_r_to_rgb

Even if the original red spectrum generates a pure red in the sRGB color space,
this might not be the case after convolution with the PSF.
Chromatic aberration could lead to more transversal chromatic aberration for smaller wavelengths,
resulting in a yellow fringe in the PSF and components in :math:`\text{psf}_{r \rightarrow g}(x, y)`.

Similarly, this applies to the G-channel with the sRGB color matching function :math:`g(\lambda)`:

.. math::
   \text{im}_{2,g\rightarrow rgb}(x, y) =
   \left[\begin{array}\,
   \text{im}_g(x, y) \otimes \text{psf}_{g\rightarrow r}(x, y)\\
   \text{im}_g(x, y) \otimes \text{psf}_{g\rightarrow g}(x, y)\\
   \text{im}_g(x, y) \otimes \text{psf}_{g\rightarrow b}(x, y)\\
   \end{array}\right]
   :label: eq_conv_rgb_g_to_rgb

And the blue channel with matching function :math:`b(\lambda)`:

.. math::
   \text{im}_{2,b\rightarrow rgb}(x, y) =
   \left[\begin{array}\,
   \text{im}_b(x, y) \otimes \text{psf}_{b\rightarrow r}(x, y)\\
   \text{im}_b(x, y) \otimes \text{psf}_{b\rightarrow g}(x, y)\\
   \text{im}_b(x, y) \otimes \text{psf}_{b\rightarrow b}(x, y)\\
   \end{array}\right]
   :label: eq_conv_rgb_b_to_rgb

The overall image :math:`\text{im}_{2,rgb \rightarrow rgb}` is obtained from the sum of all convolved R, G, B 
color components in the image. However, the mixing ratio of all channels must be considered: If the color PSFs
were all simulated with a power of one watt, this does not correspond to the correct mixing ratio for white in 
the sRGB color space. This must be adjusted so that equal parts in :math:`\text{im}_r, \text{im}_g, \text{im}_b` 
produce white in the color space.

Let :math:`a_r, a_g, a_b` be the relative mixing factors, then the final result is:

.. math::
   \text{im}_{2,rgb\rightarrow rgb}(x, y) = a_r \text{im}_{2,r\rightarrow rgb}(x, y)
   + a_g \text{im}_{2,g\rightarrow rgb}(x, y) + a_b \text{im}_{2,b\rightarrow rgb}(x, y)
   :label: eq_conv_rgb_rgb_to_rgb

The RGB color spectra and weighting factors are shown in Equation :math:numref:`r_g_b_factors`. 
Instead of applying the rescaling afterwards, the power rations could also be applied 
to the sources used to render the PSFs, so the power ratios are included in the relative R, G, B PSFs.

Convolution of a spectral homogeneous image and an RGB PSF
--------------------------------------------------------------

There is a special case when the image is spectrally homogeneous. 
Let the spectrum be :math:`s`, then:

.. math::
   \text{im}_{2,s\rightarrow rgb}(x, y) =
   \left[\begin{array}\,
   \text{im}_s(x, y) \otimes \text{psf}_{s\rightarrow r}(x, y)\\
   \text{im}_s(x, y) \otimes \text{psf}_{s\rightarrow g}(x, y)\\
   \text{im}_s(x, y) \otimes \text{psf}_{s\rightarrow b}(x, y)\\
   \end{array}\right]
   :label: eq_conv_rgb_s_to_rgb

Here, :math:`\text{im}_s` describes the spatial distribution of the intensity 
of the source that emits the spectrum :math:`s`.

Convolution of spectral homogenous image and PSF
--------------------------------------------------

For the special case where the PSF is also spectrally homogeneous, the following holds:

.. math::
   \text{im}_{2,s_1\rightarrow s_2}(x, y) = \text{im}_{s_1}(x, y) \otimes \text{psf}_{s_1\rightarrow s_2}(x, y)
   :label: eq_conv_rgb_w_to_w

Here, :math:`s_1` is the source spectrum and :math:`s_2` is the detector spectrum. Due to absorption,
they do not have to be identical. As described earlier, it is only important that both the source
and the PSF are spectrally homogeneous.

A typical example is a black and white image and a wavelength-independent PSF.

Limitations
=================================================

Limitations are described in :numref:`convolve_limitations`.

Processing Steps
==================

1. Convert the image and PSF to linear sRGB values, while including negative values.
2. For a grayscale PSF: Normalize the PSF so that the sum (= total power) equals one.
3. Downscale/interpolate the PSF so that the physical pixel sizes of the PSF and image 
   (after scaling with magnification factor) are identical.
4. Pad the PSF with zeros to ensure a defined fall-off
5. Flip the image if the magnification factor is negative
6. Pad the image according to the chosen padding method.
7. Convolve image and PSF according to the methods in Section :numref:`psf_color_handling`.
8. Convert the image back to sRGB, while performing a chosen gamut mapping.
9. Slice the image (remove padding or even trim back to original size)

The convolution takes place in sRGB coordinates because the channels are orthogonal. 
Furthermore, this color space corresponds to the target color space for monitors. 
However, the convolution must be performed as a linear operation in linear sRGB values (description see <>). 
Colors outside the color space (negative coordinates) must also be included to maintain linearity. 
If there are still negative values in the image after convolution, gamut mapping needs to be performed.

In the case of a grayscale PSF, it is automatically normalized such that, 
in combination with the :python:`normalize=False` parameter of the convolve function, the brightness/color values 
are not automatically normalized or rescaled. For colored PSFs, normalization is much more difficult because 
one would need to know how much light from the source actually reached the detector for the PSF. 
This could be implemented through metadata from RenderImage. 
However, it is questionable whether this option is even 
relevant since normalized images are generally desired.

Downscaling the PSF must be performed in a way that preserves energy. 
Additionally, it's desirable to use a method where no aliasing occurs. 
We use the scaling with the `INTER_AREA <https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html#gga5bb5a1fea74ea38e1a5445ca803ff121acf959dca2480cc694ca016b81b442ceb>`__ 
option from OpenCV in the resize function. 
The PSF must be rescaled so that the physical pixel dimensions of the image and the PSF match in both dimensions. 
Then it suffices to convolve the image as a pixel matrix, even if the pixels are non-square.

The convolution is performed in the Fourier space using the convolution theorem.
The Fourier transformation is calculated with the :func:`scipy.signal.fftconvolve` function. 
Due to the nature of the method, areas outside the image are assumed to be black. 
As a result, there is a fall-off region in the resulting image where the PSF increasingly 
convolves with black parts from the edges. This transition area is as wide as the PSF region 
where its intensities are above zero. For simplicity we assume the entire PSF width for this. 
If the user wants a different padding method, the image must be padded additionally. 
First for the user-chosen method, and a second time to avoid the dark fall-off edges.

------------

**References**

.. footbibliography::

