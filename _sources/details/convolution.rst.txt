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

In general, both the image and the PSF are wavelength-dependent and the image convolution 
must be performed individually for each wavelength:

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

The spectrum is a constant with respect to the convolution expression and can therefore 
also be multiplied with the PSF. We define the new spectrally weighted PSF :math:`\text{psf}_s(x, y, \lambda)`.

Convolution of RGB Images
--------------------------------------

Typically, the spectral profile of the image result is not of interest. 
Instead, the color coordinates for display on a monitor are more relevant. 
For instance, to calculate the red stimulus produced by the convolved image, 
it can be multiplied by the sRGB red color matching function :math:`r(\lambda)` and integrated over all wavelengths:

.. math::
   \text{im}_{2,r\rightarrow r}(x, y) 
   &= \int \Big[\text{im}_r(x, y) \otimes \text{psf}_r(x, y, \lambda)\Big] r(\lambda)\;\text{d}\lambda\\
   &= \text{im}_r(x, y) \otimes \int\text{psf}_r(x, y, \lambda) r(\lambda)\;\text{d}\lambda\\
   &= \text{im}_r(x, y) \otimes \text{psf}_{r\rightarrow r}(x, y)\\
   :label: eq_conv_rgb_r_to_r

Since only the PSF has a wavelength dependency, the image can be factored out of the integral expression. 
Because the convolution is independent of wavelength, the integration can be performed prior to convolution. 
This results in the red-to-red PSF :math:`\text{psf}_{r \rightarrow r}(x, y)`.

If the light spectra :math:`\text{im}_r(\lambda), \text{im}_g(\lambda), \text{im}_b(\lambda)` 
are selected such that they correspond to the primary spectra of sRGB, they form linearly independent color channels. 
These channels can be used to compose all colors within the sRGB color space via linear combination.
The PSF can similarly be decomposed into three individual channel PSFs that are linearly independent.

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
Chromatic aberration could lead to more transverse chromatic aberration at smaller wavelengths, 
resulting in a yellow fringe in the PSF and introducing components in :math:`\text{psf}_{r \rightarrow g}(x, y)`.

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

The overall image :math:`\text{im}_{2,rgb \rightarrow rgb}` is obtained from the sum of all convolved 
R, G, B color components in the image. However, the mixing ratio of all channels must be considered:
If the color PSFs were all simulated with a power of one watt, this does not correspond to the correct mixing 
ratio for white in the sRGB color space. 
This must be adjusted so that equal parts in :math:`\text{im}_r, \text{im}_g, \text{im}_b` 
produce white in the color space.

Let :math:`a_r, a_g, a_b` be the relative mixing factors. The final result can be expressed as:

.. math::
   \text{im}_{2,rgb\rightarrow rgb}(x, y) = a_r \text{im}_{2,r\rightarrow rgb}(x, y)
   + a_g \text{im}_{2,g\rightarrow rgb}(x, y) + a_b \text{im}_{2,b\rightarrow rgb}(x, y)
   :label: eq_conv_rgb_rgb_to_rgb

Equation :math:numref:`r_g_b_factors` illustrates the RGB color spectra and respective weighting factors. 
Rather than applying the rescaling subsequently, the power ratios might be integrated into the source rendering 
process of the PSFs. By doing this, the power ratios are inherently incorporated into the relative R, G, B PSFs.

Convolution of a spectral homogeneous image and an RGB PSF
--------------------------------------------------------------

In the special case where the image is spectrally homogeneous, let the spectrum be denoted as :math:`s`. 
The transformation can then be represented by:

.. math::
   \text{im}_{2,s\rightarrow rgb}(x, y) =
   \left[\begin{array}\,
   \text{im}_s(x, y) \otimes \text{psf}_{s\rightarrow r}(x, y)\\
   \text{im}_s(x, y) \otimes \text{psf}_{s\rightarrow g}(x, y)\\
   \text{im}_s(x, y) \otimes \text{psf}_{s\rightarrow b}(x, y)\\
   \end{array}\right]
   :label: eq_conv_rgb_s_to_rgb

In this context, :math:`\text{im}_s` describes the spatial distribution of the intensity of the source, 
which emits the spectrum :math:`s`.

Convolution of spectral homogeneous image and PSF
--------------------------------------------------

In the special scenario where both the image and the PSF are spectrally homogeneous,
the relationship can be defined as:

.. math::
   \text{im}_{2,s_1\rightarrow s_2}(x, y) = \text{im}_{s_1}(x, y) \otimes \text{psf}_{s_1\rightarrow s_2}(x, y)
   :label: eq_conv_rgb_w_to_w

In this equation, :math:`s_1` represents the source spectrum, and :math:`s_2` represents the detector spectrum.
Due to potential absorption effects, these spectra do not need to be identical. 
As explained earlier, it is crucial that both the source and the PSF exhibit spectral homogeneity.
A typical example involves a black and white image in conjunction with a wavelength-independent PSF.

Limitations
=================================================

The limitations are detailed in :numref:`convolve_limitations`.

Processing Steps
==================

1. Convert the image and PSF to linear sRGB values, while including negative values.
2. For a grayscale PSF, normalize the PSF so that its sum (total power) equals one.
3. Downscale/interpolate the PSF so that the physical pixel sizes of the PSF and the image 
   (after scaling with the magnification factor) are identical.
4. Pad the PSF with zeros to ensure a defined fall-off
5. Flip the image if the magnification factor is negative
6. Pad the image according to the chosen padding method.
7. Convolve image and PSF according to the methods in Section :numref:`psf_color_handling`.
8. Convert the image back to sRGB, applying a selected gamut mapping.
9. Slice the image to remove padding, or trim it back to its original size

The convolution is conducted in sRGB coordinates because the channels are orthogonal, and this color space 
is the target for monitors. Nevertheless, the convolution must be executed as a linear operation using 
linear sRGB values. It is also crucial to include colors outside the color space 
(negative coordinates) to maintain linearity. If negative values persist in the image post-convolution, 
gamut mapping needs to be applied.

In the case of a grayscale PSF, it is automatically normalized. 
This ensures that, when used in conjunction with the :python:`normalize=False` parameter of the convolve function, 
the brightness and color values remain unchanged. 
For colored PSFs, normalization poses a greater challenge because it requires knowing the amount of light 
from the source that actually reached the detector for the PSF. 
This could potentially be achieved using metadata from RenderImage. 
Nonetheless, the relevance of this option is debatable, as normalized images are typically preferred.

Downscaling the PSF needs to be executed in a manner that conserves energy.
Moreover, it is essential to choose a method that prevents aliasing. 
We utilize the scaling with the `INTER_AREA <https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html#gga5bb5a1fea74ea38e1a5445ca803ff121acf959dca2480cc694ca016b81b442ceb>`__ 
option from OpenCV within the resize function. 
The PSF must be rescaled so that the physical pixel dimensions of the image and the PSF align in both dimensions. 
Consequently, it becomes sufficient to convolve the image as a pixel matrix, even if the pixels are not square.

The convolution is performed in the Fourier space employing the convolution theorem. 
The Fourier transformation is calculated using the :func:`scipy.signal.fftconvolve` function.
Due to the nature of this method, areas outside the image are assumed to be black. 
Consequently, a fall-off region appears in the resulting image,
where the PSF increasingly convolves with the black portions at the edges. 
This transition area is as wide as the PSF region where its intensities exceed zero. 
For simplicity, we assume the entire PSF width for this. 
If the user desires a different padding method, additional padding must be applied to the image. 
First, the image should be padded according to the user-chosen method, and then again to prevent dark fall-off edges.

------------

**References**

.. footbibliography::

