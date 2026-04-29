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
   \text{im}_2(x, y, \lambda) = \text{im}(x, y, \lambda) \circledast \text{psf}(x, y, \lambda)
   :label: eq_conv_per_wavelength

Special Case Spectral Homogenity
--------------------------------------

In the case of so-called spectral homogeneity, the spectrum :math:`S` is consistent across the entire image 
but is scaled by a location-dependent intensity factor :math:`\text{im}_S(x, y)`. 
The concept of spectral homogeneity goes back to Barnden (1974) :footcite:`Barnden_1974`, 
who also describes this mathematical simplification. 
Ravikumar et al. (2008) :footcite:`Ravikumar_2008` also uses this to generate color images.

For an example :math:`S(\lambda)`, it holds that:

.. math::
   \text{im}_2(x, y, \lambda) 
   &= \big[S(\lambda) \,\text{im}_S(x, y)\big] \circledast \text{psf}(x, y, \lambda)\\
   &= \text{im}_S(x, y) \circledast \big[S(\lambda)\, \text{psf}(x, y, \lambda)\big]\\
   &= \text{im}_S(x, y) \circledast \text{psf}_S(x, y, \lambda)
   :label: eq_conv_special_case_spectral_homogenity

The spectrum is a constant with respect to the convolution expression and can therefore 
also be multiplied with the PSF. We define the new spectrally weighted PSF :math:`\text{psf}_S(x, y, \lambda)`.

Convolution of channel-wise spectrally homogeneous RGB image and an RGB PSF
-------------------------------------------------------------------------------

Typically, the spectral profile of the image result is of no interest. 
Instead, the color coordinates for display on a monitor are more relevant. 
For instance, to calculate the red stimulus produced by the convolved image, 
it can be multiplied by the sRGB red color matching function :math:`r(\lambda)` and integrated over all wavelengths:

.. math::
   \text{im}_{2,R\rightarrow r}(x, y) 
   &= \int \Big[\text{im}_R(x, y) \circledast \text{psf}_R(x, y, \lambda)\Big] r(\lambda)\;\text{d}\lambda\\
   &= \text{im}_R(x, y) \circledast \int\text{psf}_R(x, y, \lambda)\, r(\lambda)\;\text{d}\lambda\\
   &= \text{im}_R(x, y) \circledast \text{psf}_{R\rightarrow r}(x, y)
   :label: eq_conv_rgb_r_to_r

:math:`\text{im}_R`, :math:`\text{psf}_R` correspond to :math:`\text{im}_S`, :math:`\text{psf}_S`
from :math:numref:`eq_conv_special_case_spectral_homogenity` for a red emitting spectrum :math:`R`.
Since only the PSF has a wavelength dependency in this representation, 
the weighting image :math:`\text{im}_R` can be factored out of the integral expression. 
Because the convolution is independent of wavelength, the integration can be performed prior to convolution. 
This results in the red-to-red PSF :math:`\text{psf}_{R \rightarrow r}(x, y)`.

If the light spectra :math:`R(\lambda), G(\lambda), B(\lambda)` 
are selected such that they result in the same color impression as the primaries of sRGB, 
they form linearly independent color channels. 
These channels can be used to compose all colors within the sRGB color space via linear combination.
The PSF can similarly be decomposed into three individual channel PSFs that are linearly independent 
as the color matching curves of sRGB are applied.

The RGB color image from the red image :math:`\text{im}_R(x, y)` with spatially homogeneous :math:`R(\lambda)`
for convolution with the PSF can then be simplified and summarized as:

.. math::
   \text{im}_{2,r\rightarrow rgb}(x, y) =
   \left[\begin{array}\,
   \text{im}_R(x, y) \circledast \text{psf}_{R\rightarrow r}(x, y)\\
   \text{im}_R(x, y) \circledast \text{psf}_{R\rightarrow g}(x, y)\\
   \text{im}_R(x, y) \circledast \text{psf}_{R\rightarrow b}(x, y)
   \end{array}\right]
   :label: eq_conv_rgb_r_to_rgb

Even if the original red spectrum generates a pure red in the sRGB color space, 
this might not be the case after convolution with the PSF. 
Chromatic aberration could lead to more transverse chromatic aberration at smaller wavelengths, 
resulting in a yellow fringe in the PSF and introducing components in :math:`\text{psf}_{R \rightarrow g}(x, y)`.

Similarly, this applies to the G-channel with the sRGB color matching function :math:`g(\lambda)`:

.. math::
   \text{im}_{2,G\rightarrow rgb}(x, y) =
   \left[\begin{array}\,
   \text{im}_G(x, y) \circledast \text{psf}_{G\rightarrow r}(x, y)\\
   \text{im}_G(x, y) \circledast \text{psf}_{G\rightarrow g}(x, y)\\
   \text{im}_G(x, y) \circledast \text{psf}_{G\rightarrow b}(x, y)
   \end{array}\right]
   :label: eq_conv_rgb_g_to_rgb

And the blue channel with matching function :math:`b(\lambda)`:

.. math::
   \text{im}_{2,B\rightarrow rgb}(x, y) =
   \left[\begin{array}\,
   \text{im}_B(x, y) \circledast \text{psf}_{B\rightarrow r}(x, y)\\
   \text{im}_B(x, y) \circledast \text{psf}_{B\rightarrow g}(x, y)\\
   \text{im}_B(x, y) \circledast \text{psf}_{B\rightarrow b}(x, y)
   \end{array}\right]
   :label: eq_conv_rgb_b_to_rgb

The overall image :math:`\text{im}_{2,RGB \rightarrow rgb}` is obtained from the sum of all convolved 
R, G, B color components in the image. However, the mixing ratio of all channels must be considered:
If the color PSFs were all simulated with a power of one watt, this does not correspond to the correct mixing 
ratio for white in the sRGB color space. 
It must be adjusted so that equal parts in :math:`\text{im}_R, \text{im}_G, \text{im}_B` 
produce white in the color space.

Let :math:`a_R, a_G, a_B` be the relative mixing factors. The final result can be expressed as linear combination:

.. math::
   \text{im}_{2,RGB\rightarrow rgb}(x, y) = a_R \,\text{im}_{2,R\rightarrow rgb}(x, y)
   + a_G \,\text{im}_{2,G\rightarrow rgb}(x, y) + a_B\, \text{im}_{2,B\rightarrow rgb}(x, y)
   :label: eq_conv_rgb_rgb_to_rgb

Equation :math:numref:`r_g_b_factors` illustrates the RGB color spectra and respective weighting factors. 
Rather than applying the rescaling subsequently, the power ratios might be integrated into the relative R, G, B PSFs.

Convolution of a spectral homogeneous image and an RGB PSF
--------------------------------------------------------------

In the special case where the image is spectrally homogeneous for a single spectrum :math:`S`. 
the transformation can then be represented by:

.. math::
   \text{im}_{2,S\rightarrow rgb}(x, y) =
   \left[\begin{array}\,
   \text{im}_S(x, y) \circledast \text{psf}_{S\rightarrow r}(x, y)\\
   \text{im}_S(x, y) \circledast \text{psf}_{S\rightarrow g}(x, y)\\
   \text{im}_S(x, y) \circledast \text{psf}_{S\rightarrow b}(x, y)
   \end{array}\right]
   :label: eq_conv_rgb_s_to_rgb

In this context, :math:`\text{im}_S` describes the spatial distribution of the intensity of the source, 
which emits the spectrum :math:`S`.

Convolution of RGB image and wavelength-independent PSF
--------------------------------------------------------

With a wavelength-independent PSF, the RGB image is simply calculated 
by convolving each channel with the same :math:`\text{psf}(x, y)`: 

.. math::
   \text{im}_{2,RGB}(x, y) =
   \left[\begin{array}\,
   \text{im}_R(x, y) \circledast \text{psf}(x, y)\\
   \text{im}_G(x, y) \circledast \text{psf}(x, y)\\
   \text{im}_B(x, y) \circledast \text{psf}(x, y)
   \end{array}\right]
   :label: eq_conv_rgb_to_c

A wavelength-independent absorption in the optical setup is modelled through a scaling of the PSF.

Convolution of spectral homogeneous image and PSF
--------------------------------------------------

With both PSF and image spectrally homogeneous, equation :math:numref:`eq_conv_special_case_spectral_homogenity`
simplifies to:

.. math::
   \text{im}_2(x, y, \lambda) 
   &= \big[S_\text{im}(\lambda) \,\text{im}_{\text{S}_\text{im}}(x, y)\big] \circledast \big[s_\text{psf}(\lambda)\,\text{psf}_{\text{s}_\text{psf}}(x, y)\big]\\
   &= \underbrace{\big[\text{im}_{\text{S}_\text{im}}(x, y) \circledast \text{psf}_{\text{s}_\text{psf}}(x, y)\big]}_{:= \text{im}_2(x, y)} \underbrace{S_\text{im}(\lambda)s_\text{psf}(\lambda)}_{:= S_2(\lambda)}
   :label: eq_conv_special_case_spectral_homogenity_2

Spatial and spectral information are now independent.
In the result :math:`\text{im}_2(x, y)` encodes the intensity of each pixel, while :math:`S_2` provides the spectrum.
:math:`S_\text{im}` denotes the emitted spectrum at each pixel, while :math:`s_\text{psf}` describes the global spectral
weighting of the PSF.
As an example, the former can be a broad blue spectrum and the latter only permits wavelengths near the violet region.

With a known :math:`S_2` the corresponding sRGB color can be calculated using the :math:`r, g, b` matching functions.
After calculating :math:`\text{im}_2` by convolution this color value gets repeated at every pixel and scaled by
the corresponding :math:`\text{im}_2` in linear sRGB coordinates.

In the even simpler case, :math:`\text{im}_{\text{S}_\text{im}}(x, y)` is a black-and-white image 
(meaning :math:`S_\text{im}(\lambda)` produces a white color in the sRGB colorspace)
and the PSF is wavelength-independent (:math:`s_\text{psf}(\lambda) = 1`). 
Under these circumstances, :math:`\text{im}_2` is already the resulting black-and-white image. 
Note that this is not true for :math:`S_\text{im}(\lambda) = 1` ("equal energy illuminant"), 
as its spectrum does not produce a white color in sRGB, but a slight orange hue.

Limitations
=================================================

The limitations are detailed in :numref:`convolve_limitations`.

Processing Steps
==================

The processing steps are:

1. Convert the image and PSF to linear sRGB values, while keeping negative values
2. For a grayscale PSF, normalize the PSF so that its sum (total power) equals one
3. Downscale/interpolate the PSF so that the physical pixel sizes of the PSF and the image 
   (after scaling with the magnification factor) are identical
4. Pad the PSF with zeros to ensure a defined fall-off
5. Flip the image if the magnification factor is negative
6. Pad the image according to the chosen padding method
7. Convolve image and PSF according to the methods in Section :numref:`psf_color_handling`
8. Convert the image back to sRGB, applying a selected gamut mapping
9. Slice the image to remove padding, or trim it back to its original size

The convolution is conducted in sRGB coordinates because the channels are orthogonal, and this color space 
is the target for monitors. Nevertheless, the convolution must be executed as a linear operation using 
linear sRGB values. It is also crucial to include colors outside the color space 
(negative coordinates) to maintain linearity. If negative values persist in the image post-convolution, 
gamut mapping needs to be applied.

A grayscale PSF is power-normalized for processing.
This ensures that, when used in conjunction with the :python:`normalize=False` parameter of the convolve function, 
intensity ranges are preserved. A dark image without bright areas stays dark, even after convolution.
For colored PSFs, power normalization poses a greater challenge because it requires knowing the amount of light 
from the source that actually reached the detector for each of the RGB-PSFs. 
This could potentially be achieved using metadata from RenderImage. 
Nonetheless, the relevance of this option is debatable, as normalized color images are typically preferred.

Downscaling the PSF needs to be executed in a manner that conserves energy.
Moreover, it is essential to choose a method that prevents aliasing. 
We utilize the scaling with the `INTER_AREA <https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html#gga5bb5a1fea74ea38e1a5445ca803ff121acf959dca2480cc694ca016b81b442ceb>`__ 
option from OpenCV within the resize function. 
The PSF must be rescaled so that the physical pixel dimensions of the image and the PSF align in both dimensions. 
Consequently, it becomes sufficient to convolve the image as a pixel matrix, even when the pixels are not square.

The convolution is performed in the Fourier space, employing the convolution theorem. 
The Fourier transformation is calculated using the :func:`scipy.signal.fftconvolve` function.
Due to the implementation of this method, areas outside the image are assumed to be black. 
Consequently, a fall-off region appears in the resulting image,
where the PSF increasingly convolves with the black portions near the edges. 
This transition area is half the PSF width. 
If the user desires a different padding method, additional padding must be applied to the image. 
First, the image should be padded according to the user-chosen method, and then again to prevent dark fall-off edges.

------------

**References**

.. footbibliography::

