********************************
Spectrum and Refraction Index
********************************

Spectrum
==============

Functions
------------


**Monochromatic**

This implements a monochromatic source with wavelength :math:`\lambda_0`.

.. math::
   S(\lambda) = \delta(\lambda - \lambda_0)
   :label: eq_spectrum_mono

**Lines**

A line spectrum consist of multiple monochromatic sources, parameters are a set of power and wavelength combinations
:math:`L=\left\{(P_1,~\lambda_1),~(P_2,~\lambda_2),~\dots\right\}`

.. math::
   S(\lambda) = \sum_{(P_i, ~\lambda_i) \in L}  P_i \, \delta(\lambda - \lambda_i)
   :label: eq_spectrum_lines

**Constant**

Constant spectrum with value :math:`S_0`

.. math::
   S(\lambda) = S_0
   :label: eq_spectrum_const

Note that while the function is constant, only rays inside the visible range (default being the 380-780 nanometer range) are simulated.

**Rectangle**

Spectrum with a rectangular function :math:`\Pi(\lambda)` with bounds :math:`\lambda_0,~\lambda_1` and a scaling factor :math:`S_0`.

.. math::
   S(\lambda) = S_0\, \Pi_{[\lambda_0,~\lambda_1]}(\lambda)
   :label: eq_spectrum_rect

**Gaussian**

A gaussian spectrum is modelled with a scaling factor :math:`S_0`, a center wavelength :math:`\lambda_0` and a standard deviation :math:`\lambda_\sigma`.

.. math::
   S(\lambda) = S_0 \exp \left( -\frac{\left(\lambda - \lambda_0\right)^2}{2 \lambda^2_\sigma}\right)
   :label: eq_spectrum_gauss

**User Data/Function**

With Data/Function mode the spectrum is simply modelled by a user function/ data set. With a data set data is linearly interpolated.

.. math::
   S(\lambda) = S_F(\lambda)
   :label: eq_spectrum_user

**Blackbody**

The spectral radiance for a blackbody according to Planck's Law is: :footcite:`PlanckWiki`

.. math::
   B_\lambda (\lambda, ~T) = \frac{2 h c^2}{\lambda^5} \frac{1}{\exp\left(\frac{h  c } {\lambda k_\text{B}  T}\right) - 1}
   :label: planck_radiator

The equation contains the speed of light :math:`c`, the Planck constant :math:`h` and the Boltzmann constant :math:`k_\text{B}`:

.. math::
   c =&~ 299792458 ~\text{m/s}\\
   h =&~ 6.62607015\cdot 10^{-34} ~\text{J s}\\
   k_\text{B} =&~ 1.380649 \cdot 10^{-23} ~\text{J/K}\\

Note that :math:`\lambda` must be specified in meters.


.. topic:: Note

   The spectral radiance :math:`B_\lambda` (Power per solid angle, source area and wavelength) is given in units :math:`\text{W}/(\text{m}^3~\text{sr})`, whereas the units in this class should be :math:`\text{W/nm}` (Power per wavelength). Since :math:`B_\lambda` is constant over the source area and angle independent, converting it corresponds to a simple rescaling. 
   This is done while raytracing, where a specfic desired power is matched.


There is an option to normalize the spectrum, such that the peak value inside the visible range is equal to one.
If the peak wavelength is inside the visible range then to Stefan–Boltzmann law can be applied to calculate the normalization factor, otherwise the maximum value should lie at one of the edges of the visible range.

Color
----------

Analogously to :numref:`xyz_color_space` the tristimulus values for the light spectrum :math:`S(\lambda)` can be calculated with:

.. math::
   X &=\int_{\lambda} S(\lambda) x(\lambda) ~d \lambda \\
   Y &=\int_{\lambda} S(\lambda) y(\lambda) ~d \lambda \\
   Z &=\int_{\lambda} S(\lambda) z(\lambda) ~d \lambda
   :label: XYZ_Calc_Spectrum

From there on, typical color model conversions can be applied.


.. _random_wavelengths:
   
Random Wavelengths
--------------------


**Monochromatic**


The set of wavelengths :math:`\Lambda` is simply the monochromatic wavelength repeated many times.

.. math::
   \Lambda = \left\{\lambda_0,~\lambda_0, \dots\right\}
   :label: eq_lspectrum_random_mono

**Lines**

The sum power of all sources is

.. math::
   P = \sum_{(P_i,~\lambda_i \in L)} P_i
   :label: eq_lspectrum_lines_power

The probabilities are then 

.. math::
   p = \left\{\frac{P_i}{P} ~~:~~ (P_i, \lambda_i) \in L \right\}
   :label: eq_lspectrum_lines_p

:math:`\Lambda` is then a random variable choosing from the set :math:`\left\{\lambda_0,~\lambda_1, ~\dots\right\}` with probabilities :math:`\left\{p_1,~p_2,~\dots\right\}`.


**Constant**

With *Constant* mode, the wavelengths are chosen from within the visible range with uniform random variable :math:`\mathcal{U}`.

.. math::
   \Lambda = \mathcal{U}_{[380\,\text{nm},~780\,\text{nm}]}
   :label: eq_lspectrum_random_const

**Rectangle**

In the case of a rectangular light spectrum the random variable is a uniform variable with the bounds being equal to the rectangle bounds.

.. math::
   \Lambda = \mathcal{U}_{[\lambda_0,~\lambda_1]}
   :label: eq_lspectrum_random_rect

**Gaussian**

A gaussian with :math:`\lambda` being limited to :math:`\lambda \in [\lambda_l,~\lambda_r]` the distribution is called a *truncated gaussian distribution*.
For this function the anti-derivative integration bounds :math:`\xi_a,~\xi_b` need to be calculated first before performing the inverse transform method.

.. math::
   \xi_a =&~ \frac{1}{2}\left(1 + \text{erf}\left(\frac{\lambda_\text{l} - \lambda_0}{\sqrt{2} \lambda_\sigma}\right)\right)\\
   \xi_b =&~ \frac{1}{2}\left(1 + \text{erf}\left(\frac{\lambda_\text{r} - \lambda_0}{\sqrt{2} \lambda_\sigma}\right)\right)
   :label: gaussian_trunc_lambda_bounds
         
With these bounds the random wavelengths are then

.. math::
   \Lambda = \lambda_0 + \sqrt{2} ~ \lambda_\sigma ~  \text{erf}^{-1}\left(2\,\mathcal{U}_{[\xi_a, ~\xi_b]}-1\right)
   :label: gaussian_trunc_lambda

**User Function/ User Data / Blackbody**

For these the inverse transform method in :numref:`inverse_transform` can be applied.


Wavelength Properties
---------------------------

The following wavelength properties are available:

**Peak Wavelength**

Wavelength inside the visible range with the highest spectrum value.

**Centroid Wavelength**

The centroid wavelength (center of mass) of spectrum :math:`S(\lambda)` is defined as:

.. math::
   \lambda_\text{c} = \frac{\int S(\lambda) \lambda~\text{d}\lambda}{\int S(\lambda) ~\text{d}\lambda}

**Dominant Wavelength**

The dominant wavelength is the wavelength with the same hue as the as the spectrum.
For some colors there are is no dominant wavelength viable (for instance magenta), in these cases a complementary wavelength should be provided.
See `Dominant Wavelength <https://en.wikipedia.org/wiki/Dominant_wavelength>`__ for a visual representation.

**Complementary Wavelength**

The dominant wavelength is the wavelength with the opposite hue as the as the spectrum.
For some colors there are is no complementary wavelength viable (for instance green), in these cases a dominant wavelength should be used.
See `Dominant Wavelength <https://en.wikipedia.org/wiki/Dominant_wavelength>`__ for a visual representation.

**Full Width Half Maximum**

This is the smallest distance in wavelength units around the peak where the spectrum crosses 50% of the spectrum peak's value.
While this metric makes sense for bell-shaped curves, for more complicated curves it is not suited.


Refraction  Index
===================

.. _index_functions:

Functions
-------------

The subsequent equations describe common refractive index models used in simulation software.
They are taken from :footcite:`ComsolDispersion` and :footcite:`ZemaxHagen`.
A comprehensive list of different index models is found in :footcite:`palmer2017`.

Generally, all coefficients must be given in powers of µm, while the same is true for the wavelength input.

**Cauchy**

.. math::
   n = c_0 + \frac{c_1}{\lambda^2} + \frac{c_2}{\lambda^4} + \frac{c_3}{\lambda^6}
   :label: n_cauchy

**Conrady**

.. math::
   n = c_0+ \frac{c_1} {\lambda} + \frac{c_2} {\lambda^{3.5}}
   :label: n_conrady

**Extended**

.. math::
   n^2 = c_0+c_1 \lambda^2+ \frac{c_2} {\lambda^{2}}+ \frac{c_3} {\lambda^{4}}+ \frac{c_4} {\lambda^{6}}+ \frac{c_5} {\lambda^{8}}+ \frac{c_6} {\lambda^{10}}+\frac{c_7} {\lambda^{12}}
   :label: n_extended


**Extended2**

.. math::
   n^2 = c_0+c_1 \lambda^2+ \frac{c_2} {\lambda^{2}}+ \frac{c_3} {\lambda^{4}}+\frac{c_4} {\lambda^{6}}+\frac{c_5} {\lambda^{8}}+c_6 \lambda^4+c_7 \lambda^6
   :label: n_extended2


**Handbook of Optics 1**

.. math::
   n^2 = c_0+\frac{c_1}{\lambda^2-c_2}-c_3 \lambda^2
   :label: n_optics1


**Handbook of Optics 2**

.. math::
   n^2 = c_0+\frac{c_1 \lambda^2}{\lambda^2-c_2}-c_3 \lambda^2
   :label: n_optics2

**Herzberger**

.. math::
   \begin{align}
   n =&~ c_0+c_1 L+c_2 L^2+c_3 \lambda^2+c_4 \lambda^4+c_5 \lambda^6 \\
   &\text{ with   } L= \frac{1} {\lambda^2-0.028 {\mu m^2}}
   \end{align}
   :label: n_herzberger

**Sellmeier1**

.. math::
   n^2 = 1+\frac{c_0 \lambda^2}{\lambda^2-c_1}+\frac{c_2 \lambda^2}{\lambda^2-c_3}+\frac{c_4 \lambda^2}{\lambda^2-c_5}
   :label: n_sellmeier1 

**Sellmeier2**

.. math::
   n^2 = 1+c_0+\frac{c_1 \lambda^2}{\lambda^2-c_2^2}+\frac{c_3}{\lambda^2-c_4^2}
   :label: n_sellmeier2 

**Sellmeier3**

.. math::
   n^2 = 1+\frac{c_0 \lambda^2}{\lambda^2-c_1}+\frac{c_2 \lambda^2}{\lambda^2-c_3}+\frac{c_4 \lambda^2}{\lambda^2-c_5}+\frac{c_6 \lambda^2}{\lambda^2-c_7}
   :label: n_sellmeier3 

**Sellmeier4**

.. math::
   n^2 = c_0+\frac{c_1 \lambda^2}{\lambda^2-c_2}+\frac{c_3 \lambda^2}{\lambda^2-c_4}
   :label: n_sellmeier4 

**Sellmeier5**

.. math::
   n^2 = 1+\frac{c_0 \lambda^2}{\lambda^2-c_1}+\frac{c_2 \lambda^2}{\lambda^2-c_3}+\frac{c_4 \lambda^2}{\lambda^2-c_5}+\frac{c_6 \lambda^2}{\lambda^2-c_7}+\frac{c_8 \lambda^2}{\lambda^2-c_9}
   :label: n_sellmeier5 

**Schott**

.. math::
   n^2 = c_0+c_1 \lambda^2+\frac{c_2}{ \lambda^{2}}+\frac{c_3} {\lambda^{4}}+\frac{c_4} {\lambda^{6}}+\frac{c_5} {\lambda^{8}}
   :label: n_schott 


.. _abbe_number:

Abbe Number
--------------

The Abbe number, also called :math:`V`-number, is a simple, scalar quantity describing the optical dispersive behavior of a medium. It is calculated from the refractive indices at three different wavelength.

.. math::
   V = \frac{n_\text{c} - 1}{n_\text{s} - n_\text{l}}
   :label: abbe_eq

With :math:`n_\text{s},~n_\text{c},~n_\text{l}` are the short, center and long wavelength refraction index.
A higher Abbe number is desirable, as it corresponds to less chromatic dispersion.
In addition, most materials have a *normal dispersion*, categorized as being :math:`\frac{\text{n}}{\text{d}\lambda}<0`, so the index falls off with larger wavelengths.
The :math:`V`-number is therefore positive for such materials.
A material with :math:`V=0` is an ideal material with no dispersion.

Selecting different glasses is often done with the help of a *Abbe diagram*, where the center refractive index as well as the :math:`V`-number of the material are plotted as a scatter diagram. Examples can be found in :numref:`refraction_index_presets`.

.. _index_from_abbe:

Curve from Abbe Number
-----------------------

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




------------

**References**

.. footbibliography::

