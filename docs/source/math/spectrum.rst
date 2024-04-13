********************************
Spectrum and Refraction Index
********************************

Spectrum
==============


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


Refraction  Index
===================



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

