
*****************
Autofocus
*****************

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


