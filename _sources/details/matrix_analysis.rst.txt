
.. _ray_matrix_analysis:

*****************************
Ray Transfer Matrix Analysis
*****************************

ABCD Matrix 
=================================================

In paraxial optics, the relationships between angles :math:`\theta` and distances :math:`x` relative 
to the optical axis can be expressed linearly. 
The ABCD matrix, a fundamental concept in this context, encapsulates the linear components necessary
to compute the output parameters from the input values for a system characterized by the same matrix. 
This relationship can be represented as: :footcite:`IvanOptics`

.. math::
   \left[\begin{array}{l}
   x_2 \\
   \theta_2
   \end{array}\right]=\left[\begin{array}{ll}
   A & B \\
   C & D
   \end{array}\right]\left[\begin{array}{l}
   x_1 \\
   \theta_1
   \end{array}\right]
   :label: TMA_ABCD

.. _TMA_elements:

.. list-table:: Description of matrix elements :footcite:`IvanOptics`
   :widths: 80 300
   :header-rows: 1
   :align: center
   
   * - Element
     - Description
   * - :math:`A`
     - positional scaling
   * - :math:`B`
     - position change depending on input angle
   * - :math:`C`
     - | angle change depending on input position. 
       | Equivalent to :math:`-\frac{1}{f}`, with :math:`f` being the focal distance
   * - :math:`D`
     - angular scaling

Zero matrix elements, as described in :numref:`TMA_zero_elements`, 
hold particular importance as they indicate specific cases of imaging and focusing.

.. _TMA_zero_elements:

.. list-table:: Significance of zero matrix elements :footcite:`GillLaser,pedrotti_pedrotti_pedrotti_2006` 
   :widths: 50 200
   :header-rows: 1
   :align: center
   
   * - Case
     - Description
   * - :math:`A=0`
     - | parallel to point focussing,
       | output lies at second focal plane    
   * - :math:`B=0`
     - | point to point focussing (image of an object),
       | input and output lie at conjugate planes,
       | :math:`A` gives the image magnification
   * - :math:`C=0`
     - | parallel to parallel imaging (afocal or telescopic system),
       | :math:`D` gives us the angular magnification 
   * - :math:`D=0`
     - | point to parallel imaging (e.g. headlamp), 
       | input lies at first focal plane


An important relationship is that the determinant of an ABCD matrix, denoted here as :math:`M`, 
always equals the ratio between the refractive indices of the preceding medium :math:`n_i` 
and the subsequent medium :math:`n_o` :footcite:`pedrotti_pedrotti_pedrotti_2006`.

.. math::
   \det (M) = AD - BC = \frac{n_i}{n_o}
   :label: eq_TMA_det


Propagation through Free Space 
=================================================

An ABCD-Matrix for free space with distance :math:`d` has the following form: 
:footcite:`pedrotti_pedrotti_pedrotti_2006`

.. math::
   \text{M}_\text{s} =
   \left[\begin{array}{ll}
   1 & d \\
   0 & 1
   \end{array}\right]
   :label: TMA_free_space

Refraction on a Curved Interface 
=================================================

An ABCD matrix for free space over a distance :math:`d` is represented as follows 
:footcite:`pedrotti_pedrotti_pedrotti_2006` :

.. math::
   \text{M}_\text{c} =
   \left[\begin{array}{cc}
   1 & 0 \\
   -\frac{n_o-n_i}{R \cdot n_o} & \frac{n_i}{n_o}
   \end{array}\right]
   :label: TMA_curved_nterface

Refraction on a Flat Interface 
=================================================

When :math:`R \to \infty`, which is equivalent to a flat interface, 
the matrix simplifies to :footcite:`pedrotti_pedrotti_pedrotti_2006` :

.. math::
   \text{M}_\text{i} =
   \left[\begin{array}{cc}
   1 & 0 \\
   0 & \frac{n_i}{n_o}
   \end{array}\right]
   :label: TMA_flat_interface

Thick Lens 
=================================================

For a thick lens, several parameters are to be considered: 
The lens has a refractive index :math:`n`, front surface curvature :math:`R_1`, and back surface curvature :math:`R_2`. 
Its thickness is :math:`d`, with a medium of refractive index :math:`n_1` 
in front and a medium of refractive index :math:`n_2` behind. 
Using ray transfer matrix analysis, this system is represented by the product of the front surface matrix 
:math:`\text{M}_\text{c1}`, the free space propagation matrix :math:`\text{M}_\text{s}`, 
and the back surface matrix :math:`\text{M}_\text{c2}`. 
It is important to note that matrices are multiplied from right to left. 
The resulting matrix can be expressed as: :footcite:`Kaschke2014`

.. math::
   \text{M}_\text{thick} =&~~ \text{M}_\text{c2} \cdot \text{M}_\text{s} \cdot \text{M}_\text{c1}\\
    =&~
   \left[\begin{array}{cc}
   1 & 0 \\
   -\frac{n_2-n}{R_2 \cdot n_2} & \frac{n}{n_2}
   \end{array}\right]
   \left[\begin{array}{ll}
   1 & d \\
   0 & 1
   \end{array}\right]
   \left[\begin{array}{cc}
   1 & 0 \\
   -\frac{n-n_1}{R_1 \cdot n} & \frac{n_1}{n}
   \end{array}\right]\\
    =&~ 
   \left[\begin{array}{cc}
   1 + \frac{n_1-n}{n R_1}d & \frac{n_1}{n}d \\
    \frac{n_1 - n}{n_2 R_1}  + \frac{n-n_2}{n_2 R_2} + \frac{n_1 - n}{n R_1}\frac{n-n_2}{n_2 R_2}d 
    & \frac{n_1}{n_2} + \frac{n_1}{n}\frac{n - n_2}{n_2 R_2}d
   \end{array}\right]
   :label: TMA_thick_lens_complete

When the surrounding media are identical, i.e., :math:`n_0 := n_1 = n_2`, the matrix simplifies to:

.. math::
   \text{M}_{\text{thick},n_0}   =&~ 
   \left[\begin{array}{cc}
   1 + \frac{n_0-n}{n R_1}d & \frac{n_0}{n}d \\
   \frac{n_0 - n}{n_0} \left( \frac{1}{R_1}  - \frac{1}{R_2} + \frac{n-n_0}{n R_1 R_2}d \right) & 1
        + \frac{n - n_0}{n R_2}d
   \end{array}\right]
   :label: TMA_thick_lens_same_media


Thin Lens
====================

In general, the matrix element :math:`C` can be interpreted as the negative inverse focal length, 
:math:`-\frac{1}{f}`. For a thin lens, where :math:`d=0`, equation :math:`TMA_thick_lens_complete` simplifies to:


.. math::
    \text{M}_\text{thin} =
   \left[\begin{array}{cc}
   1 & 0 \\
   -\frac{1}{f} & \frac{n_i}{n_o}
   \end{array}\right]
   :label: TMA_thin_lens

When :math:`n_i = n_o`, resulting in :math:`D=1`, the matrix aligns with equations commonly found in the literature, 
as referenced in :footcite:`pedrotti_pedrotti_pedrotti_2006`.


Lensmaker Equation 
=================================================

For the thin lens, the element :math:`C` was equal to :math:`-\frac{1}{f}`. 
Negating this element from equation :math:numref:`TMA_thick_lens_complete` and applying :math:`-(n_1 - n) = (n - n_1)`, 
we obtain the focal length in the forward direction:

.. math::
   \frac{1}{f_2} = \frac{n-n_1}{n_2}\frac{1}{R_1} - \frac{n-n_2}{n_2}\frac{1}{R_2} + \frac{n-n_1}{n R_1}\frac{n-n_2}{n_2 R_2}d
   :label: TMA_lensmaker0

Performing the same calculations with the media and curvatures swapped yields the backward focal length:

.. math::
   f_1 = -\frac{n_1}{n_2} f_2
   :label: TMA_lensmaker_f_conv

In its expanded form, this is:

.. math::
   \frac{1}{f_1} = -\frac{n-n_1}{n_1}\frac{1}{R_1} + \frac{n-n_2}{n_1}\frac{1}{R_2} - \frac{n-n_1}{n R_1}\frac{n-n_2}{n_1 R_2}d
   :label: TMA_lensmaker1

Both equations above are consistent with :footcite:`pedrotti_pedrotti_pedrotti_2006`
For :math:`n_0 := n_1 = n_2`, we derive:

.. math::
   \frac{1}{f_2} = \frac{n-n_0}{n_0}\left(\frac{1}{R_1} - \frac{1}{R_2} + \frac{n-n_0}{n R_1 R_2}d \right)
   :label: TMA_lensmaker_common

This is the typical form of the lens maker equation :footcite:`LinsenschleiferWiki`.

Gullstrand Equation 
=================================================

Utilizing definition :math:numref:`TMA_power_alt` and equation :math:numref:`TMA_lensmaker0`, 
and defining :math:`D` as :math:`D_2` from now on, we can express:

.. math::
   D = \frac{n_2}{f_2} = \frac{n-n_1}{R_1} - \frac{n-n_2}{R_2} + \frac{n-n_1}{n R_1}\frac{n-n_2}{R_2}d
   :label: TMA_Gullstrand_base

This is equivalent to

.. math::
   D = \frac{n_2}{f_2} = \frac{n-n_1}{R_1}  + \left( - \frac{n-n_2}{R_2} \right) - \frac{n-n_1}{R_1} \cdot \left( - \frac{n-n_2}{R_2} \right) \frac{d}{n}
   :label: TMA_Gullstrand_step

With the surface optical powers :math:`D_\text{s1} = \frac{n-n_1}{R_1}` 
and :math:`D_\text{s2} = -\frac{n-n_2}{R_2}` this simplifies to:

.. math::
   D = D_\text{s1} + D_\text{s2} - D_\text{s1} D_\text{s2} \frac{d}{n}
   :label: TMA_Gullstrand

This is known as Gullstrand's equation :footcite:`GullstrandHyper,EdmundFocal`.


.. _ray_cardinal_points:

Cardinal Points 
=================================================

The following calculations are derived from :footcite:`DickenABCD` and :footcite:`pedrotti_pedrotti_pedrotti_2006`. 
Both sources also offer textual and graphical explanations of cardinal points and planes.

**Vertex Points**

The vertex points :math:`V_1` and :math:`V_2` are positioned at the optical axis 
and represent the front and back of the lens respectively.

**Principal Points**

.. math::
   P_1 =&~ V_1 - \frac{n_1 - n_2 D}{n_2 C}\\
   P_2 =&~ V_2 + \frac{1-A}{C}
   :label: TMA_principal

**Nodal Points**

.. math::
   N_1 =&~ V_1 - \frac{1-D}{C}\\
   N_2 =&~ V_2 + \frac{n_1 - n_2 A}{n_2 C}
   :label: TMA_nodal

**Focal Lengths**

Focal lengths are given by the negative inverse of :math:`C` as well as equation :math:numref:`TMA_lensmaker_f_conv`.

.. math::
   f_1 =&~ \frac{n_1}{n_2 C}\\
   f_2 =&~ -\frac{1}{C}
   :label: TMA_focal_length

**Focal Points**

.. math::
   F_1 = &~ P_1 + f_1\\
   F_2 = &~ P_2 + f_2
   :label: TMA_focal_points

**EFL, BFL, FFL**

Effective focal length (EFL), back focal length (BFL) and front focal length (FFL) are defined as follows: 

.. math::
   \text{FFL} =&~ F_1 - V_1 &=~& &\frac{D}{C}\\
   \text{BFL} =&~ F_2 - V_2 &=~& -&\frac{A}{C}\\
   \text{EFL} =&~ f_2 &=~& -&\frac{1}{C}
   :label: TMA_ffk_bfl_efl

.. _ray_power_def:

Optical Power 
=================================================

The default definition in `optrace` considers the optical power as the inverse of the geometric focal length.

.. math::
   D_1 = \frac{1}{f_1}\\
   D_2 = \frac{1}{f_2}
   :label: TMA_power_base
  
The alternative definition below has the advantage that :math:`D_\text{1n} = -D_\text{2n}` holds true 
independently of the refractive media. 

.. math::
   D_\text{1n} =&~ \frac{n_1}{f_1}\\
   D_\text{2n} =&~ \frac{n_2}{f_2}\\
   f_\text{1n} =&~ \frac{f_1}{n_1}\\
   f_\text{2n} =&~ \frac{f_2}{n_2}\\
   \text{EFL}_n =&~ \frac{f_2}{n_2}\\
   :label: TMA_power_alt

However, in this case, the focal lengths do not represent the actual distance between 
the principal plane and the focal points.
For :math:`n_1 = n_2 = 1`, both definitions are equivalent.

Lens Setups 
=================================================

To evaluate setups of :math:`N` lenses, the lens matrices :math:`\text{M}_\text{L,i}` 
and the free space matrices :math:`\text{M}_\text{s,j}` need to be multiplied. 
Here, :math:`i \in \{0, 1, \dots, N\}` and :math:`j \in \{0, 1, \dots, N-1\}` holds.

.. math::
   \text{M} = \text{M}_\text{L,N} \cdot \text{M}_\text{s,N-1} \dots \text{M}_\text{s,0} \cdot \text{M}_\text{L,0}
   :label: TMA_setup


.. _ray_image_object_distances:


Optical Center
=====================


**General Case**

The optical center is the axial position where nodal rays intersect with the optical axis,
as illustrated by :numref:`fig_optical_center`.

.. _fig_optical_center:
.. figure:: ../images/optical_center.svg
   :align: center
   :width: 550
   :class: dark-light

   Optical center of a lens.

For the yellow triangle, the following relation holds:

.. math::
   \tan \phi = \frac{x_2 - x_1}{V_2 - V_1}
   :label: eq_oc_tan_phi1

From the green triangle, we derive:

.. math::
   \tan \phi = -\frac{x_1}{o}
   :label: eq_oc_tan_phi2

Note that the minus sign was added so both equations maintain the same sign. 
Inserting :math:numref:`eq_oc_tan_phi1` into :math:numref:`eq_oc_tan_phi2` gives:

.. math::
   o = - \frac{x_1}{x_2 - x_1} \left(V_2 - V_1\right)
   :label: eq_oc_o1

The blue triangle leads to:

.. math::
   \tan \theta_1 = -\frac{x_1}{N_1 - V_1}
   :label: eq_oc_theta1

With paraxial rays, the approximation :math:`\theta_1 \approx \tan \theta_1` holds. 
Therefore, :math:`x_2` can be calculated using the ABCD matrix of the setup:

.. math::
   x_2 = A x_1 - \frac{B}{N_1 - V_1} x_1
   :label: eq_oc_x2


Inserting into :math:numref:`eq_oc_o1` gives us:

.. math::
   o = -\frac{x_1}{A x_1 - \frac{B}{N_1 - V_1} x_1 -x_1} \left(V_2 - V_1\right)
  :label: eq_oc_o2


From :math:numref:`TMA_nodal`, it follows that:

.. math::
   N_1 - V_1 = - \frac{1 - D}{C}
   :label: eq_oc_dNV

Which can also be inserted into the equation :math:numref:`eq_oc_o2`.
After some rearranging we obtain:

.. math::
   o = \frac{V_2 - V_1}{1 - A + \frac{BC}{D-1}}
   :label: eq_oc_o3

This value needs to be added to the front vertex to get the absolute position of the optical center:

.. math::
   \text{OC} = V_1 + \frac{V_2 - V_1}{1 - A + \frac{BC}{D-1}}
   :label: eq_oc_final

The requirements that were implicitly assumed include the existence of a nodal point (:math:`C \neq 0`) 
and that the input and output positions differ (:math:`x_2 \neq x_1`). 
The only scenario where it makes sense to define an optical center, despite these conditions, is for an ideal lens. 
In this case, we set :math:`\text{OC} = V_1`, although a nodal ray does not cross the optical axis at that point.
In all other cases, particularly when :math:`1 - A + \frac{BC}{D - 1} = 0` or :math:`D = 1`, 
the optical center is undefined. 
As mentioned previously, an ideal lens (:math:`A = 1, ~B=0, ~C\neq 0, ~D=1`) is an exception.


**Thick Lens/Lens Combination with Same Front and Back Medium**

With :math:`m := \frac{n - n_0}{n}d` matrix :math:numref:`TMA_thick_lens_same_media` becomes:

.. math::
   \text{M}_{\text{thick},n_0}   =&~ 
   \left[\begin{array}{cc}
   1 - \frac{m}{R_1} & \frac{n_0}{n}d \\
   \frac{n_0 - n}{n_0} \left( \frac{1}{R_1}  - \frac{1}{R_2} + \frac{m}{R_1 R_2} \right) & 1 + \frac{m}{R_2}
   \end{array}\right]
   :label: TMA_thick_lens_same_media_m

The denominator of equation :math:numref:`eq_oc_final` is then:

.. math::
   1 - A + \frac{BC}{D - 1} &= 1 - \left(1 - \frac{m}{R_1}\right) + \frac{\frac{n_0-n}{n_0}\frac{n_0}{n}d \left( \frac{1}{R_1}  - \frac{1}{R_2} + \frac{m}{R_1 R_2} \right)}{1 + \frac{m}{R_2} - 1}\\
   &= \frac{m}{R_1} - m \frac{\left( \frac{1}{R_1}  - \frac{1}{R_2} + \frac{m}{R_1 R_2} \right)}{\frac{m}{R_2}}\\
   &= \frac{m}{R_1} - \left( \frac{R_2}{R_1}  - 1 + \frac{m}{R_1} \right)\\
   &= 1 - \frac{R_2}{R_1}
   :label: eq_oc_thick_lens_denom

Leading to the final form of :math:numref:`eq_oc_final`:

.. math::
   \text{OC} = V_1 + \frac{V_2 - V_1}{1 - \frac{R_2}{R_1}}
   :label: eq_oc_radii

.. Simplifying this expression towards this form is quite labor-intensive but goes without any conditions or tricks.
.. A interested reader is free to do it by themselves or trust Wolfram Alpha with the query: `Link <https://www.wolframalpha.com/input?i=1+-+A+%2B+B*C%2F%28D+-+1%29+with+A+%3D+%281+%2B+%28n_0-n%29%2F%28n+R_1%29*d%29%2C+B+%3D+%28%28n_0%29%2F%28n%29*d%29%2C+C+%3D+%28%28n_0+-+n%29%2F%28n_0+R_1%29%2B%28n-n_0%29%2F%28n_0+R_2%29%2B%28n_0+-+n%29%2F%28n+R_1%29*%28n-n_0%29%2F%28n_0+R_2%29*d%29%2C+D+%3D+%281+%2B+%28n+-+n_0%29%2F%28n+R_2%29*d%29>`__.

Equation :math:numref:`eq_oc_radii` is consistent with the results in :footcite:`10.1117/12.805489`. 
As mentioned in :footcite:`jenkins2001fundamentals`, in this case, the optical center is completely independent 
of the wavelength and the material dispersion.

Applying the same approach to two ideal lenses with focal lengths :math:`f_1` and :math:`f_2`, 
separated by a distance :math:`d` and surrounded by the same ambient media, results in a similar form:

.. math::
   M_\text{2L} &= \left[\begin{array}{cc}
   1 & 0 \\
   \frac{1}{f_2} & 1
   \end{array}\right] \cdot\left[\begin{array}{ll}
   1 & d \\
   0 & 1
   \end{array}\right] \cdot\left[\begin{array}{cc}
   1 & 0 \\
   \frac{1}{f_1} & 1
   \end{array}\right]\\
   &= \left[\begin{array}{cc}
   1+\frac{d}{f_1} & d \\
   \frac{1}{f_1}+\frac{1}{f_2}+\frac{d}{f_1 f_2} & 1+\frac{d}{f_2}
   \end{array}\right]
   :label: eq_oc_two_lens_matrix

.. math::
   1 - A + \frac{BC}{D - 1} &= 1 - \left(1 + \frac{d}{f_1}\right) + \frac{ \frac{1}{f_1} 
   + \frac{1}{f_2} + \frac{d}{f_1 f_2} }{1 + \frac{d}{f_2} - 1} d\\
   &= -\frac{d}{f_1} + \frac{f_2}{f_1} + 1 + \frac{d}{f_1}\\
   &= 1 + \frac{f_2}{f_1}
   :label: eq_oc_two_lenses_denom


.. math::
   \text{OC} = V_1 + \frac{V_2 - V_1}{1 + \frac{f_2}{f_1}}
   :label: eq_oc_two_lenses

For :math:`R_2 = -R_1` in :math:numref:`eq_oc_radii` or :math:`f_2 = f_1` in :math:numref:`eq_oc_two_lenses` 
the optical center lies at exactly the center of the lens/lens combination.

.. _image_object_distance:

Image and Object Distances 
=================================================

**Positions**

The matrix representation for additional object distance :math:`g` and image distance :math:`b` is given by:

.. math::
   \text{M}_\text{b,g} = 
   \left[\begin{array}{ll}
   1 & b \\
   0 & 1
   \end{array}\right]
   \cdot \text{M} \cdot
   \left[\begin{array}{ll}
   1 & g \\
   0 & 1
   \end{array}\right]
   :label: TMA_image_distance_mat

In this context, the distance :math:`b` is measured relative to the lens vertex point :math:`V_2` 
and the distance :math:`g` is measured relative to :math:`V_1`, 
with both distances considered positive when oriented towards the positive z-direction.

For the imaging element, it is essential that :math:`B_{b,g} = \text{M}_{b,g}[0, 1]` equals zero. 
This condition ensures that the output ray position :math:`x_2` is independent of the input angle :math:`\theta_1`, 
relying solely on the input position :math:`x_1`.

Consequently, the condition can be expressed as:

.. math::
   B_\text{b,g} = g (A + C b) + B + D b = 0
   :label: TMA_image_distance_eq

For :math:`b, g \in \mathbb{R}`, the solution is represented as:

.. math::
   b(g) = 
    \begin{cases}
   -\frac{B + g A} {D + g C}, &~ \text{for}~~ {D + g C} \neq 0\\
   \mathbb{R} &~ \text{for}~~ {D + g C} = 0 ~~\text{and}~~ B + g A = 0\\
   \emptyset &~ \text{for}~~ {D + g C} = 0 ~~\text{and}~~ B + g A \neq 0
  \end{cases}
  :label: TMA_image_distance_solution


.. math::
   g(b) = 
    \begin{cases}
   -\frac{B + b D} {A + b C}, &~ \text{for}~~ {A + b C} \neq 0\\
   \mathbb{R} &~ \text{for}~~ {A + b C} = 0 ~~\text{and}~~ B + b D = 0\\
   \emptyset &~ \text{for}~~ {A + b C} = 0 ~~\text{and}~~ B + b D \neq 0
  \end{cases}
  :label: TMA_object_distance_solution

For special cases concerning limits approaching :math:`\pm\infty`, we derive:

.. math::
   \lim_{g \to \pm \infty} b(g) = 
     \begin{cases}
   -\frac{A}{C} &~ \text{for}~~ {C} \neq 0\\
   \mathbb{R} &~ \text{for}~~ C = 0 ~~\text{and}~~ A = 0\\
   \emptyset &~ \text{for}~~ C = 0 ~~\text{and}~~ A \neq 0
  \end{cases}
  :label: TMA_image_distance_solution_special

.. math::
   \lim_{b \to \pm \infty} g(b) = 
     \begin{cases}
   -\frac{D}{C} &~ \text{for}~~ {C} \neq 0\\
   \mathbb{R} &~ \text{for}~~ C = 0 ~~\text{and}~~ D = 0\\
   \emptyset &~ \text{for}~~ C = 0 ~~\text{and}~~ D \neq 0
  \end{cases}
  :label: TMA_object_distance_solution_special

Optrace assigns NaN (not a number) in all cases of :math:`\emptyset` and :math:`\mathbb{R}`, 
as these cases are impractical.

For a matrix :math:`\text{M} = \text{M}_\text{thin}` representing the thin lens approximation,
derived from equation :math:numref:`TMA_thin_lens`, the expressions simplify to:

.. math::
   b(g) = \frac{fg}{g-\frac{n_i}{n_o}f}
   :label: TMA_image_imaging_eq

.. math::
   g(b) = \frac{\frac{n_i}{n_o}fb}{b-f}
   :label: TMA_object_imaging_eq

These expressions are consistent with the well-known imaging equation:

.. math::
   \frac{n_o}{f} = \frac{n_i}{g} + \frac{n_o}{b}
   :label: TMA_imaging_eq_n

In the most common scenario, where :math:`n_i = 1` and :math:`n_o = 1`, 
this simplifies to: :footcite:`LinsenGleichungWiki`

.. math::
   \frac{1}{f} = \frac{1}{g} + \frac{1}{b}
   :label: TMA_imaging_eq_base

Here, :math:`f`, :math:`b`, and :math:`g` are considered positive if measured in the positive z-direction.

**Magnifications**

With a given object and image position, the combined ABCD matrix as defined in :math:numref:`TMA_image_distance_mat`
can be computed. In this matrix, element :math:`A` corresponds to the magnification factor :math:`m`, 
given that :math:`B=0` applies for this configuration.

For a thin, ideal lens, the magnification factor :math:`m` is equivalent to :math:`b/g`.

The value of this factor conveys significant optical properties. 
Specifically, if :math:`A < 0`, the image is inverted, if :math:`A > 0`, the image is upright. 
Furthermore, if :math:`\lvert A \rvert > 1`, the image experiences a size increase, 
whereas if :math:`\lvert A \rvert < 1`, the image undergoes a size decrease.

.. _pupil_calculation:

Entrance and Exit Pupils
=================================================

To calculate the entrance and exit pupils for a given optical system and aperture stop, 
the lens setup is conceptually divided into a front and a rear group. 
The pupils are defined as the image of the aperture stop within the respective group. :footcite:`GreivenStop`

.. figure:: ../images/pupil_calculation.svg
   :align: center
   :width: 850
   :class: dark-light

   Visualization of the matrix separation and different distances.

**Aperture inside setup**

When the aperture stop is positioned within the lens setup, the system and its matrix :math:`\text{M}` 
can be decomposed into three distinct parts:

.. math::
   \text{M} = \text{M}_\text{rear} \cdot \text{M}_\text{gap} \cdot \text{M}_\text{front}
   :label: eq_pupils_separation

Here, :math:`\text{M}_\text{front}` represents the matrix for all surfaces located before the aperture stop, 
while :math:`\text{M}_\text{rear}` pertains to those surfaces positioned after the stop. 
The matrix :math:`\text{M}_\text{gap}` accounts for the distance in the gap region where the stop is located 
and is not associated with either the front or rear group.

The entrance pupil is formed by imaging the stop into the front group, 
whereas the exit pupil is the image of the aperture stop in the rear group.

The object distance for calculating the exit pupil is:

.. math::
   g_\text{ex} = V_{1,\text{rear}} - z_\text{s}
   :label: eq_pupils_gex

The exit pupil image distance :math:`b_\text{ex}` is determined using :math:`\text{M}_{b,g} = \text{M}_\text{rear}` 
and following the procedure outlined in :numref:`image_object_distance`. 
Subsequently, the resulting position :math:`z_\text{ex}` is obtained as follows:

.. math::
   z_\text{ex} = V_{2,\text{rear}} + b_\text{ex}
   :label: eq_pupils_zex


For the entrance pupil, the aperture stop is imaged in the backward direction through :math:`\text{M}_\text{front}` 
by inverting the matrix:

.. math::
   \text{M}^{-1}_\text{front} &= \left[\begin{array}{cc}
       A & B \\
       C & D
       \end{array}\right]^{-1}\\
    &= \frac{1}{AD - BC}  \left[\begin{array}{cc}
       D & -B \\
       -C & A
       \end{array}\right]\\
   :label: eq_pupils_front_inverse

The object distance is negative and calculated with the back vertex of the front group: 

.. math::
   g_\text{en} = V_{2,\text{front}} - z_\text{s}
   :label: eq_pupils_gen

To compute :math:`b_\text{en}(g_\text{en})`, the procedure outlined in :numref:`image_object_distance` is utilized, 
setting :math:`\text{M}_{b,g} = \text{M}^{-1}_\text{front}`. 
The image distance :math:`b_\text{en}` is then added to the front vertex of the front 
group to determine the entrance pupil position:

.. math::
   z_\text{en} = V_{1,\text{front}} + b_\text{en}
   :label: eq_pupils_zen


**Aperture in front of setup**

When the aperture stop is located in front of all lenses, it directly serves as the entrance pupil, 
thus :math:`z_\text{en} = z_\text{s}`.
The exit pupil is then calculated by imaging through all elements using the matrix:

.. math::
   \text{M}_\text{rear} = \text{M} 
   :label: eq_pupils_rear_only

**Aperture behind of setup**

Conversely, if the aperture stop is positioned behind all lenses, 
it equates to the exit pupil, meaning :math:`z_\text{ex} = z_\text{s}`.
The entrance pupil in this scenario is calculated by imaging backwards through all elements, 
employing the procedure described previously, using the matrix:

.. math::
   \text{M}_\text{front} = \text{M} 
   :label: eq_pupils_front_only

------------

**References**

.. footbibliography::
