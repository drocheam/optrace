***********************
PSF Convolution
***********************

.. TODO Teil überarbeiten

Convolution of Colored Images
================================

Polychromatic Convolution
--------------------------------------

Im Allgemeinen muss man die Bildfaltung pro Wellenlänge durchführen, Bild und PSF sind dabei generell beide wellenlängenabhängig:

.. math::
   \text{im}_2(x, y, \lambda) = \text{im}(x, y, \lambda) \otimes \text{psf}(x, y, \lambda)
   :label: eq_conv_per_wavelength

Special Case Spectral Homogenity
--------------------------------------

Im Falle der sogenannten spektralen Homogenität ist das Spektrum im Verlauf über das gesamte Bild das gleiche, es wird jedoch mit einem ortsabhängigem Intensitätsfaktor :math:`\text{im}_r(x, y)` skaliert.
Für ein beispielhaftes Rotspektrum :math:`\text{im}_r(\lambda)` gilt:

.. math::
   \text{im}_2(x, y, \lambda) 
   &= \left(\text{im}_r(\lambda) \text{im}_r(x, y)\right) \otimes \text{psf}(x, y, \lambda)\\
   &= \text{im}_r(x, y) \otimes \left(\text{im}_r(\lambda) \text{psf}(x, y, \lambda)\right)\\
   &= \text{im}_r(x, y) \otimes \text{psf}_r(x, y, \lambda)\\
   :label: eq_conv_special_case_spectral_homogenity

Das Spektrum ist bezogen auf den Faltungsausdruck eine Konstante, kann also auch an die PSF multipliziert werden.
Wir definieren uns die neue, spektral gewertete PSF :math:`\text{psf}_r(x, y, \lambda)`.


Convolution of sRGB Images
--------------------------------------

Typischerweise ist nicht der spektrale Verlauf des Bildergebnisses interessant, sondern nur die Farbkoordinaten für eine Farbdarstellung auf dem Monitor.
Um zu errechnen, welchen Rotreiz das gefaltete Bild erzeugt, kann man es mit der color matching function :math:`r(\lambda)` multiplizieren und aufintegrieren:

.. math::
   \text{im}_{2,r\rightarrow r}(x, y) 
   &= \int \Big[\text{im}_r(x, y) \otimes \text{psf}_r(x, y, \lambda)\Big] r(\lambda)\;\text{d}\lambda\\
   &= \text{im}_r(x, y) \otimes \int\text{psf}_r(x, y, \lambda) r(\lambda)\;\text{d}\lambda\\
   &= \text{im}_r(x, y) \otimes \text{psf}_{r\rightarrow r}(x, y)\\
   :label: eq_conv_rgb_b_to_r

Aber nur die PSF hat eine Wellenlängenabhängigkeit, also lässt sich das Bild aus dem Integralausdruck herausziehen.
Da die Faltung unabhängig von der Wellenlänge ist, lässt sich die Integration auch vor der Faltung durchführen.
Wir erhalten die Rot-zu-Rot PSF :math:`\text{psf}_{r\rightarrow r}(x, y)`.

Wählt man nun die Lichtspektren :math:`\text{im}_r(\lambda), \text{im}_g(\lambda), \text{im}_b(\lambda)` so, dass
sie die gleichen Farbkoordinaten wie die Primärspektren von sRGB besitzen, so handelt es sich um linear unabhängige
Farbkanäle, mit denen sich alle Farben innerhalb des sRGB Farbraums als Linearkombination zusammenstellen lassen.
Man kann nur die PSF in drei Einzelkanal PSF zerlegen, die ebenfalls linear voneinander unabhängig sind.

Das RGB Farbbild aus dem Rotbild :math:`\text{im}_r(x, y)` mit raümlich homogenem :math:`\text{im}_r(\lambda)` 
für die Faltung mit der PSF lässt sich dann also vereinfachen und zusammenfassen zu:

.. math::
   \text{im}_{2,r\rightarrow rgb}(x, y) =
   \left[\begin{array}\,
   \text{im}_r(x, y) \otimes \text{psf}_{r\rightarrow r}(x, y)\\
   \text{im}_r(x, y) \otimes \text{psf}_{r\rightarrow g}(x, y)\\
   \text{im}_r(x, y) \otimes \text{psf}_{r\rightarrow b}(x, y)\\
   \end{array}\right]

Auch wenn das Ursprungsrotspektrum ein reines Rot in sRGB Farbraum erzeugt, so muss das durch die Faltung mit der PSF
nicht der Fall sein. Durch starke Dispersion im System können gelbe Wellenlängen unter dem Rotspektrum achsferner sein,
wodurch sich ein gelber Rand in der PSF und Anteile in :math:`\text{psf}_{r\rightarrow g}(x, y)` ergeben können.

Analog gilt für den G-Kanal mit der sRGB color matching function :math:`g(\lambda)`:

.. math::
   \text{im}_{2,g\rightarrow rgb}(x, y) =
   \left[\begin{array}\,
   \text{im}_g(x, y) \otimes \text{psf}_{g\rightarrow r}(x, y)\\
   \text{im}_g(x, y) \otimes \text{psf}_{g\rightarrow g}(x, y)\\
   \text{im}_g(x, y) \otimes \text{psf}_{g\rightarrow b}(x, y)\\
   \end{array}\right]

Analog gilt für den G-Kanal mit der sRGB color matching function :math:`b(\lambda)`:

.. math::
   \text{im}_{2,b\rightarrow rgb}(x, y) =
   \left[\begin{array}\,
   \text{im}_b(x, y) \otimes \text{psf}_{b\rightarrow r}(x, y)\\
   \text{im}_b(x, y) \otimes \text{psf}_{b\rightarrow g}(x, y)\\
   \text{im}_b(x, y) \otimes \text{psf}_{b\rightarrow b}(x, y)\\
   \end{array}\right]

Das Gesamtbild :math:`\text{im}_{2,rgb\rightarrow rgb}` aus der Summe aller gefalteten R,G,B Farbkomponenten im Bild.
Jedoch muss man das Mischverhältnis aller Kanäle beachten:
Sind die Farb PSFs alle mit einer Leistungs mit einem Watt simuliert worden, so entspricht das nicht dem korrekten
Mischverhältnis für den sRGB Farbraum. Dieses muss so angepasst sein, dass gleiche Anteile 
in :math:`\text{im}_r, \text{im}_g, \text{im}_b` weiß im Farbraum erzeugen.

Seien :math:`a_r, a_g, a_b` die relativen Mischfaktoren, wobei :math:`a_r + a_g + a_b = 1` gilt, so ist das Endresultat:

.. math::
   \text{im}_{2,rgb\rightarrow rgb}(x, y) = a_r \text{im}_{2,r\rightarrow rgb}(x, y)
   + a_g \text{im}_{2,g\rightarrow rgb}(x, y) + a_b \text{im}_{2,b\rightarrow rgb}(x, y)

Die RGB Farbspektren und Gewichtungsfaktoren sind in Abschnitt <> gezeigt.

Convolution of a spectral homogeneous image and a sRGB PSF
--------------------------------------------------------------

Es gibt einen Speziallfall, wenn das Ursprungsbild spektral homogen ist.
Sei das Spektrum :math:`s`, dann gilt:

.. math::
   \text{im}_{2,s\rightarrow rgb}(x, y) =
   \left[\begin{array}\,
   \text{im}_s(x, y) \otimes \text{psf}_{s\rightarrow r}(x, y)\\
   \text{im}_s(x, y) \otimes \text{psf}_{s\rightarrow g}(x, y)\\
   \text{im}_s(x, y) \otimes \text{psf}_{s\rightarrow b}(x, y)\\
   \end{array}\right]

Für den Spezialfall, dass das Ursprungsbild spektral homogen ist

:math:`\text{im}_w(x, y)` rein schwarz-weiß ist, 
gilt :math:`\text{im}_r(x, y) = \text{im}_g(x, y) = \text{im}_b(x, y)` und somit:

Convolution of greyscale image and PSF
------------------------------------------

Für den Spezialfall, dass auch die PSF rein schwarz-weiß ist, gilt:

.. math::
   \text{im}_{2,w\rightarrow rgb}(x, y) = \text{im}_w(x, y) \otimes \text{psf}_{w\rightarrow w}(x, y)

Wobei man für eine Darstellung im RGB Farbraum dieses Bild für jeden Kanal vervielfachen müsste.


Vorraussetzungen
=================================================


Die Einschränkungen sind in Abschnitt <> beschrieben.

Vorgehen
==================

1. Umwandeln von Bild und PSF zu linearen sRGB Werten, dabei negative Werte mitnehmen
2. PSF herunterskalieren/interpolieren, sodass die physikalischen Pixelgrößen von PSF und Bild (nach Vergrößerung/Verkleinerung mit Abbildungsmaßstab) identisch sind
3. PSF mit Nullwerten padden
4. Bild drehen, wenn Abbildungsmaßstab negativ ist.
5. Bild mit gewählter Paddingmethode padden.
6. Bilder nach Methoden von Abschnitt <> falten
7. Bild zurück nach sRGB umwandeln, dabei gamut mapping betreiben
8. Bild zurechtschneiden


Die Faltung findet in sRGB Koordinaten statt, da hier die Kanäle orthogonal zueinander sind.
Außerdem entspricht dieser Farbraum dem Zielfarbraum von Monitoren.
Jedoch muss die Faltung als lineare Operation in linearen sRGB Werten stattfinden (Beschreibung siehe <>).
Auch Farben außerhalb des Farbraums (negative Koordinaten) müssen mitgenommen werden, damit die Operation linear bleibt.
Wenn nach der Faltung immer noch negative Werte im Bild sind, kann man später gamut mapping betreiben.

Herunterskalieren der PSF muss so erfolgen, dass dies leistungserhaltend ist.
Außerdem ist ein Verfahren wünschenswert, wo kein Aliasing stattfindet.
Wir nutzen die Skalierung mit INTER_AREA Option von openCV in der resize Funktion.
Die PSF muss so umskaliert werden, dass die physikalischen Pixeldimensionen von Bild und PSF in beide Dimension übereinstimmen.
Dann genügt es das Bild as Pixelmatrix zu falten, auch wenn die Pixel nicht-quadratisches sind.

Die Faltung wird im Fourierraum über das Faltungstheorem durchgeführt.
Die Funktion scipy.fftconvolve übernimmt dies für uns.
Methodenbedingt werden Bereiche außerhalb des Bildes als schwarz angenommen.
Somit haben wir außen einen abfallen Bereich im Ergebnisbild, wo die PSF am Rand zunehmend mit schwarz faltet.
Dieser Übergangsbereich ist so breit wie der PSF Bereich, wo Intensitäten größer Null sind.
Wir nehmen hierfür die gesamte PSF Breite.
Will der Nutzer eine andere Paddingmethode, so muss das Bild mit dieser Methode zusätzlich gepadded werden.
Einmal wegen der gewünschten Methode, und das zweite Mal, da wir wieder abfallende Ränder gegen ein Schwarzbild haben.

Procedure
=================================================

The point spread function (PSF) of a optical setup can be seen as impulse response of the same.
A convolution with this PSF is equivalent to applying the transfer function of the system to an input.
Note that this procedure can only simulate some optical effects, as it assumes the PSF is spatially constant.
This ignores aberrations like coma, off-axis astigmatism, field curvature, vignetting, distortion and others.

Image convolution is done by applying the convolution theorem of the Fourier transformation:

.. math::
   g \otimes h=\mathcal{F}^{-1}\{G \cdot H\}
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

The convolved image :math:`\text{im}_2` is calculated by a two dimensional spatial convolution between the image :math:`\text{im}` and the point spread function :math:`\text{psf}`.
When done correctly, all three not only depend on the position :math:`x, y` inside the image but also the wavelength :math:`\lambda` as the image and PSF can have different spectral distributions depending on the location.

.. math::
   \text{im}_2(x, y, \lambda) &= \text{im}(x, y, \lambda) \otimes \text{psf}(x, y, \lambda)\\
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
   \text{im}_{2,r} = \int   \text{im}_2(x, y, \lambda) \cdot r(\lambda) \;\text{d}\lambda \stackrel{!}{=} \text{im}_{r}(x, y) \otimes \text{psf}_r(x, y) 
   :label: eq_conv_desired

This is done by expanding all integrals:

.. math::
   \text{im}_{2,r}(x, y) 
   &= \text{im}_{r}(x, y) \otimes \text{psf}_r(x, y)\\
   &= \iint \text{im}_r(\tau_x, \tau_y) \cdot \text{psf}_r(x-\tau_x, y-\tau_y)  \;\text{d} \tau_x \,\text{d}\tau_y\\
   &= \iint \left( \int \text{im}(\tau_x, \tau_y, \lambda) \cdot r(\lambda) \, \text{d}\lambda \cdot \int \text{psf}(x-\tau_x, y-\tau_y, \lambda) \cdot r(\lambda) \,\text{d}\lambda \right) \;\text{d} \tau_x \,\text{d}\tau_y\\
   &= \iint \left( \int \text{im}(\tau_x, \tau_y, \lambda_1) \cdot r(\lambda_1) \, \text{d}\lambda_1 \cdot \int \text{psf}(x-\tau_x, y-\tau_y, \lambda_2) \cdot r(\lambda_2) \,\text{d}\lambda_2 \right) \;\text{d} \tau_x \,\text{d}\tau_y\\
   &= \iiiint \text{im}(\tau_x, \tau_y, \lambda_1) \cdot \text{psf}(x-\tau_x, y-\tau_y, \lambda_2) \cdot r(\lambda_1) \cdot r(\lambda_2) \;\text{d}\lambda_1 \, \text{d}\lambda_2  \,\text{d} \tau_x \,\text{d}\tau_y\\
   &= \iint \left(  \iint \text{im}(\tau_x, \tau_y, \lambda_1) \cdot \text{psf}(x-\tau_x, y-\tau_y, \lambda_2) \,\text{d} \tau_x \,\text{d}\tau_y \right) \cdot r(\lambda_1) \cdot r(\lambda_2) \;\text{d}\lambda_1 \, \text{d}\lambda_2  \\
   &= \iint \Bigl[  \text{im}(x, y, \lambda_1) \otimes \text{psf}(x, y, \lambda_2)\Bigr] \cdot r(\lambda_1) \cdot r(\lambda_2) \;\text{d}\lambda_1 \, \text{d}\lambda_2\\
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
   &= \iint \Bigl[  \text{im}(x, y, \lambda_1) \otimes \text{psf}(x, y, \lambda_2)\Bigr] \cdot r(\lambda_1) \cdot r(\lambda_2) \;\text{d}\lambda_1 \, \text{d}\lambda_2\\
   &= \int \Biggl[  \left( \int\text{im}(x, y, \lambda_1) \cdot r(\lambda_1)\;\text{d}\lambda_1 \right) \otimes \text{psf}(x, y, \lambda_2)\Biggr]  \cdot r(\lambda_2) \, \text{d}\lambda_2\\
   &= \int \Biggl[  \left( \int \Bigl\{ \text{im}_\text{r}(x, y) S_\text{im,r}(\lambda_1) + \text{im}_\text{g}(x, y) S_\text{im,g}(\lambda_1) +\text{im}_\text{b}(x, y) S_\text{im,b}(\lambda_1) \Bigr\} \cdot r(\lambda_1)\;\text{d}\lambda_1 \right) \otimes \text{psf}(x, y, \lambda_2)\Biggr]  \cdot r(\lambda_2) \, \text{d}\lambda_2\\
   &= \int \Biggl[  \left( \int \text{im}_\text{r}(x, y) S_\text{im,r}(\lambda_1) \cdot r(\lambda_1)\;\text{d}\lambda_1 \right) \otimes \text{psf}(x, y, \lambda_2)\Biggr]  \cdot r(\lambda_2) \, \text{d}\lambda_2\\
   &= \int S_\text{im,r}(\lambda_1) \cdot r(\lambda_1) \, \text{d}\lambda_1 \cdot \int \Bigl[\text{im}(x, y) \otimes \text{psf}(x, y, \lambda_2)\Bigr] \cdot r(\lambda_2) \;\text{d}\lambda_2\\
   &= R_\text{im} \cdot \int   \Bigl[\text{im}(x, y) \otimes \text{psf}(x, y, \lambda_2)\Bigr] \cdot r(\lambda_2) \;\text{d}\lambda_2\\
   &= R_\text{im} \cdot \int   \text{im}_2(x, y, \lambda_2) \cdot r(\lambda_2) \;\text{d}\lambda_2\\
   :label: eq_conv_img_independent
    
This works because the spectral components for the image become independent of the other channel signals. Furthermore, the image convolution becomes independent of the wavelength :math:`\lambda_1` and this part can be integrated separately, leading to a constant factor of  :math:`R_\text{im}`.

For all this to work the convolution needs to take place in a linear color space system with the orthogonality criterion from before.
In our case the linear sRGB colorspace is applied, while also negative values are used to contain all human-visible colors, which wouldn't be the case for the typical positive-value gamut. Linearity would also be lost because of gamut clipping.
Color matching functions :math:`r, g, b` were chosen according to sRGB specifications and spectral distributions according to the procedure in :numref:`random_srgb`.




------------

**References**

.. footbibliography::

