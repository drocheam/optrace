***********************
PSF Convolution
***********************

.. TODO Teil überarbeiten und übersetzen

.. TODO auf :footcite:`Ravikumar_2008,Barnden_1974` eingehen

.. _psf_color_handling:

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
   :label: eq_conv_rgb_r_to_r

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
   :label: eq_conv_rgb_r_to_rgb

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
   :label: eq_conv_rgb_g_to_rgb

Analog gilt für den G-Kanal mit der sRGB color matching function :math:`b(\lambda)`:

.. math::
   \text{im}_{2,b\rightarrow rgb}(x, y) =
   \left[\begin{array}\,
   \text{im}_b(x, y) \otimes \text{psf}_{b\rightarrow r}(x, y)\\
   \text{im}_b(x, y) \otimes \text{psf}_{b\rightarrow g}(x, y)\\
   \text{im}_b(x, y) \otimes \text{psf}_{b\rightarrow b}(x, y)\\
   \end{array}\right]
   :label: eq_conv_rgb_b_to_rgb

Das Gesamtbild :math:`\text{im}_{2,rgb\rightarrow rgb}` aus der Summe aller gefalteten R,G,B Farbkomponenten im Bild.
Jedoch muss man das Mischverhältnis aller Kanäle beachten:
Sind die Farb PSFs alle mit einer Leistungs mit einem Watt simuliert worden, so entspricht das nicht dem korrekten
Mischverhältnis für den sRGB Farbraum. Dieses muss so angepasst sein, dass gleiche Anteile 
in :math:`\text{im}_r, \text{im}_g, \text{im}_b` weiß im Farbraum erzeugen.

Seien :math:`a_r, a_g, a_b` die relativen Mischfaktoren, wobei :math:`a_r + a_g + a_b = 1` gilt, so ist das Endresultat:

.. math::
   \text{im}_{2,rgb\rightarrow rgb}(x, y) = a_r \text{im}_{2,r\rightarrow rgb}(x, y)
   + a_g \text{im}_{2,g\rightarrow rgb}(x, y) + a_b \text{im}_{2,b\rightarrow rgb}(x, y)
   :label: eq_conv_rgb_rgb_to_rgb

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
   :label: eq_conv_rgb_s_to_rgb

Für den Spezialfall, dass das Ursprungsbild spektral homogen ist

:math:`\text{im}_w(x, y)` rein schwarz-weiß ist, 
gilt :math:`\text{im}_r(x, y) = \text{im}_g(x, y) = \text{im}_b(x, y)` und somit:

Convolution of greyscale image and PSF
------------------------------------------

Für den Spezialfall, dass auch die PSF rein schwarz-weiß ist, gilt:

.. math::
   \text{im}_{2,w\rightarrow w}(x, y) = \text{im}_w(x, y) \otimes \text{psf}_{w\rightarrow w}(x, y)
   :label: eq_conv_rgb_w_to_w

Wobei man für eine Darstellung im RGB Farbraum dieses Bild für jeden Kanal vervielfachen müsste.


Vorraussetzungen
=================================================

Die Einschränkungen sind in Abschnitt <> beschrieben.

Vorgehen
==================

1. Umwandeln von Bild und PSF zu linearen sRGB Werten, dabei negative Werte mitnehmen
2. Falls grayscale PSF: Normieren der PSF, sodass die Summe (=Gesamtleistung) Eins entspricht
3. PSF herunterskalieren/interpolieren, sodass die physikalischen Pixelgrößen von PSF und Bild (nach Vergrößerung/Verkleinerung mit Abbildungsmaßstab) identisch sind
4. PSF mit Nullwerten padden um definierten Abfall auf Null zu haben
5. Bild drehen, wenn Abbildungsmaßstab negativ ist.
6. Bild mit gewählter Paddingmethode padden.
7. Bilder nach Methoden von Abschnitt <> falten
8. Bild zurück nach sRGB umwandeln, dabei gamut mapping betreiben
9. Bild zurechtschneiden

Die Faltung findet in sRGB Koordinaten statt, da hier die Kanäle orthogonal zueinander sind.
Außerdem entspricht dieser Farbraum dem Zielfarbraum von Monitoren.
Jedoch muss die Faltung als lineare Operation in linearen sRGB Werten stattfinden (Beschreibung siehe <>).
Auch Farben außerhalb des Farbraums (negative Koordinaten) müssen mitgenommen werden, damit die Operation linear bleibt.
Wenn nach der Faltung immer noch negative Werte im Bild sind, kann man später gamut mapping betreiben.

Für den Fall einer grayscale PSF wird diese automatisch normiert, sodass in Kombination mit dem ``normalize=False``
Parameter der convolve Funktion die Helligkeits-/Farbwerte nicht automatisch normalisiert bzw. umskaliert werden.
Für farbige PSF ist die Normalisierung viel schwieriger, da man wissen müsste, wie viel Licht von der Quelle auf dem 
Detektor für die PSF wirklich ankam. Ließe sich zwar über Metadaten von RenderImage implementieren.
Jedoch fraglich, ob diese Möglichkeit überhaupt relevant ist, schließlich will man meist normalisierte Bilder haben.

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




------------

**References**

.. footbibliography::

