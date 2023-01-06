Image and Spectrum Rendering
---------------------------------


Rendering a Detector Image
_____________________________________


Rendering a Source Image
_____________________________________


Iterative Render
_______________________


Rendering a LightSpectrum
_____________________________________


Image Types
_____________________________________

* **Irradiance**: Power per area
* **Illuminance**: Luminous power per area
* **sRGB (Absolute RI)**: a human vision approximation of the image. Colors outside the gamut are saturation-clipped. Preferred sRGB-Mode for "natural" scenes.
* **sRGB (Perceptual RI)**: similar like sRGB (Absolute RI), but saturation scaling for all pixels. Preferred mode for scenes with monochromatic sources or highly dispersive optics.
* **Lightness**: human vision approximation in greyscale colors. Similar to Illuminance, but with non-linear brightness function.
* **Outside sRGB Gamut**: boolean image showing pixels outside the sRGB gamut
* **Hue**: Measure of the type of color tint (red, orange, yellow, ...) 
* **Chroma**: How colorful an area seems compared to a similar illuminated area.
* **Saturation**: How colorful an area seems compared to its brightness. Quotient of Lightness and Chroma. 

The difference between chroma and saturation is elaborately explained in :footcite:`BriggsChroma`. Due to subtle differences saturation is often put to use as light property and chroma as property for an illuminated object.


Sphere Projections
___________________________


* **Orthographic**: *Perspective Projection*, sphere area seen from far away, see :footcite:`OrthographicProjWiki`.`
* **Stereographic**: Projection keeping local angles equal, see :footcite:`SteographicProjWiki`.
* **Equidistant**: Projection keeping the radial direction from the center equal, see :footcite:`EquidistantProjWiki`.
* **Equal-Area**: Projection keeping local areas equal, see :footcite:`EqualAreaProjWiki`.

Plotting Image and Spectra
_____________________________________


Rescaling and Filtering an Image
_____________________________________


Saving, Loading and Exporting an Image
___________________________________________


Image Presets
____________________


Below you can find preset images that can be used for a ray source.

.. list-table::

   * - .. figure:: ../../optrace/ressources/images/bacteria.png
          :align: center
          :width: 300

          Colored rings on white background. Own creation. Usable as ``ot.presets.image.bacteria``
   
     - .. figure:: ../../optrace/ressources/images/ColorChecker.jpg
          :align: center
          :width: 300

          Color checker chart. Public domain image from `here <https://commons.wikimedia.org/wiki/File:X-rite_color_checker,_SahiFa_Braunschweig,_AP3Q0026_edit.jpg>`__.
          Usage with ``ot.presets.image.color_checker``

     - .. figure:: ../../optrace/ressources/images/EIA_Resolution_Chart_1956.png
          :align: center
          :width: 300

          EIA 1956 resolution chart. Public Domain image from `here <https://commons.wikimedia.org/wiki/File:EIA_Resolution_Chart_1956.svg>`__.
          Usage with ``ot.presets.image.resolution_chart``
   
   * - .. figure:: ../../optrace/ressources/images/ETDRS_Chart.png
          :align: center
          :width: 300

          ETDRS Chart standard. Public Domain Image from `here <https://commons.wikimedia.org/wiki/File:ETDRS_Chart_2.svg>`__.
          Usage with ``ot.presets.image.ETDRS_chart``
   
     - .. figure:: ../../optrace/ressources/images/ETDRS_Chart_inverted.png
          :align: center
          :width: 300
          
          ETDRS Chart standard. Edited version of the ETDRS image.
          Usage with ``ot.presets.image.ETDRS_chart_inverted``

     - .. figure:: ../../optrace/ressources/images/TestScreen_square.png
          :align: center
          :width: 300

          TV test screen. Public Domain Image from `here <https://commons.wikimedia.org/wiki/File:TestScreen_square_more_colors.svg>`__.
          Usage with ``ot.presets.image.test_screen``


Additional presets include:

* ``ot.presets.image.checkerboard``: 8x8 black and white chess-like board image
* ``ot.presets.image.ascent``: ascent image from `scipy <https://docs.scipy.org/doc/scipy/reference/generated/scipy.datasets.ascent.html>`__
* ``ot.presets.image.racoon``: racoon image from `scipy <https://docs.scipy.org/doc/scipy/reference/generated/scipy.datasets.face.html>`__


------------

**Sources**

.. footbibliography::

