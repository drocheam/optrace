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

.. _image_plots:

Plotting Image and Spectra
_____________________________________


.. _chromaticity_plots:

Chromaticity Plots
________________________


Rescaling and Filtering an Image
_____________________________________


Saving, Loading and Exporting an Image
___________________________________________


Image Presets
____________________


Below you can find preset images that can be used for a ray source.

.. list-table:: Photos of natural scenes or objects

   * - .. figure:: ../../optrace/ressources/images/cell.webp
          :align: center
          :width: 350

          Cell image for microscope examples. Usable as ``ot.presets.image.cell``.
          Image created with `Stable Diffusion <https://lexica.art/prompt/960d8351-f474-4cc0-b84b-4e9521754064>`__.
   
     - .. figure:: ../../optrace/ressources/images/group_photo.jpg
          :align: center
          :width: 250

          Group photo of managers. Usable as ``ot.presets.image.group_photo``
          Image created with `Stable Diffusion <https://lexica.art/prompt/06ba5ac6-7bfd-4ce6-8002-9d0e487b36b2>`__.
   
   * - .. figure:: ../../optrace/ressources/images/interior.jpg
          :align: center
          :width: 400

          Photo of an interior living room. Usable as ``ot.presets.image.interior``
          Image created with `Stable Diffusion <https://lexica.art/prompt/44d7e1fe-ba3b-4e73-972c-a30b95897434>`__.
   
     - .. figure:: ../../optrace/ressources/images/landscape.jpg
          :align: center
          :width: 400

          Photo of an european landscape. Usable as ``ot.presets.image.landscape``
          Image created with `Stable Diffusion <https://lexica.art/prompt/0da3a592-465e-46d6-8ee6-dfe17ddea386>`__.
   


.. list-table:: Test images for color, resolution or distortion

   * - .. figure:: ../../optrace/ressources/images/ColorChecker.jpg
          :align: center
          :width: 300

          Color checker chart. Public domain image from `here <https://commons.wikimedia.org/wiki/File:X-rite_color_checker,_SahiFa_Braunschweig,_AP3Q0026_edit.jpg>`__.
          Usage with ``ot.presets.image.color_checker``

     - .. figure:: ../../optrace/ressources/images/ETDRS_Chart.png
          :align: center
          :width: 300

          ETDRS Chart standard. Public Domain Image from `here <https://commons.wikimedia.org/wiki/File:ETDRS_Chart_2.svg>`__.
          Usage with ``ot.presets.image.ETDRS_chart``
   
   * - .. figure:: ../../optrace/ressources/images/ETDRS_Chart_inverted.png
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


------------

**Sources**

.. footbibliography::

