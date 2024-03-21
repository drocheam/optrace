
Optrace |version| |release| Documentation
============================================

.. list-table::

   * - .. figure:: images/arizona_eye_scene.png
          :align: center
          :width: 200
    
     - .. figure:: images/example_rgb_render2.svg
          :align: center
          :width: 200

     - .. figure:: ./images/example_cosine_surfaces1.png 
          :align: center
          :width: 200
     
     - .. figure:: ./images/example_keratoconus_4.svg
          :align: center
          :width: 200
     
     - .. figure:: images/example_brewster.png
          :align: center
          :width: 200
   
   * - .. figure:: ./images/example_gui_automation_1.png
          :align: center
          :width: 200

     - .. figure:: ./images/LED_illuminants.svg
          :align: center
          :width: 200
     
     - .. figure:: images/example_spherical_aberration2.png
          :align: center
          :width: 200
     
     - .. figure:: images/example_double_gauss.png
          :align: center
          :width: 200
     
     - .. figure:: ./images/rgb_render_srgb1.svg
          :align: center
          :width: 200


.. See the :ref:`example gallery <examples>` for more samples

**Overview**

.. list-table::

   * - **What this is**
        * a sequential raytracer for geometrical optics
        * a tool with presets and user defined surface shapes, ray sources and media
        * a library with an additional GUI with ray and geometry visualization
        * a renderer for colored detector images inside an optical setup
        * a program including paraxial analysis (matrix optics, position of cardinal points/planes and psf convolution)
        * a programming/scripting approach to simulation
        * free, open source software

     - **What this isn't**
        * a GUI focussed tool
        * an optics design program with lens optimization, aberration analysis and tolerancing
        * simulation incorporating wave optics, e.g. diffraction, interference
        * a non-sequential raytracer, simulating ghost images, reflections
        * a tool supporting mirror or fresnel lens optics
        * mature, bug-free software

   * - **What this is for**
        * educational purposes, demonstrating aberrations or simple optical setups
        * introductionary tool to paraxial, geometrical optics or image formation
        * simulation of simpler systems, like a prism, the eye model or a telescope
        * estimation of effects where professional software (ZEMAX, OSLO, Quadoa, ...) would be overkill for

     - 



**Similar/related Python software**

.. list-table:: 

   * - Geometrical Optics
        * `RayOptics <https://ray-optics.readthedocs.io/en/latest/>`__ by Michael Hayford. Tracing and optical design analysis tool. 
        * `rayopt <https://github.com/quartiq/rayopt>`__ by QUARTIQ. Tracing and optical design analysis tool. 
        * `RayTracing <https://github.com/DCC-Lab/RayTracing>`__ by DCC-Lab. Paraxial raytracer with beampath visualization.
   
     - Wave Optics
        * `diffractsim <https://github.com/rafael-fuente/diffractsim>`__ by Rafael de la Fuente. Waveoptics simulation of arbitrary apertures and phase holograms.
        * `poppy <https://github.com/spacetelescope/poppy>`__ by Space Telescope Science Institute. Fraunhofer and Fresnel propagation for optics.
        * `prysm <https://prysm.readthedocs.io/en/stable/index.html>`__ by Brandon Dube. Interferometer and diffraction calculations.
   
   * - Geometrical + Wave Optics
        * `opticspy <http://opticspy.org/>`__ by Xing Fan, tracing. Wave optics, aberration and Zernike polynomial analysis.
        * `raypier <https://raypier-optics.readthedocs.io/en/latest/introduction.html#the-components-of-a-raypier-model>`__ by Bryan Cole. Raytracing and beamlet propagation with 3D viewer.

     - 

.. toctree::
   :maxdepth: 1
   :numbered:
   :hidden:

   examples
   installation
   quickstart
   ./usage/usage
   ./math/math
   ./library/library
   ./development/development

