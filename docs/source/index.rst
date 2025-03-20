
Optrace |version| |release| Documentation
============================================

optrace - A scripting based optics simulation package with sequential raytracing, image rendering and a GUI frontend.

.. list-table:: 
   :class: table-borderless

   * - .. figure:: images/example_spherical_aberration2.png
          :align: center
          :width: 165
          :class: dark-light
    
     - .. figure:: images/example_rgb_render4.webp
          :align: center
          :width: 155
          :class: dark-light
  
     - .. figure:: images/example_legrand2.png
          :align: center
          :width: 165
          :class: dark-light
     
     - .. figure:: ./images/example_keratoconus_4.webp
          :align: center
          :width: 155
          :class: dark-light
     
     - .. figure:: images/example_brewster.png
          :align: center
          :width: 165
          :class: dark-light
   
   * - .. figure:: ./images/example_gui_automation_1.png
          :align: center
          :width: 165
          :class: dark-light

     - .. figure:: ./images/LED_illuminants.svg
          :align: center
          :width: 155
          :class: dark-light

     - .. figure:: images/example_double_gauss_2.png
          :align: center
          :width: 165
          :class: dark-light
     
     - .. figure:: ./images/rgb_render_srgb1.webp
          :align: center
          :width: 155
          :class: dark-light

     - .. figure:: ./images/example_cosine_surfaces1.png 
          :align: center
          :width: 165
          :class: dark-light

Overview
----------------------

.. list-table::
   :class: table-borderless

   * - **Features**
        * sequential raytracing for geometrical optics
        * rendering colored detector images
        * paraxial analysis (matrix optics, cardinal points/planes and PSF convolution)
        * presets and user defined surface shapes, ray sources and media
        * an additional GUI for visualization
        * programming/scripting approach to simulation
        * high performance of 0.11 s / surface / million rays (:ref:`details <benchmarking>`)
        * free and open source software

     - **What it CAN'T do**
        * coding-free simulations
        * wave optics, including diffraction and interference
        * non-sequential scenes, simulating ghost images, reflections
        * mirror or fresnel lens optics
        * scattering and polarization dependent media
        * lens optimization, aberration analysis and tolerancing

   * - |

       **Purpose/Use Cases**
        * educational purposes, demonstrating aberrations or simple optical setups
        * introductionary tool to paraxial, geometrical optics or image formation
        * simulation of simpler systems, like a prism, the eye model or a telescope
        * estimation of effects where professional software (ZEMAX, OSLO, Quadoa, ...) is overkill for

     - 


Similar Software
-------------------------------------

.. list-table:: 
   :class: table-borderless

   * - Geometrical Optics
        * `RayOptics <https://ray-optics.readthedocs.io/en/latest/>`__ by Michael Hayford. 
          Tracing and optical design analysis tool. 
        * `rayopt <https://github.com/quartiq/rayopt>`__ by QUARTIQ. 
          Tracing and optical design analysis tool. 
        * `RayTracing <https://github.com/DCC-Lab/RayTracing>`__ by DCC-Lab. 
          Paraxial raytracer with beampath visualization.
   
     - Wave Optics
        * `diffractsim <https://github.com/rafael-fuente/diffractsim>`__ by Rafael de la Fuente. 
          Waveoptics simulation of arbitrary apertures and phase holograms.
        * `poppy <https://github.com/spacetelescope/poppy>`__ by Space Telescope Science Institute. 
          Fraunhofer and Fresnel propagation for optics.
        * `prysm <https://prysm.readthedocs.io/en/stable/index.html>`__ by Brandon Dube. 
          Interferometer and diffraction calculations.
   
   * - |

       Geometrical + Wave Optics
        * `opticspy <http://opticspy.org/>`__ by Xing Fan. 
          Tracing, Wave optics, aberration and Zernike polynomial analysis.
        * `raypier <https://raypier-optics.readthedocs.io/en/latest/introduction.html#the-components-of-a-raypier-model>`__ by Bryan Cole. 
          Raytracing and beamlet propagation with 3D viewer.

     - 

.. toctree::
   :maxdepth: 1
   :numbered:
   :hidden:

   examples
   installation
   quickstart
   ./usage/index
   ./details/index
   ./reference/index
   ./development/index

