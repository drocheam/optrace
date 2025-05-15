
Optrace |version| |release| Documentation
============================================

.. figure:: ./images/header.webp
  :align: center
  :width: 100%
  :class: dark-light
    

Overview
----------------------

optrace (OPtics rayTRACEr) is a scripting based optics simulation package, developed at the Institute 
for Applied Optics and Electronics at the TH Köln - University of Applied Science in Cologne, Germany. 
It features sequential raytracing, image rendering capabilities and a graphical user interface.
This tool is designed with a focus on image simulation with accurate color handling, 
differentiating it from other raytracers that may prioritize engineering workflows.

optrace is suitable for educational purposes, enabling the creation of interactive applications 
with custom user interfaces through Python scripting and comprehensive documentation. 
Its interactive 3D scene viewer allows for hands-on exploration of optical principles.

Furthermore, optrace's automation features and extensibility, including support for custom surfaces and materials, 
make it a viable tool for research applications. 

.. Examples of research works utilizing this library include <> and <>.

**Features**
 * Free and open source software
 * Programming/scripting approach to simulation
 * Sequential raytracing for geometrical optics
 * Rendering of colored detector images
 * Paraxial analysis (matrix optics, cardinal points/planes and PSF convolution)
 * Includes preset and user-definable surface shapes, ray sources, and media
 * An additional GUI with an interactive 3D scene viewer
 * Automation capabilities
 * High performance of 0.11 s / surface / million rays (:ref:`details <benchmarking>`)
 * Comprehensive documentation

**Limitations**
 * Coding-free simulations are not supported
 * Wave optics effects such as diffraction and interference are not included
 * No non-sequential raytracing for simulating ghost images and reflections
 * Mirror or fresnel lens optics are not supported
 * No modelling of scattering effects or polarization-dependent media
 * No functionality for lens optimization, aberration analysis, and tolerancing

**Purpose/Use Cases**
 * Educational purposes, demonstrating aberrations or simple optical setups
 * Introductory tool to paraxial, geometrical optics or image formation
 * Simulation of simpler systems: Prism, eye model, telescope, ...
 * Estimation of effects where professional software (ZEMAX, OSLO, Quadoa, ...) is overkill for


Similar Software
-------------------------------------

Geometrical Optics
 * `RayOptics <https://ray-optics.readthedocs.io/en/latest/>`__ by Michael Hayford. 
   Tracing and optical design analysis tool. 
 * `rayopt <https://github.com/quartiq/rayopt>`__ by QUARTIQ. 
   Tracing and optical design analysis tool. 
 * `RayTracing <https://github.com/DCC-Lab/RayTracing>`__ by DCC-Lab. 
   Paraxial raytracer with beampath visualization.
 * `Optiland <https://optiland.readthedocs.io/en/latest/index.html>`__ by Harrison Kramer.
   Optical design and analysis framework.

Wave Optics
 * `diffractsim <https://github.com/rafael-fuente/diffractsim>`__ by Rafael de la Fuente. 
   Waveoptics simulation of arbitrary apertures and phase holograms.
 * `poppy <https://github.com/spacetelescope/poppy>`__ by Space Telescope Science Institute. 
   Fraunhofer and Fresnel propagation for optics.
 * `prysm <https://prysm.readthedocs.io/en/stable/index.html>`__ by Brandon Dube. 
   Interferometer and diffraction calculations.
   
Geometrical + Wave Optics
 * `opticspy <http://opticspy.org/>`__ by Xing Fan. 
   Tracing, wave optics, aberration and Zernike polynomial analysis.
 * `raypier <https://raypier-optics.readthedocs.io/en/latest/introduction.html#the-components-of-a-raypier-model>`__ by Bryan Cole. 
   Raytracing and beamlet propagation with 3D viewer.

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

