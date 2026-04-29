
Optrace |version| |release| Documentation
============================================

.. figure:: ./images/header.webp
  :align: center
  :width: 100%
  :class: dark-light
    

Overview
----------------------

optrace (OPtics rayTRACEr) is a scripting based optics simulation package, developed at the Institute 
for Applied Optics and Electronics at the TH Köln -- University of Applied Science in Cologne, Germany. 
It features sequential raytracing, image rendering capabilities and a graphical user interface.
This tool is designed with a focus on image simulation with accurate color handling, 
differentiating it from other raytracers that may prioritize engineering workflows.

optrace is suitable for educational purposes, enabling the creation of interactive applications 
through Python scripting and comprehensive documentation. 
Its interactive 3D scene viewer allows for hands-on exploration of optical principles.

Furthermore, optrace's automation features and extensibility, including support for custom surfaces and materials, 
make it a viable tool for research applications.
As such, optrace has already been used for the simulation of intraocular lenses in two peer-reviewed publications 
in the journal *Translational Vision Science & Technology*
(`Paper 1 <https://doi.org/10.1167/tvst.13.8.33>`__, `Paper 2 <https://doi.org/10.1167/tvst.14.12.33>`__).


**Features**
 * Sequential Monte-Carlo raytracing for geometrical optics
 * Rendering of RGB detector images, including dispersion and filter effects of the optical setup
 * Functions for paraxial analysis (matrix optics, cardinal points/planes and PSF convolution)
 * Includes various presets and user-definable surface shapes, ray sources, and media
 * Import of basic ``.zmx`` ZEMAX geometries (surfaces, media, positions)
 * Optional edge diffraction approximation with :ref:`hurb_details`
 * Optional GUI with an interactive 3D scene viewer
 * Free and open source software
 * Programming/scripting approach to optics simulation
 * Automation capabilities
 * High performance of 85 ms / surface / million rays (:ref:`details <benchmarking>`)
 * Comprehensive documentation

**Limitations**
 * Coding-free simulations ("GUI only") are not supported
 * Wave optics effects such as diffraction and interference are not included
 * No non-sequential raytracing for simulating ghost images and reflections
 * Mirror or Fresnel lens optics are not supported
 * No modelling of scattering effects, gradient index materials, or birefringent media
 * No functionality for lens optimization, aberration analysis, and tolerancing

**Purpose/Use Cases**
 * Educational purposes, demonstrating aberrations or simple optical setups
 * Introductory tool to paraxial, geometrical optics or image formation
 * Simulation of simpler systems: Prism, eye model, telescope, ...
 * Realistic color image rendering in imaging setups
 * Estimation of effects where professional software (ZEMAX, OSLO, Quadoa, ...) is overkill for


Similar Software
-------------------------------------

Geometrical Optics
 * `RayTracing <https://github.com/DCC-Lab/RayTracing>`__ by DCC-Lab. 
   Paraxial raytracer with beampath visualization.
 * `RayOptics <https://ray-optics.readthedocs.io/en/latest/>`__ by Michael Hayford. 
   Tracing and optical design analysis tool. 
 * `rayopt <https://github.com/quartiq/rayopt>`__ by QUARTIQ. 
   Tracing and optical design analysis tool. 
 * `tracepy <https://github.com/TracePy-Org/tracepy/>`__ by Gavin Niendorf.
   Simple raytracing tool with optimization features.
 * `Optiland <https://optiland.readthedocs.io/en/latest/index.html>`__ by Kramer Harrison.
   Comprehensive optical design and analysis framework with a 3D viewer.

Wave Optics
 * `diffractsim <https://github.com/rafael-fuente/diffractsim>`__ by Rafael de la Fuente. 
   Waveoptics simulation of arbitrary apertures and phase holograms.
 * `poppy <https://github.com/spacetelescope/poppy>`__ by Space Telescope Science Institute. 
   Fraunhofer and Fresnel propagation for optics.
 * `prysm <https://prysm.readthedocs.io/en/stable/index.html>`__ by Brandon Dube. 
   Interferometer and diffraction simulations.
   
Geometrical + Wave Optics
 * `opticspy <https://github.com/Sterncat/opticspy>`__ by Xing Fan. 
   Tracing, wave optics, aberration and Zernike polynomial analysis.
 * `raypier <https://raypier-optics.readthedocs.io/>`__ by Bryan Cole. 
   Raytracing and beamlet propagation with a 3D viewer.
 * `PAOS <https://paos.readthedocs.io/en/latest/index.html>`__ by Andrea Bocchieria, Lorenzo V. Mugnaia, and Enzo Pascale.
   Paraxial raytracing and Fresnel approximation wave propagation.

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

