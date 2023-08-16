################
Overview
################


.. figure:: images/example_double_gauss.png
   :align: center
   :width: 700


**What this is**

* a sequential raytracer for geometrical optics
* a tool with presets and user defined surface shapes, ray sources and media
* a library with an additional GUI with ray and geometry visualization
* a renderer for colored detector images inside an optical setup
* a program including paraxial analysis (matrix optics, position of cardinal points/planes and psf convolution)
* a programming/scripting approach to simulation
* free, open source software


**What this isn't**

* a GUI focussed tool
* an optics design program with lens optimization, aberration analysis and tolerancing
* simulation incorporating wave optics, e.g. diffraction, interference
* a non-sequential raytracer, simulating ghost images, reflections
* a tool supporting mirror or fresnel lens optics
* mature, bug-free software


**Who/what is this for**

* educational purposes, demonstrating aberrations or simple optical setups
* introductionary tool to paraxial, geometrical optics or image formation
* simulation of simpler systems, like a prism, the eye model or a telescope
* estimation of effects where professional software (ZEMAX, OSLO, Quadoa, ...) would be overkill for


See the :ref:`example gallery <examples>` for some functionality samples


---------------------------------

**Similar/related Python software**

* Geometrical Optics
    * `RayOptics <https://ray-optics.readthedocs.io/en/latest/>`__ by Michael Hayford. Tracing and optical design analysis tool. 
    * `rayopt <https://github.com/quartiq/rayopt>`__ by QUARTIQ. Tracing and optical design analysis tool. 
    * `RayTracing <https://github.com/DCC-Lab/RayTracing>`__ by DCC-Lab. Paraxial raytracer with beampath visualization.

* Wave Optics
    * `diffractsim <https://github.com/rafael-fuente/diffractsim>`__ by Rafael de la Fuente. Waveoptics simulation of arbitrary apertures and phase holograms.
    * `poppy <https://github.com/spacetelescope/poppy>`__ by Space Telescope Science Institute. Fraunhofer and Fresnel propagation for optics.
    * `prysm <https://prysm.readthedocs.io/en/stable/index.html>`__ by Brandon Dube. Interferometer and diffraction calculations.

* Geometrical + Wave Optics
    * `opticspy <http://opticspy.org/>`__ by Xing Fan, tracing. Wave optics, aberration and Zernike polynomial analysis.
    * `raypier <https://raypier-optics.readthedocs.io/en/latest/introduction.html#the-components-of-a-raypier-model>`__ by Bryan Cole. Raytracing and beamlet propagation with 3D viewer.
