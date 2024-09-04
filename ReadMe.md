# Optrace 1.5.9
## Geometrical OPtics rayTRACEr and image simulation

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/downloads/release/python-3100/)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/downloads/release/python-3110/)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org/downloads/release/python-3120/)
[![Windows Linux Mac](https://shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-blue)]()
<!--TODO add license-->

[![Testing](https://github.com/drocheam/optrace/actions/workflows/tox_test.yml/badge.svg)](https://github.com/drocheam/optrace/actions/workflows/tox_test.yml)
[![Documentation](https://github.com/drocheam/optrace/actions/workflows/gen_docs.yml/badge.svg)](https://github.com/drocheam/optrace/actions/workflows/gen_docs.yml)

<img src="./docs/source/images/example_spherical_aberration2.png" width="18.2%"></img>
<img src="./docs/source/images/example_rgb_render2.svg" width="17%"></img>
<img src="./docs/source/images/example_legrand2.png" width="18.2%"></img>
<img src="./docs/source/images/example_keratoconus_4.svg" width="17%"></img>
<img src="./docs/source/images/example_brewster.png" width="18.2%"></img>
   
<img src="./docs/source/images/example_gui_automation_1.png" width="18.2%"></img>
<img src="./docs/source/images/LED_illuminants.svg" width="17%"></img>
<img src="./docs/source/images/example_double_gauss_2.png" width="18.2%"></img>
<img src="./docs/source/images/rgb_render_srgb1.svg" width="17%"></img>
<img src="./docs/source/images/example_cosine_surfaces1.png" width="18.2%"></img>

## Overview

 **Features**
  * a sequential raytracer for geometrical optics
  * a tool with presets and user defined surface shapes, ray sources and media
  * a library with an additional GUI with ray and geometry visualization
  * a renderer for colored detector images inside an optical setup
  * a program including paraxial analysis (matrix optics, position of cardinal points/planes and psf convolution)
  * a programming/scripting approach to simulation
  * free, open source software

 **Non-Features**
  * a GUI focussed tool
  * an optics design program with lens optimization, aberration analysis and tolerancing
  * simulation incorporating wave optics, e.g. diffraction, interference
  * a non-sequential raytracer, simulating ghost images, reflections
  * a tool supporting mirror or fresnel lens optics

 **Purpose/Use Cases**
  * educational purposes, demonstrating aberrations or simple optical setups
  * introductionary tool to paraxial, geometrical optics or image formation
  * simulation of simpler systems, like a prism, the eye model or a telescope
  * estimation of effects where professional software (ZEMAX, OSLO, Quadoa, ...) would be overkill for


## Examples

An example gallery is found at <>.

## Documentation

See <>

## Installation

This library requires Python 3.10, 3.11 or 3.12.

Clone/download the repository and run the following command from a terminal within the folder:

``pip install .``

More details on the installation process and the requirements are found at <>.


## Similar software

- Geometrical Optics
   * [RayOptics](https://ray-optics.readthedocs.io/en/latest/) by Michael Hayford. Tracing and optical design analysis tool. 
   * [rayopt](https://github.com/quartiq/rayopt) by QUARTIQ. Tracing and optical design analysis tool. 
   * [RayTracing](https://github.com/DCC-Lab/RayTracing) by DCC-Lab. Paraxial raytracer with beampath visualization.

- Wave Optics
   * [diffractsim](https://github.com/rafael-fuente/diffractsim) by Rafael de la Fuente. Waveoptics simulation of arbitrary apertures and phase holograms.
   * [poppy](https://github.com/spacetelescope/poppy) by Space Telescope Science Institute. Fraunhofer and Fresnel propagation for optics.
   * [prysm](https://prysm.readthedocs.io/en/stable/index.html) by Brandon Dube. Interferometer and diffraction calculations.

- Geometrical + Wave Optics
   * [opticspy](http://opticspy.org/) by Xing Fan, tracing. Wave optics, aberration and Zernike polynomial analysis.
   * [raypier](https://raypier-optics.readthedocs.io/en/latest/introduction.html#the-components-of-a-raypier-model) by Bryan Cole. Raytracing and beamlet propagation with 3D viewer.
