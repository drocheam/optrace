# Optrace
## Geometrical OPtics rayTRACEr and image simulation

[![Python 3.10, 3.11, 3.12, 3.13](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org/downloads/release/python-3127/)
[![Windows Linux Mac](https://shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-blue)]()
[![License MIT](https://img.shields.io/badge/license-MIT-blue)](https://github.com/drocheam/optrace/blob/main/LICENSE)
<!-- TODO add version tag, such as https://img.shields.io/github/v/tag/drocheam/optrace?label=version see https://shields.io/badges/git-hub-tag -->

[![Testing](https://github.com/drocheam/optrace/actions/workflows/tox_test.yml/badge.svg)](https://github.com/drocheam/optrace/actions/workflows/tox_test.yml)
[![Documentation](https://github.com/drocheam/optrace/actions/workflows/gen_docs.yml/badge.svg)](https://github.com/drocheam/optrace/actions/workflows/gen_docs.yml)
<!-- TODO link tag to documentation -->

<!-- TODO add coverage info (via third party site?), lines of code, comment/docstring quality etc. -->

<img src="./docs/source/images/example_spherical_aberration2.png" width="18.2%"></img>
<img src="./docs/source/images/example_rgb_render4.webp" width="17%"></img>
<img src="./docs/source/images/example_legrand2.png" width="18.2%"></img>
<img src="./docs/source/images/example_keratoconus_4.webp" width="17%"></img>
<img src="./docs/source/images/example_brewster.png" width="18.2%"></img>
   
<img src="./docs/source/images/example_gui_automation_1.png" width="18.2%"></img>
<img src="./docs/source/images/LED_illuminants.svg" width="17%"></img>
<img src="./docs/source/images/example_double_gauss_2.png" width="18.2%"></img>
<img src="./docs/source/images/rgb_render_srgb1.webp" width="17%"></img>
<img src="./docs/source/images/example_cosine_surfaces1.png" width="18.2%"></img>

<!-- TODO images are clickable, redirect to example page -->

## Overview

 **Features**
  * sequential raytracing for geometrical optics
  * rendering colored detector images
  * paraxial analysis (matrix optics, cardinal points/planes and PSF convolution)
  * presets and user defined surface shapes, ray sources and media
  * an additional GUI for visualization
  * programming/scripting approach to simulation
  * high performance of 0.11 s / surface / million rays
  * free and open source software

<!--  ^--- TODO add benchmarking details when documentation is online-->

 **What it CAN'T do**
  * coding-free simulations
  * wave optics, including diffraction and interference
  * non-sequential scenes, simulating ghost images, reflections
  * mirror or Fresnel lens optics
  * scattering and polarization dependent media
  * lens optimization, aberration analysis and tolerancing

 **Purpose/Use Cases**
  * educational purposes, demonstrating aberrations or simple optical setups
  * introductory tool to paraxial, geometrical optics and image formation
  * simulation of simpler systems, like a prism, the eye model or a telescope
  * estimation of effects where professional software (ZEMAX, OSLO, Quadoa, ...) is overkill for

## Examples

Coming soon.

## Documentation

Coming soon.

## Installation

Make sure Python 3.10, 3.11, 3.12 or 3.13 are installed on your system

**Installing the latest official release**

1. Download the optrace .tar.gz archive from https://github.com/drocheam/optrace/releases/latest
2. Open a terminal
3. Run `pip install <path to archive>`, where `<path to archive>` is the path to the archive downloaded in the first step

**Installing the current git version**

1. Open a terminal
2. Clone the whole project repository using git: `git clone https://github.com/drocheam/optrace/`
3. Change the directory into the cloned folder: `cd optrace`
4. Install using `pip install .`


<!-- ## Contributing-->

## License

This project is published under an MIT License.

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
   * [opticspy](http://opticspy.org/) by Xing Fan. Tracing, wave optics, aberration and Zernike polynomial analysis.
   * [raypier](https://raypier-optics.readthedocs.io/en/latest/introduction.html#the-components-of-a-raypier-model) by Bryan Cole. Raytracing and beamlet propagation with 3D viewer.

