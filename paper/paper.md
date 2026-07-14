---
title: 'optrace: A Python library for 3D optical raytracing and polychromatic image synthesis'
tags:
  - Python
  - optics
  - imaging
  - image simulation
  - convolution
  - raytracing
authors:
  - name: Damian Mendroch
    orcid: 0009-0003-1332-2527
    equal-contrib: false
    affiliation: 1
affiliations:
 - name: Institute for Applied Optics and Electronics, TH Köln – University of Applied Sciences, Cologne, Germany
   index: 1
date: 14 July 2026
bibliography: paper.bib

---

# Summary

Alongside experimental setups, simulation is a key component of optical design and analysis.
Physics-based modeling tools allow for the approximation of real-world optical problems with significantly greater flexibility, cost-efficiency, and versatility compared to classical prototyping and measurement.
The engineering aspect of such tools places a strong emphasis on optimizing setups in terms of aberrations, cost, and feasibility. 
In addition, another key aspect is the demonstration and explanation of optical effects:
This is not only relevant for educational purposes but also for an application-oriented visualization of system properties.
Examples include realistically rendered images through camera lenses or simulations of a patient’s vision with an artificial intraocular lens.
In this context, there is a gap in available open-source tools that this work aims to address.

For this purpose, we provide a package that enables the simulation of color images through raytracing, even for more complex setups, and includes analysis tools such as a 3D viewer, paraxial calculation tools, and automated scripting.

# Statement of need

optrace offers the following features:

 * Sequential raytracing of freely definable geometries

 * Rendering of grayscale or color images on detector surfaces

 * Iterative search for the focal position

 * Alternative image simulation mode by PSF convolution

 * Paraxial analysis involving focal positions, image distances, and pupils
 
 * User-definable spectra, refractive indices, geometries, surfaces, simulation images, and a variety of presets

 * Import of basic .zmx Zemax geometry files and .agf material catalogues

 * Optional edge diffraction approximation using Heisenberg uncertainty ray bending [@Heinisch_1971]

The presented tool has multiple use cases:
One application is the educational demonstration of aberrations and fundamental optical configurations.
Here, it serves as an introductory framework for paraxial optics, geometric optics, and image formation.
Furthermore, the software enables the simulation of schematic optical systems, such as eye models, prisms, and telescopes.
With these features, it offers an accessible alternative to professional-grade optical design suites for preliminary system estimations.

# State of the field                                                                                                                  

There is a wide variety of open-source projects available for the optics domain.
While these tools are more constrained compared to the proprietary suites such as Ansys Zemax OpticStudio, Lambda Research Corporation OSLO, and Quadoa by Quadoa Optical Systems, they can provide targeted and easily accessible simulation capabilities.
Geometric optical libraries, focusing on ray-based analysis, include rayopt [@Quartiq_2020], Optiland [@Harrison_2026], RayOptics [@Hayford_2025], RayTracing [@DCC_Lab_2026], and tracepy [@Niendorf_2025].
Purely wave-optical simulations are facilitated by frameworks such as diffractsim [@Herrezuelo_2022], poppy [@poppy_ascl], and prysm [@Dube_2019].
Furthermore, hybrid architectures that integrate both physical principles are provided by opticspy [@Fan_2015], raypier [@Cole_2021], and PAOS [@Bocchieri_2024].

Despite the availability of these projects, there remains a notable gap regarding the simulation of realistic, polychromatic imagery within a 3D optics environment featuring custom-defined surfaces.
Furthermore, no existing package is known to consolidate paraxial analysis, sequential raytracing, and PSF-based image simulation with an interactive graphical user interface and 3D visualization within a unified framework.


# Software design

The primary design decision involves the definition of the functional scope and its inherent constraints.
Central to this framework is the simulation of geometric optics via sequential raytracing.
Consequently, wave-optical modeling and non-sequential configurations are excluded.
Physical phenomena such as diffraction, interference, and reflections are beyond the current simulation capabilities.
Furthermore, `optrace` does not provide tools for aberration analysis or automated geometry optimization.
This specialized functional focus reduces the systemic complexity and allows for prioritizing the computational efficiency of the implemented methods.

In contrast to the GUI-centric workflows of various commercial applications, geometry definition and simulation follow a scripting-based approach.
The graphical user interface is entirely optional, facilitating a high degree of automation, modularity, and seamless integration with external libraries and existing codebases.
Regarding internal modularity, the core tracing functionality is included within the `optrace` namespace, distinct from the plotting (`optrace.plots`) and GUI (`optrace.gui`) modules.
This decoupled architecture ensures that components remain optional, interchangeable, and easily extensible.

Significant emphasis is placed on usability and a low barrier to entry.
To this end, the library is supported by comprehensive online documentation and an extensive suite of practical examples.
Furthermore, it includes a wide variety of presets for images, spectra, refractive indices, and surface geometries.
Next, the Python programming language was selected due to its widespread adoption in the scientific community and its vast ecosystem of specialized libraries.

To maintain high performance despite the high-level nature of the language, computationally intensive components are offloaded to optimized numerical libraries, including NumPy [@harris2020array], SciPy [@2020SciPy-NMeth], and OpenCV [@opencv_library].
Performance is further enhanced through the implementation of multithreading for resource-heavy operations and memory-efficient data structures for ray parameter management.
Additionally, analytical solutions are prioritized over numerical or iterative methods to ensure both precision and speed.

To maintain a specialized functional scope, the library delegates a multitude of tasks to established external libraries.
2D-visualization is handled by matplotlib [@Hunter_2007], while 3D rendering and spatial data representation leverage VTK [@vtkBook] and pyvista [@sullivan2019pyvista].
User interface related libraries comprise PySide [@pyside6], pyvistaqt [@pyvista_qt_2026], TraitsUI [@traitsui_2023], and pyqtdarktheme-fork [@pyqtdarktheme_fork_2026].
Additional dependencies include chardet [@chardet_2026] for character encoding detection and tqdm [@casper_da_costa_luis_2026_18473238] for progress visualization.

The preceding descriptions provide only a concise overview of the library.
Comprehensive application examples are available in the [Examples section](https://drocheam.github.io/optrace/examples.html) of the online documentation, complemented by a detailed [User Guide](https://drocheam.github.io/optrace/usage/index.html).
Furthermore, extensive resources regarding the [API](https://drocheam.github.io/optrace/reference/index.html) and the underlying mathematical and technical [implementation details](https://drocheam.github.io/optrace/details/index.html) are provided for further reference.


# Research impact statement

Research utilizing `optrace` has already resulted in two peer-reviewed publications for the simulation of intraocular lenses (IOLs) [@Mendroch_2024; @Mendroch_2025].
The first work focused on the optical characterization of the IOL by analyzing the refractive power and raytracing behavior.
The second work expanded this scope by integrating these lenses in a full-eye model and generating realistic polychromatic retinal images.

Beyond research, the software is actively utilized at the Institute of Applied Optics and Electronics, TH Köln – University of Applied Sciences, Germany, to create teaching materials and perform basic simulation tasks.

By providing access to this versatile optics tool, we aim to support the broader academic community and enhance both research and educational applications in the field of optics.

# AI usage disclosure

No generative AI tools were used in the development of this software.
AI tools were solely used for proofreading, linguistic refinement and polishing of the project documentation and this paper.
All produced content has been reviewed by a human reader.

# Acknowledgements

The author would like to thank Prof. Dr. Stefan Altmeyer and Prof. Dr. Uwe Oberheide from the Institute of Applied Optics and Electronics at TH Köln – University of Applied Sciences for providing the original inspiration and valuable conceptual guidance for this project.

# References
