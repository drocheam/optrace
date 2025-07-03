Testing
---------------

.. role:: python(code)
  :language: python
  :class: highlight

.. role:: bash(code)
  :language: bash
  :class: highlight


Functionality Testing
_______________________


The goal of the functionality testing is to check both typical features as well as edge cases.
Not only should the code "run", but also produce correct results.
This is guaranteed by many automated test cases.

Some examples include:

* a nearly ideal lens produces a distinct focal point at the correct focal distance
* a growing number of rays decreases sampling noise
* user-built surfaces modeling a spherical surface behaving the same as the in-built type
* sampling of directions, color, positions produces the correct distribution on both the source and destination 
* GUI interactions execute the expected actions
* images are correctly loaded and saved
* typical phenomena and aberrations of geometrical optics can be reproduced (spherical and chromatic aberration, 
  vignetting, dispersion, astigmatism, coma, ...)
* image formation by convolving produces correct results even the images must be scaled or interpolated
* PSF convolution and raytracing produce comparable results
* simple :bash:`.zmx` and :bash:`.agf` files are handled and imported correctly 
* ... many more

Some edge cases include:

* normalizing a black image does not lead to divisions by zero
* requesting billions of rays prints an error message instead of crashing the PC due to too much requested RAM
* collisions of lenses are detected
* moving geometry elements outside of the setup bounding box is not possible
* check of value validity (no negative powers, wavelengths outside the visible range, ...)
* plotting/loading/tracing of a single pixel image
* an optical setup without lenses, filters or apertures is traced correctly
* ... many more

Test files can be found under `tests/ <https://github.com/drocheam/optrace/blob/main/tests/>`_, 
starting with filename :bash:`test_`
Testing is performed with :mod:`pytest`.
To ensure that it is performed on a "clean" and defined environment, it is run by `tox <https://tox.wiki/en/latest/>`_, 
this creates a virtual python environment where the library is installed and tested.
With its default call, :bash:`tox` calls the testing environment that includes all test cases.
To lower RAM usage, tox calls test groups instead of all tests at once.

The workflow file `tox_test.yml <https://github.com/drocheam/optrace/blob/main/.github/workflows/tox_test.yml>`_ 
runs the functionality tests on a defined system and python version as github actions.
By default this is done on pushed code changes, but the action can also be run manually.


Coverage Testing
_______________________

:mod:`coverage` is used to check if all code lines and branches were tested.
It wraps around pytest inside the tox test environment and reports on executed and missing lines.
The output gets represented as both html or as text output (inside the terminal or github action output).

.. _benchmarking:

Benchmark Testing
__________________

Tracing performance depends on the number of surfaces and rays as well as the complexity of the surfaces.
Spherical surfaces are easier to trace then complex user defined ones.
For systems with a low number of surfaces the ray generation at the source becomes dominant.

The benchmarking file `tests/benchmark.py <https://github.com/drocheam/optrace/blob/main/tests/benchmark.py>`_
is an adapted version of the :ref:`example_microscope` example.
It includes 57 light-interacting surfaces (one ray source, one aperture, 27 lenses with front and back surface each, 
one outline surface of the bounding box). Detectors are excluded as they don't interact with the rays unless
image rendering is executed.
Turning off the polarization calculation (with :attr:`Raytracer.no_pol <optrace.tracer.raytracer.Raytracer.no_pol>`)
leads to a significant speedup.

The raytracing on my system (Arch Linux 6.13, i7-1360P notebook CPU, 16GB RAM, Python 3.13) results in:

.. table:: Performance Comparison. Result values are in seconds / surface / million rays.
   :widths: 60 110 110
   :width: 600px

   +-------+----------------------+---------------------+
   | Cores | Without Polarization | With Polarization   |
   +=======+======================+=====================+
   |   1   |        0.262         |        0.417        |
   +-------+----------------------+---------------------+
   |   2   |        0.122         |        0.171        |
   +-------+----------------------+---------------------+
   |   4   |        0.069         |        0.125        |
   +-------+----------------------+---------------------+
   |   8   |        0.059         |        0.108        |
   +-------+----------------------+---------------------+
   |  16   |        0.057         |        0.101        |
   +-------+----------------------+---------------------+

So there is not much gain in using 16 over 8 cores.

.. TODO how to handle these points:

Test Cases
__________________

* :mod:`doctest` for documentation strings in some non-class functions
* test cases are handled with :mod:`unittest`, testing is however typically done by :mod:`pytest` 
  that loads and executed the created TestSuites
* :python:`TraceGUI` has a :python:`debug` method, which runs a separate thread, from which actions are executed:
   * Note that each UI actions needs to be run in the main thread, for this :python:`TraceGUI` 
     provides :python:`_do_in_main` and :python:`_set_in_main` methods
   * generally we want to do an action, after the old one has finished.
     For this :python:`TraceGUI._wait_for_idle` is implemented.


Documentation Testing
__________________________________

The workflow file `doc_test.yml <https://github.com/drocheam/optrace/blob/main/.github/workflows/doc_test.yml>`_ 
runs multiple test environments. This includes:

* :bash:`tox -e docsbuildcheck`: Testing of correct documentation building while not permitting warning
* :bash:`tox -e linkcheck`: Testing of all documentation links
* :bash:`tox -e doctest`: Testing of most documentation code examples (using :mod:`doctest`)

Python Version Testing
__________________________________

Testing of the compatibility with multiple python versions is done with the 
`pyver_comp.yml <https://github.com/drocheam/optrace/blob/main/.github/workflows/pyver_comp.yml>`_ workflow.
It executes a subset of tests (:bash:`tox -e fast`) for multiple python main versions in an Ubuntu runner.

Platform Testing
__________________________________

Platform testing is done with the 
`os_comp.yml <https://github.com/drocheam/optrace/blob/main/.github/workflows/os_comp.yml>`_ workflow.
It executes the os tox environment, which runs a subset of all available tests.
These tests include relevant possibly platform-dependent cases, including:

* installation
* loading and saving of files
* handling of filenames and paths
* loading sensitivity and image presets (which also are files inside the library)
* opening the gui and plots
* enforcing a specific float bit width
* detecting the number of cores and multithreading

The workflow is executed with the current python version and both the latest macOC and Windows runners.
Linux is not included, as all other workflows already run on Ubuntu.

Github Workflows
________________________

* see `.github/workflows/ <https://github.com/drocheam/optrace/blob/main/.github/workflows/>`_
* Actions: Main testing, OS Compatibility, Older Python Version Compatibility
* all get run on a push the repository or can get run manually


Manual Tests
_____________________________________________________

Unfortunately, not all tests can be automated.
The following test cases need to be handled manually:

**Checking that** :func:`optrace.plots.block <optrace.plots.misc_plots.block>` **actually pauses the program execution**

This can't be tested automatically, as it would halt testing. 
Maybe there is a way by using multithreading or multiprocessing and a timer?
What about coverage information in such cases?

**Correct and nice formatting inside plots and the GUI**

Can only be tested by a human viewer.


**Usage in Python Notebooks or inline IDEs such as Spyder**

* Installation:
    * see https://docs.spyder-ide.org/current/installation.html
    * :bash:`python3 -m venv spyder-env`
    * :bash:`source spyder-env/bin/activate`
* Testing:
    * make sure pyplot windows are displayed correctly (plot viewer) with enough dpi
    * make sure this is also the case for plots generated by the GUI


.. _tox_file:

Tox configuration in `tox.toml <https://github.com/drocheam/optrace/blob/main/tox.toml>`_
___________________________________________________________________________________________

.. literalinclude:: ../../../tox.toml
   :language: toml
   :linenos:


**Notes**

* tox-ignore-env-name-mismatch allows us to reuse the tox env for all actions
* parallelized testing with pytest-xdist could be possible, but we wouldn't gain much from it as the heaviest
  tasks are already multithreaded
* when gui tests fail on wayland, first run :bash:`xhost +`


