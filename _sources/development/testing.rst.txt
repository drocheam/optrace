Testing
---------------

.. role:: python(code)
  :language: python
  :class: highlight

.. role:: bash(code)
  :language: bash
  :class: highlight


Overview
_______________________

Extensive tests are done on the functionality, documentation and website of optrace.
Testing is done with `tox <https://tox.wiki/en/latest/>`_ and :mod:`pytest`.
Packages `tox <https://tox.wiki/en/latest/>`_ as well as 
`tox-ignore-env-name-mismatch <https://github.com/masenf/tox-ignore-env-name-mismatch>`_ need to be installed to run
different tests environments.
All other requirements are automatically installed while running tox.
The configuration can be found in Section :numref:`tox_file`.

Test can be run locally or through GitHub actions.
The workflows are located in `.github/workflows/ <https://github.com/drocheam/optrace/blob/main/.github/workflows/>`_,
while the last action runs can be seen in `GitHub Actions <https://github.com/drocheam/optrace/actions>`_. 

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
The number of used threads is controlled by setting an environment variable
as described in section :numref:`number_of_thread_specification`.

The raytracing on my system (Arch Linux 6.15.8, i7-1360P notebook CPU, 16GB RAM, Python 3.13) results in:

.. table:: Performance Comparison. Result values are in seconds / surface / million rays.
   :widths: 60 110 110
   :width: 600px

   +-------+----------------------+---------------------+
   | Cores | Without Polarization | With Polarization   |
   +=======+======================+=====================+
   |   1   |        0.148         |        0.218        |
   +-------+----------------------+---------------------+
   |   2   |        0.082         |        0.115        |
   +-------+----------------------+---------------------+
   |   4   |        0.053         |        0.085        |
   +-------+----------------------+---------------------+
   |   8   |        0.045         |        0.082        |
   +-------+----------------------+---------------------+
   |  12   |        0.043         |        0.078        |
   +-------+----------------------+---------------------+
   |  16   |        0.046         |        0.073        |
   +-------+----------------------+---------------------+

Performance plateaus at 8-16 cores for the "without polarization" case, but still seems to improve for the second case.
Reasons could be thermal throttling or other bottlenecks.
On my machine, 8 cores could be a compromise between CPU usage and performance.

Documentation Testing
__________________________________

The workflow file `doc_test.yml <https://github.com/drocheam/optrace/blob/main/.github/workflows/doc_test.yml>`_ 
runs multiple test environments. This includes:

* :bash:`tox -e docsbuildcheck`: Testing of correct documentation building while not permitting warning
* :bash:`tox -e linkcheck`: Testing of all documentation links (currently deactived, due to GitHub rate limits)
* :bash:`tox -e doctest`: Testing of most documentation code examples (using :mod:`doctest`)

Python Version Testing
__________________________________

Testing of the compatibility with multiple python versions is done with the 
`pyver_comp.yml <https://github.com/drocheam/optrace/blob/main/.github/workflows/pyver_comp.yml>`_ workflow.
It executes a subset of tests (:bash:`tox -e fast`) for multiple python main versions in an Ubuntu runner.

Installation Testing
__________________________________

Weekly tests that try to install the package and execute some quick tests.
Ensures that requirements and available PyPI packages are up-to-date or need changes.
The workflow is located in 
`install_test.yml <https://github.com/drocheam/optrace/blob/main/.github/workflows/install_test.yml>`_.

Platform Testing
__________________________________

Platform testing is done with the 
`os_comp.yml <https://github.com/drocheam/optrace/blob/main/.github/workflows/os_comp.yml>`_ workflow.
It executes the os tox environment (run with :bash:`tox -e os`), which runs a subset of all available tests.
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

Update Action Dependencies
_____________________________

The dependencies inside the GitHub Actions are kept up-to-date with the 
`Dependabot <https://docs.github.com/en/code-security/getting-started/dependabot-quickstart-guide>`__.
It is run weekly an opens pull request if there a newer versions of an action available.

Website tests
________________________

Documentation website tests are done with the 
`website_test.yml <https://github.com/drocheam/optrace/blob/main/.github/workflows/website_test.yml>` 
workflow that runs the 
`tests/test_website.sh <https://github.com/drocheam/optrace/blob/main/tests/test_website.sh>`__ bash script.
The test check that the site is reachable, includes most important parts (index, impressum, robots, sitemap), 
does not set cookies and does not load any external resources.
The two last points are required for more privacy and GDPR compliance.
Runs are triggered either manually, weekly or after the page deployment.

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

The pytest configuration is located in the ``pyproject.toml`` in Section :numref:`pyproject_toml`.


**Notes**

* parallelized testing with pytest-xdist could be possible, but we wouldn't gain much from it as the heaviest
  tasks are already multithreaded


* `pytest-xvfb <https://github.com/The-Compiler/pytest-xvfb>`_ 
  uses xvfb as headless-display. Use the option :bash:`--no-xvfb` to actually see the plots/windows.

* using `pytest-timeout <https://pypi.org/project/pytest-timeout/>`_ for adding timeouts

* using `pytest-random-order <https://pypi.org/project/pytest-random-order/>`_ for running tests in random order

* `tox-ignore-env-name-mismatch <https://github.com/masenf/tox-ignore-env-name-mismatch>`_ is required so multiple
  tox environments are able to use the same Python virtualenv.

* when GUI tests fail on wayland, first run :bash:`xhost +`

* some tests are excluded in GitHub actions, as there issues with the headless displays

