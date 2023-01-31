Testing
---------------


**Goal**

* automated testing of functionality and edge cases
* 100% coverage, all branches and source code lines get executed at least once
* testing of the installation of the required packages on a clean system
* testing of multiple platforms (Linux, Windows, macOS)
* test multithreading with different number of cores
* future goal: support of multiple python versions (>= Python 3.10)


**Main Testing**

* done by `tox <https://tox.wiki/en/latest/>`_, this creates a virtual environment where the library is installed and tested
* `pytest <https://docs.pytest.org/en/7.2.x/>`_ runs all test cases
* `coverage <https://coverage.readthedocs.io>`_ wraps around ``pytest`` and reports on executed and missing lines
* see the ``tox.ini`` file for details
* test files can be found under ``./tests/``, starting with filename ``test_``


**Test Cases**

* `doctest <https://docs.python.org/3/library/doctest.html>`_ for documentation strings in some non-class functions
* test cases are handled with ``unittest``, testing is however typically done by ``pytest`` that loads and executed the created TestSuites
* ``TraceGUI`` has a ``debug`` method, which runs a separate thread, from which actions are executed:
   * Note that each UI actions needs to be run in the main thread, for this ``TraceGUI`` provides ``_do_in_main`` and ``_set_in_main methods``
   * generally we want to do an action, after the old one has finished. For this ``TraceGUI._wait_for_idle`` is implemented.


**Platform and version testing**

* platform tests are done with a github workflow, that executes some platform dependent test. These include:
   * loading and saving of files
   * handling of filenames and paths
   * opening the gui and plots
   * detecting the number of cores
* python version testing: currently does the same as platform testing, however some libraries don't support Python 3.11 yet


**Github Workflows**

* see ``.github/workflows/``
* Actions: Main testing, OS Compatibility, Python 3.11 Compatibility
* all get run on a push the repository or can get run manually


**Tox configuration in** ``tox.ini``

.. literalinclude:: ../../../tox.ini

