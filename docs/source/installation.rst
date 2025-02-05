.. _installation:

################
Installation
################

.. role:: python(code)
  :language: python
  :class: highlight

.. role:: bash(code)
  :language: bash
  :class: highlight

Make sure Python 3.10, 3.11, 3.12 or 3.13 are installed on your system

**Installing the latest official release**

#. Find the newest release at the `Releases page <https://github.com/drocheam/optrace/releases>`__ and download the .tar.gz archive from the Assets section
#. Open a terminal
#. Run :bash:`pip install <path to archive>`, where :bash:`<path to archive>` is the path to the archive downloaded in the first step

**Installing the current git version**

#. Open a terminal
#. Clone the whole project repository using git: :bash:`git clone https://github.com/drocheam/optrace/`
#. Change the directory into the cloned folder: :bash:`cd optrace`
#. Install using :bash:`pip install .`


**External dependencies**

Below you can find a list of external dependencies that will be automatically installed. 

.. list-table:: 
   :widths: 250 600 250
   :header-rows: 1
   :align: left

   * - Dependency
     - Description
     - License
   * - `chardet <https://chardet.readthedocs.io/en/latest/>`_
     - automatic detection of file encoding
     - LGPLv2.1
   * - `mayavi <https://docs.enthought.com/mayavi/mayavi/>`_
     - 3D plotting library
     - BSD 3-Clause
   * - `matplotlib <https://matplotlib.org/stable/users/index>`_
     - 2D plotting library
     - PSF
   * - `numexpr <https://numexpr.readthedocs.io/en/latest/>`_
     - sped-up and optimized calculations
     - MIT
   * - `numpy <https://numpy.org/doc/stable/user/index.html#user>`_
     - matrix calculations
     - BSD 3-Clause
   * - `opencv-python-headless <https://pypi.org/project/opencv-python-headless/>`_
     - image loading, saving and resizing
     - Apache 2.0
   * - `progressbar2 <https://pypi.org/project/progressbar2/>`_
     - animated progressbar in standard output (terminal)
     - BSD 3-Clause
   * - `pyqtdarktheme <https://pypi.org/project/pyqtdarktheme/>`_
     - setting a QT dark theme
     - MIT
   * - `pyside6 <https://wiki.qt.io/Qt_for_Python>`_
     - Qt GUI backend
     - LGPLv3
   * - `scipy <https://scipy.github.io/devdocs/tutorial/index.html#user-guide>`_
     - specialized methods for interpolation and math
     - BSD 3-Clause


**Troubleshooting**

* When `mayavi <https://pypi.org/project/mayavi/>`__ fails installing `vtk <https://pypi.org/project/vtk/>`_, try to install vtk first

* When the installation of vtk fails, try to install from a list of other wheels from `here <https://docs.vtk.org/en/latest/advanced/available_python_wheels.html>`__.

* Consult the `mayavi issues <https://github.com/enthought/mayavi/issues>`__, `vtk issues <https://gitlab.kitware.com/vtk/vtk/-/issues>`__ or `PySide issues <https://bugreports.qt.io/projects/PYSIDE/issues/>`__ for current problems and solutions

* In many cases forcing the installation of a specific library version (e.g. vtk) circumvents issues of newer releases. The syntax is shown `here <https://stackoverflow.com/questions/5226311/installing-specific-package-version-with-pip/5226504#5226504>`__. Often older releases are hosted outside of PyPi, so you might try to locate the packages first.

