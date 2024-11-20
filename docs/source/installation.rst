.. _installation:

################
Installation
################

.. role:: python(code)
  :language: python
  :class: highlight

This library requires Python 3.10, 3.11 or 3.12.

**Instructions**

#. make sure you have Python 3.10, 3.11 or 3.12 installed on your system
#. download/clone the whole project repository. This is the folder that includes the ``optrace``-folder as well as the ``setup.py`` file.
#. open a terminal in the downloaded folder and run :python:`pip install .` :math:`\leftarrow` Note that the dot is part of the command.

This automatically runs the file ``setup.py``, which provides all the required information for the installation.
After this, `optrace` should be installed on your system. 


**External dependencies**

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

* When mayavi fails installing vtk, try to install vtk first

* When the installation of vtk fails, try to install from a list of other wheels from `here <https://docs.vtk.org/en/latest/advanced/available_python_wheels.html>`__.

* Consult the `mayavi issues <https://github.com/enthought/mayavi/issues>`__, `vtk issues <https://gitlab.kitware.com/vtk/vtk/-/issues>`__ or `PySide issues <https://bugreports.qt.io/projects/PYSIDE/issues/>`__ for current problems and solutions

* In many cases forcing the installation of a specific library version (e.g. vtk) circumvents issues of newer releases. The syntax is shown `here <https://stackoverflow.com/questions/5226311/installing-specific-package-version-with-pip/5226504#5226504>`__. Often older releases are hosted outside of PyPi, so you might try to locate the packages first.

