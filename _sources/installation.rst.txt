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

Make sure Python 3.11, 3.12 or 3.13 are installed on your system


**Installing the latest official release**

#. Make sure python and pip are installed
#. Download the optrace .tar.gz archive 
   from the `latest release <https://github.com/drocheam/optrace/releases/latest>`__
#. Run :bash:`pip install <archive>` from a terminal, 
   where :bash:`<archive>` is the path of the downloaded file 

**Installing the latest git version**

#. Make sure python and pip are installed
#. Open a terminal and run: :bash:`pip install "optrace @ git+https://github.com/drocheam/optrace.git"`


.. TODO note about limitation with PyPi

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
   * - `numpy <https://numpy.org/doc/stable/user/index.html#user>`_
     - matrix calculations
     - BSD 3-Clause
   * - `opencv-python-headless <https://pypi.org/project/opencv-python-headless/>`_
     - image loading, saving and resizing
     - Apache 2.0
   * - `pyqtdarktheme-fork <https://pypi.org/project/pyqtdarktheme-fork/>`_
     - setting a QT dark theme
     - MIT
   * - `pyside6 <https://wiki.qt.io/Qt_for_Python>`_
     - Qt GUI backend
     - LGPLv3
   * - `scipy <https://scipy.github.io/devdocs/tutorial/index.html#user-guide>`_
     - specialized methods for interpolation and math
     - BSD 3-Clause
   * - `tqdm <https://pypi.org/project/tqdm/>`_
     - animated progressbar
     - MPL-2.0 and MIT


**Troubleshooting**

* When `mayavi <https://pypi.org/project/mayavi/>`__ fails installing `vtk <https://pypi.org/project/vtk/>`_, 
  try to install vtk first

* In many cases forcing the installation of a specific library version (e.g. vtk) circumvents issues of newer releases. 
  The syntax is: :bash:`pip install --force-reinstall -v "some-package==1.2.2"`.
  Often older releases are hosted outside of PyPi, so you might try to locate the packages first.
  A list of other wheels for vtk is found `here <https://docs.vtk.org/en/latest/advanced/available_python_wheels.html>`__.

* Consult the `mayavi issues <https://github.com/enthought/mayavi/issues>`__, 
  `vtk issues <https://gitlab.kitware.com/vtk/vtk/-/issues>`__ 
  or `PySide issues <https://bugreports.qt.io/projects/PYSIDE/issues/>`__ for current problems and solutions

* Installing mayavi with no cache and without isolated building can help, see `here <https://github.com/enthought/mayavi/issues/1325#issuecomment-2537662062>`__:
  :bash:`pip install mayavi --no-cache-dir --verbose  --no-build-isolation`


