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

Make sure Python 3.11, 3.12 or 3.13 and pip are installed on your system

**Installing the latest official release**

#. Download the optrace .tar.gz archive 
   from the `latest release <https://github.com/drocheam/optrace/releases/latest>`__
#. Run :bash:`pip install <archive>` from a terminal, 
   where :bash:`<archive>` is the path of the downloaded file 

**Installing the latest git version**

Open a terminal and run: :bash:`pip install "optrace @ git+https://github.com/drocheam/optrace.git"`

**Installing from a package index**

optrace currently isn't included in any official package index.
This project is not affiliated with a package of the same name on PyPI.

**Dependencies**

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
   * - `pyvista <https://docs.pyvista.org>`_
     - easier Pythonic interface to VTK
     - MIT
   * - `pyvistaqt <https://pypi.org/project/pyvistaqt/>`_
     - placing a vtk-widget inside Qt
     - MIT
   * - `scipy <https://scipy.github.io/devdocs/tutorial/index.html#user-guide>`_
     - specialized methods for interpolation and math
     - BSD 3-Clause
   * - `tqdm <https://pypi.org/project/tqdm/>`_
     - animated progressbar
     - MPL-2.0 and MIT
   * - `traitsui <https://docs.enthought.com/traitsui/>`_
     - traits-capable windowing framework
     - BSD 3-Clause
   * - `vtk <https://pypi.org/project/vtk/>`_
     - visualization toolkit for 3D graphics
     - BSD 3-Clause


**Troubleshooting**

* In many cases forcing the installation of a specific library version (e.g. vtk) circumvents issues of newer releases. 
  The syntax is: :bash:`pip install --force-reinstall -v "some-package==1.2.2"`.
  Often older releases are hosted outside of PyPi, so you might try to locate the packages first.
  A list of other wheels for vtk is found 
  `here <https://docs.vtk.org/en/latest/advanced/available_python_wheels.html>`__.

* Consult the `pyvista issues <https://github.com/pyvista/pyvista/issues>`__, 
  `vtk issues <https://gitlab.kitware.com/vtk/vtk/-/issues>`__ 
  or `PySide issues <https://bugreports.qt.io/projects/PYSIDE/issues/>`__ for current problems and solutions


