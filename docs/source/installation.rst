.. _installation:

################
Installation
################

.. role:: python(code)
  :language: python
  :class: highlight

This library requires Python 3.10.

**Instructions**

#. make sure you have Python 3.10 installed on your system
#. download/clone the whole project repository. This is the folder that includes the ``optrace``-folder as well as the ``setup.py`` file.
#. open a terminal in the downloaded folder and run :python:`pip install .` :math:`\leftarrow` Note that the dot is part of the command.

This automatically runs the file ``setup.py``, which provides all the required information for the installation.
After this, ``optrace`` should be installed on your system. 

**External dependencies**

.. list-table:: 
   :widths: 200 350
   :header-rows: 1
   :align: left

   * - Dependency
     - Needed for/as
   * - `chardet <https://chardet.readthedocs.io/en/latest/>`_
     - automatic detection of file encoding
   * - `mayavi <https://docs.enthought.com/mayavi/mayavi/>`_
     - 3D plotting library
   * - `matplotlib <https://matplotlib.org/stable/users/index>`_
     - 2D plotting library
   * - `numexpr <https://numexpr.readthedocs.io/projects/NumExpr3/en/latest/user_guide.html>`_
     - sped-up and optimized calculations
   * - `numpy <https://numpy.org/doc/stable/user/index.html#user>`_
     - matrix calculations
   * - `Pillow <https://pillow.readthedocs.io/en/stable/>`_
     - image loading and converting
   * - `progressbar2 <https://pypi.org/project/progressbar2/>`_
     - animated progressbar in standard output (terminal)
   * - `PyQt5 <https://pypi.org/project/PyQt5/>`_
     - GUI backend for mayavi
   * - `scipy <https://scipy.github.io/devdocs/tutorial/index.html#user-guide>`_
     - specialized methods for interpolation and math
   * - `traitsui <https://docs.enthought.com/traitsui/>`_
     - trait framework for mayavi
   * - `vtk <https://pypi.org/project/vtk/>`_
     - 3D toolkit needed for mayavi

