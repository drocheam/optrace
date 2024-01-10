Documentation
---------------


**Goal**

* website documenting library functionality and source code
* mathematical description of methods, including images, equations and sources
* installation instruction and simple functionality overview


**Source Code**

* `typing hints <https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html>`_ for almost all functions
* `docstring <https://peps.python.org/pep-0257/>`_ comments for functions and important variables
* comments have `reStructuredText <https://docutils.sourceforge.io/docs/user/rst/quickref.html>`_ syntax


**Documentation**

* see documentation guidelines in :numref:`guidelines`
* generate source code documentation using `Sphinx <https://www.sphinx-doc.org/en/master/>`_.
* functionality description is done using `reStructuredText <https://docutils.sourceforge.io/docs/user/rst/quickref.html>`_ and embedded in the rest of the documentation
* documentation also includes information on testing, library dependencies, changelog and documentation
* documentation source files can be found under ``./docs/source`` and consist of ``.rst`` files
* run ``make html`` from the ``doc`` folder to generate the html documentation (requires the `GNU make utility <https://www.gnu.org/software/make/>`_)
* build files get created in ``./docs/build/html``, open the documentation by e.g. ``firefox ./docs/build/html/index.html``
* check code snippets with ``sphinx.ext.doctest`` and ``make doctest``

**Workflow**

* `tox <https://tox.wiki/en/latest/>`_ environment ``docs``, call from main folder using ``tox -e docs``, see the ``tox.ini`` files for details
* this runs the following:
   * the `Sphinx <https://www.sphinx-doc.org/en/master/>`_ html documentation compilation
   * a bash script generating the changelog
   * a bash script generating the source file toctree


**Sphinx configuration**

* see ``.docs/source/conf.py`` for details
* Sphinx extensions 
   * `autodoc <https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html>`_ for automatic source code documentation generation
   * `intersphinx <https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html>`_ for linking documentation of other packages
   * `mathjax <https://www.sphinx-doc.org/en/master/usage/extensions/math.html>`_ for LaTeX equations
   * `sphinxcontrib.bibtex <https://sphinxcontrib-bibtex.readthedocs.io/en/latest/>`_ for a bibliography
* `pyramid <https://sphinx-themes.org/sample-sites/default-pyramid/>`_ Sphinx theme, however we overwrite the text font family by including the  ``./docs/sources/_static/custom.css`` file. This also centrally aligns the equation label vertically.


**Content of** ``./docs/source/conf.py``

.. literalinclude:: ../conf.py

