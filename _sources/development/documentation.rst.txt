Documentation
---------------

.. role:: python(code)
  :language: python
  :class: highlight

.. role:: bash(code)
  :language: bash
  :class: highlight

Overview
_______________________

The documentation should contain information about the project, its usage as well as implementation and programming details.
All should be presented as easily navigable webpage which includes figures, equations and code snippets.

To allow for a simple and up-to-date code reference the process should be automatic, by generating it from `typing hints <https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html>`_ and  `docstrings <https://peps.python.org/pep-0257/>`_. 


Workflow
_______________________

The `gen_docs.yml <https://github.com/drocheam/optrace/blob/main/.github/workflows/gen_docs.yml>`_ action compiles the documentation and publishes it in the `gh-pages <https://github.com/drocheam/optrace/tree/gh-pages>`__ branch. In the future, this content will be published on a webpage.

Internally the workflow executes the :bash:`docs` environment for `tox <https://tox.wiki/en/latest/>`_ (see the :ref:`tox file <tox_file>`), which builds the documentation with the help of `Sphinx <https://www.sphinx-doc.org/en/master/>`_.
The steps also include an automatic generation of a changelog and a source file toctree.

Documentation Generation
__________________________________

The documentation is created with `Sphinx <https://www.sphinx-doc.org/en/master/>`_, which uses `reStructuredText <https://docutils.sourceforge.io/docs/user/rst/quickref.html>`_ as markup language. 
The html builder of Sphinx compiles a webpage including the self-written documentation, automatically generated code information, as well as indices for searching.

The following extensions are used:

.. list-table::
   :align: left

   * - `sphinx.ext.doctest <https://www.sphinx-doc.org/en/master/usage/extensions/doctest.html>`_ 
     - automatic testing of documentation code examples
   * - `autodoc <https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html>`_ 
     - automatic source code documentation generation
   * - `intersphinx <https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html>`_ 
     - linking to the documentation of external packages
   * - `mathjax <https://www.sphinx-doc.org/en/master/usage/extensions/math.html>`_ 
     - LaTeX equations
   * - `sphinxcontrib.bibtex <https://sphinxcontrib-bibtex.readthedocs.io/en/latest/>`_ 
     - bibliography management
   * - `PyData Sphinx Theme <https://pydata-sphinx-theme.readthedocs.io/en/stable/>`_
     - A modern looking theme

Additionally, a custom css (`docs/source/_static/css/custom.css <https://github.com/drocheam/optrace/blob/main/docs/source/_static/css/custom.css>`_) adapts the formatting to our needs.

Documentation source files can be found under `docs/source/ <https://github.com/drocheam/optrace/blob/main/docs/source/>`_
The Sphinx configuration can be found below.

**Content of** `docs/source/conf.py <https://github.com/drocheam/optrace/blob/main/docs/source/conf.py>`_

.. literalinclude:: ../conf.py

