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

To allow for a simple and up-to-date code reference the process should be automatic, by generating it from 
`typing hints <https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html>`_ 
and  `docstrings <https://peps.python.org/pep-0257/>`_. 


Workflow
_______________________

The `gen_docs.yml <https://github.com/drocheam/optrace/blob/main/.github/workflows/gen_docs.yml>`_ 
action compiles the documentation and publishes 
it in the `gh-pages <https://github.com/drocheam/optrace/tree/gh-pages>`__ branch. 
The `pages-build-deployment <https://github.com/drocheam/optrace/actions/workflows/pages/pages-build-deployment>`_ 
action deploys the documentation on the website.

Internally the workflow executes the :bash:`docs` environment for `tox <https://tox.wiki/en/latest/>`_ 
(see the :ref:`tox file <tox_file>`), which builds the documentation 
with the help of `Sphinx <https://www.sphinx-doc.org/en/master/>`_.
The steps also include an automatic generation of a changelog, source file toctree and sitemap.

Documentation Generation
__________________________________

The documentation is created with `Sphinx <https://www.sphinx-doc.org/en/master/>`_, 
which uses `reStructuredText <https://docutils.sourceforge.io/docs/user/rst/quickref.html>`_ as markup language. 
The html builder of Sphinx compiles a webpage including the self-written documentation, 
automatically generated code information, as well as indices for searching.

The following extensions are used:

.. list-table::
   :align: left

   * - `doctest <https://www.sphinx-doc.org/en/master/usage/extensions/doctest.html>`_ 
     - automatic testing of documentation code examples
   * - `autodoc <https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html>`_ 
     - automatic source code documentation generation
   * - `intersphinx <https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html>`_ 
     - linking to the documentation of external packages
   * - `mathjax <https://www.sphinx-doc.org/en/master/usage/extensions/math.html>`_ 
     - LaTeX equations
   * - `sphinxcontrib.bibtex <https://sphinxcontrib-bibtex.readthedocs.io/en/latest/>`_ 
     - bibliography management
   * - `Shibuya Sphinx Theme <https://shibuya.lepture.com/>`_
     - A modern looking theme
   * - `Sphinx Sitemap <https://sphinx-sitemap.readthedocs.io/en/latest/index.html>`_
     - Generate sitemaps while building the documentation
   * - `Sphinx Mathjax Offline <https://pypi.org/project/sphinx-mathjax-offline/>`_
     - Provides the mathjax js and font files without having to rely on an external CDN
   * - `sphinx-notfound-page <https://sphinx-notfound-page.readthedocs.io/en/latest/>`_
     - Fixes some issues with 404 pages

Additionally, a custom css (`docs/source/_static/css/custom.css <https://github.com/drocheam/optrace/blob/main/docs/source/_static/css/custom.css>`_) adapts the formatting to our needs.

Documentation source files can be found under `docs/source/ <https://github.com/drocheam/optrace/blob/main/docs/source/>`_
The Sphinx configuration can be found below.

**Content of** `docs/source/conf.py <https://github.com/drocheam/optrace/blob/main/docs/source/conf.py>`_

.. literalinclude:: ../conf.py
   :linenos:

