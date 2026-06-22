Packaging
-----------------

.. role:: bash(code)
  :language: bash
  :class: highlight

Overview
___________________

Releases are located `here <https://github.com/drocheam/optrace/releases>`_.
The `release.yml <https://github.com/drocheam/optrace/blob/main/.github/workflows/release.yml>`_ workflow 
creates a :bash:`.tar.gz` archive that can be installed with pip.
The compressed package should only include metadata and the library itself, 
while documentation and testing data should be excluded.
Unfortunately, tests are included by default and need to be explicitly 
excluded with a `MANIFEST.in <https://github.com/drocheam/optrace/blob/main/MANIFEST.in>`_.

.. _pyproject_toml:

Configuration
_______________________

Requirements and metadata for the library are defined in the 
`pyproject.toml <https://github.com/drocheam/optrace/blob/main/pyproject.toml>`_ file.
This is a more modern successor to a :bash:`setup.py` file.

**Content of** `pyproject.toml <https://github.com/drocheam/optrace/blob/main/pyproject.toml>`_

.. literalinclude:: ../../../pyproject.toml
   :language: toml
   :linenos:

