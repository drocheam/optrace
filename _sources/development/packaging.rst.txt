Packaging
-----------------

.. role:: bash(code)
  :language: bash
  :class: highlight

The `release.yml <https://github.com/drocheam/optrace/blob/main/.github/workflows/release.yml>`_ workflow creates a :bash:`.tar.gz` archive that can be installed with pip.
Releases are located `here <https://github.com/drocheam/optrace/releases>`_.

The compressed package should only include the library itself and metadata, while documentation and testing should be excluded.
Unfortunately, tests are included by default and need to be explicitly excluded with a `MANIFEST.in <https://github.com/drocheam/optrace/blob/main/MANIFEST.in>`_.

Requirements and metadata for the library are defined in the `pyproject.toml <https://github.com/drocheam/optrace/blob/main/pyproject.toml>`_ file.
This is a more modern successor to a :bash:`setup.py` file.

**Content of** `pyproject.toml <https://github.com/drocheam/optrace/blob/main/pyproject.toml>`_

.. literalinclude:: ../../../pyproject.toml

