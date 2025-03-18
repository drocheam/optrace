Packaging
-----------------

.. role:: bash(code)
  :language: bash
  :class: highlight

The `release.yml <https://github.com/drocheam/optrace/blob/main/.github/workflows/release.yml>`_ workflow 
creates a :bash:`.tar.gz` archive that can be installed with pip.
Releases are located `here <https://github.com/drocheam/optrace/releases>`_.

The compressed package should only include the library itself and metadata, 
while documentation and testing should be excluded.
Unfortunately, tests are included by default and need to be explicitly 
excluded with a `MANIFEST.in <https://github.com/drocheam/optrace/blob/main/MANIFEST.in>`_.

Requirements and metadata for the library are defined in the 
`pyproject.toml <https://github.com/drocheam/optrace/blob/main/pyproject.toml>`_ file.
This is a more modern successor to a :bash:`setup.py` file.

**Content of** `pyproject.toml <https://github.com/drocheam/optrace/blob/main/pyproject.toml>`_

.. literalinclude:: ../../../pyproject.toml

**Notes**

* mayavi git version needed as PyPi version rarely gets updated
  see `<https://stackoverflow.com/questions/68073819/pypi-install-requires-direct-links>`_
  and `<https://stackoverflow.com/questions/40065321/how-to-include-git-dependencies-in-setup-py-for-pip-installation#comment115929654_65527149>`_
  or `<https://stackoverflow.com/a/54894359>`_
* MANIFEST.in with content :bash:`recursive-exclude tests *` required, 
  so tests directory gets excluded from dist. See `<https://stackoverflow.com/a/72821651>`_
  and `<https://stackoverflow.com/a/48912748>`_ at comment from bogeymin


