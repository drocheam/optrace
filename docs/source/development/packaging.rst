Packaging
-----------------

``setup.py`` is used as script for the installation using ``setuptools``.

**Content of** ``setup.py``

.. literalinclude:: ../../../pyproject.toml

Make sure to ``include_package_data`` to also include image and text files needed for the ``./optrace/tracer/presets``.
The classifiers are needed if the package should ever be hosted on `PyPi <https://pypi.org/>`_.
