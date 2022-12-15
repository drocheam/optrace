Packaging
-----------------

Package metadata is defined in its own python file.  
This information is loaded while creating the documentation and when packaging.

**Content of** ``./optrace/__metadata__.py``
 
.. literalinclude:: ../../optrace/__metadata__.py


``setup.py`` is used as cript for the installation using ``setuptools``.

**Content of** ``setup.py``

.. literalinclude:: ../../setup.py

Make sure to ``include_package_data`` to also include image and text files needed for the ``./optrace/tracer/presets``.
The classifiers are needed if the package should ever be hosted on `PyPi <https://pypi.org/>`_.
