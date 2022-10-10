from setuptools import setup
import os

# older vtk wheels:
# https://github.com/pyvista/pyvista-wheels

if __name__ == '__main__':

    # load package information
    path = os.path.join("optrace", "__metadata__.py")
    with open(path) as f:
        exec(f.read())

    # metadata is found in ./optrace/__metadata__.py

    setup(
            name=__name__,
            version=__version__,
            author=__author__,
            author_email=__email__,
            url=__url__,
            license=__license__,
            description=__description__,
            keywords=__keywords__,
            #
            classifiers=["Development Status :: 4 - Beta",
                         "Intended Audience :: Developers",
                         "Intended Audience :: Education",
                         "Intended Audience :: Science/Research",
                         "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
                         "Natural Language :: English",
                         "Operating System :: OS Independent",
                         "Operating System :: Microsoft :: Windows",
                         "Operating System :: POSIX",
                         "Operating System :: Unix",
                         "Operating System :: MacOS",
                         "Programming Language :: Python",
                         "Topic :: Scientific/Engineering",
                         "Topic :: Scientific/Engineering :: Physics",
                         "Topic :: Scientific/Engineering :: Visualization",
                         "Topic :: Software Development",
                         "Topic :: Software Development :: Libraries"],
            #
            python_requires='>=3.10, <3.11',
            packages=["optrace", "optrace.tracer", "optrace.tracer.color", "optrace.tracer.geometry",
                      "optrace.tracer.spectrum", "optrace.tracer.presets", "optrace.gui", "optrace.plots"],
            install_requires=['numpy>1.21', 'numexpr', 'scipy', 'Pillow', 'vtk', 'PyQt5', 'matplotlib', 'progressbar2', 'mayavi'],
            tests_require=['pynput', 'tox', 'colorio'],
            package_data={"": ["*.png", "*.jpg", "*.csv"]},   # includes tables and images
            include_package_data=True,
        )

