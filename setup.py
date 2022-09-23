from setuptools import setup

# older vtk wheels:
# https://github.com/pyvista/pyvista-wheels

if __name__ == '__main__':
    setup(
            name='optrace',
            version="0.9.8",
            author="Damian Mendroch",
            author_email="damian.mendroch@th-koeln.de",
            url='http://github.com/drocheam/optrace/',
            #
            #
            description='Sequental raytracing geometrical optics simulation package',
            keywords=["simulation", "optics", "raytracing"],
            license="GPLv3",
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
            #
            python_requires='>3.10',
            packages=["optrace", "optrace.tracer", "optrace.tracer.geometry",
                      "optrace.tracer.spectrum", "optrace.tracer.presets", "optrace.gui", "optrace.plots"],
            # install_requires=['numpy>1.21', 'numexpr', 'scipy', 'Pillow', 'traits>=6.1',
                              # 'envisage', 'traitsui', 'pyface', 'pyside2', 'vtk', 'PyQt5', 'matplotlib', 'progressbar2', 'mayavi'],
            install_requires=['numpy>1.21', 'numexpr', 'scipy', 'Pillow', 'vtk', 'PyQt5', 'matplotlib', 'progressbar2', 'mayavi'],
            tests_require=['pynput', 'tox'],
            package_data={"": ["*.png", "*.jpg", "*.csv"]},   # include preset images
            include_package_data=True,
        )

