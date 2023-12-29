from setuptools import setup

# older vtk wheels:
# https://github.com/pyvista/pyvista-wheels

if __name__ == '__main__':

    setup(name = "optrace",
          author = "Damian Mendroch",
          credits = [],
          license = "not specified yet",
          version = "1.5.1",
          maintainer = "Damian Mendroch",
          email = "damian.mendroch@th-koeln.de",
          status = "Beta",
          url = "http://github.com/drocheam/optrace/",
          description = "An optics simulation package with sequential raytracing,"
                            " image rendering and a gui frontend",
          keywords = ["simulation", "optics", "raytracing"],
          #
          classifiers=["Development Status :: 4 - Beta",
                       "Intended Audience :: Developers",
                       "Intended Audience :: Education",
                       "Intended Audience :: Science/Research",
                       # "License :: OSI Approved :: # TODO add license
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
                    "optrace.tracer.geometry.surface","optrace.tracer.geometry.marker", 
                    "optrace.tracer.geometry.volume", "optrace.tracer.spectrum", "optrace.tracer.presets", 
                    "optrace.ressources", "optrace.ressources.images", "optrace.gui", "optrace.plots"],
          install_requires=['numpy>1.21', 'numexpr', 'chardet', 'scipy>=1.10', 'pyface<8', 'opencv-python-headless',
                            'vtk', 'PyQt5', 'matplotlib', 'progressbar2', 'mayavi'],
          # TODO loosen pyface restriction if ValueEditor and CodeEditor start to work again in mayavi
          # see https://github.com/enthought/pyface/releases/tag/8.0.0
          # see https://github.com/enthought/mayavi/pull/1255
          # see https://github.com/enthought/mayavi/issues/1252
          #
          tests_require=['pyautogui', 'pytest', 'colour-science', 'requests', 'mock', 'pytest-random-order'],
          package_data={"": ["*.png", "*.jpg", "*.webp", "*.csv"]},   # includes tables and images
          include_package_data=True,
        )

