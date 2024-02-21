from setuptools import setup
import os

# older vtk wheels:
# https://github.com/pyvista/pyvista-wheels

if __name__ == '__main__':
    
    # load package information
    metadata = {}
    path = os.path.join("optrace", "metadata.py")
    with open(path, "r") as f:
        exec(f.read(), metadata)

    # setup options
    setup(name = metadata["name"],
          author = metadata["author"],
          credits = metadata["credits"],
          license = metadata["license"],
          version = metadata["version"],
          maintainer = metadata["maintainer"],
          email = metadata["email"],
          status = metadata["status"],
          url = metadata["url"],
          description = metadata["description"],
          keywords = metadata["keywords"],
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
          python_requires='>=3.10, <3.12',
          packages=["optrace", "optrace.tracer", "optrace.tracer.color", "optrace.tracer.geometry", 
                    "optrace.tracer.geometry.surface","optrace.tracer.geometry.marker", 
                    "optrace.tracer.geometry.volume", "optrace.tracer.spectrum", "optrace.tracer.presets", 
                    "optrace.ressources", "optrace.ressources.images", "optrace.gui", "optrace.plots"],
          install_requires=['numpy>1.21', 'numexpr', 'chardet', 'scipy>=1.10', 'pyface', 'opencv-python-headless',
                            'vtk<9.3.0', 'PyQt5', 'matplotlib', 'progressbar2', 'mayavi @ git+https://github.com/enthought/mayavi'],
          # TODO loosen pyface restriction if ValueEditor and CodeEditor start to work again in mayavi
          # see https://github.com/enthought/pyface/releases/tag/8.0.0
          # see https://github.com/enthought/mayavi/pull/1255
          # see https://github.com/enthought/mayavi/issues/1252
          #
          tests_require=['pyautogui', 'pytest', 'colour-science', 'requests', 'pytest-random-order'],
          package_data={"": ["*.png", "*.jpg", "*.webp", "*.csv"]},   # includes tables and images
          include_package_data=True,
        )

