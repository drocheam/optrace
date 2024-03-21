# Configuration file for the Sphinx documentation builder.
#

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import datetime

sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath(os.path.join('..', '..')))


# -- Project information -----------------------------------------------------

# load package information
metadata = {}
path = os.path.join("..", "..", "optrace", "metadata.py")
with open(path, "r") as f:
    exec(f.read(), metadata)

# assign metadata
project = metadata["name"]
release = metadata["version"]
author = metadata["author"]
copyright = f"{datetime.date.today().year}, {author}"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
        'sphinx.ext.autodoc',
        'sphinx.ext.intersphinx',
        'sphinx.ext.mathjax',
        'sphinx.ext.doctest',
        'sphinxcontrib.bibtex',
        # 'sphinxcontrib.inkscapeconverter',  # convert svg to pdf for latex output
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

bibtex_bibfiles = ['bib.bib']
bibtex_default_style = 'unsrt'

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# needs sphinx-rtd-theme installed
# html_theme = 'pyramid'
html_theme = "pydata_sphinx_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_css_files = ['css/custom.css']

html_theme_options = {
    "secondary_sidebar_items": ["page-toc"],  # add navigation to secondary sidebar
    "navbar_align": "content",
    "icon_links": [{"name": "GitHub",
                    "url": "https://github.com/drocheam/optrace",
                    "icon": "fa-brands fa-square-github",
                    "logo": "",
                    "type": "fontawesome",},],
    "pygment_dark_style": "github-dark",
}

# html_context = {
   # "default_mode": "light"  # force light mode
# }

# primary side bar
html_sidebars = {
  "**": ["sidebar-nav-bs"],
  "index": ["sidebar-nav-bs-index", "indices"],
}

numfig = True
math_numfig = True

# links to libraries
intersphinx_mapping = {'python': ('http://docs.python.org/3', None),
                       'numpy': ('http://numpy.org/doc/stable', None),
                       'scipy': ('http://docs.scipy.org/doc/scipy/', None),
                       'traits': ('https://docs.enthought.com/traits/', None),
                       'matplotlib': ('http://matplotlib.org/stable', None),
                       'mayavi': ('https://docs.enthought.com/mayavi/mayavi', None)}

autodoc_default_options = {
	'members': True,
    'undoc-members': True,
    'special-members' : '__call__, __eq__, __ne__',
    'exclude-members': '__weakref__'
}

autodoc_member_order = 'groupwise'

# move docstring from __init__ to class
autoclass_content = "init"

# move typehints from header to description
autodoc_typehints = "description"


# -- Misc Options -------------------------------------------------

# ignore links that seemingly only work in a browser and not in automated tests
linkcheck_ignore = ['https://doc.comsol.com/6.1/docserver/#!/com.comsol.help.roptics/roptics_ug_optics.6.54.html',
                    'https://doi.org/10.1002/9783527648962.app1',
                    'https://www.publicdomainpictures.net/en/view-image.php',
                    'https://www.pexels.com/photo/sliced-fruits-on-tray-1132047/',
                    'https://www.pexels.com/photo/photo-of-people-standing-near-blackboard-3184393/',
                    'https://www.pexels.com/photo/green-island-in-the-middle-of-the-lake-during-daytime-724963/',
                    'https://www.pexels.com/photo/green-2-seat-sofa-1918291/',
                    'https://www.pexels.com/photo/documents-on-wooden-surface-95916/',
                    'https://www.pexels.com/photo/cars-on-street-during-night-time-3158562/',
                    'https://www.edmundoptics.com/knowledge-center/tech-tools/focal-length/',
                    'https://doi.org/10.1080/713818864',
                    'https://doi.org/10.1080/10867651.1997.10487479']

# only check doctest blocks
doctest_test_doctest_blocks = ""

