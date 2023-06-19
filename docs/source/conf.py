# Configuration file for the Sphinx documentation builder.
#

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath(os.path.join('..', '..')))


# -- Project information -----------------------------------------------------

# load from file
path = os.path.join("..", "..", "optrace", "__metadata__.py")
with open(path) as f:
    exec(f.read())

# assign
project = __name__
copyright = __copyright__.replace("Copyright ", "")
author = __author__
release = __version__


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
    "secondary_sidebar_items": ["page-toc", "sidebar-nav-bs-auto"],  # add navigation to secondary sidebar
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

# turn off primary side bar
html_sidebars = {
  "**": []
}

numfig = True
math_numfig = True

# links to libraries
intersphinx_mapping = {'python': ('http://docs.python.org/3', None),
                       'numpy': ('http://numpy.org/doc/stable', None),
                       'scipy': ('http://docs.scipy.org/doc/scipy/', None),
                       'traits': ('https://docs.enthought.com/traits/', None),
                       'matplotlib': ('http://matplotlib.org/stable', None),
                       'mayavi': ('https://docs.enthought.com/mayavi/mayavi', None),
                       'PIL': ('https://pillow.readthedocs.io/en/stable/', None)}

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


# -- Options for misc output -------------------------------------------------

# ignore links that only work in a browser
linkcheck_ignore = ['https://doc.comsol.com/6.1/docserver/#!/com.comsol.help.roptics/roptics_ug_optics.6.54.htm',
                    'https://doi.org/10.1002/9783527648962.app1']

# only check doctest blocks
doctest_test_doctest_blocks = ""
