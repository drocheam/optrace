# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
# sys.path.append(['../../optrace', '../../'
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../..'))



# -- Project information -----------------------------------------------------

project = 'Optrace'
copyright = '2022, Damian Mendroch'
author = 'Damian Mendroch'

# The full version, including alpha/beta/rc tags
release = '0.7.2'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
        'sphinx.ext.autodoc',
        'sphinx.ext.intersphinx',
        'sphinx_rtd_theme',
        'sphinx.ext.mathjax',
        'sphinxcontrib.bibtex',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

bibtex_bibfiles = ['bib.bib']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# needs sphinx-rtd-theme installed
html_theme = 'sphinx_rtd_theme'
# html_theme = 'sphinx_pdj_theme'
# html_theme = 'classic'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

def setup(app):
    app.add_css_file('css/custom.css')

html_theme_options = {
    # "rightsidebar": "false",
    # "relbarbgcolor": "GoldenRod",
    # "footerbgcolor": "GoldenRod",
    # "sidebarbgcolor": "black",
    "stickysidebar": True,
    "body_max_width" : 1200
}


numfig = True
math_numfig = True

# links to libraries
intersphinx_mapping = {'python': ('http://docs.python.org/3', None),
                       'numpy': ('http://docs.scipy.org/doc/numpy', None),
                       'scipy': ('http://docs.scipy.org/doc/scipy/reference', None),
                       'matplotlib': ('http://matplotlib.org/stable', None)}


autodoc_default_options = {
	'members': True,
    'undoc-members': True,
    'special-members' : '__call__, __eq__, __ne__, __setattr__',
    'exclude-members': '__weakref__'
}

autodoc_member_order = 'groupwise'

# move docstring from __init__ to class
autoclass_content = "init"

# move typehints from header to description
autodoc_typehints = "description"
