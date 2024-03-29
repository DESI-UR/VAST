# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
import os
import re

import sphinx_rtd_theme

sys.path.insert(0, os.path.abspath('../'))
import vast

# -- Project information -----------------------------------------------------

project = 'VAST'
copyright = "2022, Kelly A. Douglass, Dahlia Veyrat, Stephen W. O'Neill Jr., Segev BenZvi, Fatima Zaidouni, Michaela Guzzetti"
author = "Kelly A. Douglass, Dahlia Veyrat, Stephen W. O'Neill Jr., Segev BenZvi, Fatima Zaidouni, Michaela Guzzetti"

# The short X.Y version
version = re.match(r"v?(\d+\.\d+)", vast._version.__version__).group(1)
# The full version, including alpha/beta/rc tags
release = vast._version.__version__

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'numpydoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx_rtd_theme',
    'sphinxarg.ext'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['.templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

#html_logo = 'filename.png'

# -- Options for autodoc/napoleon -----------------------------------------

# Napoleon settings (see https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html)
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_rtype = False
autoclass_content = 'both'


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['.static']
