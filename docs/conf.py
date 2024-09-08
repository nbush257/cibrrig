import os
import sys
sys.path.insert(0, os.path.abspath('../cibrrig')) 

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'cibrrig'
copyright = '2024, Nicholas Bush'
author = 'Nicholas Bush'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',   # For extracting docstrings
    'sphinx.ext.napoleon',  # To support NumPy and Google-style docstrings
    'sphinx.ext.viewcode',  # Adds links to highlighted source code
    'sphinx.ext.doctest',
    'myst_parser',            # Use markdown files in addition to .rst
    'sphinx_rtd_theme',
    'sphinx.ext.autosummary',
]
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

autoapi_dirs = ['.../cibrrig']
master_doc = 'index'


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
