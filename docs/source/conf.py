# Configuration file for the Sphinx documentation builder.

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------
project = 'Hill Climber'
copyright = '2025, Hill Climber Contributors'
author = 'Hill Climber Contributors'
release = '2.1.1'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
    'sphinx.ext.mathjax',
    'nbsphinx',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# -- Napoleon settings -------------------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# -- Autodoc settings --------------------------------------------------------
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# Suppress warnings about duplicate dataclass field documentation
suppress_warnings = ['ref.python']

# Control dataclass documentation to avoid duplicates
autodoc_typehints = 'description'
autodoc_class_signature = 'separated'

# -- NBSphinx settings -------------------------------------------------------
nbsphinx_execute = 'never'  # Don't execute notebooks during build
nbsphinx_allow_errors = True  # Continue on errors
nbsphinx_timeout = 60

# Add notebooks directory to exclude from source processing
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Add path to notebooks
import os
nbsphinx_link_target_root = os.path.abspath('../..')

# -- Autosummary settings ----------------------------------------------------
autosummary_generate = True
