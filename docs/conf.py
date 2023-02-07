# MuseX documentation build configuration file, created by
# sphinx-quickstart on Mon Feb  5 09:38:29 2018.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))
from pkg_resources import get_distribution

# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
# needs_sphinx = '1.0'

language = 'en'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'numpydoc',
    'nbsphinx',
    'sphinx_automodapi.automodapi',
    'sphinx_automodapi.smart_resolver'
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://docs.scipy.org/doc/numpy/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference/', None),
    'matplotlib': ('https://matplotlib.org/', None),
    'astropy': ('http://docs.astropy.org/en/stable/', None),
    'mpdaf': ('https://mpdaf.readthedocs.io/en/stable/', None),
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = 'pyetc'
copyright = '2023, Roland Bacon'
author = 'Roland Bacon'

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
release = get_distribution('pyetc').version
# for example take major/minor
version = '.'.join(release.split('.')[:2])

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False

# autodoc_default_flags = ['members', 'special-members',
#                          'inherited-members']
autodoc_member_order = 'bysource'

autosummary_generate = True

automodapi_toctreedirnm = 'api'
automodsumm_inherited_members = True

numpydoc_class_members_toctree = False
numpydoc_show_class_members = False
numpydoc_xref_param_type = True
numpydoc_xref_ignore = {'type', 'optional', 'default', 'or', 'of', 'method'}
numpydoc_xref_aliases = {
    # python
    'sequence': ':term:`python:sequence`',
    'iterable': ':term:`python:iterable`',
    'string': 'str',
    # numpy
    'array': 'numpy.ndarray',
    'dtype': 'numpy.dtype',
    'ndarray': 'numpy.ndarray',
    'array-like': ':term:`numpy:array_like`',
    'array_like': ':term:`numpy:array_like`',
}
# numpydoc_use_plots = True

# Class documentation should contain *both* the class docstring and
# the __init__ docstring
autoclass_content = "both"

# The reST default role (used for this markup: `text`) to use for all
# documents. Set to the "smart" one.
default_role = 'obj'


# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
# html_theme_options = {}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# This is required for the alabaster theme
# refs: http://alabaster.readthedocs.io/en/latest/installation.html#sidebars
html_sidebars = {
    '**': [
        'relations.html',  # needs 'show_related': True theme option to display
        'searchbox.html',
    ]
}
