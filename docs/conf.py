'''numpy-stl Sphinx configuration.'''

import datetime

from stl import __about__ as metadata

# -- Project information
project = 'numpy-stl'
author = metadata.__author__
version = release = metadata.__version__
copyright = f'2014-{datetime.date.today().year}, {author}'

# -- Extensions
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]

# -- Theme
html_theme = 'furo'

# -- Napoleon (Google-style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_param = True
napoleon_use_rtype = True

# -- Intersphinx
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
}

# -- Autodoc
autodoc_member_order = 'bysource'
autodoc_typehints = 'description'
