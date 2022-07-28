# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
import inspect
from os.path import relpath, dirname
import sys
import fitlins


# -- Project information -----------------------------------------------------

project = 'FitLins'
copyright = '2022, Center for Reproducible Neuroscience'
author = 'Center for Reproducible Neuroscience'
version = fitlins.__version__

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'sphinx.ext.linkcode',
    'sphinxarg.ext',  # argparse extension
    'sphinxcontrib.apidoc',
    'texext.math_dollar',
    'nbsphinx',
    'myst_parser',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

apidoc_module_dir = '../fitlins'
apidoc_output_dir = 'api'
apidoc_separate_modules = True
apidoc_extra_args = ['-H', 'API']

source_suffix = ['.rst', '.md']

intersphinx_mapping = {
    'nilearn': ('https://nilearn.github.io/stable/', None),
    'afni': ('https://afni.nimh.nih.gov/pub/dist/doc/htmldoc/', None)
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'alabaster'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

example_gallery_config = {
    'examples_dirs': '../examples/notebooks',
    'gallery_dirs': 'examples/notebooks',
    'pattern': '.+.ipynb',
    'disable_warnings': False,
    'dont_preprocess': ['../examples/notebooks/ds003_sample_analysis.ipynb'],
    'toctree_depth': 1,
    }


# -----------------------------------------------------------------------------
# Source code links
# -----------------------------------------------------------------------------

def linkcode_resolve(domain, info):
    """
    Determine the URL corresponding to Python object
    """
    if domain != 'py':
        return None

    modname = info['module']
    fullname = info['fullname']

    submod = sys.modules.get(modname)
    if submod is None:
        return None

    obj = submod
    for part in fullname.split('.'):
        try:
            obj = getattr(obj, part)
        except Exception:
            return None

    obj = inspect.unwrap(obj)

    try:
        fn = inspect.getsourcefile(obj)
    except Exception:
        fn = None
    if not fn:
        return None

    try:
        source, lineno = inspect.getsourcelines(obj)
    except Exception:
        linespec = ""
    else:
        linespec = "#L%d-L%d" % (lineno, lineno + len(source) - 1)

    fn = relpath(fn, start=dirname(fitlins.__file__))

    ver = fitlins.__version__
    if 'dev' in ver:
        ver = 'dev'
    return f"https://github.com/poldracklab/fitlins/blob/{ver}/fitlins/{fn}{linespec}"
