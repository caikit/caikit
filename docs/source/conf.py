# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# Standard
# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join("..", "..")))


# -- Project information -----------------------------------------------------

project = "Caikit"
copyright = "2023, The Caikit Authors"
author = "The Caikit Authors"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    # Generate complete API docs by parsing source code
    "autoapi.extension",
    # Add links to source from generated docs
    "sphinx.ext.viewcode",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "caikit/config"]

# -- autoapi configuration ---------------------------------------------------

# Language of source code to parse
autoapi_type = "python"

# Source code to parse to generate API docs relative to 'docs/source' directory
autoapi_dirs = [os.path.join("..", "..", "caikit")]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []

# Support external links to specific versions of the files in the Github repo
branch = os.environ.get("READTHEDOCS_VERSION")
if branch is None or branch == "latest":
    branch = "main"

REPO = "caikit/caikit"
scm_raw_web = "https://raw.githubusercontent.com/" + REPO + branch
scm_web = "https://github.com/" + REPO + "blob/" + branch

# Store variables in the epilogue so they are globally available.
rst_epilog = """
.. |SCM_WEB| replace:: {s}
.. |SCM_RAW_WEB| replace:: {sr}
.. |SCM_BRANCH| replace:: {b}
""".format(
    s=scm_web, sr=scm_raw_web, b=branch
)

# used to have links to repo files
extlinks = {
    "scm_raw_web": (scm_raw_web + "/%s", "scm_raw_web"),
    "scm_web": (scm_web + "/%s", "scm_web"),
}
