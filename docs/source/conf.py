# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


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
# import maro.backends.frame
import os
import sys

from recommonmark.parser import CommonMarkParser

sys.path.insert(0, os.path.abspath("../.."))


os.environ["APIDOC_GEN"] = os.environ.get("APIDOC_GEN", "True")
# -- Project information -----------------------------------------------------

project = "maro"
copyright = "2020 Microsoft"
author = "MARO Team"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named "sphinx.ext.*") or your custom
# ones.

extensions = ["recommonmark",
              "sphinx.ext.autodoc",
              "sphinx.ext.coverage",
              "sphinx.ext.napoleon",
              "sphinx.ext.viewcode",
              "sphinx_markdown_tables",
              "sphinx_copybutton",
              ]

napoleon_google_docstring = True
napoleon_use_param = False
napoleon_use_ivar = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
if os.environ["APIDOC_GEN"] == "True":
    exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
else:
    exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "apidoc"]


# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False

# -- Options for HTML output -------------------------------------------------
# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = "alabaster"
html_theme = "sphinx_book_theme"

html_theme_options = {
    "repository_url": "https://github.com/microsoft/maro",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "path_to_docs": "doc/source",
    "home_page_in_toc": True,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ["_static"]
html_favicon = "images/fav32x32.ico"
html_title = "latest"
html_logo = "images/logo.svg"
htmlhelp_basename = "MaroDoc"

source_parsers = {
    ".md": CommonMarkParser,
}

source_suffix = [".md", ".rst"]
