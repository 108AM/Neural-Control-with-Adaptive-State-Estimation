"""Sphinx configuration for the Neural Control with Adaptive State Estimation docs."""

from __future__ import annotations

import sys
from pathlib import Path

# -- Path setup --------------------------------------------------------------
# The week_1 directory is not an installable package, so make its modules
# importable by autodoc.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src" / "week_1"))
sys.path.insert(0, str(_REPO_ROOT / "src"))

# -- Project information -----------------------------------------------------
project = "Neural Control with Adaptive State Estimation"
author = "Jerry Liu, Asmee Mishra, Freddie McElwaine-John, Tahmid Azam"
copyright = "2026, %s" % author
release = "0.1.0"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns: list[str] = []
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

myst_enable_extensions = [
    "dollarmath",  # $...$ and $$...$$ math in .md files
    "colon_fence",  # ::: directive shorthand
]

# -- Autodoc -----------------------------------------------------------------
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "show-inheritance": True,
    "undoc-members": False,
}
autodoc_typehints = "none"
autosummary_generate = True

# -- Intersphinx -------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "seaborn": ("https://seaborn.pydata.org/", None),
}

# -- HTML output -------------------------------------------------------------
html_theme = "furo"
html_static_path = ["_static"]
html_title = project
html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "source_repository": "https://github.com/JerryLiu0911/Neural-Control-with-Adaptive-State-Estimation/",
    "source_branch": "main",
    "source_directory": "docs/source/",
}

# -- sphinx-copybutton -------------------------------------------------------
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True
