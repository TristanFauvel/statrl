"""Sphinx configuration for the statrl documentation site."""

import os
import sys

# Fallback for local builds without `pip install .`: the repo directory itself is the
# `statrl` package, so put its PARENT on the path to make `import statrl` resolve.
sys.path.insert(0, os.path.abspath("../../.."))

# -- Project information ------------------------------------------------------
project = "statrl"
copyright = "2026, StatisticalRL"
author = "StatisticalRL"
release = "0.0.1"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",   # scikit-learn-style summary tables + per-object pages
    "sphinx.ext.napoleon",      # NumPy- and Google-style docstrings (no numpydoc dep)
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.doctest",       # runs the >>> blocks in the narrative pages
    "sphinx.ext.coverage",      # reports objects autodoc never rendered
    "sphinx_design",            # grid cards on the landing page
]

autosummary_generate = True
templates_path = ["_templates"]

# Nothing is mocked. `pip install .` pulls in gymnasium, so the modules import for
# real — which the doctest builder needs: a mocked `gymnasium.utils.seeding` returns
# a Mock where the environments expect a generator, and every example that resets an
# environment would fail.

# `make coverage` lists documented-but-unrendered objects.
coverage_show_missing_items = True

# Test only explicit `.. doctest::` directives, i.e. the narrative pages. Docstring
# `Examples` sections render as plain `>>>` blocks and would be executed here in an
# empty namespace, where the names their own module defines are not bound. They are
# covered instead by `pytest --doctest-modules src`, which runs each one inside its
# module. Both run in CI, so between them every example is executed.
doctest_test_doctest_blocks = ""

# Napoleon handles the predominant NumPy style; a few docstrings use rst field lists,
# which autodoc parses natively.
napoleon_numpy_docstring = True
napoleon_google_docstring = False

autodoc_default_options = {
    "members": True,
    # Every public object now carries a docstring, so rendering undocumented
    # members would only produce empty stubs that make the reference look
    # more complete than it is.
    "undoc-members": False,
    "show-inheritance": True,
}

# Types come from the hand-written numpydoc `Parameters` sections, which say more
# than a bare annotation can ("ndarray of shape (nbArms,)"). Rendering annotations
# as well would duplicate them, and their unqualified names are ambiguous anyway:
# `BanditAgent` names two different classes across the stochastic settings.
autodoc_typehints = "none"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "gymnasium": ("https://gymnasium.farama.org/", None),
}

exclude_patterns = ["_build"]

# -- HTML output (scikit-learn's theme) --------------------------------------
html_theme = "pydata_sphinx_theme"
html_title = "statrl"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/StatisticalRL/statrl",
            "icon": "fa-brands fa-github",
        },
    ],
    "navbar_align": "left",
    "show_prev_next": True,
    "navigation_with_keys": False,
}
