from pathlib import Path

from jupyter_cache import get_cache

project = "mimosa-ml"
author = "Simon Lejoly"
copyright = "2026, Simon Lejoly"

extensions = [
    "myst_nb",
    "sphinx_togglebutton",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
]

myst_enable_extensions = ["dollarmath", "amsmath", "colon_fence"]

# jupyter-cache creates its directory and SQLite database lazily, on first access. Under
# `sphinx-build -j auto` (Read the Docs' default command) several workers do that at the same time
# and the build breaks (FileExistsError, then "table settings already exists"). Fixing the path and
# initialising the cache here, in the parent process, gets it done before the fork.
nb_execution_cache_path = str(Path(__file__).parent / "_build" / ".jupyter_cache")
get_cache(nb_execution_cache_path).db

nb_execution_mode = "cache"
nb_execution_timeout = 600          # generous: JIT compilation plus Cholesky factorisations
nb_execution_raise_on_error = True  # a broken example fails the build

# jax-tqdm's progress bar (and its TqdmWarning when ipywidgets isn't installed) goes to stderr,
# which is of no use in a static page.
nb_output_stderr = "remove"

html_theme = "sphinx_book_theme"
html_theme_options = {
    "repository_url": "https://github.com/SimLej18/mimosa-ml",
    "repository_branch": "main",
    "path_to_docs": "docs",
    "use_repository_button": True,
    "launch_buttons": {"colab_url": "https://colab.research.google.com"},
}

exclude_patterns = ["_build", "**.ipynb_checkpoints"]