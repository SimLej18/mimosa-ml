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

nb_execution_mode = "cache"
nb_execution_timeout = 600          # JIT + Cholesky, sois large
nb_execution_raise_on_error = True  # un exemple cassé fait échouer le build

html_theme = "sphinx_book_theme"
html_theme_options = {
    "repository_url": "https://github.com/SimLej18/mimosa-ml",
    "repository_branch": "main",
    "path_to_docs": "docs",
    "use_repository_button": True,
    "launch_buttons": {"colab_url": "https://colab.research.google.com"},
}

exclude_patterns = ["_build", "**.ipynb_checkpoints"]