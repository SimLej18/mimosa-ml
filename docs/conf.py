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

# jupyter-cache initialise son dossier et sa base SQLite paresseusement, au premier accès. Sous
# `sphinx-build -j auto` (la commande par défaut de Read the Docs), plusieurs workers le font en
# même temps et le build casse (FileExistsError, puis "table settings already exists").
# On fixe le chemin et on initialise le cache ici, dans le process parent, avant le fork.
nb_execution_cache_path = str(Path(__file__).parent / "_build" / ".jupyter_cache")
get_cache(nb_execution_cache_path).db

nb_execution_mode = "cache"
nb_execution_timeout = 600          # JIT + Cholesky, sois large
nb_execution_raise_on_error = True  # un exemple cassé fait échouer le build

# La barre de progression de jax-tqdm (et son TqdmWarning quand ipywidgets n'est pas installé)
# passe par stderr : inutile dans une page statique.
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