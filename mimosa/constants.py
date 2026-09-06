"""
Package-wide numerical constants.

This module imports nothing from `mimosa`, so every other module may import it without risking a
cycle through the package root.
"""

import jax.numpy as jnp

__all__ = ["DEFAULT_JITTER", "PAD_INDEX"]

DEFAULT_JITTER = jnp.asarray(1e-8)

# Index standing for "this input point is not on the grid". Out of bounds for any grid, so a scatter
# drops it and a gather clamps it. Weakly typed, so it adopts the dtype of the mappings it is
# written into. See `mimosa.mappings` for the full contract.
PAD_INDEX: int = int(jnp.iinfo(jnp.int32).max)
