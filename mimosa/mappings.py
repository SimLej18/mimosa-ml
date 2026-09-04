"""
Map the input points of every task onto the points of a grid.

A mapping is the bridge between a `Dataset`'s inputs and a `Grid`: it gives, for each task input
point, the index of the grid point that represents it. `mimosa.grid`'s builders delegate to an
`InputMapper` to produce the `mappings` field of the `Grid` they return.

Padding
-------
A task input point that does not resolve to a grid point -- a NaN point padding a variable-length
task, or a point simply absent from the grid -- maps to `PAD_INDEX`. As it out of bounds, it is dropped
by a scatter (`mimosa.hyperpost`) and clamped by a gather (`mimosa.prediction`, `mimosa.nll`).

To avoid overflow errors, it is recommended not to perform arithmetics (mainly addition) on mappings.
"""
from abc import abstractmethod
from jax import Array, vmap
import equinox as eqx

from mimosa import PAD_INDEX
from mimosa.linalg import find_exact_mappings


class InputMapper(eqx.Module):
    """
    Base class for mapping the input points of every task onto grid points.
    """
    @abstractmethod
    def __call__(self, points: Array, inputs: Array, *args, **kwargs) -> Array:
        """
        Map each of `inputs`' points to its index in `points`.

        Parameters
        ----------
        points
            Grid points to map `inputs` onto, of shape `(G, I)`, as returned by
            `mimosa.grid.GridBuilder.compute_points`.
        inputs
            Input points of every task, of shape `(#T, N, I)`. A padding point (NaN on every input
            dimension) maps to `PAD_INDEX`.

        Returns
        -------
        Index of each of `inputs`' points in `points`, of shape `(#T, N)`.
        """
        ...


class ExactInputMapper(InputMapper):
    """
    Map every input point to the grid point it is exactly equal to, by binary search.

    Default mapper of `mimosa.grid`'s builders. Suited to a grid that contains the tasks' own input
    points, such as one built by `mimosa.grid.UnionGrid`: a point that is not bit-for-bit equal to a
    grid point maps to `PAD_INDEX`, and is thus ignored during training/prediction.

    jit- and vmap-compatible.
    """
    def __call__(self, points: Array, inputs: Array, *args, **kwargs) -> Array:
        """
        See `InputMapper.__call__`.
        """
        return vmap(lambda task_inputs: find_exact_mappings(points, task_inputs))(inputs)
