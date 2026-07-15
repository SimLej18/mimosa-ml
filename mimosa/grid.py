"""
Build grids of input points from task inputs, and map each task's inputs onto the grid.
"""
from abc import abstractmethod
import jax.numpy as jnp
from jax import Array
import equinox as eqx

from mimosa.linalg import compute_mapping, lexicographic_sort
from mimosa.data_structures import Grid


class GridBuilder(eqx.Module):
    """
    Base class for building a Grid from task inputs.
    """
    def __call__(self, inputs: Array, *args, **kwargs) -> Grid:
        """
        Build the full Grid from `inputs`: `compute_points` followed by `compute_mappings`.

        Parameters
        ----------
        inputs
            Input points of every task.

        Returns
        -------
        Grid of points and mappings of `inputs` onto it.
        """
        points = self.compute_points(inputs, *args, **kwargs)
        mappings = self.compute_mappings(points, inputs, *args, **kwargs)
        return Grid(points=points, mappings=mappings)

    @abstractmethod
    def compute_points(self, inputs: Array, *args, **kwargs) -> Array:
        """
        Build the grid of points spanning `inputs`.

        Parameters
        ----------
        inputs
            Input points of every task.

        Returns
        -------
        Grid points.
        """
        ...

    @abstractmethod
    def compute_mappings(self, points: Array, inputs: Array, *args, **kwargs) -> Array:
        """
        Map each of `inputs`' points to its index in `points`.

        Parameters
        ----------
        points
            Grid points to map `inputs` onto, as returned by `compute_points`.
        inputs
            Input points of every task.

        Returns
        -------
        Index of each of `inputs`' points in `points`.
        """
        ...


class UnionGrid(GridBuilder):
    """
    Grid formed by the union of the unique input points across all tasks.
    """
    def compute_points(self, inputs: Array, *args, **kwargs) -> Array:
        """
        See `GridBuilder.compute_points`.

        Not jit-compatible: relies on `jnp.unique`, whose output shape depends on `inputs`' values,
        not just its shape.
        """
        if inputs.shape[-1] == 1:
            return jnp.sort(jnp.unique(inputs.reshape(-1)))[..., None]  # (G, 1)
        return lexicographic_sort(jnp.unique(inputs.reshape(-1, inputs.shape[-1]), axis=0))

    def compute_mappings(self, points: Array, inputs: Array, *args, **kwargs) -> Array:
        """
        See `GridBuilder.compute_mappings`.
        """
        return compute_mapping(points, inputs)
