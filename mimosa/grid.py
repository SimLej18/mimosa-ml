"""
Build grids of input points from task inputs, and map each task's inputs onto the grid.
"""
from abc import abstractmethod
import jax.numpy as jnp
from jax import Array
import equinox as eqx

from mimosa import PAD_INDEX
from mimosa.linalg import lexicographic_sort
from mimosa.mappings import InputMapper, ExactInputMapper
from mimosa.data_structures import Grid, Dataset, ModelConfig


class GridBuilder(eqx.Module):
    """
    Base class for building a Grid from task inputs.

    Attributes
    ----------
    input_mapper
        Maps each task's input points onto the grid points. Defaults to `ExactInputMapper`.
    """
    input_mapper: InputMapper = ExactInputMapper()

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
            Input points of every task. A point padding a variable-length task is NaN on every input
            dimension, and must be excluded from the grid.

        Returns
        -------
        Grid points.
        """
        ...

    def compute_mappings(self, points: Array, inputs: Array, *args, **kwargs) -> Array:
        """
        Map each of `inputs`' points to its index in `points`, with `input_mapper`.

        Parameters
        ----------
        points
            Grid points to map `inputs` onto, as returned by `compute_points`.
        inputs
            Input points of every task. A padding point (NaN on every input dimension) maps to
            `mimosa.mappings.PAD_INDEX`.

        Returns
        -------
        Index of each of `inputs`' points in `points`.
        """
        return self.input_mapper(points, inputs, *args, **kwargs)


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
        points = inputs.reshape(-1, inputs.shape[-1])
        points = points[~jnp.any(jnp.isnan(points), axis=-1)]
        if points.shape[-1] == 1:
            return jnp.sort(jnp.unique(points.reshape(-1)))[..., None]  # (G, 1)
        return lexicographic_sort(jnp.unique(points, axis=0))


def _unique_points(points: Array) -> Array:
    """
    Sorted, unique, NaN-dropped points from a flat `(n, I)` array. Building block shared by
    `MultiOutputUnionGrid`'s branches (mirrors `UnionGrid.compute_points`' own inline logic).
    """
    points = points[~jnp.any(jnp.isnan(points), axis=-1)]
    if points.shape[-1] == 1:
        return jnp.sort(jnp.unique(points.reshape(-1)))[..., None]
    return lexicographic_sort(jnp.unique(points, axis=0))


class MultiOutputUnionGrid(eqx.Module):
    """
    Multi-output analogue of `UnionGrid`: grid formed by the union of the unique input points
    across all tasks, adapted to `config`'s `isotopic_output_in_grid`/`isotopic_output_in_tasks`.

    Only useful for a genuinely multi-output `Dataset` (`dims.O > 1`); a single-output one should
    use `UnionGrid` directly.

    Not jit-compatible: relies on `jnp.unique`, whose output shape depends on `dataset`'s values,
    not just its shape.

    Attributes
    ----------
    input_mapper
        Maps each task's input points onto the grid points. Defaults to `ExactInputMapper`.
    """
    input_mapper: InputMapper = ExactInputMapper()

    def __call__(self, dataset: Dataset, config: ModelConfig) -> Grid:
        """
        Build the full multi-output Grid from `dataset`, mapping every task's (and every output's)
        input points onto it, block-major over `O*G` (or `sum(G_o)` when heterotopic) like
        `mimosa.synthetic.generate_grid`/`sample_inputs`.

        Parameters
        ----------
        dataset
            Dataset whose inputs (and, when `output_ids` isn't None, output ids) the grid is built
            from. `O` is inferred from `dataset` itself: `dataset.outputs.shape[1] //
            dataset.inputs.shape[1]` if `dataset.output_ids is None` (every output shares the same
            input locations, so it can't be read off `output_ids`), else `dataset.output_ids.max() + 1`.
        config
            Model configuration, used for its `isotopic_output_in_grid`/`isotopic_output_in_tasks`
            fields.

        Returns
        -------
        Grid of points, output_ids and mappings of `dataset`'s inputs onto it. A padding point
        (NaN on every input dimension) maps to `mimosa.mappings.PAD_INDEX`, i.e. outside the whole
        grid rather than merely outside the output-major block it would otherwise fall into.

        Raises
        ------
        ValueError
            If `config.isotopic_output_in_tasks` is True while `config.isotopic_output_in_grid` is
            False -- a heterotopic grid can't share task input locations across outputs.
        """
        if config.isotopic_output_in_tasks and not config.isotopic_output_in_grid:
            raise ValueError("Cannot have isotopic_output_in_tasks with a heterotopic grid.")

        if dataset.output_ids is None:
            n_outputs = dataset.outputs.shape[1] // dataset.inputs.shape[1]
        else:
            n_outputs = int(dataset.output_ids.max()) + 1

        if config.isotopic_output_in_grid:
            return self._shared_grid(dataset.inputs, dataset.output_ids, n_outputs)
        return self._per_output_grid(dataset.inputs, dataset.output_ids, n_outputs, dataset.outputs.shape[0])

    def _shared_grid(self, inputs: Array, output_ids: None | Array, n_outputs: int) -> Grid:
        """
        `isotopic_output_in_grid=True`: every output shares one pool of grid points, whether or
        not tasks also share input locations across outputs (`output_ids is None` or not).
        """
        points = _unique_points(inputs.reshape(-1, inputs.shape[-1]))
        G = len(points)

        base = self.input_mapper(points, inputs)  # (#T, N or O*N)
        is_pad = jnp.any(jnp.isnan(inputs), axis=-1)  # same shape as base
        base = jnp.where(is_pad, 0, base)  # offsets below must not be added to PAD_INDEX

        if output_ids is None:
            # isotopic_output_in_tasks: the same N points are reused for every output, block-major.
            N = inputs.shape[1]
            offset = jnp.repeat(jnp.arange(n_outputs), N) * G
            mappings = jnp.tile(base, n_outputs) + offset  # (#T, O*N)
            is_pad = jnp.tile(is_pad, n_outputs)
        else:
            mappings = base + output_ids * G  # each row already carries its own output id

        mappings = jnp.where(is_pad, PAD_INDEX, mappings)  # outside the whole O*G object, not just its block
        return Grid(points=points, output_ids=None, mappings=mappings)

    def _per_output_grid(self, inputs: Array, output_ids: Array, n_outputs: int, T: int) -> Grid:
        """
        `isotopic_output_in_grid=False`: each output gets its own pool of grid points (`output_ids`
        selects which rows of `inputs` belong to which output).
        """
        inputs = jnp.broadcast_to(inputs, (T,) + inputs.shape[1:])
        output_ids = jnp.broadcast_to(output_ids, (T, inputs.shape[1]))
        is_pad_row = jnp.any(jnp.isnan(inputs), axis=-1)  # (T, oN)

        block_points = [_unique_points(inputs[output_ids == o]) for o in range(n_outputs)]
        points = jnp.concatenate(block_points, axis=0)
        grid_output_ids = jnp.concatenate([jnp.full((len(p),), o) for o, p in enumerate(block_points)])
        starts = jnp.concatenate([jnp.zeros((1,), dtype=int), jnp.cumsum(jnp.array([len(p) for p in block_points]))[:-1]])

        mappings = jnp.full((T, inputs.shape[1]), PAD_INDEX)
        for o, pts in enumerate(block_points):
            local = self.input_mapper(pts, inputs)  # (T, oN)
            belongs = (output_ids == o) & ~is_pad_row & (local < len(pts))
            local = jnp.where(belongs, local, 0)  # `starts[o]` below must not be added to PAD_INDEX
            mappings = jnp.where(belongs, local + starts[o], mappings)

        return Grid(points=points, output_ids=grid_output_ids, mappings=mappings)
