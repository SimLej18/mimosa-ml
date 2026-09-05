"""
Build grids of input points from task inputs, and map each task's inputs onto the grid.
"""
from abc import abstractmethod
from dataclasses import replace
from itertools import accumulate

import jax.numpy as jnp
from jax import Array
import equinox as eqx

from mimosa.constants import PAD_INDEX
from mimosa.linalg import lexicographic_sort
from mimosa.kmeans import minibatch_kmeans
from mimosa.mappings import InputMapper, ExactInputMapper, NearestInputMapper
from mimosa.data_structures import Grid, Dataset, ModelConfig

__all__ = [
    "GridBuilder", "UnionGrid", "RegularGrid", "KMeansGrid", "MultiOutputGridBuilder", "MultiOutputUnionGrid",
    "MergedGrid"
]


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


class RegularGrid(GridBuilder):
    """
    Grid of `n_points` evenly-spaced points per input dimension, spanning `bounds`.

    Its points rarely coincide with an observed location, so under `ExactInputMapper` its mappings
    are essentially all `PAD_INDEX`: a grid to predict and plot on rather than to fit on. Merge it
    with a `UnionGrid` (see `MergedGrid`) to get both.

    Attributes
    ----------
    bounds
        `(min, max)` per input dimension, so `len(bounds)` is the grid's input dimension.
    n_points
        Number of points per input dimension, so `n_points ** len(bounds)` in total.
    """
    bounds: tuple[tuple[float, float], ...] = eqx.field(static=True, kw_only=True)
    n_points: int = eqx.field(static=True, default=100)

    def compute_points(self, inputs: None | Array = None, *args, **kwargs) -> Array:
        """
        See `GridBuilder.compute_points`. The points come from `bounds` alone, so `inputs` is only
        checked against them and may be left out to build the grid without a dataset.
        """
        if inputs is not None and inputs.shape[-1] != len(self.bounds):
            raise ValueError(f"Got {len(self.bounds)} bounds for {inputs.shape[-1]} input dimensions.")

        axes = [jnp.linspace(low, high, self.n_points) for low, high in self.bounds]
        # meshgrid(indexing="ij") varies the last axis fastest, so the rows come out lexicographically
        # sorted, as `ExactInputMapper` requires of grid points.
        return jnp.stack(jnp.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, len(self.bounds))


class KMeansGrid(GridBuilder):
    """
    Grid of `n_points` k-means centers over the tasks' pooled input points.

    Where `UnionGrid` sizes the grid at the number of distinct input points -- which continuous
    inputs, or several input dimensions, push past what a dense `(O*G, O*G)` covariance can hold --
    this fixes the budget at `n_points` and spends it minimising the distance from each input point
    to the grid point standing for it. Its points do not coincide with the tasks' own, so it is
    fitted through `NearestInputMapper` rather than `ExactInputMapper`.

    Not jit-compatible: `minibatch_kmeans` runs on the non-NaN input points, whose count depends on
    `inputs`' values rather than on its shape.

    Attributes
    ----------
    prng_key
        `jax.random` PRNG key, forwarded to `minibatch_kmeans`.
    n_points
        Number of centers, i.e. the grid budget.
    batch_size
        Input points drawn per `minibatch_kmeans` step and, unless `input_mapper` carries its own
        `chunk_size`, mapped per step afterwards. Each step holds a `(batch_size, n_points)`
        distance matrix. Defaults to `n_points`; lower it if that does not fit in memory, at some
        cost in center quality, and raise it to spend memory on better centers.

    Examples
    --------
    >>> grid = KMeansGrid(prng_key=key, n_points=256)(dataset.inputs)
    >>> distances = mapping_distances(grid.points, dataset.inputs, grid.mappings)  # what it cost
    """
    prng_key: Array = eqx.field(kw_only=True)
    input_mapper: InputMapper = NearestInputMapper()
    n_points: int = eqx.field(static=True, kw_only=True)
    batch_size: None | int = eqx.field(static=True, default=None)

    @property
    def _resolved_batch_size(self) -> int:
        """
        `batch_size`, or the whole grid budget when it is left unset.
        """
        return self.n_points if self.batch_size is None else self.batch_size

    def compute_points(self, inputs: Array, *args, **kwargs) -> Array:
        """
        See `GridBuilder.compute_points`. Centers are sorted lexicographically, like every other
        builder's points.

        Raises
        ------
        ValueError
            If there are fewer non-NaN input points than centers to place on them.
        """
        points = inputs.reshape(-1, inputs.shape[-1])
        points = points[~jnp.any(jnp.isnan(points), axis=-1)]
        if self.n_points > len(points):
            raise ValueError(f"Cannot place {self.n_points} centers on {len(points)} input points.")

        centers = minibatch_kmeans(self.prng_key, points, self.n_points, self._resolved_batch_size)
        return lexicographic_sort(centers)

    def compute_mappings(self, points: Array, inputs: Array, *args, **kwargs) -> Array:
        """
        See `GridBuilder.compute_mappings`. A `NearestInputMapper` left to size its own chunks
        inherits `batch_size`, so one setting bounds the memory of both passes over the inputs.
        """
        mapper = self.input_mapper
        if isinstance(mapper, NearestInputMapper) and mapper.chunk_size is None:
            mapper = replace(mapper, chunk_size=self._resolved_batch_size)
        return mapper(points, inputs, *args, **kwargs)


def _unique_points(points: Array, return_inverse: bool = False) -> Array | tuple[Array, Array]:
    """
    Sorted, unique, NaN-dropped points from a flat `(n, I)` array. Building block shared by
    `MultiOutputUnionGrid`'s branches (mirrors `UnionGrid.compute_points`' own inline logic).

    Parameters
    ----------
    points
        Points to deduplicate, of shape `(n, I)`.
    return_inverse
        Also return where each of `points`' rows landed in the result.

    Returns
    -------
    unique
        The unique points, sorted lexicographically.
    inverse
        Only when `return_inverse`: index of each of `points`' rows in `unique`, or `PAD_INDEX` for
        a NaN row, which has no unique point to land on.
    """
    kept = ~jnp.any(jnp.isnan(points), axis=-1)
    filtered = points[kept]

    if filtered.shape[-1] == 1:
        unique, inverse = jnp.unique(filtered.reshape(-1), return_inverse=True)
        unique = unique[..., None]
        order = jnp.arange(len(unique))  # jnp.unique already sorts a 1-D array
    else:
        unique, inverse = jnp.unique(filtered, axis=0, return_inverse=True)
        order = jnp.lexsort(unique.T[::-1])  # lexicographic_sort's permutation, needed to fix up `inverse`
        unique = unique[order]

    if not return_inverse:
        return unique

    rank = jnp.zeros(len(order), dtype=int).at[order].set(jnp.arange(len(order)))
    inverse = jnp.full(len(points), PAD_INDEX).at[jnp.flatnonzero(kept)].set(rank[inverse.ravel()])
    return unique, inverse


class MultiOutputGridBuilder(eqx.Module):
    """
    Base class for building a multi-output `Grid`.

    Attributes
    ----------
    n_outputs
        Number of correlated outputs the built grid spans.
    input_mapper
        Maps each task's input points onto the grid points. Defaults to `ExactInputMapper`.
    """
    n_outputs: int = eqx.field(static=True)
    input_mapper: InputMapper = ExactInputMapper()


class MultiOutputUnionGrid(MultiOutputGridBuilder):
    """
    Multi-output analogue of `UnionGrid`: grid formed by the union of the unique input points
    across all tasks, adapted to `config`'s `isotopic_output_in_grid`/`isotopic_output_in_tasks`.

    Only useful for a genuinely multi-output `Dataset` (`dims.O > 1`); a single-output one should
    use `UnionGrid` directly.

    Not jit-compatible: relies on `jnp.unique`, whose output shape depends on `dataset`'s values,
    not just its shape.
    """
    def __call__(self, dataset: Dataset, config: ModelConfig) -> Grid:
        """
        Build the full multi-output Grid from `dataset`, mapping every task's (and every output's)
        input points onto it, block-major over `O*G` (or `sum(G_o)` when heterotopic) like
        `mimosa.synthetic.generate_grid`/`sample_inputs`.

        Parameters
        ----------
        dataset
            Dataset whose inputs (and, when `output_ids` isn't None, output ids) the grid is built
            from.
        config
            Model configuration, used for its `isotopic_output_in_grid`/`isotopic_output_in_tasks`
            fields.

        Returns
        -------
        Grid of points, output_ids, mappings of `dataset`'s inputs onto it, and `n_outputs`. A padding
        point (NaN on every input dimension) maps to `mimosa.mappings.PAD_INDEX`, i.e. outside the
        whole grid rather than merely outside the output-major block it would otherwise fall into.

        Raises
        ------
        ValueError
            If `config.isotopic_output_in_tasks` is True while `config.isotopic_output_in_grid` is
            False -- a heterotopic grid can't share task input locations across outputs.
        """
        if config.isotopic_output_in_tasks and not config.isotopic_output_in_grid:
            raise ValueError("Cannot have isotopic_output_in_tasks with a heterotopic grid.")

        if config.isotopic_output_in_grid:
            return self._shared_grid(dataset.inputs, dataset.output_ids)
        return self._per_output_grid(dataset.inputs, dataset.output_ids, dataset.outputs.shape[0])

    def _shared_grid(self, inputs: Array, output_ids: None | Array) -> Grid:
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
            offset = jnp.repeat(jnp.arange(self.n_outputs), N) * G
            mappings = jnp.tile(base, self.n_outputs) + offset  # (#T, O*N)
            is_pad = jnp.tile(is_pad, self.n_outputs)
        else:
            mappings = base + output_ids * G  # each row already carries its own output id

        mappings = jnp.where(is_pad, PAD_INDEX, mappings)  # outside the whole O*G object, not just its block
        return Grid(points=points, output_ids=None, mappings=mappings, n_outputs=self.n_outputs)

    def _per_output_grid(self, inputs: Array, output_ids: Array, T: int) -> Grid:
        """
        `isotopic_output_in_grid=False`: each output gets its own pool of grid points (`output_ids`
        selects which rows of `inputs` belong to which output).
        """
        inputs = jnp.broadcast_to(inputs, (T,) + inputs.shape[1:])
        output_ids = jnp.broadcast_to(output_ids, (T, inputs.shape[1]))
        is_pad_row = jnp.any(jnp.isnan(inputs), axis=-1)  # (T, oN)

        block_points = [_unique_points(inputs[output_ids == o]) for o in range(self.n_outputs)]
        points = jnp.concatenate(block_points, axis=0)
        grid_output_ids = jnp.concatenate([jnp.full((len(p),), o) for o, p in enumerate(block_points)])
        starts = jnp.concatenate([jnp.zeros((1,), dtype=int), jnp.cumsum(jnp.array([len(p) for p in block_points]))[:-1]])

        mappings = jnp.full((T, inputs.shape[1]), PAD_INDEX)
        for o, pts in enumerate(block_points):
            local = self.input_mapper(pts, inputs)  # (T, oN)
            belongs = (output_ids == o) & ~is_pad_row & (local < len(pts))
            local = jnp.where(belongs, local, 0)  # `starts[o]` below must not be added to PAD_INDEX
            mappings = jnp.where(belongs, local + starts[o], mappings)

        return Grid(points=points, output_ids=grid_output_ids, mappings=mappings,
                    n_outputs=self.n_outputs)


class MergedGrid(Grid):
    """
    Grid pooling several grids' points, and the index of each source in it.

    Points are the sorted unique union of the sources', per output when they carry `output_ids`.
    Equality is exact: a point two sources only nearly agree on stays two points. Mappings are taken
    from the first source that has them and only re-expressed in the merged indexing -- never
    recomputed, never combined -- so argument order says which grid carries the observations.

    Not jit-compatible: relies on `jnp.unique`, whose output shape depends on the grids' values, not
    just their shapes.

    Attributes
    ----------
    sources
        For each merged grid, the index of its own distribution axis in the merged one -- exactly
        the selector `mimosa.data_structures.MultivariateNormal.marginal` takes.

    Examples
    --------
    Fit on the observed locations, but predict and plot on an even grid:

    >>> union, regular = UnionGrid()(inputs), RegularGrid(bounds=((-2.5, 2.5),), n_points=200)(inputs)
    >>> merged = MergedGrid(union, regular)  # keeps `union`'s mappings, so still fits the tasks
    >>> hyperposterior = model.hyperpost(dataset, merged, mixture, parameters)
    >>> plot_clusters(regular, dims, hyperposterior=hyperposterior.marginal(merged.sources[1]))
    """
    sources: tuple[Array, ...] = ()

    def __init__(self, *grids: Grid):
        """
        Parameters
        ----------
        grids
            Grids to merge, agreeing on their `n_outputs` and on whether they label their points.

        Raises
        ------
        ValueError
            If no grid is given, if the grids disagree on either count, or if one maps past the
            axis its own fields describe, i.e. spans more outputs than its `n_outputs` says.
        """
        outputs = {grid.n_outputs for grid in grids}
        labelled = {grid.output_ids is not None for grid in grids}
        if len(outputs) != 1 or len(labelled) != 1:
            raise ValueError("Merge at least one grid, all over the same number of outputs and "
                             "either all labelling their points (`output_ids`) or none.")
        self.n_outputs = outputs.pop()
        # A labelled grid already holds one block per output; an unlabelled one has its outputs
        # share `points`, so its distribution axis is `n_outputs` blocks of it.
        n_blocks = 1 if labelled == {True} else self.n_outputs

        for grid in grids:
            if grid.mappings is not None and jnp.any(
                    (grid.mappings != PAD_INDEX) & (grid.mappings >= n_blocks * len(grid.points))):
                raise ValueError(f"A grid maps past its own {n_blocks} * {len(grid.points)} points. "
                                 "A grid spanning several outputs must say so through `n_outputs`.")

        if labelled == {True}:
            # An output id prepended as a leading input dimension makes the lexicographic dedup pool
            # per output and lay the result out block-major, like `MultiOutputUnionGrid`.
            keyed = [jnp.concatenate([grid.output_ids[:, None], grid.points], axis=1) for grid in grids]
        else:
            keyed = [grid.points for grid in grids]

        unique, inverse = _unique_points(jnp.concatenate(keyed, axis=0), return_inverse=True)
        self.points, self.output_ids = (unique[:, 1:], unique[:, 0].astype(int)) \
            if labelled == {True} else (unique, None)

        bounds = list(accumulate(len(grid.points) for grid in grids))[:-1]
        self.sources = tuple(_over_outputs(index, n_blocks, len(self.points))
                             for index in jnp.split(inverse, bounds))

        self.mappings = next(
            # PAD_INDEX is out of bounds, so the gather clamps it onto the sentinel appended here.
            (jnp.append(index, PAD_INDEX)[grid.mappings]
             for grid, index in zip(grids, self.sources) if grid.mappings is not None),
            None,
        )


def _over_outputs(index: Array, n_blocks: int, n_points: int) -> Array:
    """
    Lift `index`, into one block of `n_points`, into the `n_blocks * n_points` block-major axis.
    """
    if n_blocks == 1:
        return index
    return (index + n_points * jnp.arange(n_blocks)[:, None]).ravel()
