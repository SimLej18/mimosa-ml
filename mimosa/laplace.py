"""
Laplace matching (Hennig-style): a preprocessing step turning non-Gaussian observations into
Gaussian ones, so the rest of the pipeline can assume a Gaussian likelihood everywhere.

Each approximator groups observations, fits the conjugate posterior of the matching
exponential-family likelihood per group (Beta for Binomial, Gamma for Poisson/Exponential), and
moment-matches it with a Gaussian in the transformed, natural-parameter space. `wrap` returns a new
`Dataset` with those Gaussian means in `outputs` and variances in `known_output_noise`;
`unwrap` maps samples back to the original space. Feed those variances to the model with
`mimosa.synthetic.known_noise_kernel`.

Grouping is controlled by `interval`: 0 treats every observation as its own group; any other value
bins the inputs on a lattice shared by every task.

"""

from abc import abstractmethod

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.nn
from jax import Array
from jaxtyping import Bool, Float, Int

from mimosa.data_structures import Dataset


class LaplaceApproximator(eqx.Module):
    """
    Base for the three moment-matching families: groups observations, reduces each group to
    sufficient statistics `(total, count)`, rebuilds the Dataset. Subclasses supply `link`
    (`(total, count) -> (mean, variance)`) and its elementwise inverse `unwrap`.

    Attributes
    ----------
    interval
        Aggregation bin width. 0 disables binning: every point is its own group, returned
        unchanged. Otherwise bin edges of width `interval` anchor at the global (cross-task) input
        minimum, so the lattice -- and the aggregated points -- is identical for every task. Empty
        bins are dropped; a bin empty for one task only becomes NaN padding there.
    use_bin_centers
        If True (default), a bin is represented by its center; if False, by the mean of the inputs
        falling in it, pooled across tasks so the representative point stays shared. Ignored when
        `interval == 0`.
    prior_count
        Pseudo-observations added to the conjugate posterior, i.e. the prior itself:
        `Beta(prior_count, prior_count)` for binomial data, `Gamma(prior_count, prior_count)` for
        the others. 1/2 (default) is Jeffreys' prior, 1 the uniform one.

        Also the numerical guard against a saturated group (all successes, or no event: a
        parameter at 0, where the link diverges). Use `1e-6` to match the reference R
        implementation.

    Notes
    -----
    With `I > 1` inputs the lattice is the product of the per-dimension bin counts, addressed by
    raveled per-dimension indices. Kept bins never exceed the number of observed points, but that
    address space grows as `prod(n_bins)`, so a small `interval` on many dimensions can overflow
    it.
    """
    interval: float = eqx.field(static=True)
    use_bin_centers: bool = eqx.field(static=True, default=True)
    prior_count: float = eqx.field(static=True, default=0.5)

    @abstractmethod
    def link(self, total: Float[Array, "..."], count: Float[Array, "..."]
             ) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
        """
        Moment-match one group's conjugate posterior with a Gaussian in the transformed space.

        Parameters
        ----------
        total
            Sum of the group's observed values.
        count
            Number of observed values in the group.

        Returns
        -------
        mean, variance
        """
        ...

    @abstractmethod
    def unwrap(self, x: Float[Array, "..."]) -> Float[Array, "..."]:
        """
        Map values from the transformed space back to the observation space, elementwise.

        Applies to samples, not a Dataset: the inverse link is non-linear so it does not commute
        with averaging.

        Parameters
        ----------
        x
            Transformed-space values, any shape.

        Returns
        -------
        `x` mapped back to observation space, same shape.
        """
        ...

    def wrap(self, dataset: Dataset) -> Dataset:
        """
        Turn raw observations into Gaussian ones.

        Not jit-compatible when `interval != 0`: which bins are occupied depends on the input
        values, not just their shape. This is preprocessing, so it runs once, eagerly.

        Parameters
        ----------
        dataset
            Dataset of raw observations.

        Returns
        -------
        Dataset whose `outputs` are moment-matched Gaussian means, `known_output_noise` their
        variances.

        Raises
        ------
        NotImplementedError
            If `interval != 0` and outputs don't share their input locations
            (`dataset.output_ids is not None`).
        """
        outputs = dataset.outputs

        if self.interval == 0.0:
            observed = ~jnp.isnan(outputs)
            mean, variance = self._masked_link(jnp.nan_to_num(outputs), observed.astype(outputs.dtype))
            return Dataset(inputs=dataset.inputs, outputs=mean, known_output_noise=variance,
                           output_ids=dataset.output_ids)

        if dataset.output_ids is not None:
            raise NotImplementedError(
                "Binned Laplace matching requires outputs sharing their input locations "
                "(`dataset.output_ids is None`). Use `interval=0` for heterotopic outputs.")

        T, ON, C = outputs.shape
        N = dataset.inputs.shape[1]
        O = ON // N

        # O correlated outputs share one input block: folding O into the task axis reuses the
        # single-output path, since each (task, output) pair is independent groups over the same
        # lattice.
        R = T * O
        inputs = jnp.repeat(jnp.broadcast_to(dataset.inputs, (T,) + dataset.inputs.shape[1:]), O, axis=0)
        values = outputs.reshape(R, N, C)
        observed = ~jnp.isnan(values)

        points, bin_ix, n_kept = self._lattice(inputs, observed.any(axis=-1))
        total, count = self._scatter(bin_ix, observed, values, R, n_kept)

        mean, variance = self._masked_link(total, count)
        return Dataset(inputs=points[None, ...],
                       outputs=mean.reshape(T, O * n_kept, C),
                       known_output_noise=variance.reshape(T, O * n_kept, C))

    @staticmethod
    def _scatter(bin_ix: Int[Array, "R N"], observed: Bool[Array, "R N C"],
                 values: Float[Array, "R N C"], R: int, n_kept: int
                ) -> tuple[Float[Array, "R B C"], Float[Array, "R B C"]]:
        """
        Reduce every (row, bin) group to its sufficient statistics `(total, count)` in one pass.

        Each point's (row, bin) address is flattened into a single global segment id, and `total`
        and `count` are computed together as the two halves of one `segment_sum` -- half the
        scatter cost of accumulating them separately. The trailing, out-of-range bin (see
        `_lattice`) is sliced off after.
        """
        C = values.shape[-1]
        seg = jnp.arange(R)[:, None] * (n_kept + 1) + bin_ix
        stat = jnp.concatenate([jnp.where(observed, values, 0.0), observed.astype(values.dtype)], axis=-1)
        acc = jax.ops.segment_sum(stat.reshape(-1, 2 * C), seg.reshape(-1), num_segments=R * (n_kept + 1))
        acc = acc.reshape(R, n_kept + 1, 2 * C)[:, :n_kept]
        return acc[..., :C], acc[..., C:]

    def _lattice(self, inputs: Float[Array, "R N I"], usable: Bool[Array, "R N"]
                 ) -> tuple[Float[Array, "B I"], Int[Array, "R N"], int]:
        """
        Bin `inputs` on the lattice shared by every row, dropping bins nobody occupies.

        Parameters
        ----------
        inputs
            Input points of every (task, output) row.
        usable
            Whether each point carries at least one observed channel.

        Returns
        -------
        points
            Representative point of each kept bin, shape `(B, I)`.
        bin_ix
            Bin index of each point, shape `(R, N)`. Unaggregated points (padded, or with a NaN
            input) get `B`, an out-of-range dump index that callers slice off.
        n_kept
            Number of kept bins, `B`.
        """
        usable = usable & ~jnp.isnan(inputs).any(axis=-1)
        clean = jnp.where(usable[..., None], inputs, jnp.nan)

        flat_inputs = clean.reshape(-1, clean.shape[-1])
        lo, hi = jnp.nanmin(flat_inputs, axis=0), jnp.nanmax(flat_inputs, axis=0)
        n_bins = tuple(max(int(n), 1) for n in jnp.ceil((hi - lo) / self.interval))

        # Bins are half-open, so a point sitting exactly on `hi` would open a degenerate bin of
        # its own; the clip folds it back into the last full one instead.
        per_dim = jnp.clip(jnp.floor((clean - lo) / self.interval).astype(int),
                           0, jnp.asarray(n_bins) - 1)
        address = jnp.ravel_multi_index(tuple(per_dim[..., d] for d in range(len(n_bins))),
                                        n_bins, mode="clip")
        address = jnp.where(usable, address, -1)

        # `jnp.unique` sorts ascending, so an unusable point's dump address (-1) is always the
        # first unique value when present: one call gives both the occupied bins and, via
        # `return_inverse`, each point's index into them (shifted by the dropped dump slot).
        vals, inv = jnp.unique(address, return_inverse=True)
        n_dump = int(jnp.sum(vals < 0))  # 0 or 1: `jnp.unique` sorts, so the dump address comes first
        occupied = vals[n_dump:]
        n_kept = int(occupied.shape[0])
        bin_ix = jnp.where(usable, inv.reshape(usable.shape) - n_dump, n_kept)

        if self.use_bin_centers:
            coords = jnp.stack(jnp.unravel_index(occupied, n_bins), axis=-1).astype(inputs.dtype)
            return lo + (coords + 0.5) * self.interval, bin_ix, n_kept

        # Pooled across rows, so the representative point stays the same for every task.
        stat = jnp.concatenate([jnp.nan_to_num(clean), usable[..., None].astype(inputs.dtype)], axis=-1)
        acc = jax.ops.segment_sum(stat.reshape(-1, stat.shape[-1]), bin_ix.reshape(-1),
                                  num_segments=n_kept + 1)[:n_kept]
        return acc[:, :-1] / acc[:, -1:], bin_ix, n_kept

    def _masked_link(self, total: Float[Array, "T ON C"], count: Float[Array, "T ON C"]
                     ) -> tuple[Float[Array, "T ON C"], Float[Array, "T ON C"]]:
        """
        Apply `link`, then restore NaN padding wherever a group holds no observation at all.

        `link` stays finite on an empty group (`prior_count` alone keeps it so), so emptiness must
        be masked explicitly -- otherwise a missing point would silently become a real observation
        carrying nothing but the prior.
        """
        mean, variance = self.link(total, count)
        empty = count == 0
        return jnp.where(empty, jnp.nan, mean), jnp.where(empty, jnp.nan, variance)


class IdentityLaplaceApproximator(eqx.Module):
    """
    No-op approximator, for data that is already Gaussian. Honours the same `wrap`/`unwrap`
    contract as `LaplaceApproximator` but carries no binning and no prior, so it does not inherit
    from it.
    """

    def wrap(self, dataset: Dataset) -> Dataset:
        """See `LaplaceApproximator.wrap`. Returns `dataset` unchanged."""
        return dataset

    def unwrap(self, x: Float[Array, "..."]) -> Float[Array, "..."]:
        """See `LaplaceApproximator.unwrap`. Returns `x` unchanged."""
        return x


class BinomialLaplaceApproximator(LaplaceApproximator):
    """
    Laplace matching for Binomial observations (0/1, or boolean cast to float): a Beta posterior
    over the success probability, moment-matched with a Gaussian over its log-odds. A group's
    successes are `total`, its failures `count - total`.
    """
    def link(self, total: Float[Array, "..."], count: Float[Array, "..."]
             ) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
        """See `LaplaceApproximator.link`. `Beta(alpha, beta)`, matched over log-odds."""
        alpha = total + self.prior_count
        beta = count - total + self.prior_count
        return jnp.log(alpha) - jnp.log(beta), (count + 2 * self.prior_count) / (alpha * beta)

    def unwrap(self, x: Float[Array, "..."]) -> Float[Array, "..."]:
        """See `LaplaceApproximator.unwrap`. Log-odds back to a probability."""
        return jax.nn.sigmoid(x)


class GammaLaplaceApproximator(LaplaceApproximator):
    """
    Base for the two families whose conjugate posterior is a Gamma -- Poisson and Exponential.
    They share the link exactly and differ only in how `unwrap` reads the transformed value.
    """
    def link(self, total: Float[Array, "..."], count: Float[Array, "..."]
             ) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
        """See `LaplaceApproximator.link`. `Gamma(total, count)`, matched over log-rate."""
        shape = total + self.prior_count
        return jnp.log(shape) - jnp.log(count + self.prior_count), 1.0 / shape


class PoissonLaplaceApproximator(GammaLaplaceApproximator):
    """Laplace matching for Poisson counts: a Gamma posterior over the rate, matched over its log."""

    def unwrap(self, x: Float[Array, "..."]) -> Float[Array, "..."]:
        """See `LaplaceApproximator.unwrap`. Log-rate back to a rate."""
        return jnp.exp(x)


class ExponentialLaplaceApproximator(GammaLaplaceApproximator):
    """
    Laplace matching for Exponential durations: a Gamma posterior over the rate, matched over its
    log. `unwrap` returns the mean duration, i.e. the inverse of that rate.
    """
    def unwrap(self, x: Float[Array, "..."]) -> Float[Array, "..."]:
        """See `LaplaceApproximator.unwrap`. Log-rate back to a mean duration."""
        return jnp.exp(-x)
