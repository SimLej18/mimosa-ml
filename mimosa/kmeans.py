"""
Soft k-means in JAX with k-means++ initialisation.

Stopping criterion is a fixed number of iterations (`lax.scan`), and output is "soft":
responsibilities (mixture weights) of shape `(N, k)`, each row summing to 1. Since softmax is
strictly positive, cluster weights never vanish, so there is no "empty cluster" problem
unlike hard k-means.

`soft_kmeans` materialises the full `(N, k)` responsibilities, which is the binding constraint for
a large `k`. `minibatch_kmeans` trades them for `O(batch_size * k)` memory, independent of `N`, and
returns centers alone; assignments for all `N` points come from a separate
`mimosa.linalg.find_nearest_mappings` pass.
"""

from jax import Array, lax, nn, random, vmap
import jax.numpy as jnp

from mimosa.linalg import sq_dists

__all__ = ["kmeanspp_init", "soft_kmeans", "default_stiffness", "minibatch_kmeans"]


def kmeanspp_init(key: Array, X: Array, k: int) -> Array:
	"""
	Sample `k` initial centers via k-means++ (probability proportional to squared distance).

	Requires `N >= k` and the points not all identical (otherwise the sampling distribution
	degenerates to zero everywhere).

	Parameters
	----------
	key
		`jax.random` PRNG key.
	X
		Points to sample centers from. Shape `(N, D)`.
	k
		Number of centers to sample.

	Returns
	-------
	Sampled centers. Shape `(k, D)`.
	"""
	n = X.shape[0]
	key, sub = random.split(key)
	c0 = X[random.randint(sub, (), 0, n)]
	d2 = jnp.sum((X - c0) ** 2, axis=1)  # D² to the nearest center

	def pick(carry, _):
		key, d2 = carry
		key, sub = random.split(key)
		i = random.choice(sub, n, p=d2 / d2.sum())
		d2 = jnp.minimum(d2, jnp.sum((X - X[i]) ** 2, axis=1))
		return (key, d2), X[i]

	_, rest = lax.scan(pick, (key, d2), None, length=k - 1)
	return jnp.concatenate([c0[None], rest], axis=0)  # (k, D)


def soft_kmeans(
	key: Array, X: Array, k: int, n_iters: int = 50, stiffness: float = 1.0, n_restarts: int = 8
) -> tuple[Array, Array]:
	"""
	Soft k-means (isotropic EM) with k-means++ initialisation, best of `n_restarts` runs.

	Parameters
	----------
	key
		`jax.random` PRNG key (split into one k-means++ init per restart).
	X
		Points to cluster. Shape `(N, D)`.
	k
		Number of clusters.
	n_iters
		Number of Lloyd iterations.
	stiffness
		Softmax inverse-temperature β. β -> +inf recovers hard k-means; β -> 0 gives uniform
		memberships. Tune relative to the scale of `X` (β acts on squared distances).
	n_restarts
		Number of independently initialised runs. All run in parallel under `vmap`, and the one
		reaching the lowest free energy is returned.

	Returns
	-------
	centers
		Cluster centers of the best restart. Shape `(k, D)`.
	resp
		Its responsibilities, each row summing to 1. Shape `(N, k)`.
	"""

	def fit(key: Array) -> tuple[Array, Array, Array]:
		def lloyd(C, _):
			R = nn.softmax(-stiffness * sq_dists(X, C), axis=1)  # E-step: (N, k)
			C = (R.T @ X) / R.sum(0)[:, None]  # M-step: (k, D)
			return C, None

		C, _ = lax.scan(lloyd, kmeanspp_init(key, X, k), None, length=n_iters)
		logits = -stiffness * sq_dists(X, C)
		# Free energy the E-step attains, up to the constant factor 1/β: the objective Lloyd
		# descends here, so restarts are ranked by what they actually optimise. Hard inertia
		# (Σ min d²) only agrees with it as β -> +inf.
		return C, nn.softmax(logits, axis=1), -nn.logsumexp(logits, axis=1).sum()

	C, R, free_energy = vmap(fit)(random.split(key, n_restarts))
	best = jnp.argmin(free_energy)
	return C[best], R[best]


def default_stiffness(X: Array, k: int) -> Array:
	"""
	Stiffness tied to the spacing of a `k`-point grid over `X`: `1 / spacing**2`, where
	`spacing = width / k**(1/D)` and `width` is the geometric-mean side of `X`'s bounding box.

	A stiffness fixed to the scale of `X` alone makes the softmax span ever more centers as `k`
	grows, until every center is pulled toward the global mean and the clustering collapses. Tying
	the softmax kernel width to the distance between neighbouring centers instead keeps a center's
	responsibilities local across a range of `k`.

	Parameters
	----------
	X
		Points to cluster. Shape `(N, D)`.
	k
		Number of clusters the stiffness is tuned for.

	Returns
	-------
	Scalar stiffness.
	"""
	D = X.shape[-1]
	width = jnp.prod(jnp.max(X, axis=0) - jnp.min(X, axis=0)) ** (1.0 / D)
	spacing = width / k ** (1.0 / D)
	return 1.0 / spacing**2


def minibatch_kmeans(
	key: Array, X: Array, k: int, batch_size: int, n_iters: int = 100, stiffness: Array | float | None = None
) -> Array:
	"""
	Mini-batch soft k-means (Sculley 2010): `n_iters` E/M steps, each on a fresh random batch of
	`batch_size` rows of `X`, so peak memory is `O(batch_size * k)` regardless of `N`.

	Each center keeps a running total responsibility `v` and moves a fraction
	`batch_responsibility / v` of the way to the batch's responsibility-weighted mean. The step size
	decays as `v` accumulates, so the centers converge rather than tracking whichever batch landed
	last.

	Only `batch_size * n_iters` points are ever visited, so runtime is flat in `N` and the
	approximation is set by `batch_size`, not by how much data exists. `batch_size ~ 2 * k` is a
	reasonable rule; under-sizing costs in proportion to `k / batch_size`.

	k-means++ initialisation (`kmeanspp_init`) still reads all of `X`, but at `O(N)` per step rather
	than `O(N*k)`.

	jit- and vmap-compatible for a fixed `X.shape`, `k`, `batch_size` and `n_iters`.

	Parameters
	----------
	key
		`jax.random` PRNG key, split between initialisation and the batch stream.
	X
		Points to cluster. Shape `(N, D)`.
	k
		Number of clusters.
	batch_size
		Rows sampled, with replacement, per iteration.
	n_iters
		Number of E/M steps.
	stiffness
		Softmax inverse-temperature β, as in `soft_kmeans`. `None` uses `default_stiffness(X, k)`.

	Returns
	-------
	Cluster centers. Shape `(k, D)`.
	"""
	beta = default_stiffness(X, k) if stiffness is None else stiffness
	key, init_key = random.split(key)

	def step(carry, batch_key):
		C, v = carry
		Xb = X[random.randint(batch_key, (batch_size,), 0, X.shape[0])]
		R = nn.softmax(-beta * sq_dists(Xb, C), axis=1)  # E-step: (batch_size, k)
		weight = R.sum(0)  # (k,)
		v = v + weight
		# At a large β the softmax underflows a far-away center's weight to exactly 0.0. Both
		# weights are non-negative, so a zero `v` implies a zero `weight` and hence a zero step:
		# guarding the denominators leaves such a center unmoved rather than NaN.
		batch_mean = (R.T @ Xb) / jnp.where(weight > 0, weight, 1.0)[:, None]
		step_size = weight / jnp.where(v > 0, v, 1.0)
		return (C + step_size[:, None] * (batch_mean - C), v), None  # M-step: running weighted mean

	init = (kmeanspp_init(init_key, X, k), jnp.zeros(k, dtype=X.dtype))
	(C, _), _ = lax.scan(step, init, random.split(key, n_iters))
	return C
