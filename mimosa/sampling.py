"""
Sample from a multivariate normal distribution (used to sample Gaussian process realisations).
"""
import jax.random as jr
import jax.numpy as jnp
from jax import Array

from mimosa import DEFAULT_JITTER


def sample_gp(key: Array, mean: Array, cov: Array, jitter: Array = DEFAULT_JITTER) -> Array:
	"""
	Sample from a multivariate normal distribution, with jitter added to the covariance's
	diagonal for numerical stability.

	Parameters
	----------
	key
		`jax.random` PRNG key.
	mean
		Mean of the distribution. Shape `(..., N)`.
	cov
		Covariance of the distribution. Shape `(..., N, N)`.
	jitter
		Diagonal jitter added to the covariance before sampling.

	Returns
	-------
	Sampled values. Shape `(..., N)`.
	"""
	return jr.multivariate_normal(key, mean, cov + jitter*jnp.eye(cov.shape[-1]))
