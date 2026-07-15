"""
Negative log-likelihoods used to optimise cluster and task hyperparameters: a standard
multivariate-normal NLL (`mvn_nll`), and its Magma-algorithm variant with a trace-correction
term accounting for the mean-process's posterior uncertainty (`magma_nll`).
"""

import jax.numpy as jnp
import jax.scipy as jsp
from jax import vmap, Array
import equinox as eqx

from mimosa.linalg import cho_factor, cho_solve
from mimosa.data_structures import Dataset, Grid, Hyperposterior, Hyperprior
from mimosa import DEFAULT_JITTER


def single_output_mvn_nll(value: Array, mean: Array, cov: Array, jitter: Array = DEFAULT_JITTER) -> Array:
	"""
	Negative log-likelihood of a multivariate normal distribution, for a single output.

	Handles padded data: missing points are read from NaNs in `value`.

	Parameters
	----------
	value
		Observed values for this output. Shape `(F*N,)`.
	mean
		Mean of the distribution, for this output. Shape `(F*N,)`.
	cov
		Covariance of the distribution, for this output. Shape `(F*N, F*N)`.
	jitter
		Diagonal jitter added before Cholesky factorization, for numerical stability.

	Returns
	-------
	Negative log-likelihood. Scalar.
	"""
	nan_mask = jnp.isnan(value)  # (F*N,)

	cov = jnp.where(nan_mask[None, :] | nan_mask[:, None], jnp.eye(cov.shape[-1]), cov)
	cov_l = cho_factor(cov, jitter=jitter)  # Shape (F*N, F*N)
	diff = jnp.where(nan_mask, 0., value - mean)  # Shape (F*N,)
	y = cho_solve(cov_l, diff[:, None])[:, 0]  # Shape (F*N,)

	data_fit = jnp.sum(diff * y)
	penalty = 2 * jnp.sum(jnp.log(jnp.diagonal(cov_l)))
	constant = (value.shape[0] - jnp.sum(nan_mask)) * jnp.log(2 * jnp.pi)

	return 0.5 * (data_fit + penalty + constant)


def mvn_nll(values: Array, mean: Array, cov: Array, jitter: Array = DEFAULT_JITTER) -> Array:
	"""
	Negative log-likelihood of a multivariate normal distribution, vmapped across outputs.

	See `single_output_mvn_nll`.

	Parameters
	----------
	values
		Observed values for each output. Shape `(F*N, O)`.
	mean
		Mean of the distribution. Shape `(O, F*N)`, with `O=1` if `shared_output_hps`.
	cov
		Covariance of the distribution. Shape `(O, F*N, F*N)`, with `O=1` if `shared_output_hps`.
	jitter
		Diagonal jitter added before Cholesky factorization, for numerical stability.

	Returns
	-------
	Negative log-likelihood of each output. Shape `(O,)`.
	"""
	return vmap(single_output_mvn_nll, in_axes=(0, 0, 0, None))(values.T, mean, cov, jitter)


def single_output_trace_correction(value: Array, cov: Array, post_cov: Array, jitter: Array = DEFAULT_JITTER) -> Array:
	"""
	Trace correction term that adapts the negative log-likelihood of a MVN to the Magma algorithm,
	for a single output: `0.5 * trace(post_cov @ inv(cov))`.

	Handles padded data: missing points are read from NaNs in `value`.

	Parameters
	----------
	value
		Observed values for this output, used only for their NaN pattern (missing points). Shape `(F*N,)`.
	cov
		Covariance of the task or mean process, for this output. Shape `(F*N, F*N)`.
	post_cov
		Posterior covariance of a specific mean process, for this output. Shape `(F*N, F*N)`.
	jitter
		Diagonal jitter added before Cholesky factorization, for numerical stability.

	Returns
	-------
	Trace correction term. Scalar.
	"""
	nan_mask = jnp.isnan(value)  # (F*N,)
	nan_mask_2d = nan_mask[None, :] | nan_mask[:, None]
	eye = jnp.eye(cov.shape[-1])

	post_cov = jnp.where(nan_mask_2d, eye, post_cov)
	post_cov_l = cho_factor(post_cov, jitter=jitter)  # Shape (F*N, F*N)
	cov = jnp.where(nan_mask_2d, eye, cov)
	cov_l = cho_factor(cov, jitter=jitter)  # Shape (F*N, F*N)

	v = jsp.linalg.solve_triangular(cov_l, post_cov_l, lower=True)
	return 0.5 * (jnp.sum(v ** 2) - jnp.sum(nan_mask))


def trace_correction(values: Array, cov: Array, post_cov: Array, jitter: Array = DEFAULT_JITTER) -> Array:
	"""
	Trace correction term that adapts the negative log-likelihood of a MVN to the Magma algorithm,
	vmapped across outputs.

	See `single_output_trace_correction`.

	Parameters
	----------
	values
		Observed values for each output, used only for their NaN pattern (missing points). Shape `(F*N, O)`.
	cov
		Covariance of the task or mean process. Shape `(O, F*N, F*N)`, with `O=1` if `shared_output_hps`.
	post_cov
		Posterior covariance of a specific mean process. Shape `(O, F*N, F*N)`, with `O=1` if `shared_output_hps`.
	jitter
		Diagonal jitter added before Cholesky factorization, for numerical stability.

	Returns
	-------
	Trace correction term of each output. Shape `(O,)`.
	"""
	O = values.shape[-1]
	cov = jnp.broadcast_to(cov, (O,) + cov.shape[-2:])
	post_cov = jnp.broadcast_to(post_cov, (O,) + post_cov.shape[-2:])
	return vmap(single_output_trace_correction, in_axes=(0, 0, 0, None))(values.T, cov, post_cov, jitter)


def magma_nll(values: Array, mean: Array, cov: Array, post_cov: Array, jitter: Array = DEFAULT_JITTER) -> Array:
	"""
	Full negative log-likelihood of a mean process in the Magma algorithm: `mvn_nll` plus the
	trace-correction term.

	Parameters
	----------
	values
		Observed values for each output. Shape `(F*N, O)`.
	mean
		Posterior mean of a specific mean process. Shape `(O, F*G)`, with `O=1` if `shared_output_hps`.
	cov
		Covariance of the task or mean process. Shape `(O, F*N, F*N)`, with `O=1` if `shared_output_hps`.
	post_cov
		Posterior covariance of a specific mean process. Shape `(O, F*G, F*G)`, with `O=1` if `shared_output_hps`.
	jitter
		Diagonal jitter added before Cholesky factorization, for numerical stability.

	Returns
	-------
	Negative log-likelihood of each output. Shape `(O,)`.
	"""
	return mvn_nll(values, mean, cov, jitter=jitter) + trace_correction(values, cov, post_cov, jitter=jitter)


def clusters_nlls(hyperposterior: Hyperposterior, hyperprior: Hyperprior,
                  jitter: Array = DEFAULT_JITTER) -> Array:
	"""
	Negative log-likelihood of every mean-process, for each output, under its prior.

	Parameters
	----------
	hyperposterior
		Posterior distribution over each mean-process's values at the grid points.
	hyperprior
		Prior distribution over each mean-process's values at the grid points.
	jitter
		Diagonal jitter added before Cholesky factorization, for numerical stability.

	Returns
	-------
	Negative log-likelihood of every mean-process, for each output. Shape `(K, O)`.
	"""
	hyperprior = Hyperprior(
		mean=jnp.broadcast_to(hyperprior.mean, hyperposterior.mean.shape),
		covariance=jnp.broadcast_to(hyperprior.covariance, hyperposterior.covariance.shape)
	)

	return vmap(magma_nll, in_axes=(0, 0, 0, 0, None))(hyperposterior.mean.mT, hyperprior.mean, hyperprior.covariance, hyperposterior.covariance, jitter)


def tasks_nlls(dataset: Dataset, grid: Grid, task_covs: Array, hyperposterior: Hyperposterior,
               jitter: Array = DEFAULT_JITTER) -> Array:
	"""
	Negative log-likelihood of every task, under each mean-process, for each output.

	Parameters
	----------
	dataset
		Dataset whose tasks' likelihoods are computed.
	grid
		Grid of points and mappings of `dataset`'s inputs onto it.
	task_covs
		Task covariance (including noise) of every task. Shape `(T, K, O, F*N, F*N)`, with `T=1` if
		`shared_task_hps`, `K=1` if `shared_cluster_hps` and `O=1` if `shared_output_hps`.
	hyperposterior
		Posterior distribution over each mean-process's values at the grid points.
	jitter
		Diagonal jitter added before Cholesky factorization, for numerical stability.

	Returns
	-------
	Negative log-likelihood of every task, under each mean-process, for each output. Shape `(T, K, O)`.
	"""
	# A nice trick we can use in this function is that it can just be a vmap over `full_nll`, providing only the right
	# portions of post_means and post_covs to each task depending on the mappings.
	task_covs = jnp.broadcast_to(task_covs, (dataset.outputs.shape[0],)+hyperposterior.covariance.shape[:-2]+task_covs.shape[-2:])

	if dataset.inputs.shape[0] == 1:  # no vmap over mappings
		return vmap(
			lambda o, k_t_c: vmap(
				lambda p_m, p_c, t_c:
					magma_nll(
						o,
						p_m[:, grid.mappings[0]],
						t_c,
						p_c[:, grid.mappings[0], :][:, :, grid.mappings[0]],
						jitter))(hyperposterior.mean, hyperposterior.covariance, k_t_c))(dataset.outputs, task_covs)
	return vmap(
		lambda o, m, k_t_c: vmap(
			lambda p_m, p_c, t_c:
				magma_nll(
					o,
					p_m[:, m],
					t_c,
					p_c[:, m, :][:, :, m],
					jitter))(hyperposterior.mean, hyperposterior.covariance, k_t_c))(dataset.outputs, grid.mappings, task_covs)


class ClusterNLL(eqx.Module):
	"""
	Callable wrapper around `clusters_nlls`, as an `equinox.Module`.
	"""
	def __call__(self, hyperposterior: Hyperposterior, hyperprior: Hyperprior, jitter: Array = DEFAULT_JITTER) -> Array:
		"""
		See `clusters_nlls`.
		"""
		return clusters_nlls(hyperposterior, hyperprior, jitter)

class TaskNLL(eqx.Module):
	"""
	Callable wrapper around `tasks_nlls`, as an `equinox.Module`.
	"""
	def __call__(self, dataset: Dataset, grid: Grid, task_covs: Array, hyperposterior: Hyperposterior,
				jitter: Array = DEFAULT_JITTER) -> Array:
		"""
		See `tasks_nlls`.
		"""
		return tasks_nlls(dataset, grid, task_covs, hyperposterior, jitter)
