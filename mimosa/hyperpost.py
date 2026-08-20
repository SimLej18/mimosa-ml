"""
Compute the hyperposterior: the posterior distribution over each mean-process's values at the grid
points, given a Dataset, a soft-clustering Mixture, and model Parameters.

`single_channel_hyperpost` computes it for one mean-process and one channel dimension;
`single_cluster_hyperpost` vmaps it across channel dimensions; `hyperpost` vmaps it across mean-processes.
"""

import jax.numpy as jnp
from jax import Array, vmap
import equinox as eqx

from mimosa.linalg import cho_factor, cho_solve
from mimosa.data_structures import Parameters, Dataset, Grid, Mixture, Hyperprior, Hyperposterior
from mimosa import DEFAULT_JITTER


def single_channel_hyperpost(outputs: Array, grid: Grid, responsibilities: Array,
                            hyperprior: Hyperprior, task_covs: Array,
                            jitter: Array = DEFAULT_JITTER) -> Hyperposterior:
	"""
	Compute the hyperposterior for a single mean-process and a single channel dimension.

	Parameters
	----------
	outputs
		Output values of every task, for this channel. Shape `(T, O*N)`.
	grid
		Grid of points and mappings of dataset's inputs onto it.
	responsibilities
		Responsibility of each task towards this mean-process. Shape `(T,)`.
	hyperprior
		Prior distribution over this mean-process's values at the grid points.
	task_covs
		Task covariance (including noise) of every task, for this channel. Shape `(T, O*N, O*N)`.
	jitter
		Diagonal jitter added before Cholesky factorizations, for numerical stability.

	Returns
	-------
	Hyperposterior over this mean-process's values at the grid points, for this channel.
	"""
	big_eye = jnp.eye(hyperprior.covariance.shape[-1])  # hyperprior is always dense over O*G, unlike grid.points which may collapse to G
	small_eye = jnp.eye(outputs.shape[-1])

	# Cluster covariance inversion
	cluster_cov_l = cho_factor(hyperprior.covariance, jitter=jitter)  # Shape (G, G)
	cluster_cov_inv = cho_solve(cluster_cov_l, big_eye)

	# Task covariances inversion -- masked from this channel's own NaNs (task_covs itself is always NaN-free)
	nan_mask = jnp.isnan(outputs)  # (T, O*N)
	nan_mask_2d = nan_mask[:, None, :] | nan_mask[:, :, None]  # (T, O*N, O*N)
	task_covs_padded = jnp.where(nan_mask_2d, small_eye, task_covs)  # Padding
	task_covs_l = cho_factor(task_covs_padded, jitter=jitter)  # Shape (T, O*N, O*N)  with T=1 if shared_inputs_in_tasks and shared_task_hps
	task_covs_inv = cho_solve(task_covs_l, jnp.broadcast_to(small_eye, task_covs_l.shape))
	task_covs_inv -= jnp.where(nan_mask_2d, task_covs_inv, 0)  # Correction on the diagonal
	task_covs_inv *= responsibilities[:, None, None]  # Apply mixture coefficients

	# Mapping to full grid
	mappings = jnp.broadcast_to(grid.mappings, (outputs.shape[0], grid.mappings.shape[1]))
	task_covs_inv = jnp.zeros_like(big_eye).at[mappings[:, :, None], mappings[:, None, :]].add(task_covs_inv)  # Shape (O*G, O*G)

	# Sum mean and task covariances and compute Cholesky factor of the posterior covariance
	post_covs_inv = cho_factor(cluster_cov_inv + task_covs_inv, jitter=jitter)  # Shape (O*G, O*G)
	post_cov = cho_solve(post_covs_inv, big_eye)  # Shape (O*G, O*G)

	# --- Posterior mean ---
	# Compute prior means
	prior_mean = cho_solve(cluster_cov_l, hyperprior.mean)  # Shape (O*G)
	task_means = cho_solve(jnp.broadcast_to(task_covs_l, (outputs.shape[0],)+task_covs_l.shape[1:]), jnp.nan_to_num(outputs))  # Shape (T, O*N)
	task_means *= responsibilities[:, None]  # Shape (T, O*N)
	task_means = jnp.zeros(big_eye.shape[0]).at[mappings].add(task_means)  # Shape (O*G)

	full_mean = prior_mean + task_means  # Shape (O*G)
	post_mean = cho_solve(post_covs_inv, full_mean)

	return Hyperposterior(mean=post_mean, covariance=post_cov)


def single_cluster_hyperpost(outputs: Array, grid: Grid, responsibilities: Array,
                             hyperprior: Hyperprior, task_covs: Array,
                             jitter: Array = DEFAULT_JITTER) -> Hyperposterior:
	"""
	Compute the hyperposterior for a single mean-process, vmapped across channel dimensions.

	Parameters
	----------
	outputs
		Output values of every task, for this mean-process. Shape `(T, O*N, C)`.
	grid
		Grid of points and mappings of `dataset`'s inputs onto it.
	responsibilities
		Responsibility of each task towards this mean-process. Shape `(T,)`.
	hyperprior
		Prior distribution over this mean-process's values at the grid points, batched over channel
		dimensions (or a single shared prior, if `shared_channel_hps`).
	task_covs
		Task covariance (including noise) of every task, batched over channel dimensions
		(or shared across channels, if no `channel_hp_in_tasks`).
	jitter
		Diagonal jitter added before Cholesky factorizations, for numerical stability.

	Returns
	-------
	Hyperposterior over this mean-process's values at the grid points, batched over channel dimensions.
	"""
	if hyperprior.mean.shape[0] == 1:  # Shared channel HPs
		if task_covs.shape[1] == 1:  # No channel_hp_in_tasks
			f = vmap(single_channel_hyperpost, in_axes=(0, None, None, None, None, None))
			return f(outputs.T.mT, grid, responsibilities, hyperprior[0], task_covs.swapaxes(0, 1)[0], jitter)
		else:
			f = vmap(single_channel_hyperpost, in_axes=(0, None, None, None, 0, None))
			return f(outputs.T.mT, grid, responsibilities, hyperprior[0], task_covs.swapaxes(0, 1), jitter)

	else:  # Distinct channel HPs
		if task_covs.shape[1] == 1:  # No channel_hp_in_tasks
			f = vmap(single_channel_hyperpost, in_axes=(0, None, None, 0, None, None))
			return f(outputs.T.mT, grid, responsibilities, hyperprior, task_covs.swapaxes(0, 1)[0], jitter)
		else:
			f = vmap(single_channel_hyperpost, in_axes=(0, None, None, 0, 0, None))
			return f(outputs.T.mT, grid, responsibilities, hyperprior, task_covs.swapaxes(0, 1), jitter)


def hyperpost(dataset: Dataset, grid: Grid, mixture: Mixture, parameters: Parameters,
             jitter: Array = DEFAULT_JITTER) -> Hyperposterior:
	"""
	Compute the hyperposterior over every mean-process's values at the grid points.

	Parameters
	----------
	dataset
		Dataset used to fit the model.
	grid
		Grid of points and mappings of `dataset`'s inputs onto it.
	mixture
		Soft-clustering of `dataset`'s tasks into mean-processes.
	parameters
		Model parameters (mean, kernels) used to compute the priors and task covariances.
	jitter
		Diagonal jitter added before Cholesky factorizations, for numerical stability.

	Returns
	-------
	Hyperposterior over every mean-process's values at the grid points, batched over mean-processes
	(and channel dimensions).

	Examples
	--------
	>>> import jax.random as jr
	>>> from kernax import ZeroMean, VarianceKernel, SEKernel, WhiteNoiseKernel
	>>> from mimosa.data_structures import Dimensions, Parameters, ModelConfig
	>>> from mimosa.synthetic import generate_data
	>>> dims = Dimensions(T=3, K=1, I=1, C=1, O=1, N=5, G=5)
	>>> parameters = Parameters(
	...     cluster_mean=ZeroMean(),
	...     cluster_kernel=VarianceKernel(1.0) * SEKernel(length_scale=1.0),
	...     task_kernel=VarianceKernel(0.5) * SEKernel(length_scale=1.0),
	...     noise_kernel=WhiteNoiseKernel(noise=0.1),
	... )
	>>> dataset, grid, _, mixture, sampled_params, *_ = generate_data(jr.PRNGKey(0), dims, parameters, ModelConfig())
	>>> hyperposterior = hyperpost(dataset, grid, mixture, sampled_params)
	>>> hyperposterior.mean.shape
	(1, 1, 5)
	"""
	hyperprior = Hyperprior(parameters.cluster_mean(grid.points, output_ids=grid.output_ids), parameters.cluster_kernel(grid.points, output_ids=grid.output_ids))
	# hyperprior mean has shape (K, C, O*G) with K=1 if shared_cluster_hps and C=1 if shared_channel_hps
	# hyperprior cov has shape (K, C, O*G, O*G) with K=1 if shared_cluster_hps and C=1 if shared_channel_hps

	if dataset.inputs.shape[0] == 1:
		output_ids = dataset.output_ids[0] if dataset.output_ids is not None else None
		task_covs = parameters.task_kernel(dataset.clean_inputs[0], output_ids=output_ids) + parameters.noise_kernel(dataset.clean_inputs[0], output_ids=output_ids)  # Shape: (T, K, C, O*N, O*N) with
	else:
		task_covs = parameters.task_kernel(dataset.clean_inputs, output_ids=dataset.output_ids) + parameters.noise_kernel(dataset.clean_inputs, output_ids=dataset.output_ids)

	# Shape: (T, K, C, O*N, O*N) with
	# T=1 if shared_inputs_in_tasks, shared_task_hps and no cluster_specific_task_hps
	# K=1 if shared_cluster_hps
	# C=1 if shared_channel_hps

	if hyperprior.mean.shape[0] == 1:  # Shared cluster HPs
		if task_covs.shape[1] == 1:  # No cluster_specific_task_hps
			f = vmap(single_cluster_hyperpost, in_axes=(None, None, 0, None, None, None))
			return f(dataset.outputs, grid, mixture.responsibilities.T, hyperprior[0], task_covs.swapaxes(0, 1)[0], jitter)
		else:
			f = vmap(single_cluster_hyperpost, in_axes=(None, None, 0, None, 0, None))
			return f(dataset.outputs, grid, mixture.responsibilities.T, hyperprior[0], task_covs.swapaxes(0, 1), jitter)

	else:  # Distinct cluster HPs
		if task_covs.shape[1] == 1:  # No cluster_specific_task_hps
			f = vmap(single_cluster_hyperpost, in_axes=(None, None, 0, 0, None, None))
			return f(dataset.outputs, grid, mixture.responsibilities.T, hyperprior, task_covs.swapaxes(0, 1)[0], jitter)
		else:
			f = vmap(single_cluster_hyperpost, in_axes=(None, None, 0, 0, 0, None))
			return f(dataset.outputs, grid, mixture.responsibilities.T, hyperprior, task_covs.swapaxes(0, 1), jitter)


class Hyperpost(eqx.Module):
	"""
	Callable wrapper around `hyperpost`, as an `equinox.Module`.
	"""
	def __call__(self, dataset: Dataset, grid: Grid, mixture: Mixture, parameters: Parameters,
				 jitter: Array = DEFAULT_JITTER) -> Hyperposterior:
		"""
		See `hyperpost`.

		Examples
		--------
		Continuing from the example of `hyperpost`:

		>>> hyperposterior = Hyperpost()
		>>> hyperposterior(dataset, grid, mixture, sampled_params).mean.shape
		(1, 1, 5)
		"""
		return hyperpost(dataset, grid, mixture, parameters, jitter)
