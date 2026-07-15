"""
Compute the hyperposterior: the posterior distribution over each mean-process's values at the grid
points, given a Dataset, a soft-clustering Mixture, and model Parameters.

`single_output_hyperpost` computes it for one mean-process and one output dimension;
`single_cluster_hyperpost` vmaps it across output dimensions; `hyperpost` vmaps it across mean-processes.
"""

import jax.numpy as jnp
from jax import Array, vmap
import equinox as eqx

from mimosa.linalg import cho_factor, cho_solve
from mimosa.data_structures import Parameters, Dataset, Grid, Mixture, Hyperprior, Hyperposterior
from mimosa import DEFAULT_JITTER


def single_output_hyperpost(outputs: Array, grid: Grid, responsibilities: Array,
                            hyperprior: Hyperprior, task_covs: Array,
                            jitter: Array = DEFAULT_JITTER) -> Hyperposterior:
	"""
	Compute the hyperposterior for a single mean-process and a single output dimension.

	Parameters
	----------
	outputs
		Output values of every task, for this output dimension. Shape `(T, F*N)`.
	grid
		Grid of points and mappings of dataset's inputs onto it.
	responsibilities
		Responsibility of each task towards this mean-process. Shape `(T,)`.
	hyperprior
		Prior distribution over this mean-process's values at the grid points.
	task_covs
		Task covariance (including noise) of every task, for this output dimension. Shape `(T, F*N, F*N)`.
	jitter
		Diagonal jitter added before Cholesky factorizations, for numerical stability.

	Returns
	-------
	Hyperposterior over this mean-process's values at the grid points, for this output dimension.
	"""
	big_eye = jnp.eye(grid.points.shape[0])
	small_eye = jnp.eye(outputs.shape[-1])

	# Cluster covariance inversion
	cluster_cov_l = cho_factor(hyperprior.covariance, jitter=jitter)  # Shape (G, G)
	cluster_cov_inv = cho_solve(cluster_cov_l, big_eye)

	# Task covariances inversion -- masked from this output's own NaNs (task_covs itself is always NaN-free)
	nan_mask = jnp.isnan(outputs)  # (T, F*N)
	nan_mask_2d = nan_mask[:, None, :] | nan_mask[:, :, None]  # (T, F*N, F*N)
	task_covs_padded = jnp.where(nan_mask_2d, small_eye, task_covs)  # Padding
	task_covs_l = cho_factor(task_covs_padded, jitter=jitter)  # Shape (T, F*N, F*N)  with T=1 if shared_inputs_in_tasks and shared_task_hps
	task_covs_inv = cho_solve(task_covs_l, jnp.broadcast_to(small_eye, task_covs_l.shape))
	task_covs_inv -= jnp.where(nan_mask_2d, task_covs_inv, 0)  # Correction on the diagonal
	task_covs_inv *= responsibilities[:, None, None]  # Apply mixture coefficients

	# Mapping to full grid
	mappings = jnp.broadcast_to(grid.mappings, (outputs.shape[0], grid.mappings.shape[1]))
	task_covs_inv = jnp.zeros((len(grid.points), len(grid.points))).at[mappings[:, :, None], mappings[:, None, :]].add(task_covs_inv)  # Shape (F*G, F*G)

	# Sum mean and task covariances and compute Cholesky factor of the posterior covariance
	post_covs_inv = cho_factor(cluster_cov_inv + task_covs_inv, jitter=jitter)  # Shape (F*G, F*G)
	post_cov = cho_solve(post_covs_inv, big_eye)  # Shape (F*G, F*G)

	# --- Posterior mean ---
	# Compute prior means
	prior_mean = cho_solve(cluster_cov_l, hyperprior.mean)  # Shape (F*G)
	task_means = cho_solve(jnp.broadcast_to(task_covs_l, (outputs.shape[0],)+task_covs_l.shape[1:]), jnp.nan_to_num(outputs))  # Shape (T, F*N)
	task_means *= responsibilities[:, None]  # Shape (T, F*N)
	task_means = jnp.zeros((len(grid.points),)).at[mappings].add(task_means)  # Shape (F*G)

	full_mean = prior_mean + task_means  # Shape (F*G)
	post_mean = cho_solve(post_covs_inv, full_mean)

	return Hyperposterior(mean=post_mean, covariance=post_cov)


def single_cluster_hyperpost(outputs: Array, grid: Grid, responsibilities: Array,
                             hyperprior: Hyperprior, task_covs: Array,
                             jitter: Array = DEFAULT_JITTER) -> Hyperposterior:
	"""
	Compute the hyperposterior for a single mean-process, vmapped across output dimensions.

	Parameters
	----------
	outputs
		Output values of every task, for this mean-process. Shape `(T, F*N, O)`.
	grid
		Grid of points and mappings of `dataset`'s inputs onto it.
	responsibilities
		Responsibility of each task towards this mean-process. Shape `(T,)`.
	hyperprior
		Prior distribution over this mean-process's values at the grid points, batched over output
		dimensions (or a single shared prior, if `shared_output_hps`).
	task_covs
		Task covariance (including noise) of every task, batched over output dimensions
		(or shared across outputs, if no `output_hp_in_tasks`).
	jitter
		Diagonal jitter added before Cholesky factorizations, for numerical stability.

	Returns
	-------
	Hyperposterior over this mean-process's values at the grid points, batched over output dimensions.
	"""
	if hyperprior.mean.shape[0] == 1:  # Shared outputs HPs
		if task_covs.shape[1] == 1:  # No output_hp_in_tasks
			f = vmap(single_output_hyperpost, in_axes=(0, None, None, None, None, None))
			return f(outputs.T.mT, grid, responsibilities, hyperprior[0], task_covs.swapaxes(0, 1)[0], jitter)
		else:
			f = vmap(single_output_hyperpost, in_axes=(0, None, None, None, 0, None))
			return f(outputs.T.mT, grid, responsibilities, hyperprior[0], task_covs.swapaxes(0, 1), jitter)

	else:  # Distinct output HPs
		if task_covs.shape[1] == 1:  # No output_hp_in_tasks
			f = vmap(single_output_hyperpost, in_axes=(0, None, None, 0, None, None))
			return f(outputs.T.mT, grid, responsibilities, hyperprior, task_covs.swapaxes(0, 1)[0], jitter)
		else:
			f = vmap(single_output_hyperpost, in_axes=(0, None, None, 0, 0, None))
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
	(and output dimensions).

	Examples
	--------
	>>> import jax.random as jr
	>>> from kernax import ZeroMean, VarianceKernel, SEKernel, WhiteNoiseKernel
	>>> from mimosa-ml.data_structures import Dimensions, Parameters, ModelConfig
	>>> from mimosa-ml.synthetic import generate_data
	>>> dims = Dimensions(T=3, K=1, I=1, O=1, F=1, N=5, G=5)
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
	hyperprior = Hyperprior(parameters.cluster_mean(grid.points), parameters.cluster_kernel(grid.points))
	# hyperprior mean has shape (K, O, F*G) with K=1 if shared_cluster_hps and O=1 if shared_output_hps
	# hyperprior cov has shape (K, O, F*G, F*G) with K=1 if shared_cluster_hps and O=1 if shared_output_hps

	if dataset.inputs.shape[0] == 1:
		task_covs = parameters.task_kernel(dataset.inputs[0]) + parameters.noise_kernel(dataset.inputs[0])  # Shape: (T, K, O, F*N, F*N) with
	else:
		task_covs = parameters.task_kernel(dataset.inputs) + parameters.noise_kernel(dataset.inputs)

	# Shape: (T, K, O, F*N, F*N) with
	# T=1 if shared_inputs_in_tasks, shared_task_hps and no cluster_specific_task_hps
	# K=1 if shared_cluster_hps
	# O=1 if shared_output_hps

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
