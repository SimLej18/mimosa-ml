"""
Predict outputs at the grid points, for every task and every mean-process, by Gaussian process
conditioning of each task on its observed values and on the mean-process hyperposterior.
"""

import jax.numpy as jnp
import jax.lax as lax
from jax import vmap, Array
import equinox as eqx

from mimosa.linalg import cho_factor
from mimosa.data_structures import (Parameters, Dataset, Grid, Hyperposterior, PredictionMeanBlocks,
                                    PredictionCovBlocks, MultivariateNormal)
from mimosa import DEFAULT_JITTER


def predict_task_output(output_obs: Array,
                        post_mean_blocks: PredictionMeanBlocks,
                        cov_blocks: PredictionCovBlocks,
                        jitter: Array = DEFAULT_JITTER) -> MultivariateNormal:
	"""
	Predict a single task's output at the grid points, under a single mean-process, by GP
	conditioning on the task's observed values.

	Handles padded data: missing points are read from NaNs in `output_obs`.

	Parameters
	----------
	output_obs
		Observed values of this output, for this task. Shape `(F*N,)`.
	post_mean_blocks
		Mean-process hyperposterior mean at the task's observed points and at the grid points,
		for this output.
	cov_blocks
		Combined mean-process and task covariance (observed/grid/cross blocks), for this output.
	jitter
		Diagonal jitter added before Cholesky factorization, for numerical stability.

	Returns
	-------
	Predicted distribution over this task's output at the grid points.
	"""
	padding_mask_1D = ~jnp.isnan(output_obs)[:, None]
	padding_mask_2D = padding_mask_1D & padding_mask_1D.T

	gamma_obs = jnp.where(padding_mask_2D, cov_blocks.cov_obs, jnp.eye(len(cov_blocks.cov_obs)))
	gamma_crossed = jnp.where(padding_mask_1D, cov_blocks.cov_crossed, 0.)

	L = cho_factor(gamma_obs, jitter=jitter)
	z = lax.linalg.triangular_solve(L, gamma_crossed, left_side=True, lower=True)
	y = \
	lax.linalg.triangular_solve(L, (jnp.nan_to_num(output_obs) - post_mean_blocks.mean_obs)[:, None], left_side=True,
	                            lower=True)[:, 0]

	pred_mean = post_mean_blocks.mean_grid + (z.T @ y)
	pred_cov = cov_blocks.cov_grid - (z.T @ z)

	return MultivariateNormal(mean=pred_mean, covariance=pred_cov)


def predict_task_in_cluster(output_obs: Array,
                            post_mean_blocks: PredictionMeanBlocks,
                            cov_blocks: PredictionCovBlocks,
                            jitter: Array = DEFAULT_JITTER) -> MultivariateNormal:
	"""
	Predict a single task's outputs at the grid points, under a single mean-process, vmapped
	across outputs.

	See `predict_task_output`.

	Parameters
	----------
	output_obs
		Observed values of this task, for every output. Shape `(F*N, O)`.
	post_mean_blocks
		Mean-process hyperposterior mean at the task's observed points and at the grid points,
		batched over output dimensions.
	cov_blocks
		Combined mean-process and task covariance (observed/grid/cross blocks), batched over
		output dimensions.
	jitter
		Diagonal jitter added before Cholesky factorization, for numerical stability.

	Returns
	-------
	Predicted distribution over this task's outputs at the grid points, batched over output dimensions.
	"""
	if cov_blocks.cov_obs.shape[0] == 1:
		return (vmap(
			predict_task_output,
			in_axes=(0, 0, None, None))
		        (output_obs.T, post_mean_blocks, cov_blocks[0], jitter))
	else:
		return (vmap(
			predict_task_output,
			in_axes=(0, 0, 0, None))
		        (output_obs.T, post_mean_blocks, cov_blocks, jitter))


def predict_clusters(task_outputs: Array,
                     mappings: Array,
                     hyperposterior: Hyperposterior,
                     task_cov_blocks: PredictionCovBlocks,
                     jitter: Array = DEFAULT_JITTER) -> MultivariateNormal:
	"""
	Predict a single task's outputs at the grid points, under every mean-process, vmapped across
	mean-processes.

	See `predict_task_in_cluster`.

	Parameters
	----------
	task_outputs
		Observed values of this task, for every output. Shape `(F*N, O)`.
	mappings
		Index of this task's observed points in the grid.
	hyperposterior
		Posterior distribution over every mean-process's values at the grid points.
	task_cov_blocks
		Task covariance (including noise), split into observed/grid/cross blocks, batched over
		mean-processes (or shared, if `task_cov_blocks.cov_obs.shape[0] == 1`).
	jitter
		Diagonal jitter added before Cholesky factorization, for numerical stability.

	Returns
	-------
	Predicted distribution over this task's outputs at the grid points, batched over mean-processes
	(and output dimensions).
	"""
	post_mean_obs = hyperposterior.mean[:, :, mappings]
	post_cov_obs = hyperposterior.covariance[:, :, mappings, :][:, :, :, mappings]
	post_cov_crossed = hyperposterior.covariance[:, :, mappings, :]

	cov_blocks = PredictionCovBlocks(
		cov_obs=post_cov_obs + task_cov_blocks.cov_obs,
		cov_grid=hyperposterior.covariance + task_cov_blocks.cov_grid,
		cov_crossed=post_cov_crossed + task_cov_blocks.cov_crossed
	)

	post_mean_blocks = PredictionMeanBlocks(
		mean_obs=post_mean_obs,
		mean_grid=hyperposterior.mean
	)

	if cov_blocks.cov_obs.shape[0] == 1:
		return vmap(
			predict_task_in_cluster,
			in_axes=(None, 0, None, None)
		)(task_outputs, post_mean_blocks, cov_blocks[0], jitter)
	return vmap(
		predict_task_in_cluster,
		in_axes=(None, 0, 0, None)
	)(task_outputs, post_mean_blocks, cov_blocks, jitter)


def predict(dataset: Dataset,
            grid: Grid,
            hyperposterior: Hyperposterior,
            parameters: Parameters,
            jitter: Array = DEFAULT_JITTER) -> MultivariateNormal:
	"""
	Predict every task's outputs at the grid points, under every mean-process.

	See `predict_clusters`.

	Parameters
	----------
	dataset
		Dataset whose tasks are predicted.
	grid
		Grid of points and mappings of `dataset`'s inputs onto it.
	hyperposterior
		Posterior distribution over every mean-process's values at the grid points.
	parameters
		Model parameters (task/noise kernels) used to compute the task covariances.
	jitter
		Diagonal jitter added before Cholesky factorization, for numerical stability.

	Returns
	-------
	Predicted distribution over every task's outputs at the grid points, batched over tasks,
	mean-processes and output dimensions.
	"""
	if dataset.inputs.shape[0] == 1:
		extended_grid = grid.points
		task_cov_blocks = PredictionCovBlocks(
			cov_obs=parameters.task_kernel(dataset.inputs[0]) + parameters.noise_kernel(dataset.inputs[0]),
			cov_grid=parameters.task_kernel(extended_grid),
			cov_crossed=parameters.task_kernel(dataset.inputs[0], extended_grid),
		)
	else:
		extended_grid = jnp.broadcast_to(grid.points, dataset.inputs.shape[:1] + grid.points.shape)
		task_cov_blocks = PredictionCovBlocks(
			cov_obs=parameters.task_kernel(dataset.inputs) + parameters.noise_kernel(dataset.inputs),
			cov_grid=parameters.task_kernel(extended_grid),
			cov_crossed=parameters.task_kernel(dataset.inputs, extended_grid),
		)

	mappings = grid.mappings[0] if dataset.inputs.shape[0] == 1 else grid.mappings

	if task_cov_blocks.cov_obs.shape[0] == 1:
		return vmap(
			predict_clusters,
			in_axes=(0,
			         0 if mappings.ndim == 2 else None,
			         None,
			         None,
			         None),
		)(dataset.outputs,
		  mappings,
		  hyperposterior,
		  task_cov_blocks[0],
		  jitter)

	return vmap(
		predict_clusters,
		in_axes=(0,
		         0 if mappings.ndim == 2 else None,
		         None,
		         0,
		         None),
	)(dataset.outputs,
	  mappings,
	  hyperposterior,
	  task_cov_blocks,
	  jitter)


class Predictor(eqx.Module):
	"""
	Callable wrapper around `predict`, as an `equinox.Module`.
	"""
	def __call__(self, dataset: Dataset,
            grid: Grid,
            hyperposterior: Hyperposterior,
            parameters: Parameters,
            jitter: Array = DEFAULT_JITTER) -> MultivariateNormal:
		"""
		See `predict`.
		"""
		return predict(dataset, grid, hyperposterior, parameters, jitter)
