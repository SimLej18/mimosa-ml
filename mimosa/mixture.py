"""
Update and initialise the soft-clustering Mixture of tasks into mean-processes.
"""

from jax import Array
from jax.nn import softmax
import jax.numpy as jnp
import equinox as eqx
from kernax import AbstractKernel

from mimosa.nll import tasks_nlls
from mimosa.kmeans import soft_kmeans
from mimosa.data_structures import Dataset, Grid, Hyperposterior, Mixture
from mimosa import DEFAULT_JITTER


class MixtureInitialiser(eqx.Module):
	"""
	Base class for initialising a Mixture from a Dataset.
	"""
	def __call__(self, dataset: Dataset) -> Mixture:
		"""
		Initialise a Mixture from `dataset`.

		Parameters
		----------
		dataset
			Dataset to initialise a Mixture for.

		Returns
		-------
		Initial Mixture.
		"""
		...

class KMeansMixtureInitialiser(MixtureInitialiser):
	"""
	Initialise a Mixture by soft k-means clustering of per-task output summary statistics.

	Attributes
	----------
	prng_key
		`jax.random` PRNG key.
	n_clusters
		Number of mean-processes to initialise responsibilities for.
	"""
	prng_key: Array
	n_clusters: int

	def __init__(self, prng_key, n_clusters: int):
		self.prng_key = prng_key
		self.n_clusters = n_clusters

	def __call__(self, dataset: Dataset) -> Mixture:
		"""
		Cluster tasks by soft k-means over each task's per-output min, max, mean and std.

		See `MixtureInitialiser.__call__`.
		"""
		# Outputs: shape (T, N, O)
		features = jnp.stack((
			jnp.nanmin(dataset.outputs, axis=1),
			jnp.nanmax(dataset.outputs, axis=1),
			jnp.nanmean(dataset.outputs, axis=1),
			jnp.nanstd(dataset.outputs, axis=1))).reshape((len(dataset.outputs), -1))

		_, resp = soft_kmeans(self.prng_key, features, self.n_clusters)
		return Mixture(proportions=jnp.ones(self.n_clusters)/self.n_clusters, responsibilities=resp)


def update_mixture(dataset: Dataset, grid: Grid, task_kernel: AbstractKernel, hyperposterior: Hyperposterior,
                   mixture: Mixture, jitter: Array = DEFAULT_JITTER) -> Mixture:
	"""
	Update the tasks' responsibilities towards each mean-process, given the current hyperposterior.

	Parameters
	----------
	dataset
		Dataset whose tasks are being clustered.
	grid
		Grid of points and mappings of `dataset`'s inputs onto it.
	task_kernel
		Task covariance kernel (including noise).
	hyperposterior
		Current posterior distribution over each mean-process's values at the grid points.
	mixture
		Current mixture, whose proportions are kept and responsibilities are recomputed.
	jitter
		Diagonal jitter added before Cholesky factorizations, for numerical stability.

	Returns
	-------
	Mixture with updated responsibilities, proportional to each task's likelihood under each mean-process.
	"""
	if dataset.inputs.shape[0] == 1:
		task_llhs = jnp.sum(tasks_nlls(dataset, grid, task_kernel(dataset.inputs[0]), hyperposterior, jitter=jitter), axis=-1)
	else:
		task_llhs = jnp.sum(tasks_nlls(dataset, grid, task_kernel(dataset.inputs), hyperposterior, jitter=jitter), axis=-1)
	return Mixture(proportions=mixture.proportions, responsibilities=softmax(jnp.log(mixture.proportions[None, :]) - task_llhs, axis=1))


class MixtureUpdater(eqx.Module):
	"""
	Callable wrapper around `update_mixture`, as an `equinox.Module`.
	"""
	def __call__(self, dataset: Dataset, grid: Grid, task_kernel: AbstractKernel, hyperposterior: Hyperposterior,
				mixture: Mixture, jitter: Array = DEFAULT_JITTER) -> Mixture:
		"""
		See `update_mixture`.
		"""
		return update_mixture(dataset, grid, task_kernel, hyperposterior, mixture, jitter=jitter)
