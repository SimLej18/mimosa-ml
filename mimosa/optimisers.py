"""
Maximum likelihood optimisation of the cluster and task hyperparameters, against their
respective negative log-likelihoods (see `mimosa-ml.nll`).
"""

import jax.numpy as jnp
from jax import Array
import optimistix as optx
from equinox import combine
import equinox as eqx
from kernax import AbstractMean, AbstractKernel

from mimosa.nll import clusters_nlls, tasks_nlls, ClusterNLL, TaskNLL
from mimosa.data_structures import Dataset, Grid, Hyperprior, Hyperposterior, Mixture
from mimosa import DEFAULT_JITTER


def optimise_clusters(
		cluster_mean: AbstractMean, cluster_kernel: AbstractKernel,
		hyperposterior: Hyperposterior, grid: Grid,
		solver: optx.AbstractMinimiser = optx.LBFGS(atol=1e-4, rtol=1e-4), jitter: Array = DEFAULT_JITTER,
		cluster_mean_frozen: AbstractMean | None = None, cluster_kernel_frozen: AbstractKernel | None = None) -> optx.Solution:
	"""
	Maximum-likelihood optimisation of the cluster mean and kernel hyperparameters, against
	`clusters_nlls`.

	`cluster_mean_frozen`/`cluster_kernel_frozen` let part of the hyperparameters be held fixed
	during optimisation (via `equinox.combine`), while the rest is optimised.

	Parameters
	----------
	cluster_mean
		Mean function to optimise (or its non-frozen part, if `cluster_mean_frozen` is given).
	cluster_kernel
		Kernel to optimise (or its non-frozen part, if `cluster_kernel_frozen` is given).
	hyperposterior
		Posterior distribution over each mean-process's values at the grid points.
	grid
		Grid of points and mappings of tasks' inputs onto it.
	solver
		Optimistix minimiser.
	jitter
		Diagonal jitter added before Cholesky factorizations, for numerical stability.
	cluster_mean_frozen
		Part of `cluster_mean` to hold fixed during optimisation, if any.
	cluster_kernel_frozen
		Part of `cluster_kernel` to hold fixed during optimisation, if any.

	Returns
	-------
	Optimistix solution, whose `.value` is the optimised `(cluster_mean, cluster_kernel)` pair.
	"""

	def loss_fn(params, frozen):
		mean = params[0] if frozen[0] is None else combine(params[0], frozen[0])
		kern = params[1] if frozen[1] is None else combine(params[1], frozen[1])

		hyperprior = Hyperprior(mean=mean(grid.points), covariance=kern(grid.points))

		return clusters_nlls(hyperposterior, hyperprior, jitter=jitter).sum()


	params = (cluster_mean, cluster_kernel)
	frozen = (cluster_mean_frozen, cluster_kernel_frozen)

	return optx.minimise(loss_fn, solver, params, frozen, throw=False)

def optimise_tasks(
		task_kernel: AbstractKernel,
		dataset: Dataset, grid: Grid, hyperposterior: Hyperposterior, mixture: Mixture,
		solver: optx.AbstractMinimiser = optx.LBFGS(atol=1e-4, rtol=1e-4), jitter: Array = DEFAULT_JITTER,
		task_kernel_frozen: AbstractKernel | None = None) -> optx.Solution:
	"""
	Maximum-likelihood optimisation of the task (and noise) kernel hyperparameters, against
	`tasks_nlls`, weighted by each task's mixture responsibilities.

	`task_kernel_frozen` lets part of the hyperparameters be held fixed during optimisation
	(via `equinox.combine`), while the rest is optimised.

	Parameters
	----------
	task_kernel
		Task kernel to optimise (or its non-frozen part, if `task_kernel_frozen` is given). Typically
		`task_kernel + noise_kernel`, so both are optimised jointly.
	dataset
		Dataset whose tasks' hyperparameters are optimised.
	grid
		Grid of points and mappings of `dataset`'s inputs onto it.
	hyperposterior
		Posterior distribution over each mean-process's values at the grid points.
	mixture
		Mixture whose responsibilities weight each task's contribution to the loss.
	solver
		Optimistix minimiser.
	jitter
		Diagonal jitter added before Cholesky factorizations, for numerical stability.
	task_kernel_frozen
		Part of `task_kernel` to hold fixed during optimisation, if any.

	Returns
	-------
	Optimistix solution, whose `.value` is the optimised `task_kernel`.
	"""

	def loss_fn(params, frozen):
		kern = params if frozen is None else combine(params, frozen)

		return (tasks_nlls(
			dataset,
			grid,
			kern(dataset.inputs[0]) if dataset.inputs.shape[0] == 1 else kern(dataset.inputs),
			hyperposterior,
			jitter=jitter) * mixture.responsibilities[..., None]).sum()

	return optx.minimise(loss_fn, solver, task_kernel, task_kernel_frozen, throw=False)


class ClusterOptimiser(eqx.Module):
	"""
	Maximum-likelihood optimiser for the cluster mean and kernel hyperparameters, as an
	`equinox.Module`. See `optimise_clusters`.

	Attributes
	----------
	solver
		Optimistix minimiser.
	nll
		Negative log-likelihood to optimise against.
	throw
		If True, raise on optimisation failure instead of returning a failed `optx.Solution`.
	"""
	solver: optx.AbstractMinimiser
	nll: ClusterNLL
	throw: bool

	def __init__(self, solver: optx.AbstractMinimiser, nll: ClusterNLL, throw: bool = False):
		self.solver = solver
		self.nll = nll
		self.throw = throw

	def __call__(self, cluster_mean: AbstractMean, cluster_kernel: AbstractKernel,
	             hyperposterior: Hyperposterior, grid: Grid,
	             jitter: Array = DEFAULT_JITTER,
	             cluster_mean_frozen: AbstractMean | None = None,
	             cluster_kernel_frozen: AbstractKernel | None = None) -> optx.Solution:
		"""
		See `optimise_clusters`.
		"""

		def loss_fn(params, frozen):
			mean = params[0] if frozen[0] is None else combine(params[0], frozen[0])
			kern = params[1] if frozen[1] is None else combine(params[1], frozen[1])

			hyperprior = Hyperprior(mean=mean(grid.points), covariance=kern(grid.points))

			return self.nll(hyperposterior, hyperprior, jitter=jitter).sum()

		params = (cluster_mean, cluster_kernel)
		frozen = (cluster_mean_frozen, cluster_kernel_frozen)

		return optx.minimise(loss_fn, self.solver, params, frozen, throw=self.throw)

class TaskOptimiser(eqx.Module):
	"""
	Maximum-likelihood optimiser for the task (and noise) kernel hyperparameters, as an
	`equinox.Module`. See `optimise_tasks`.

	Attributes
	----------
	solver
		Optimistix minimiser.
	nll
		Negative log-likelihood to optimise against.
	throw
		If True, raise on optimisation failure instead of returning a failed `optx.Solution`.
	"""
	solver: optx.AbstractMinimiser
	nll: TaskNLL
	throw: bool

	def __init__(self, solver: optx.AbstractMinimiser, nll: TaskNLL, throw: bool = False):
		self.solver = solver
		self.nll = nll
		self.throw = throw

	def __call__(self, task_kernel: AbstractKernel,
				 dataset: Dataset, grid: Grid, hyperposterior: Hyperposterior, mixture: Mixture,
				 jitter: Array = DEFAULT_JITTER, task_kernel_frozen: AbstractKernel | None = None) -> optx.Solution:
		"""
		See `optimise_tasks`.
		"""

		def loss_fn(params, frozen):
			kern = params if frozen is None else combine(params, frozen)

			task_covs = kern(dataset.inputs[0]) if dataset.inputs.shape[0] == 1 else kern(dataset.inputs)

			return (self.nll(dataset, grid, task_covs, hyperposterior, jitter=jitter) * mixture.responsibilities[..., None]).sum()

		return optx.minimise(loss_fn, self.solver, task_kernel, task_kernel_frozen, throw=self.throw)
