"""
Data structures used throughout the package: dimensions and configuration, model parameters and priors,
datasets and grids, and prediction outputs.

Structures are plain dataclasses, or `equinox.Module` when they must be passed to jitted functions.
"""

from dataclasses import dataclass, fields
from jaxtyping import Array, Float, Int, jaxtyped
import jax.numpy as jnp
from beartype import beartype as typechecker
import equinox as eqx
from kernax import AbstractMean, AbstractKernel


@dataclass(frozen=True)
class Dimensions:
	"""
	Dimensions of the data that the model will generate, learn, or predict.

	Attributes
	----------
	T
		Number of tasks.
	K
		Number of clusters in the mixture.
	I
		Dimensionality of input points.
	O
		Dimensionality of output points.
	F
		Number of correlated features.
	N
		Number of points observed by the largest task.
	G
		Number of points in the full grid.

	Raises
	------
	ValueError
		If any dimension is non-positive, if there are more clusters than tasks,
		or more points per task than in the grid.
	"""
	T: int
	K: int
	I: int
	O: int
	F: int
	N: int
	G: int

	def __post_init__(self):
		# Check positivity
		def all_positive():
			return all(getattr(self, field.name) > 0 for field in fields(self))

		if not all_positive():
			raise ValueError("All dimensions must be positive integers.")

		# Check for contradicting dimensions
		if self.K > self.T:
			raise ValueError("Cannot have more clusters than tasks.")
		if self.N > self.G:
			raise ValueError("Cannot have more points in tasks than in the full grid")


@dataclass(frozen=True)
class ModelConfig:
	"""
	Configuration of the model, used to know which parameters are batched.

	Attributes
	----------
	shared_task_hps
		If True, task-kernel hyperparameters are shared across tasks; if False, each task has its own.
	shared_cluster_hps
		If True, cluster-kernel hyperparameters are shared across mean-processes; if False, each mean-process has its own.
	shared_output_hps
		If True, hyperparameters are shared across output dimensions; if False, each output has its own.
	shared_features_hps
		If True, hyperparameters are shared across features; if False, each feature has its own.
	cluster_specific_task_hps
		If True, task-kernel hyperparameters may additionally vary by cluster assignment, independently of `shared_task_hps`.
	isotopic_tasks
		If True, all tasks share the same input locations.
	isotopic_features
		If True, all features share the same input locations.
	"""
	shared_task_hps: bool = True
	shared_cluster_hps: bool = True
	shared_output_hps: bool = True
	shared_features_hps: bool = True
	cluster_specific_task_hps: bool = True
	isotopic_tasks: bool = True
	isotopic_features: bool = True


@dataclass(frozen=True)
class DataRemovalConfig:
	"""
	Configuration for the random removal of data points in a Dataset.

	Attributes
	----------
	max_missing
		Maximum number of missing points per task.
	random_missing_count
		If True, the number of missing points is drawn randomly in [0, `max_missing`]; if False, it is fixed to `max_missing`.
	same_missing_across_outputs
		If True, a missing point is missing for every output; if False, missingness is drawn independently per output.
	same_missing_across_features
		If True, a missing point is missing for every feature; if False, missingness is drawn independently per feature.
	"""
	max_missing: int
	random_missing_count: bool = False
	same_missing_across_outputs: bool = True
	same_missing_across_features: bool = True


def validate_model_config(model_config: ModelConfig, dimensions: Dimensions) -> None:
	"""
	Check that the model configuration is compatible with the dimensions.

	Parameters
	----------
	model_config
		Configuration to validate.
	dimensions
		Dimensions to validate the configuration against.

	Raises
	------
	ValueError
		If a hyperparameter is configured to vary along an axis of size 1.
	"""
	if not model_config.shared_task_hps and dimensions.T == 1:
		raise ValueError("Cannot have distinct task hyperparameters with only one task.")
	if not model_config.shared_cluster_hps and dimensions.K == 1:
		raise ValueError("Cannot have distinct cluster hyperparameters with only one cluster.")
	if not model_config.shared_output_hps and dimensions.O == 1:
		raise ValueError("Cannot have distinct output hyperparameters with only one output.")
	if not model_config.shared_features_hps and dimensions.F == 1:
		raise ValueError("Cannot have distinct feature hyperparameters with only one feature.")


class Parameters(eqx.Module):
	"""
	Cluster mean, cluster kernel, task kernel and noise kernel of the model.

	Attributes
	----------
	cluster_mean
		Mean function of the mean-processes.
	cluster_kernel
		Covariance kernel of the mean-processes.
	task_kernel
		Covariance kernel modelling the deviation of each task from its mean-process.
	noise_kernel
		Covariance kernel modelling the observation noise.
	"""
	cluster_mean: AbstractMean
	cluster_kernel: AbstractKernel
	task_kernel: AbstractKernel
	noise_kernel: AbstractKernel


@dataclass(frozen=True)
class ParameterPriors:
	"""
	Priors for each parameter of the model, used to generate the hyperparameters or to perform maximum-a-posteriori
	inference.

	Attributes
	----------
	cluster_mean_priors
		Priors for the mean function's hyperparameters.
	cluster_kernel_priors
		Priors for the mean-process kernel's hyperparameters.
	task_kernel_priors
		Priors for the task kernel's hyperparameters.
	noise_kernel_priors
		Priors for the noise kernel's hyperparameters.
	"""
	cluster_mean_priors: dict
	cluster_kernel_priors: dict
	task_kernel_priors: dict
	noise_kernel_priors: dict


@jaxtyped(typechecker=typechecker)
class Dataset(eqx.Module):
	"""
	Dataset regrouping the inputs and outputs of all tasks.

	Attributes
	----------
	inputs
		Input points of each task. Shape is `(T, N, I)` when features share the same input locations
		(isotopic features), or `(T, F*N, I)` otherwise.
	outputs
		Output values of each task, at each feature and input point.
	known_output_noise
		Known observation noise, when available.
	"""
	inputs: Float[Array, "#T N I"] | Float[Array, "#T FN I"]
	outputs: Float[Array, "T FN O"]
	known_output_noise: None | Float[Array, "T FN O"] = None


@jaxtyped(typechecker=typechecker)
class Grid(eqx.Module):
	"""
	Grid points and mappings of each task's inputs on the grid.

	Attributes
	----------
	points
		Input points of the grid.
	mappings
		Index of each task's input points in `points`.
	"""
	points: Float[Array, "FG I"]
	mappings:  Int[Array, "#T N"]


@jaxtyped(typechecker=typechecker)
class MultivariateNormal(eqx.Module):
	"""
	Multivariate normal distribution over a P-dimensional vector, possibly batched.

	Attributes
	----------
	mean
		Mean vector.
	covariance
		Covariance matrix.
	"""
	mean: Float[Array, "... P"]
	covariance: Float[Array, "... P P"]

	def __getitem__(self, item):
		"""
		Index `mean` and `covariance` jointly along the batch dimensions.
		"""
		return MultivariateNormal(mean=self.mean[item], covariance=self.covariance[item])


@jaxtyped(typechecker=typechecker)
class Hyperprior(MultivariateNormal):
	"""
	Prior distribution over the mean-process values at grid points, before observing data.
	"""
	mean: Float[Array, "*B FG"]
	covariance: Float[Array, "*B FG FG"]


@jaxtyped(typechecker=typechecker)
class Hyperposterior(MultivariateNormal):
	"""
	Posterior distribution over the mean-process values at grid points, after observing data.
	"""
	mean: Float[Array, "*B FG"]
	covariance: Float[Array, "*B FG FG"]


@jaxtyped(typechecker=typechecker)
class Mixture(eqx.Module):
	"""
	Soft-clustering of the tasks into mean-processes.

	Attributes
	----------
	proportions
		Mixture weight of each mean-process.
	responsibilities
		Probability of each task belonging to each mean-process.
	"""
	proportions: Float[Array, "K"]
	responsibilities: Float[Array, "T K"]

	@property
	def assignments(self) -> Float[Array, "T"]:
		"""
		Hard cluster assignment of each task, i.e. its most likely mean-process.
		"""
		return jnp.argmax(self.responsibilities, axis=1)


@jaxtyped(typechecker=typechecker)
class PredictionMeanBlocks(eqx.Module):
	"""
	Predicted mean, split into blocks for observed points and grid points.

	Attributes
	----------
	mean_obs
		Predicted mean at the observed input points.
	mean_grid
		Predicted mean at the grid points.
	"""
	mean_obs: Float[Array, "*B FN"]
	mean_grid: Float[Array, "*B FG"]

	def __getitem__(self, item):
		"""
		Index `mean_obs` and `mean_grid` jointly along the batch dimensions.
		"""
		return PredictionMeanBlocks(mean_obs=self.mean_obs[item], mean_grid=self.mean_grid[item])


@jaxtyped(typechecker=typechecker)
class PredictionCovBlocks(eqx.Module):
	"""
	Predicted covariance, split into blocks for observed points, grid points, and their cross-covariance.

	Attributes
	----------
	cov_obs
		Covariance among the observed input points.
	cov_grid
		Covariance among the grid points.
	cov_crossed
		Cross-covariance between the observed input points and the grid points.
	"""
	cov_obs: Float[Array, "*B FN FN"]
	cov_grid: Float[Array, "*B FG FG"]
	cov_crossed: Float[Array, "*B FN FG"]

	def __getitem__(self, item):
		"""
		Index `cov_obs`, `cov_grid` and `cov_crossed` jointly along the batch dimensions.
		"""
		return PredictionCovBlocks(cov_obs=self.cov_obs[item], cov_grid=self.cov_grid[item], cov_crossed=self.cov_crossed[item])
