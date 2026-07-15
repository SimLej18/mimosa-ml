"""
Generate synthetic multi-task, multi-cluster datasets from GP priors (`generate_data`), and remove
data points at random to simulate missingness (`RandomDataRemover`).
"""

from abc import abstractmethod
import jax.random as jr
import jax.numpy as jnp
from jax import vmap, Array
import equinox as eqx
from kernax import BatchModule, AbstractKernel, AbstractMean, AbstractModule
from kernax.hp_sampling import sample_hps_from_uniform_priors

from mimosa.data_structures import Dimensions, Parameters, ParameterPriors, ModelConfig, Hyperprior, Mixture, Dataset, \
	Grid, MultivariateNormal, DataRemovalConfig
from mimosa.linalg import compute_mapping
from mimosa.sampling import sample_gp
from mimosa import DEFAULT_JITTER


def generate_grid(G: int, n_dims: int, bounds: tuple[int, int]) -> Array:
	"""
	Build a regular grid spanning `bounds` in every dimension, with as close to `G` points as possible.

	Parameters
	----------
	G
		Desired total number of grid points. The actual grid uses the number of points per input
		dimension whose `n_dims`-th power is closest to `G`.
	n_dims
		Number of input dimensions.
	bounds
		Min and max value of the grid, applied to every dimension.

	Returns
	-------
	Grid points. Shape `(grid_size ** n_dims, n_dims)`.
	"""
	grid_size = max(round(G ** (1 / n_dims)), 1)
	axis = jnp.linspace(bounds[0], bounds[1], grid_size)
	grids = jnp.meshgrid(*([axis] * n_dims), indexing='ij')
	return jnp.stack(grids, axis=-1).reshape(-1, n_dims)


def sample_inputs(key: Array, grid: Array, dims: Dimensions, config: ModelConfig) -> tuple[Array, Array]:
	"""
	Sample `dims.N` input points per task (or per feature) from `grid`, without replacement, and
	compute each sampled point's index (mapping) in `grid`.

	Sampling structure follows `config.isotopic_tasks`/`config.isotopic_features`: shared or distinct
	sampled points across tasks, and across features.

	Parameters
	----------
	key
		`jax.random` PRNG key.
	grid
		Grid of points to sample from.
	dims
		Dimensions of the dataset to generate.
	config
		Model configuration, used for its `isotopic_tasks`/`isotopic_features` fields.

	Returns
	-------
	inputs
		Sampled input points.
	mappings
		Index of each sampled point in `grid`.
	"""
	if config.isotopic_tasks:
		if config.isotopic_features:
			inputs = jr.choice(key, grid, (dims.N,), replace=False)[None, ...]
			mappings = compute_mapping(grid, inputs[0])[None, ...]
		else:
			inputs = vmap(lambda k: jr.choice(k, grid, (dims.N,), replace=False))(jr.split(key, dims.F))
			mappings = vmap(lambda i: compute_mapping(grid, i))(inputs)
	else:
		# FIXME: in multi-features, inputs should be concatenated, not stacked
		if config.isotopic_features:
			inputs = vmap(lambda k: jr.choice(k, grid, (dims.N,), replace=False))(jr.split(key, dims.T))
			mappings = vmap(lambda i: compute_mapping(grid, i))(inputs)
		else:
			inputs = vmap(lambda k1: vmap(lambda k2: jr.choice(k2, grid, (dims.N,), replace=False))(jr.split(k1, dims.F)))(jr.split(key, dims.T))
			mappings = vmap(vmap(lambda i: compute_mapping(grid, i)))(inputs)
	return inputs, mappings


def build_mean(
		mean: AbstractMean,
		dims: Dimensions,
		config: ModelConfig) -> AbstractModule:
	"""
	Batch `mean` across output dimensions and mean-processes, according to `config`'s HP-sharing
	flags, for synthetic data generation.

	`mean` should be the "base" mean, i.e. the one used if all HPs were shared.

	Parameters
	----------
	mean
		Base mean function to batch.
	dims
		Dimensions of the dataset to generate.
	config
		Model configuration, used for its `shared_output_hps`/`shared_cluster_hps` fields.

	Returns
	-------
	Batched mean function, with independent hyperparameters per output/mean-process where configured.
	"""
	# multi-output HPs
	if not config.shared_output_hps:
		mean = BatchModule(mean, batch_size=dims.O, batch_in_axes=0, batch_over_inputs=False)
	else:
		mean = BatchModule(mean, batch_size=1, batch_in_axes=None, batch_over_inputs=False)

	# cluster HPs
	if not config.shared_cluster_hps:
		mean = BatchModule(mean, batch_size=dims.K, batch_in_axes=0, batch_over_inputs=False)
	else:
		mean = BatchModule(mean, batch_size=1, batch_in_axes=None, batch_over_inputs=False)

	return mean


def build_mean_kernel(
		mean_kernel: AbstractKernel,
		dims: Dimensions,
		config: ModelConfig) -> AbstractModule:
	"""
	Batch `mean_kernel` across output dimensions and mean-processes, according to `config`'s
	HP-sharing flags, for synthetic data generation.

	`mean_kernel` should be the "base" kernel, i.e. the one used if all HPs were shared. If
	`dims.F > 1`, it should already be wrapped in a `BlockKernel` to handle the multi-feature
	structure (this function doesn't manage feature-related config).

	Parameters
	----------
	mean_kernel
		Base kernel to batch.
	dims
		Dimensions of the dataset to generate.
	config
		Model configuration, used for its `shared_output_hps`/`shared_cluster_hps` fields.

	Returns
	-------
	Batched kernel, with independent hyperparameters per output/mean-process where configured.
	"""
	# multi-output HPs
	if not config.shared_output_hps:
		mean_kernel = BatchModule(mean_kernel, batch_size=dims.O, batch_in_axes=0, batch_over_inputs=False)
	else:
		mean_kernel = BatchModule(mean_kernel, batch_size=1, batch_in_axes=None, batch_over_inputs=False)

	# cluster HPs
	if not config.shared_cluster_hps:
		mean_kernel = BatchModule(mean_kernel, batch_size=dims.K, batch_in_axes=0, batch_over_inputs=False)
	else:
		mean_kernel = BatchModule(mean_kernel, batch_size=1, batch_in_axes=None, batch_over_inputs=False)

	return mean_kernel


def build_task_kernel(
		task_kernel: AbstractKernel,
		dims: Dimensions,
		config: ModelConfig) -> AbstractModule:
	"""
	Batch `task_kernel` across output dimensions, mean-processes and tasks, according to `config`'s
	HP-sharing flags, for synthetic data generation. Used for both the task and noise kernels.

	`task_kernel` should be the "base" kernel, i.e. the one used if all HPs were shared. If
	`dims.F > 1`, it should already be wrapped in a `BlockKernel` to handle the multi-feature
	structure (this function doesn't manage feature-related config).

	Parameters
	----------
	task_kernel
		Base kernel to batch.
	dims
		Dimensions of the dataset to generate.
	config
		Model configuration, used for its `shared_output_hps`, `cluster_specific_task_hps`,
		`shared_task_hps` and `isotopic_tasks` fields.

	Returns
	-------
	Batched kernel, with independent hyperparameters per output/mean-process/task where configured.
	"""
	# multi-output HPs
	if not config.shared_output_hps:
		task_kernel = BatchModule(task_kernel, batch_size=dims.O, batch_in_axes=0, batch_over_inputs=False)
	else:
		task_kernel = BatchModule(task_kernel, batch_size=1, batch_in_axes=None, batch_over_inputs=False)

	# cluster HPs
	if config.cluster_specific_task_hps:
		task_kernel = BatchModule(task_kernel, batch_size=dims.K, batch_in_axes=0, batch_over_inputs=False)
	else:
		task_kernel = BatchModule(task_kernel, batch_size=1, batch_in_axes=None, batch_over_inputs=False)

	# task HPs
	if config.shared_task_hps:
		if config.isotopic_tasks:
			task_kernel = BatchModule(task_kernel, batch_size=1, batch_in_axes=None, batch_over_inputs=False)
		else:
			task_kernel = BatchModule(task_kernel, batch_size=1, batch_in_axes=None, batch_over_inputs=True)
	else:
		if config.isotopic_tasks:
			task_kernel = BatchModule(task_kernel, batch_size=dims.T, batch_in_axes=0, batch_over_inputs=False)
		else:
			task_kernel = BatchModule(task_kernel, batch_size=dims.T, batch_in_axes=0, batch_over_inputs=True)

	return task_kernel


def build_parameters(parameters: Parameters, dims: Dimensions, config: ModelConfig) -> Parameters:
	"""
	Batch every field of `parameters` (cluster mean/kernel, task/noise kernel) according to
	`config`'s hyperparameter-sharing flags.

	`parameters` should hold the "base" mean/kernels, i.e. the ones used if all HPs were shared. Self
	-contained: unlike `build_mean`/`build_mean_kernel`/`build_task_kernel`, doesn't require calling
	each one separately, so it can be reused outside `generate_data` (e.g. to build a model's initial
	parameters from a `ModelConfig`).

	Parameters
	----------
	parameters
		Base cluster mean/kernel and task/noise kernel to batch.
	dims
		Dimensions of the dataset to generate or fit.
	config
		Model configuration, used for its HP-sharing flags (see `build_mean`/`build_mean_kernel`/
		`build_task_kernel`).

	Returns
	-------
	`parameters` with every field batched, with independent hyperparameters per output/cluster/task
	where configured.
	"""
	return Parameters(
		cluster_mean=build_mean(parameters.cluster_mean, dims, config),
		cluster_kernel=build_mean_kernel(parameters.cluster_kernel, dims, config),
		task_kernel=build_task_kernel(parameters.task_kernel, dims, config),
		noise_kernel=build_task_kernel(parameters.noise_kernel, dims, config),
	)


def sample_parameters_from_priors(key: Array, parameters: Parameters, priors: ParameterPriors) -> Parameters:
	"""
	Sample every field of `parameters` (cluster mean/kernel, task/noise kernel) uniformly from
	`priors`.

	`parameters` should already be batched (e.g. via `build_parameters`), so that hyperparameters
	are sampled independently wherever `priors` and the batching structure allow. Self-contained: can
	be reused outside `generate_data` (e.g. to sample a model's initial parameters).

	Parameters
	----------
	key
		`jax.random` PRNG key.
	parameters
		Batched cluster mean/kernel and task/noise kernel whose hyperparameters are resampled.
	priors
		Min/max bounds for each parameter of `parameters`, used to sample its hyperparameters.

	Returns
	-------
	`parameters` with every field's hyperparameters resampled from `priors`.
	"""
	subkey1, subkey2, subkey3, subkey4 = jr.split(key, 4)
	return Parameters(
		cluster_mean=sample_hps_from_uniform_priors(subkey1, parameters.cluster_mean, priors.cluster_mean_priors),
		cluster_kernel=sample_hps_from_uniform_priors(subkey2, parameters.cluster_kernel, priors.cluster_kernel_priors),
		task_kernel=sample_hps_from_uniform_priors(subkey3, parameters.task_kernel, priors.task_kernel_priors),
		noise_kernel=sample_hps_from_uniform_priors(subkey4, parameters.noise_kernel, priors.noise_kernel_priors),
	)


def generate_data(
		key: Array,
		dims: Dimensions,
		parameters: Parameters,
		config: ModelConfig,
		priors: ParameterPriors | None = None,
		input_range: tuple[int, int] = (-50, 50),
		jitter: Array = DEFAULT_JITTER
) -> tuple[Dataset, Grid, Hyperprior, Mixture, Parameters, Array, MultivariateNormal]:
	"""
	Generate a synthetic multi-task, multi-cluster dataset from GP priors.

	Parameters
	----------
	key
		`jax.random` PRNG key.
	dims
		Dimensions of the dataset to generate.
	parameters
		Cluster mean/kernel and task/noise kernels, used as priors to sample the cluster processes.
	config
		Model configuration: hyperparameter-sharing structure (task/cluster/output/feature) and
		input-sampling structure (isotopic tasks/features).
	priors
		Min/max bounds for each parameter of `parameters`, used to sample its hyperparameters.
		If None, hyperparameters are left unchanged.
	input_range
		Min and max value for input points, applied to every input dimension.
	jitter
		Diagonal jitter added before Cholesky factorizations, for numerical stability.

	Returns
	-------
	dataset
		Generated inputs and outputs.
	grid
		Grid of points and each task's mapping onto it.
	hyperprior
		Prior distribution over each mean-process's values at the grid points.
	mixture
		Cluster proportions and hard task-to-cluster assignments.
	parameters
		Sampled Parameters (cluster mean/kernel, task/noise kernels) used for generation.
	cluster_means
		Sampled mean-process values at the grid points. Shape `(K, O, F*G)`.
	tasks
		Task processes' mean and covariance, evaluated at each task's sampled input points.

	Notes
	-----
	Multi-feature (`dims.F > 1`) generation is not yet supported.
	"""
	# TODO: adapt for multi-feature
	
	# Step 1: generate the grid
	grid = generate_grid(dims.G, dims.I, input_range)  # Shape (G, I) where G = grid_size**I

	# Step 2: sample the input grid
	inputs, mappings = sample_inputs(key, grid, dims, config)  # Varying shapes

	# Step 3: batch kernels
	parameters = build_parameters(parameters, dims, config)

	# Step 4: sample HPs from priors
	if priors is not None:
		key, subkey = jr.split(key)
		parameters = sample_parameters_from_priors(subkey, parameters, priors)

	# Step 5: sample mean processes for each cluster from the mean and mean kernel, evaluated on the grid
	# Adapt grid if we are in multi-feature and features don't share inputs, to create a separate grid for each feature
	if not config.isotopic_features:
		grid = jnp.tile(grid, (dims.F,) + (1,) * grid.ndim)  # Shape (F*G, I)

	# mean has shape (K, O, F*G), cov has shape (K, O, F*G, F*G)
	hyperprior = Hyperprior(mean=parameters.cluster_mean(grid), covariance=parameters.cluster_kernel(grid))

	if config.shared_output_hps:
		sample_outputs = vmap(lambda k, m, c: sample_gp(k, m[0], c[0], jitter=jitter), in_axes=(0, None, None))
		if config.shared_cluster_hps:
			sample_clusters = vmap(lambda k, m, c: sample_outputs(k, m[0], c[0]), in_axes=(0, None, None))
		else:
			sample_clusters = vmap(lambda k, m, c: sample_outputs(k, m, c), in_axes=(0, 0, 0))
	else:
		sample_outputs = vmap(lambda k, m, c: sample_gp(k, m, c, jitter=jitter), in_axes=(0, 0, 0))
		if config.shared_cluster_hps:
			sample_clusters = vmap(lambda k, m, c: sample_outputs(k, m[0], c[0]), in_axes=(0, None, None))
		else:
			sample_clusters = vmap(lambda k, m, c: sample_outputs(k, m, c), in_axes=(0, 0, 0))
	key, subkey = jr.split(key)
	subkeys = jr.split(subkey, (dims.K, dims.O))

	cluster_means = sample_clusters(subkeys, hyperprior.mean, hyperprior.covariance)  # Shape (K, O, F*G)

	# Step 6: assign tasks to clusters
	proportions = jnp.repeat(1/dims.K, dims.K)
	responsibilities = jnp.eye(dims.K)[jnp.array(jnp.floor(jnp.arange(dims.T) / dims.T * dims.K), dtype=int)]  # Shape (T, K)
	mixture = Mixture(proportions=proportions, responsibilities=responsibilities)

	# Step 7: sample task processes for each task from the task kernel, evaluated on the task inputs
	task_means_on_grid = cluster_means[jnp.argmax(mixture.responsibilities, axis=1), ...]  # Shape (T, O, F*G)
	if config.isotopic_tasks:
		task_means = vmap(lambda t_m, m: t_m[:, m], in_axes=(0, None))(task_means_on_grid, mappings[0])  # Shape (T, O, F*N)
	else:
		task_means = vmap(lambda t_m, m: t_m[:, m], in_axes=(0, 0))(task_means_on_grid, mappings)  # Shape (T, O, F*N)

	if config.isotopic_tasks:
		task_covs = parameters.task_kernel(inputs[0]) + parameters.noise_kernel(inputs[0])
	else:
		task_covs = parameters.task_kernel(inputs) + parameters.noise_kernel(inputs)
	# Shape (T, K, O, F*N, F*N), with T=1 if shared_task_hps, K=1 if not cluster_specific_task_hps and O=1 if shared_output_hps

	if config.cluster_specific_task_hps:
		# Select covariance from the "right" cluster for each task
		task_covs = task_covs[jnp.arange(len(task_covs)),jnp.argmax( mixture.responsibilities, axis=1)]  # Shape (T, O, F*N, F*N) with T=1 if shared_task_hps and O=1 if shared_output_hps
	else:
		task_covs = task_covs[:, 0, ...]  # Shape (T, O, F*N, F*N) with T=1 if shared_task_hps and O=1 if shared_output_hps

	tasks = MultivariateNormal(mean=task_means, covariance=task_covs)

	if config.shared_output_hps:
		sample_outputs = vmap(lambda k, m, c: sample_gp(k, m, c[0], jitter=jitter), in_axes=(0, 0, None))
		if config.isotopic_tasks and config.shared_task_hps:
			sample_tasks = vmap(lambda k, m, c: sample_outputs(k, m, c[0]), in_axes=(0, 0, None))
		else:
			sample_tasks = vmap(lambda k, m, c: sample_outputs(k, m, c), in_axes=(0, 0, 0))
	else:
		sample_outputs = vmap(lambda k, m, c: sample_gp(k, m, c, jitter=jitter), in_axes=(0, 0, 0))
		if config.isotopic_tasks and config.shared_task_hps:
			sample_tasks = vmap(lambda k, m, c: sample_outputs(k, m, c[0]), in_axes=(0, 0, None))
		else:
			sample_tasks = vmap(lambda k, m, c: sample_outputs(k, m, c), in_axes=(0, 0, 0))
	key, subkey = jr.split(key)
	subkeys = jr.split(subkey, (dims.T, dims.O))

	outputs = sample_tasks(subkeys, task_means, task_covs).mT  # Shape (T, F*N, O)

	dataset = Dataset(inputs=inputs, outputs=outputs)
	grid = Grid(points=grid, mappings=mappings)

	return dataset, grid, hyperprior, mixture, parameters, cluster_means, tasks


class AbstractDataRemover(eqx.Module):
	"""
	Base class for modules that remove data points from a Dataset generated by `generate_data`.
	"""
	@abstractmethod
	def __call__(self, key: Array, dataset: Dataset, config: DataRemovalConfig,
				grid: Grid | None = None) -> Dataset | tuple[Dataset, Grid]:
		"""
		Remove data points from `dataset`, according to `config`.

		Parameters
		----------
		key
			`jax.random` PRNG key.
		dataset
			Dataset to remove points from.
		config
			Removal configuration.
		grid
			Kept for API parity, passed through unchanged: a missing point is marked by NaN in
			`dataset.outputs` only, inputs/grid are never touched.

		Returns
		-------
		Dataset with points removed, or `(Dataset, Grid)` if `grid` was given.
		"""
		...


class RandomDataRemover(AbstractDataRemover):
	"""
	Removes data points at random, per `DataRemovalConfig`. A missing point is marked by NaN in
	`dataset.outputs`; this masking is read downstream in `mimosa-ml.hyperpost`, `mimosa-ml.nll` and
	`mimosa-ml.prediction`.
	"""
	def __call__(self, key: Array, dataset: Dataset, config: DataRemovalConfig,
				grid: Grid | None = None) -> Dataset | tuple[Dataset, Grid]:
		"""
		See `AbstractDataRemover.__call__`.
		"""
		# TODO: adapt for multi-feature
		T, N, O = dataset.outputs.shape

		# 1: random remove_mask, shape (T, N, O). "Which k of N points are removed" is drawn without
		# replacement by ranking iid uniform scores per row and keeping the k lowest ranks: no python loop,
		# no vmap, works whether the row is (task,) or (task, output).
		shape = (T, N) if config.same_missing_across_outputs else (T, O, N)
		key_scores, key_counts = jr.split(key)
		ranks = jnp.argsort(jnp.argsort(jr.uniform(key_scores, shape), axis=-1), axis=-1)
		counts = jr.randint(key_counts, shape[:-1], 0, config.max_missing + 1) if config.random_missing_count \
			else jnp.full(shape[:-1], config.max_missing)
		selected = ranks < counts[..., None]  # shape (T, N) or (T, O, N)

		remove_mask = jnp.broadcast_to(selected[..., None], (T, N, O)) if config.same_missing_across_outputs \
			else jnp.moveaxis(selected, 1, 2)  # (T, O, N) -> (T, N, O)

		# 2: remove outputs (and known noise, if any) in one line
		outputs = jnp.where(remove_mask, jnp.nan, dataset.outputs)
		outputs_known_noise = None if dataset.known_output_noise is None \
			else jnp.where(remove_mask, jnp.nan, dataset.known_output_noise)

		dataset = Dataset(inputs=dataset.inputs, outputs=outputs, known_output_noise=outputs_known_noise)
		return (dataset, grid) if grid is not None else dataset
