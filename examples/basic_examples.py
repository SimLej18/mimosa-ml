"""
Basic usage example of mimosa-ml.

Walks through the full pipeline: configure dimensions -> generate a synthetic dataset and remove
some points at random -> plot the raw dataset -> fit a BasicModel -> plot the fitted clusters ->
predict a task's output -> plot the prediction (mean + confidence interval) and samples from it.

Meant to be run cell-by-cell (e.g. in PyCharm/VSCode's "#%%" notebook mode), or as a plain script.
"""
import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_disable_jit", False)
import jax.random as jr
import jax.numpy as jnp
from jax import vmap
import matplotlib.pyplot as plt

from kernax import ZeroMean, VarianceKernel, SEKernel, WhiteNoiseKernel

from mimosa import (
	Dimensions, ModelConfig, DataRemovalConfig, Parameters,
	BasicModel, generate_data, RandomDataRemover, save_csv, load_csv, build_parameters,
)
from mimosa.grid import UnionGrid
from mimosa.plot import plot_dataset, plot_clusters, plot_single_task_prediction
from mimosa.sampling import sample_gp

key = jr.PRNGKey(0)
plt.rcParams['figure.dpi']=300

#%% 1. Configuration
# Dimensions: T tasks, K clusters, I/O input/output dims, N points observed
# per task, G points in the full grid.
dims = Dimensions(T=32, K=2, I=1, O=2, F=1, N=50, G=150)

# ModelConfig controls which hyperparameters are shared (across tasks/clusters/outputs/features) and
# whether tasks/features share input locations.
model_config = ModelConfig(
	shared_task_hps=True,
	shared_cluster_hps=True,
	shared_output_hps=True,
	cluster_specific_task_hps=False,
	isotopic_tasks=False,
)

# How many points to remove at random per task, to simulate missing data.
removal_config = DataRemovalConfig(max_missing=5, random_missing_count=True, same_missing_across_outputs=False)

#%% 2. Generative parameters
# These are the "true" parameters used to synthesise the toy dataset below. Swap any kernel/mean for
# another kernax one (e.g. MaternKernel, PeriodicKernel, LinearMean, ...) to change the shape of the
# data generated.
true_params = Parameters(
	cluster_mean=ZeroMean(),
	cluster_kernel=VarianceKernel(5.0) * SEKernel(length_scale=.5),
	task_kernel=VarianceKernel(1.0) * SEKernel(length_scale=.4),
	noise_kernel=WhiteNoiseKernel(noise=.05),
)

#%% 3. Generate synthetic data, then remove points at random
key, gen_key, removal_key = jr.split(key, 3)

dataset, grid, hyperprior, true_mixture, true_params, cluster_means, tasks = generate_data(
	gen_key, dims, true_params, model_config, input_range=(-2.5, 2.5)
)
dataset = RandomDataRemover()(removal_key, dataset, removal_config)

#%% 3bis. Alternatively, you can load a dataset from a local file through load_csv.
save_csv("./dummy.csv", dataset)
dataset = load_csv("./dummy.csv")

#%% 4. Plot the raw dataset (coloured by each task's true cluster)
fig, ax = plot_dataset(dataset, mixture=true_mixture, figsize=(8*dims.O, 6))
fig.suptitle("Synthetic dataset (colored by true cluster)")
plt.show()

#%% 5. Instantiate the model
# n_clusters can differ from the true K above (the model doesn't know it); jitter is the numerical
# stabiliser added before Cholesky factorizations, only increase it if you hit factorization errors.
key, model_key = jr.split(key)
model = BasicModel(prng_key=model_key, n_clusters=dims.K)

# Starting guess for the parameters to fit. In practice these would be a rough, uninformed guess
# rather than the true generative ones — feel free to try different starting kernels/values here.
init_params = Parameters(
		cluster_mean=ZeroMean(),
		cluster_kernel=VarianceKernel(2.0) * SEKernel(length_scale=1.),
		task_kernel=VarianceKernel(.5) * SEKernel(length_scale=.5),
		noise_kernel=WhiteNoiseKernel(noise=0.5))

# build_parameters batches the base kernels/mean below to match model_config's sharing structure
# (same helper generate_data uses internally), so their shapes line up with what model.fit expects.
init_params = build_parameters(init_params, dims, model_config)

# proportions of the dataset in each cluster, a priori
mixture_proportions = jnp.repeat(1 / dims.K, dims.K)  # fixed, equal weight per cluster

#%% 6. Fit
# Grid construction (union of every task's input points) isn't jit-compatible, so it's built once
# here by the caller, outside of fit/predict, rather than owned by the model — see
# mimosa.grid.GridBuilder. Swap UnionGrid for another GridBuilder to change how the grid is built.
fitted_grid = UnionGrid()(dataset.inputs)

fitted_params, fitted_mixture = model.fit(dataset, fitted_grid, mixture_proportions, init_params, n_iter=50)

#%% 7. Plot the fitted clusters (mean-processes)
hyperposterior = model.hyperpost(dataset, fitted_grid, fitted_mixture, fitted_params, jitter=model.jitter)

fig, ax = plot_clusters(fitted_grid, hyperposterior=hyperposterior, figsize=(8*dims.O, 6))
fig.suptitle("Fitted clusters (mean-processes)")
plt.show()

#%% 8. Predict
predictions = model.predict(dataset, fitted_grid, fitted_mixture, fitted_params)  # MultivariateNormal, batched (T, K, O, G)

t_id, o_id = 0, 0
k_id = int(fitted_mixture.assignments[t_id])  # task's dominant cluster
prediction = predictions[t_id, k_id, o_id]

#%% 9. Plot the prediction: observed points, cluster means, and predictive mean + confidence interval
fig, ax = plot_single_task_prediction(
	dataset, fitted_grid, hyperposterior, fitted_mixture, t_id, o_id, prediction=prediction, figsize=(8*dims.O, 6)
)
fig.suptitle(f"Prediction — task {t_id}, output {o_id}")
plt.show()

#%% 10. Draw samples from the prediction and plot them alongside it
key, sample_key = jr.split(key)
n_samples = 64
sample_keys = jr.split(sample_key, n_samples)
samples = vmap(lambda k: sample_gp(k, prediction.mean, prediction.covariance))(sample_keys)  # (S, G)

fig, ax = plot_single_task_prediction(
	dataset, fitted_grid, hyperposterior, fitted_mixture, t_id, o_id, samples=samples, figsize=(8*dims.O, 6)
)
fig.suptitle(f"Prediction samples — task {t_id}, output {o_id}")
plt.show()
