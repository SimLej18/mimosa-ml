# %% tags=["remove-cell"]
import importlib.util, subprocess, sys
if importlib.util.find_spec("mimosa-ml") is None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "mimosa-ml"], check=True)
# %% [markdown]
"""
# Basic usage of Mimosa

This example walks through the full pipeline on a synthetic dataset: configure the dimensions,
generate data and remove some points at random, fit a `BasicModel`, then predict a task and sample
from that prediction.

Written using jupytext's py:percent format. This script can be run cell-by-cell or as a usual Python
script.
"""

# %% [markdown]
"""
## Getting started

First, the usual imports and configs:
"""

# %%
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

key = jr.PRNGKey(42)
plt.rcParams['figure.dpi']=300

# %% [markdown]
"""
Throughout the framework, Mimosa keeps track of the dimensions of the datasets and the shape of the
model parameters via two objects: `Dimensions` and `ModelConfig`. Here we also generate the data
ourselves, so these two objects describe both the data we create and the model we will fit on it.
"""

# %% 1. Configuration
# Dimensions: T tasks, K clusters, I input dims, C channels, O correlated outputs, N points
# observed per task, G points in the full grid.
dims = Dimensions(T=32, K=2, I=1, C=2, O=1, N=50, G=150)

# ModelConfig controls which hyperparameters are shared (across tasks/clusters/channels/outputs) and
# whether tasks/outputs share input locations.
model_config = ModelConfig(
	shared_task_hps=True,
	shared_cluster_hps=True,
	shared_channel_hps=True,
	cluster_specific_task_hps=False,
	isotopic_tasks=False,
)

# How many points to remove at random per task, to simulate missing data.
removal_config = DataRemovalConfig(max_missing=5, random_missing_count=True, same_missing_across_channels=False)

# %% [markdown]
"""
A model is described by 4 parameters:
* `cluster_mean`: the mean function of the mean-processes, i.e. the long-term trend clusters are
  centered on
* `cluster_kernel`: the kernel of the mean-processes, i.e. their "shape" (smooth, wiggly, periodic...)
* `task_kernel`: the kernel of the tasks, i.e. how they vary around their mean-process
* `noise_kernel`: the noise on the observed points

Below, they play the role of the *true* parameters used to synthesise the dataset.
"""

# %% 2. Generative parameters
# These are the "true" parameters used to synthesise the toy dataset below. Swap any kernel/mean for
# another kernax one (e.g. MaternKernel, PeriodicKernel, LinearMean, ...) to change the shape of the
# data generated.
true_params = Parameters(
	cluster_mean=ZeroMean(),
	cluster_kernel=VarianceKernel(5.0) * SEKernel(length_scale=.5),
	task_kernel=VarianceKernel(1.0) * SEKernel(length_scale=.4),
	noise_kernel=WhiteNoiseKernel(noise=.05),
)

# %% 3. Generate synthetic data, then remove points at random
key, gen_key, removal_key = jr.split(key, 3)

dataset, grid, hyperprior, true_mixture, true_params, cluster_means, tasks = generate_data(
	gen_key, dims, true_params, model_config, input_range=[(-2.5, 2.5)]
)
dataset = RandomDataRemover()(removal_key, dataset, removal_config)

# %% 3bis. Alternatively, you can load a dataset from a local file through load_csv.
# Check load_csv's doc or open the csv file to see the expected file format.
save_csv("data/dummy.csv", dataset)
dataset = load_csv("data/dummy.csv")

# %% 4. Plot the raw dataset (coloured by each task's true cluster)
fig, ax = plot_dataset(dataset, dims, mixture=true_mixture, figsize=(8 * dims.C, 6))
fig.suptitle("Synthetic dataset (colored by true cluster)")
plt.show()

# %% [markdown]
"""
## Training the model

The model knows nothing about the parameters above: it starts from its own guess and optimises it.
The initial values do not matter too much, but their *structure* does, which is what
`build_parameters` takes care of.
"""

# %% 5. Instantiate the model
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

# %% 6. Fit
# Grid construction (union of every task's input points) isn't jit-compatible, so it's built once
# here by the caller, outside of fit/predict, rather than owned by the model — see
# mimosa.grid.GridBuilder. Swap UnionGrid for another GridBuilder to change how the grid is built.
fitted_grid = UnionGrid()(dataset.inputs)

fitted_params, fitted_mixture = model.fit(dataset, fitted_grid, mixture_proportions, init_params, n_iter=50)

# %% 7. Plot the fitted clusters (mean-processes)
hyperposterior = model.hyperpost(dataset, fitted_grid, fitted_mixture, fitted_params, jitter=model.jitter)

fig, ax = plot_dataset(dataset, dims, mixture=true_mixture, figsize=(8 * dims.C, 6), alpha=.1)
fig, ax = plot_clusters(fitted_grid, dims, hyperposterior=hyperposterior, figsize=(8 * dims.C, 6), fig=fig, ax=ax)
fig.suptitle("Fitted clusters (mean-processes) on the dataset")
plt.show()

# %% [markdown]
"""
## Predicting

Predictions are multimodal: the model returns one Gaussian process per cluster for each task. Here we
simply keep the one of the task's most probable cluster.

Try changing `t_id` to see another task, and `k_id` to see what the prediction would look like if the
task belonged to that cluster instead!
"""

# %% 8. Predict
predictions = model.predict(dataset, fitted_grid, fitted_mixture, fitted_params)  # MultivariateNormal, batched (T, K, C, O*G)

t_id, c_id = 0, 0
k_id = int(fitted_mixture.assignments[t_id])  # task's dominant cluster
prediction = predictions[t_id, k_id, c_id]

# %% 9. Plot the prediction: observed points, cluster means, and predictive mean + confidence interval
fig, ax = plot_single_task_prediction(
	dataset, fitted_grid, dims, hyperposterior, fitted_mixture, t_id, c_id, prediction=prediction, figsize=(8 * dims.C, 6)
)
fig.suptitle(f"Prediction — task {t_id}, channel {c_id}")
plt.show()

# %% 10. Draw samples from the prediction and plot them alongside it
key, sample_key = jr.split(key)
n_samples = 64
sample_keys = jr.split(sample_key, n_samples)
samples = vmap(lambda k: sample_gp(k, prediction.mean, prediction.covariance))(sample_keys)  # (S, O*G)

fig, ax = plot_single_task_prediction(
	dataset, fitted_grid, dims, hyperposterior, fitted_mixture, t_id, c_id, samples=samples, figsize=(8 * dims.C, 6)
)
fig.suptitle(f"Prediction samples — task {t_id}, channel {c_id}")
plt.show()
