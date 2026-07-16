"""
Plotting utilities for mimosa-ml, using matplotlib with a seaborn-style theme.

Composable plotting functions for Dataset points (`plot_output`/`plot_task`/`plot_dataset`),
mean-processes (`plot_single_cluster_single_output`/`plot_single_cluster`/`plot_clusters`), and
per-task predictions (`plot_single_task_prediction`).

Convention
----------
Subplots: columns = outputs (`O`), rows = features (`F`, left at 1 for now since multi-feature
generation isn't supported yet). Every plotting function accepts optional `fig`/`ax` so plots can
be composed into larger figures (see `_get_fig_ax`). Selector arguments (`t_id`, `k_id`, `o_id`,
`f_id`) pick a single task/cluster/output/feature index (int), or "all" (default) to plot every one.

Limitations
-----------
Only 1D inputs (`I == 1`) and single-feature datasets (`F == 1`) are supported, matching the current
limitations of `mimosa.synthetic.generate_data`.
"""

from __future__ import annotations

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from jax import Array

from mimosa.data_structures import Dataset, Grid, Hyperposterior, Hyperprior, Mixture, MultivariateNormal

plt.style.use("seaborn-v0_8-whitegrid")


IdArg = int | Literal["all"]

_F = 1  # multi-feature not supported yet in generate_data; F is always 1
_DEFAULT_SCATTER_KWARGS = {"s": 15, "alpha": 0.7}


def _resolve_ids(id_arg: IdArg, size: int) -> list[int]:
	"""
	Resolve a t_id/k_id/o_id/f_id argument into a list of indices: every index if "all",
	or a single-element list if an int.
	"""
	if id_arg == "all":
		return list(range(size))
	if isinstance(id_arg, int):
		if not (0 <= id_arg < size):
			raise ValueError(f"Index {id_arg} out of range for size {size}.")
		return [id_arg]
	raise TypeError(f"Expected 'all' or int, got {id_arg!r}.")


def _get_fig_ax(fig, ax, nrows: int, ncols: int, figsize: tuple[float, float] | None = None):
	"""
	Get or create a (fig, ax) pair with an `nrows` x `ncols` grid of axes, returning `ax` as a 2D
	array regardless of grid size. Reuses `fig`/`ax` if given, so plots can be composed together.

	Either way, the figure's layout engine is set to "constrained": unlike a one-shot
	`fig.tight_layout()` call, it keeps titles, axis labels and figure-level legends from
	overlapping (or from being clipped at the figure's edge) even as more elements are added later
	by composing further plots onto the same `fig`/`ax`, or by the caller (e.g. `fig.suptitle`).
	"""
	if ax is not None:
		ax_arr = np.atleast_2d(ax)
		if fig is None:
			fig = ax_arr.flat[0].figure
		fig.set_layout_engine("constrained")
		return fig, ax_arr

	if figsize is None:
		figsize = (4 * ncols, 3 * nrows)
	fig, ax = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False, layout="constrained")
	return fig, ax


def _mvn_dims(hyperprior, hyperposterior) -> tuple[int, int]:
	"""
	Infer the true `(K, O)` of a mean-process distribution from whichever of `hyperprior`/
	`hyperposterior` is given. Prefers `hyperposterior`: `hyperprior` may collapse a batch axis to
	1 under shared hyperparameters (broadcast, see `_mvn_cell`), while `hyperposterior` (e.g. from
	`mimosa.hyperpost.hyperpost`) is always fully expanded.
	"""
	ref = hyperposterior if hyperposterior is not None else hyperprior
	return ref.mean.shape[0], ref.mean.shape[1]


def _mvn_cell(obj, k_id: int, o_id: int):
	"""
	Index a Hyperprior/Hyperposterior's `(K, O, ...)` mean/covariance at `(k_id, o_id)`,
	broadcasting any axis of size 1 (shared hyperparameters) to index 0 instead.
	"""
	k = k_id if obj.mean.shape[0] > 1 else 0
	o = o_id if obj.mean.shape[1] > 1 else 0
	return obj.mean[k, o], obj.covariance[k, o]


def _task_xy(dataset: Dataset, t_id: int, o_id: int):
	"""
	Extract a single task's observed `(x, y)` points for a single output, as numpy arrays with
	missing (NaN) points already dropped.
	"""
	# inputs shape (#T, N, 1), #T is 1 (isotopic tasks) or T
	x = np.asarray(dataset.inputs[0 if dataset.inputs.shape[0] == 1 else t_id, :, 0])
	y = np.asarray(dataset.outputs[t_id, :, o_id])
	mask = ~np.isnan(y)
	return x[mask], y[mask]


def _cluster_palette(K: int) -> list:
	"""
	Build one color per cluster index in `[0, K)`. Shared by `plot_dataset` (colored by mixture
	assignment) and `plot_clusters` (colored by cluster index), so the two can be composed on the
	same axes with matching colors.
	"""
	cmap = plt.get_cmap("tab10" if K <= 10 else "tab20")
	return [cmap(k % cmap.N) for k in range(K)]


def plot_output(
	dataset: Dataset,
	t_id: int,
	o_id: int,
	f_id: IdArg = "all",
	fig=None,
	ax=None,
	figsize: tuple[float, float] | None = None,
	color="C0",
	**scatter_kwargs,
):
	"""
	Scatter-plot a single task's observed points for a single output, one subplot per feature.

	x-axis is the input value (only 1D inputs, `I == 1`, are supported for now), y-axis is the
	output value.

	Parameters
	----------
	dataset
		Dataset to plot, as returned by `generate_data`.
	t_id
		Index of the task to plot.
	o_id
		Index of the output to plot.
	f_id
		"all" (default) or int, restrict the plot to a single feature. Has no effect yet since
		multi-feature datasets aren't supported (F is always 1).
	fig, ax
		Existing figure/axes to draw on, to combine with other plots. If given, `ax` must already
		have shape `(len(f_id), 1)`. A new figure/axes grid is created if None.
	figsize
		Passed to `plt.subplots` when a new figure is created.
	color
		Color of this task's points.
	**scatter_kwargs
		Extra keyword arguments forwarded to `ax.scatter`, overriding the defaults (s=15, alpha=0.7).

	Returns
	-------
	fig, ax
		The (possibly newly created) figure and 2D array of axes, shape `(len(f_id), 1)`.
	"""
	if dataset.inputs.shape[-1] != 1:
		raise NotImplementedError("plot_output only supports 1D inputs (I=1) for now.")

	f_ids = _resolve_ids(f_id, _F)

	fig, ax = _get_fig_ax(fig, ax, len(f_ids), 1, figsize=figsize)

	x, y = _task_xy(dataset, t_id, o_id)

	kwargs = _DEFAULT_SCATTER_KWARGS | scatter_kwargs

	for row, f in enumerate(f_ids):
		a = ax[row, 0]
		a.scatter(x, y, color=color, **kwargs)
		a.set_title(f"output {o_id}" + (f", feature {f}" if len(f_ids) > 1 else ""))
		a.set_xlabel("input")
		a.set_ylabel("output value")

	return fig, ax


def plot_task(
	dataset: Dataset,
	t_id: int,
	o_id: IdArg = "all",
	f_id: IdArg = "all",
	fig=None,
	ax=None,
	figsize: tuple[float, float] | None = None,
	color="C0",
	**scatter_kwargs,
):
	"""
	Scatter-plot a single task's observed points, looping `plot_output` over outputs.

	Parameters
	----------
	dataset
		Dataset to plot, as returned by `generate_data`.
	t_id
		Index of the task to plot.
	o_id, f_id
		"all" (default) or int, restrict the plot to a single output/feature. `f_id` has no effect
		yet since multi-feature datasets aren't supported (F is always 1).
	fig, ax
		Existing figure/axes to draw on, to combine with other plots. If given, `ax` must already
		have shape `(len(f_id), len(o_id))`. A new figure/axes grid is created if None.
	figsize
		Passed to `plt.subplots` when a new figure is created.
	color
		Color of this task's points.
	**scatter_kwargs
		Extra keyword arguments forwarded to `ax.scatter`, overriding the defaults (s=15, alpha=0.7).

	Returns
	-------
	fig, ax
		The (possibly newly created) figure and 2D array of axes, shape `(len(f_id), len(o_id))`.
	"""
	O = dataset.outputs.shape[-1]

	o_ids = _resolve_ids(o_id, O)
	f_ids = _resolve_ids(f_id, _F)

	fig, ax = _get_fig_ax(fig, ax, len(f_ids), len(o_ids), figsize=figsize)

	for col, o in enumerate(o_ids):
		plot_output(dataset, t_id, o, f_id=f_id, fig=fig, ax=ax[:, col:col + 1], color=color, **scatter_kwargs)

	return fig, ax


def plot_dataset(
	dataset: Dataset,
	mixture: Mixture | None = None,
	t_id: IdArg = "all",
	o_id: IdArg = "all",
	f_id: IdArg = "all",
	fig=None,
	ax=None,
	figsize: tuple[float, float] | None = None,
	legend: bool = True,
	**scatter_kwargs,
):
	"""
	Scatter-plot a Dataset's observed points, looping `plot_task` over tasks. Points are colored
	by each task's hard cluster assignment if `mixture` is given.

	Parameters
	----------
	dataset
		Dataset to plot, as returned by `generate_data`.
	mixture
		Cluster assignments used to color tasks. If None, every task shares one color.
	t_id, o_id, f_id
		"all" (default) or int, restrict the plot to a single task/output/feature. `f_id` has no
		effect yet since multi-feature datasets aren't supported (F is always 1).
	fig, ax
		Existing figure/axes to draw on, to combine with other plots. If given, `ax` must already
		have shape `(len(f_id), len(o_id))`. A new figure/axes grid is created if None.
	figsize
		Passed to `plt.subplots` when a new figure is created.
	legend
		If True and `mixture` is given, add a legend mapping colors to cluster indices.
	**scatter_kwargs
		Extra keyword arguments forwarded to `ax.scatter`, overriding the defaults (s=15, alpha=0.7).

	Returns
	-------
	fig, ax
		The (possibly newly created) figure and 2D array of axes, shape `(len(f_id), len(o_id))`.
	"""
	T, _, O = dataset.outputs.shape

	t_ids = _resolve_ids(t_id, T)
	o_ids = _resolve_ids(o_id, O)
	f_ids = _resolve_ids(f_id, _F)

	fig, ax = _get_fig_ax(fig, ax, len(f_ids), len(o_ids), figsize=figsize)

	if mixture is None:
		colors = {t: "C0" for t in t_ids}
		handles = []
	else:
		K = mixture.proportions.shape[0]
		palette = _cluster_palette(K)
		assignments = np.asarray(mixture.assignments)
		colors = {t: palette[assignments[t]] for t in t_ids}
		handles = [
			plt.Line2D([0], [0], marker="o", linestyle="", color=palette[k], label=f"cluster {k}")
			for k in range(K)
		]

	for t in t_ids:
		plot_task(dataset, t, o_id=o_id, f_id=f_id, fig=fig, ax=ax, color=colors[t], **scatter_kwargs)

	if legend and handles:
		fig.legend(handles=handles, loc="outside right center")

	return fig, ax


def plot_single_cluster_single_output(
	grid: Grid,
	k_id: int,
	o_id: int,
	hyperprior: Hyperprior | None = None,
	hyperposterior: Hyperposterior | None = None,
	f_id: IdArg = "all",
	fig=None,
	ax=None,
	figsize: tuple[float, float] | None = None,
	color="C0",
	ci_scale: float = 1.96,
	ci_alpha: float = 0.2,
	**line_kwargs,
):
	"""
	Plot a single mean-process's values at the grid points, for a single output, one subplot per
	feature: `hyperprior.mean` as a dashed line, `hyperposterior.mean` as a solid line, and a
	confidence interval shaded from the diagonal of `hyperposterior.covariance`.

	Parameters
	----------
	grid
		Grid of points the mean-process is evaluated at (only 1D inputs, `I == 1`, supported for
		now). x-axis of the plot.
	k_id
		Index of the mean-process (cluster) to plot.
	o_id
		Index of the output to plot.
	hyperprior
		Prior distribution over the mean-process's grid values, shape `(K, O, F*G)`/`(K, O, F*G, F*G)`.
		Plotted as a dashed line if given; skipped otherwise.
	hyperposterior
		Posterior distribution over the mean-process's grid values, same shape as `hyperprior`.
		Plotted as a solid line with a shaded confidence interval if given; skipped otherwise.
	f_id
		"all" (default) or int, restrict the plot to a single feature. Has no effect yet since
		multi-feature datasets aren't supported (F is always 1).
	fig, ax
		Existing figure/axes to draw on, to combine with other plots. If given, `ax` must already
		have shape `(len(f_id), 1)`. A new figure/axes grid is created if None.
	figsize
		Passed to `plt.subplots` when a new figure is created.
	color
		Color of this cluster's prior/posterior lines and confidence interval.
	ci_scale
		Number of standard deviations spanned by the shaded confidence interval (default 1.96, ~95%).
	ci_alpha
		Opacity of the shaded confidence interval.
	**line_kwargs
		Extra keyword arguments forwarded to `ax.plot` for both the prior and posterior lines.

	Returns
	-------
	fig, ax
		The (possibly newly created) figure and 2D array of axes, shape `(len(f_id), 1)`.
	"""
	if hyperprior is None and hyperposterior is None:
		raise ValueError("At least one of hyperprior/hyperposterior must be given.")
	if grid.points.shape[-1] != 1:
		raise NotImplementedError("plot_single_cluster_single_output only supports 1D inputs (I=1) for now.")

	f_ids = _resolve_ids(f_id, _F)

	fig, ax = _get_fig_ax(fig, ax, len(f_ids), 1, figsize=figsize)

	x = np.asarray(grid.points[:, 0])

	for row, f in enumerate(f_ids):
		a = ax[row, 0]
		if hyperprior is not None:
			prior_mean, _ = _mvn_cell(hyperprior, k_id, o_id)
			a.plot(x, np.asarray(prior_mean), linestyle="--", color=color, **line_kwargs)
		if hyperposterior is not None:
			post_mean, post_cov = _mvn_cell(hyperposterior, k_id, o_id)
			post_mean = np.asarray(post_mean)
			post_std = np.sqrt(np.diagonal(np.asarray(post_cov)))
			a.plot(x, post_mean, linestyle="-", color=color, **line_kwargs)
			a.fill_between(x, post_mean - ci_scale * post_std, post_mean + ci_scale * post_std,
							color=color, alpha=ci_alpha, linewidth=0)
		a.set_title(f"output {o_id}" + (f", feature {f}" if len(f_ids) > 1 else ""))
		a.set_xlabel("input")
		a.set_ylabel("output value")

	return fig, ax


def plot_single_cluster(
	grid: Grid,
	k_id: int,
	o_id: IdArg = "all",
	hyperprior: Hyperprior | None = None,
	hyperposterior: Hyperposterior | None = None,
	f_id: IdArg = "all",
	fig=None,
	ax=None,
	figsize: tuple[float, float] | None = None,
	color="C0",
	ci_scale: float = 1.96,
	ci_alpha: float = 0.2,
	**line_kwargs,
):
	"""
	Plot a single mean-process's values at the grid points, looping `plot_single_cluster_single_output`
	over outputs.

	Parameters
	----------
	grid
		Grid of points the mean-process is evaluated at.
	k_id
		Index of the mean-process (cluster) to plot.
	o_id, f_id
		"all" (default) or int, restrict the plot to a single output/feature. `f_id` has no effect
		yet since multi-feature datasets aren't supported (F is always 1).
	hyperprior, hyperposterior
		See `plot_single_cluster_single_output`. At least one must be given.
	fig, ax
		Existing figure/axes to draw on, to combine with other plots. If given, `ax` must already
		have shape `(len(f_id), len(o_id))`. A new figure/axes grid is created if None.
	figsize
		Passed to `plt.subplots` when a new figure is created.
	color
		Color of this cluster's prior/posterior lines and confidence interval.
	ci_scale, ci_alpha
		See `plot_single_cluster_single_output`.
	**line_kwargs
		Extra keyword arguments forwarded to `ax.plot` for both the prior and posterior lines.

	Returns
	-------
	fig, ax
		The (possibly newly created) figure and 2D array of axes, shape `(len(f_id), len(o_id))`.
	"""
	if hyperprior is None and hyperposterior is None:
		raise ValueError("At least one of hyperprior/hyperposterior must be given.")
	_, O = _mvn_dims(hyperprior, hyperposterior)

	o_ids = _resolve_ids(o_id, O)
	f_ids = _resolve_ids(f_id, _F)

	fig, ax = _get_fig_ax(fig, ax, len(f_ids), len(o_ids), figsize=figsize)

	for col, o in enumerate(o_ids):
		plot_single_cluster_single_output(
			grid, k_id, o, hyperprior=hyperprior, hyperposterior=hyperposterior, f_id=f_id,
			fig=fig, ax=ax[:, col:col + 1], color=color, ci_scale=ci_scale, ci_alpha=ci_alpha, **line_kwargs)

	return fig, ax


def plot_clusters(
	grid: Grid,
	k_id: IdArg = "all",
	o_id: IdArg = "all",
	f_id: IdArg = "all",
	hyperprior: Hyperprior | None = None,
	hyperposterior: Hyperposterior | None = None,
	fig=None,
	ax=None,
	figsize: tuple[float, float] | None = None,
	legend: bool = True,
	ci_scale: float = 1.96,
	ci_alpha: float = 0.2,
	**line_kwargs,
):
	"""
	Plot every mean-process's values at the grid points, looping `plot_single_cluster` over
	clusters. Each cluster gets its own color, matching `plot_dataset`'s mixture-based coloring so
	the two can be composed on the same axes.

	Parameters
	----------
	grid
		Grid of points the mean-processes are evaluated at.
	k_id, o_id, f_id
		"all" (default) or int, restrict the plot to a single cluster/output/feature. `f_id` has no
		effect yet since multi-feature datasets aren't supported (F is always 1).
	hyperprior, hyperposterior
		See `plot_single_cluster_single_output`. At least one must be given.
	fig, ax
		Existing figure/axes to draw on, to combine with other plots (e.g. `plot_dataset`). If
		given, `ax` must already have shape `(len(f_id), len(o_id))`. A new figure/axes grid is
		created if None.
	figsize
		Passed to `plt.subplots` when a new figure is created.
	legend
		If True, add a legend mapping colors to cluster indices. Set to False on one of the calls
		when composing with `plot_dataset` on the same figure, to avoid a duplicate legend.
	ci_scale, ci_alpha
		See `plot_single_cluster_single_output`.
	**line_kwargs
		Extra keyword arguments forwarded to `ax.plot` for both the prior and posterior lines.

	Returns
	-------
	fig, ax
		The (possibly newly created) figure and 2D array of axes, shape `(len(f_id), len(o_id))`.
	"""
	if hyperprior is None and hyperposterior is None:
		raise ValueError("At least one of hyperprior/hyperposterior must be given.")
	K, O = _mvn_dims(hyperprior, hyperposterior)

	k_ids = _resolve_ids(k_id, K)
	o_ids = _resolve_ids(o_id, O)
	f_ids = _resolve_ids(f_id, _F)

	fig, ax = _get_fig_ax(fig, ax, len(f_ids), len(o_ids), figsize=figsize)

	palette = _cluster_palette(K)

	for k in k_ids:
		plot_single_cluster(
			grid, k, o_id=o_id, hyperprior=hyperprior, hyperposterior=hyperposterior, f_id=f_id,
			fig=fig, ax=ax, color=palette[k], ci_scale=ci_scale, ci_alpha=ci_alpha, **line_kwargs)

	if legend:
		handles = [
			plt.Line2D([0], [0], color=palette[k], label=f"cluster {k}")
			for k in k_ids
		]
		fig.legend(handles=handles, loc="outside right center")

	return fig, ax


def plot_single_task_prediction(
	dataset: Dataset,
	grid: Grid,
	hyperposterior: Hyperposterior,
	mixture: Mixture,
	t_id: int,
	o_id: int,
	prediction: MultivariateNormal | None = None,
	samples: Array | None = None,
	f_id: IdArg = "all",
	fig=None,
	ax=None,
	figsize: tuple[float, float] | None = None,
	legend: bool = True,
	point_color="black",
	prediction_color="black",
	sample_color="gray",
	ci_scale: float = 1.96,
	ci_alpha: float = 0.2,
	sample_alpha: float = 0.15,
	**scatter_kwargs,
):
	"""
	Plot a single task's prediction for a single output, one subplot per feature: the task's
	observed points (scatter), every mean-process's hyperposterior mean (dashed lines, one per
	cluster, colored and made transparent by that cluster's mixture coefficient for this task), and
	optionally the task's predictive distribution (solid mean line + shaded confidence interval)
	and samples drawn from it (thin lines, all the same color/alpha).

	Parameters
	----------
	dataset
		Dataset the task's observed points are read from.
	grid
		Grid of points the hyperposterior/prediction are evaluated at (only 1D inputs, `I == 1`,
		supported for now). x-axis of the plot.
	hyperposterior
		Posterior distribution over every mean-process's values at the grid points.
	mixture
		Soft-clustering of the dataset's tasks into mean-processes, used to set each hyperposterior
		line's alpha to this task's mixture coefficient towards that cluster.
	t_id
		Index of the task to plot.
	o_id
		Index of the output to plot.
	prediction
		This task's predictive distribution at the grid points (e.g. from
		`mimosa.prediction.predict_task_output`), for a single output and mean-process. Plotted as a
		solid mean line with a shaded confidence interval if given; skipped otherwise.
	samples
		Samples drawn from `prediction`, shape `(S, F*G)`. Plotted as thin lines, all the same
		color/alpha, if given; skipped otherwise.
	f_id
		"all" (default) or int, restrict the plot to a single feature. Has no effect yet since
		multi-feature datasets aren't supported (F is always 1).
	fig, ax
		Existing figure/axes to draw on, to combine with other plots. If given, `ax` must already
		have shape `(len(f_id), 1)`. A new figure/axes grid is created if None.
	figsize
		Passed to `plt.subplots` when a new figure is created.
	legend
		If True, add a legend mapping colors to cluster indices, with this task's mixture
		coefficient (%) towards each.
	point_color
		Color of the observed points.
	prediction_color
		Color of the predictive mean line and confidence interval.
	sample_color
		Color of the prediction samples.
	ci_scale
		Number of standard deviations spanned by the shaded confidence interval (default 1.96, ~95%).
	ci_alpha
		Opacity of the shaded confidence interval.
	sample_alpha
		Opacity of each sample line.
	**scatter_kwargs
		Extra keyword arguments forwarded to `ax.scatter` for the observed points, overriding the
		defaults (s=15, alpha=0.7).

	Returns
	-------
	fig, ax
		The (possibly newly created) figure and 2D array of axes, shape `(len(f_id), 1)`.
	"""
	if dataset.inputs.shape[-1] != 1 or grid.points.shape[-1] != 1:
		raise NotImplementedError("plot_single_task_prediction only supports 1D inputs (I=1) for now.")

	f_ids = _resolve_ids(f_id, _F)

	fig, ax = _get_fig_ax(fig, ax, len(f_ids), 1, figsize=figsize)

	x_obs, y_obs = _task_xy(dataset, t_id, o_id)
	x_grid = np.asarray(grid.points[:, 0])

	K = hyperposterior.mean.shape[0]
	palette = _cluster_palette(K)
	weights = np.asarray(mixture.responsibilities[t_id])

	scatter_kwargs = _DEFAULT_SCATTER_KWARGS | scatter_kwargs

	for row, f in enumerate(f_ids):
		a = ax[row, 0]

		if samples is not None:
			for s in np.asarray(samples):
				a.plot(x_grid, s, color=sample_color, alpha=sample_alpha, linewidth=1)

		for k in range(K):
			cluster_mean, _ = _mvn_cell(hyperposterior, k, o_id)
			a.plot(x_grid, np.asarray(cluster_mean), linestyle="--", color=palette[k], alpha=float(weights[k]))

		if prediction is not None:
			pred_mean = np.asarray(prediction.mean)
			pred_std = np.sqrt(np.diagonal(np.asarray(prediction.covariance)))
			a.plot(x_grid, pred_mean, linestyle="-", color=prediction_color)
			a.fill_between(x_grid, pred_mean - ci_scale * pred_std, pred_mean + ci_scale * pred_std,
							color=prediction_color, alpha=ci_alpha, linewidth=0)

		a.scatter(x_obs, y_obs, color=point_color, **scatter_kwargs)

		a.set_title(f"task {t_id}, output {o_id}" + (f", feature {f}" if len(f_ids) > 1 else ""))
		a.set_xlabel("input")
		a.set_ylabel("output value")

	if legend:
		handles = [
			plt.Line2D([0], [0], linestyle="--", color=palette[k], label=f"cluster {k} ({100 * float(weights[k]):.0f}%)")
			for k in range(K)
		]
		fig.legend(handles=handles, loc="outside right center")

	return fig, ax