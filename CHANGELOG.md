# Changelog

All notable changes to `mimosa-ml` are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html). MIMOSA is pre-1.0 and in alpha:
breaking changes may land in any minor release. Version-lock your dependency, and read this file
before upgrading.

---

## [v0.3.0-alpha] — 2026-08-31

Laplace matching: non-Gaussian observations (binary, counts, durations) can now be fitted, by
moment-matching them into Gaussian pseudo-observations before the pipeline runs. **Experimental** —
the API may change in any release.

### Added

* `mimosa.laplace`, with `BinomialLaplaceApproximator`, `PoissonLaplaceApproximator`,
  `ExponentialLaplaceApproximator` and the no-op `IdentityLaplaceApproximator`. `wrap(dataset)`
  returns a `Dataset` of Gaussian means with their variances in `known_output_noise`;
  `unwrap(samples)` maps predictions back to observation space. Observations can be binned along
  the inputs with `interval`, and the conjugate prior tuned with `prior_count`.
* `mimosa.synthetic.known_noise_kernel`, to feed those per-point variances to the model as a
  non-trainable noise term.
* `examples/binary_classif_example.py`, end-to-end on Bernoulli data.

### Breaking changes

* `BasicModel` no longer holds a `laplace_approximation`, and no longer wraps the dataset in
  `fit`/`predict`. Call `approximator.wrap(dataset)` yourself before fitting. Gaussian data is
  unaffected.
* `PredictionCovBlocks` batch dimensions are now broadcastable (`#*B`): only `cov_obs` carries the
  noise kernel, so it may batch along an axis the grid blocks are shared along.

---

## [v0.2.0] — 2026-08-26

The multi-output release. MIMOSA now learns **correlations between outputs** of a vector-valued
function, on top of the existing multi-task/multi-cluster machinery — using the multi-output kernels
of [Kernax](https://github.com/SimLej18/kernax-ml) (`ICMKernel`, `LCMKernel`, `ConvolutionKernel`,
`BlockMean`, `BlockDiagKernel`) throughout the full pipeline: data generation, grid construction,
fitting, hyperposterior, prediction, plotting and CSV I/O.

This required a **vocabulary change** (see *Breaking changes* below) and touches almost every public
signature. Read the migration guide at the end of this section before upgrading.

### Breaking changes

#### 1. Terminology: `output` → `channel`, `feature` → `output`

MIMOSA's old "output" (a dimension of the observed vector, modelled independently) and "feature"
(a correlated dimension) did not match the usual multi-output GP convention. They have been renamed
to **channel** and **output** respectively, everywhere.

| v0.1.x | v0.2.0 | Meaning |
| --- | --- | --- |
| `Dimensions.O` | `Dimensions.C` | Number of channels (dimensionality of an observation) |
| `Dimensions.F` | `Dimensions.O` | Number of correlated outputs |
| `ModelConfig.shared_output_hps` | `ModelConfig.shared_channel_hps` | Share hyper-parameters across channels |
| `ModelConfig.shared_features_hps` | *(removed)* | Superseded by the multi-output kernels themselves |
| `ModelConfig.isotopic_features` | `ModelConfig.isotopic_output_in_tasks` / `isotopic_output_in_grid` | Split in two — see below |
| `DataRemovalConfig.same_missing_across_outputs` | `DataRemovalConfig.same_missing_across_channels` | Missingness shared across channels |
| `DataRemovalConfig.same_missing_across_features` | `DataRemovalConfig.same_missing_across_outputs` | Missingness shared across outputs (**default is now `False`**) |
| `plot.plot_output` | `plot.plot_channel` | |
| `plot.plot_single_cluster_single_output` | `plot.plot_single_cluster_single_channel` | |
| `hyperpost.single_output_hyperpost` | `hyperpost.single_channel_hyperpost` | |
| `nll.single_output_mvn_nll` | `nll.single_channel_mvn_nll` | |
| `nll.single_output_trace_correction` | `nll.single_channel_trace_correction` | |
| `prediction.predict_task_output` | `prediction.predict_task_channel` | |
| `plot.*(..., o_id=...)` | `plot.*(..., c_id=...)` | The old `o_id` selector now selects a *channel*; the new `o_id` selects an *output* |

⚠️ `Dimensions` still takes seven ints, but positions 4 and 5 swapped meaning
(`T, K, I, O, F, N, G` → `T, K, I, C, O, N, G`). Positional construction will **not** raise — it will
silently mean something else. Audit every `Dimensions(...)` call, or switch to keyword arguments.

#### 2. Every `mimosa.plot` function now takes `dims` as its second positional argument

Plots are laid out as a grid of *outputs* × *channels*, which cannot be recovered from the arrays
alone, so a `Dimensions` is now required:

```python
# v0.1.x
fig, ax = plot_dataset(dataset, mixture=mixture)
fig, ax = plot_clusters(grid, hyperposterior=hyperposterior)
fig, ax = plot_single_task_prediction(dataset, grid, hyperposterior, mixture, t_id, o_id)

# v0.2.0
fig, ax = plot_dataset(dataset, dims, mixture=mixture)
fig, ax = plot_clusters(grid, dims, hyperposterior=hyperposterior)
fig, ax = plot_single_task_prediction(dataset, grid, dims, hyperposterior, mixture, t_id, c_id)
```

#### 3. `Grid` field order changed

`Grid` gained an `output_ids` field, inserted **before** `mappings`:

```python
# v0.1.x
Grid(points, mappings)                      # mappings was the 2nd positional field, and mandatory
# v0.2.0
Grid(points, output_ids=None, mappings=None)  # both now optional, mappings is 3rd
```

Positional `Grid(points, mappings)` now assigns `mappings` to `output_ids`. Use keywords.

#### 4. `Dataset` gained `output_ids`, and a `clean_inputs` property

`Dataset(inputs, outputs, known_output_noise=None, output_ids=None)`. `output_ids` labels which
output each input row belongs to; it is `None` exactly when every output shares the same input
locations. Existing 3-argument construction is unaffected.

`Dataset.inputs` may hold NaN padding for variable-length tasks. **Never feed `inputs` to a kernel** —
use `dataset.clean_inputs`, which replaces the padding with `0`. `jnp.where` does not stop NaN in the
VJP (`0 * NaN = NaN`), so a single padded point turns every kernel hyper-parameter's gradient into
NaN and freezes the optimiser. Conversely, never pass `clean_inputs` to a `GridBuilder`: `0` is a real
input location and would add a spurious grid point.

#### 5. `generate_data(input_range=...)` now takes a list of ranges

One `(min, max)` per output, applied to every input dimension. A single-element list applies to all
outputs. Default is `[(-50, 50)]`.

```python
generate_data(key, dims, params, config, input_range=(-2.5, 2.5))    # v0.1.x
generate_data(key, dims, params, config, input_range=[(-2.5, 2.5)])  # v0.2.0
```

#### 6. CSV format

Output columns are now self-describing: `Output<o>_<c>` and (optional) `Noise<o>_<c>`, with `<o>` the
output index and `<c>` the channel index, both 1-based. Two sentinels are now distinguished within an
`Output<o>_<c>` cell:

* **empty field** — that coordinate is *not* on output `o`'s input axis for that task;
* **`nan`** — the coordinate is on the axis, but the value is unobserved.

This is what lets `load_csv` *recover* rather than guess whether outputs are isotopic or heterotopic
from a single file.

**Files written by v0.1.x still load**: flat `Output*` columns with arbitrary suffixes are read as one
output with one channel per column — which is exactly their old meaning under the new vocabulary.
Files written by v0.2.0 will *not* load in v0.1.x.

#### 7. `kernax-ml >= 0.7.5a0` is now required

`Parameters` fields are annotated `MeanLike`/`KernelLike` (was `AbstractMean`/`AbstractKernel`), and
the whole pipeline calls kernels with `output_ids=` / `output_ids2=`, which older Kernax does not
accept.

### Added

* **Multi-output correlation learning**, end to end. Wrap the mean and kernels in a Kernax
  multi-output kernel and everything else follows:

  ```python
  from kernax import ZeroMean, VarianceKernel, SEKernel, WhiteNoiseKernel, BlockMean, ICMKernel, BlockDiagKernel

  params = Parameters(
      cluster_mean=BlockMean(ZeroMean(), n_outputs=dims.O, output_hps_in_axes=None),
      cluster_kernel=ICMKernel(VarianceKernel(5.0) * SEKernel(length_scale=.5), n_outputs=dims.O, n_latent=dims.O - 1),
      task_kernel=ICMKernel(VarianceKernel(1.0) * SEKernel(length_scale=.4), n_outputs=dims.O, n_latent=dims.O - 1),
      noise_kernel=BlockDiagKernel(WhiteNoiseKernel(noise=.05), n_outputs=dims.O, output_hps_in_axes=None),
  )
  ```

  `BlockMean`/`BlockDiagKernel` broadcast independently per output; `ICMKernel`/`LCMKernel`/
  `ConvolutionKernel` additionally learn inter-output correlations (`n_latent < dims.O` gives a
  low-rank coregionalisation).

* `mimosa.grid.MultiOutputUnionGrid` — multi-output counterpart of `UnionGrid`. Takes the `Dataset`
  *and* the `ModelConfig` (it needs to know whether outputs share grid/task input locations):

  ```python
  fitted_grid = MultiOutputUnionGrid()(dataset, model_config)
  ```

  Like `UnionGrid`, it is not jit-compatible; build it once, outside `fit`/`predict`.

* `ModelConfig.isotopic_output_in_tasks` and `ModelConfig.isotopic_output_in_grid` (both default
  `True`), replacing the single `isotopic_features` flag. `isotopic_output_in_tasks=True` with
  `isotopic_output_in_grid=False` is rejected — task inputs cannot be isotopic when sampled from a
  heterotopic grid.
* `Dataset.output_ids`, `Grid.output_ids`, `Dataset.clean_inputs` (see *Breaking changes* 3–4).
* `BasicModel(..., n_outputs=...)` and `KMeansMixtureInitialiser(..., n_outputs=...)`. The k-means
  mixture initialisation now summarises each task **per output** (min/max/mean/std per output, per
  channel) instead of pooling outputs into one summary — pooling makes two tasks differing in a
  single output look nearly identical, which is exactly the case a multi-output model exists for.
  Only needed when outputs do *not* share input locations; otherwise the count is read off the
  `Dataset`'s shapes.
* Multi-output CSV I/O: `save_single_csv`, `load_single_csv`, `split_into_single_output_datasets`,
  `merge_multioutput_datasets`, and an `output_groups` override on `load_csv`/`load_single_csv` for
  foreign files whose columns aren't named `Output<o>_<c>`. `save_csv`/`load_csv` now also accept a
  **sequence of paths**, storing one file per output (in output-index order) — cheaper than one wide
  file for heterotopic data, which otherwise pays an `O`-times empty-column cost.
* Multi-output synthetic data: `generate_grid`/`sample_inputs` build heterotopic grids and per-output
  task inputs; `RandomDataRemover` draws missingness per output and/or per channel per
  `DataRemovalConfig`.
* Multi-output plotting: every `mimosa.plot` function takes an `o_id` selector (default `"all"`) and
  lays out outputs as rows, channels as columns.
* `examples/basic_mo_example.py` — the full multi-output pipeline, cell by cell.

### Changed

* `validate_model_config` gained checks for the new flags: distinct channel hyper-parameters with
  `C == 1`, multi-output grids/task inputs with `O == 1`, and the isotopic-tasks/heterotopic-grid
  combination above.
* `RandomDataRemover` raises when `same_missing_across_outputs=True` but outputs don't share input
  locations — sharing missingness across outputs is meaningless there.
* k-means initialisation now zero-fills non-finite summary features. A task with no surviving
  observation for some output used to contribute a NaN feature, poisoning every distance in the
  k-means rather than just that one coordinate.
* Publish workflow no longer installs `twine` (unused).

### Fixed

* **NaN gradients from padded inputs.** `hyperpost`, `predict`, `update_mixture`, `optimise_tasks`,
  `TaskOptimiser` and `ClusterOptimiser` now feed kernels `dataset.clean_inputs` instead of
  `dataset.inputs`. Previously, any variable-length task (padded with NaN) produced NaN gradients for
  every kernel hyper-parameter and stalled the optimiser.
* **Cluster optimisation ignored grid output ids.** `optimise_clusters`/`ClusterOptimiser` now pass
  `grid.output_ids` when evaluating the mean and cluster kernel on the grid.
* **Cross-covariance between labelled and unlabelled points.** Kernax multi-output kernels require
  both sides of a two-argument call to carry `output_ids`, or neither. `predict` now tiles and labels
  the grid side when the grid is an unlabelled isotopic pool but the observations are labelled
  (`mimosa.prediction._cross_grid`).
* **k-means summary statistics were reshaped, not transposed.** The old
  `jnp.stack(...).reshape((T, -1))` put the statistic on the leading axis, so each row ended up
  holding one statistic belonging to four different tasks. Statistics are now concatenated along the
  feature axis.

### Migration guide (v0.1.x → v0.2.0)

1. Rename in your own code, in this order (the two renames collide — do `output` → `channel` first,
   then `feature` → `output`):
   * `Dimensions(O=...)` → `Dimensions(C=...)`, then `Dimensions(F=...)` → `Dimensions(O=...)`.
     Prefer keyword arguments: positional construction silently changes meaning.
   * `shared_output_hps` → `shared_channel_hps`; drop `shared_features_hps`.
   * `isotopic_features` → `isotopic_output_in_tasks` (and set `isotopic_output_in_grid` if you want
     per-output grids).
   * `same_missing_across_outputs` → `same_missing_across_channels`;
     `same_missing_across_features` → `same_missing_across_outputs`.
   * `plot_output` → `plot_channel`; `plot_single_cluster_single_output` →
     `plot_single_cluster_single_channel`; `o_id=` → `c_id=` in every plot call.
2. Add `dims` as the second positional argument of every `mimosa.plot` call.
3. Wrap `input_range` in a list: `(-2.5, 2.5)` → `[(-2.5, 2.5)]`.
4. Replace positional `Grid(points, mappings)` with `Grid(points=..., mappings=...)`.
5. Upgrade Kernax: `pip install -U "kernax-ml>=0.7.5a0"`.
6. If (and only if) you want correlated outputs: set `dims.O > 1`, wrap your mean/kernels in
   `BlockMean`/`BlockDiagKernel`/`ICMKernel`/`LCMKernel`/`ConvolutionKernel`, and build the grid with
   `MultiOutputUnionGrid()(dataset, model_config)` instead of `UnionGrid()(dataset.inputs)`.
   Single-output code needs none of this — `dims.O = 1` keeps the v0.1.x behaviour.

Existing CSV files load unchanged. Files written by v0.2.0 are not readable by v0.1.x.

---

## [v0.1.1-alpha] — 2026-07-27

### Added

* Progress bar in `model.fit()`, via `jax-tqdm`.
* A basic `.gitignore`.

### Fixed

* `save_csv`/`load_csv` now support tasks of varying length, as intended.
* Grid and mapping construction with multi-dimensional inputs and/or NaN-marked missing inputs.

---

## [v0.1.0-alpha] — 2026-07-16

First public release. Multi-task Gaussian processes over unaligned sampling grids, clustering of
tasks as a mixture of Magma GPs, multi-dimensional inputs and (uncorrelated) outputs, probabilistic
predictions with uncertainty quantification, Kernax kernel/mean integration, and full JAX/Equinox
compatibility for `vmap`/`grad`/`jit`.

[v0.3.0-alpha]: https://github.com/SimLej18/mimosa/releases/tag/v0.3.0-alpha
[v0.2.0]: https://github.com/SimLej18/mimosa/releases/tag/v0.2.0
[v0.1.1-alpha]: https://github.com/SimLej18/mimosa/releases/tag/v0.1.1-alpha
[v0.1.0-alpha]: https://github.com/SimLej18/mimosa/releases/tag/v0.1.0-alpha
