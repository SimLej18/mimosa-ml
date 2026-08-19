"""
Read and write Dataset objects to/from external file formats.

Multi-output datasets (`dataset.output_ids is not None`, or several equal-size output blocks when
it's `None`) are stored as one CSV per output rather than as extra columns: `save_csv`/`load_csv`
dispatch to `save_single_csv`/`load_single_csv` for a single path, or to
`split_into_single_output_datasets`/`merge_multioutput_datasets` for a sequence of paths (one per
output, in output-index order).
"""

from collections.abc import Sequence
from pathlib import Path
import numpy as np
import polars as pl
import jax.numpy as jnp

from mimosa.data_structures import Dataset


def save_single_csv(csv_path: str | Path, dataset: Dataset) -> None:
	"""
	Write a single-output Dataset to a CSV file, in "pivoted" form: one row per point.

	Columns: "TaskID", "Input<suffix>" (one per input dim), "Output<suffix>" (one per channel).
	A point missing on every channel (NaN padding a variable-length task) is dropped entirely, so
	tasks can end up with a different number of rows. A point missing on only some channels keeps its
	row, with "nan" written for the missing channels.

	Parameters
	----------
	csv_path
		Path to write the CSV file to.
	dataset
		Single-output Dataset to write (see `split_into_single_output_datasets` for multi-output ones).
	"""
	T, N, C = dataset.outputs.shape
	I = dataset.inputs.shape[-1]

	inputs = np.broadcast_to(np.asarray(dataset.inputs), (T, N, I)).reshape(T * N, I)
	outputs = np.asarray(dataset.outputs).reshape(T * N, C)
	task_ids = np.repeat(np.arange(T), N)

	keep = ~np.all(np.isnan(outputs), axis=-1)
	task_ids, inputs, outputs = task_ids[keep], inputs[keep], outputs[keep]

	columns = ["TaskID"] + [f"Input{i + 1}" for i in range(I)] + [f"Output{c + 1}" for c in range(C)]
	df = pl.DataFrame(np.column_stack([task_ids, inputs, outputs]), schema=columns)
	df.write_csv(csv_path)


def split_into_single_output_datasets(dataset: Dataset) -> list[Dataset]:
	"""
	Split a multi-output Dataset into one single-output Dataset per output, in output-index order.
	Inverse of `merge_multioutput_datasets`.

	Parameters
	----------
	dataset
		Dataset to split.

	Returns
	-------
	One Dataset per output, each re-padded to that output's own longest task. A single-output
	`dataset` (`output_ids is None` and only one implied output) is returned as `[dataset]`.
	"""
	T, oN, C = dataset.outputs.shape
	outputs = np.asarray(dataset.outputs)
	known_noise = None if dataset.known_output_noise is None else np.asarray(dataset.known_output_noise)

	if dataset.output_ids is None:
		# isotopic_output_in_tasks: every output has the same N points and shares the same inputs
		# -- N is `inputs`' own row count (never O*N-collapsed like `outputs`), no dims/config needed.
		N = dataset.inputs.shape[1]
		O = oN // N
		if O == 1:
			return [dataset]
		return [
			Dataset(
				inputs=dataset.inputs,
				outputs=jnp.asarray(outputs[:, o * N:(o + 1) * N, :]),
				known_output_noise=None if known_noise is None else jnp.asarray(known_noise[:, o * N:(o + 1) * N, :]),
			)
			for o in range(O)
		]

	# Heterotopic outputs: each may have a different number of points, varying per task.
	inputs = np.broadcast_to(np.asarray(dataset.inputs), (T, oN, dataset.inputs.shape[-1]))
	output_ids = np.broadcast_to(np.asarray(dataset.output_ids), (T, oN))
	O = int(output_ids.max()) + 1

	datasets = []
	for o in range(O):
		mask = output_ids == o  # (T, oN)
		rank = np.cumsum(mask, axis=1) - 1  # each True row's position within output o's compacted block
		N_o = int(mask.sum(axis=1).max())

		out_inputs = np.full((T, N_o, inputs.shape[-1]), np.nan)
		out_outputs = np.full((T, N_o, C), np.nan)
		out_noise = None if known_noise is None else np.full((T, N_o, C), np.nan)

		task_idx, row_idx = np.nonzero(mask)
		dest = rank[task_idx, row_idx]
		out_inputs[task_idx, dest] = inputs[task_idx, row_idx]
		out_outputs[task_idx, dest] = outputs[task_idx, row_idx]
		if known_noise is not None:
			out_noise[task_idx, dest] = known_noise[task_idx, row_idx]

		datasets.append(Dataset(
			inputs=jnp.asarray(out_inputs),
			outputs=jnp.asarray(out_outputs),
			known_output_noise=None if out_noise is None else jnp.asarray(out_noise),
		))
	return datasets


def save_csv(csv_path: str | Path | Sequence[str | Path], dataset: Dataset) -> None:
	"""
	Write a Dataset to CSV. A single path writes one file (see `save_single_csv`); a sequence of
	paths splits `dataset` into one file per output (see `split_into_single_output_datasets`), in
	output-index order.

	Parameters
	----------
	csv_path
		Path (single-output dataset) or one path per output, in output-index order.
	dataset
		Dataset to write.

	Raises
	------
	ValueError
		If `csv_path` is a sequence whose length doesn't match `dataset`'s number of outputs.
	"""
	if isinstance(csv_path, (str, Path)):
		save_single_csv(csv_path, dataset)
		return

	csv_paths = list(csv_path)
	datasets = split_into_single_output_datasets(dataset)
	if len(datasets) != len(csv_paths):
		raise ValueError(f"Dataset has {len(datasets)} output(s), but got {len(csv_paths)} csv path(s).")
	for path, single_output_dataset in zip(csv_paths, datasets):
		save_single_csv(path, single_output_dataset)


def load_single_csv(csv_path: str | Path) -> Dataset:
	"""
	Read a single-output Dataset from a CSV file, in "pivoted" form: one row per point.

	Columns: "TaskID", "Input<suffix>" (one per input dim), "Output<suffix>" (one per channel).
	Suffixes are arbitrary; only the "Input"/"Output" prefix matters. Any other column is ignored.
	Missing values ("nan") are read as NaN.

	Tasks may have a different number of rows. They're padded to the longest task to build a
	uniform-shape Dataset: padding points are NaN on every input and every channel.

	Parameters
	----------
	csv_path
		Path to read the CSV file from.

	Returns
	-------
	Dataset built from the CSV's "Input*"/"Output*" columns, padded to the longest task.

	Raises
	------
	ValueError
		If the CSV is missing a "TaskID" column, or has no "Input*"/"Output*" column.
	"""
	df = pl.read_csv(csv_path)

	if "TaskID" not in df.columns:
		raise ValueError("CSV must contain a 'TaskID' column.")
	input_cols = [c for c in df.columns if c.startswith("Input")]
	channel_cols = [c for c in df.columns if c.startswith("Output")]
	if not input_cols:
		raise ValueError("CSV must contain at least one 'Input*' column.")
	if not channel_cols:
		raise ValueError("CSV must contain at least one 'Output*' column.")

	task_ids = df["TaskID"].to_numpy()
	_, counts = np.unique(task_ids, return_counts=True)
	T, N = counts.size, counts.max()

	sort_idx = np.argsort(task_ids, kind="stable")
	inputs_sorted = df.select(input_cols).to_numpy()[sort_idx]
	outputs_sorted = df.select(channel_cols).to_numpy()[sort_idx]

	starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
	task_idx = np.repeat(np.arange(T), counts)
	pos_in_task = np.arange(task_ids.size) - np.repeat(starts, counts)

	I, C = len(input_cols), len(channel_cols)
	inputs = np.full((T, N, I), np.nan)
	outputs = np.full((T, N, C), np.nan)
	inputs[task_idx, pos_in_task] = inputs_sorted
	outputs[task_idx, pos_in_task] = outputs_sorted

	return Dataset(inputs=jnp.asarray(inputs), outputs=jnp.asarray(outputs))


def merge_multioutput_datasets(datasets: Sequence[Dataset]) -> Dataset:
	"""
	Merge single-output Datasets (e.g. from `load_single_csv`), in output-index order, into one
	multi-output Dataset. Inverse of `split_into_single_output_datasets`.

	Every dataset must describe the same tasks, in the same order -- `load_csv` guarantees this
	(by validating every CSV shares the same set of `TaskID`s) before calling `load_single_csv` on
	each file and this function on the results; `Dataset` itself doesn't carry `TaskID`s, so this
	function can't check or realign them itself.

	Parameters
	----------
	datasets
		One Dataset per output, in output-index order, all describing the same tasks in the same order.

	Returns
	-------
	Merged multi-output Dataset, with `output_ids` marking each row's source dataset.
	"""
	if len(datasets) == 1:
		return datasets[0]

	inputs = jnp.concatenate([d.inputs for d in datasets], axis=1)
	outputs = jnp.concatenate([d.outputs for d in datasets], axis=1)

	if all(d.known_output_noise is None for d in datasets):
		known_output_noise = None
	else:
		known_output_noise = jnp.concatenate([
			d.known_output_noise if d.known_output_noise is not None else jnp.full(d.outputs.shape, jnp.nan)
			for d in datasets
		], axis=1)

	output_ids = jnp.concatenate([jnp.full((d.outputs.shape[1],), o, dtype=int) for o, d in enumerate(datasets)])[None, :]

	return Dataset(inputs=inputs, outputs=outputs, known_output_noise=known_output_noise, output_ids=output_ids)


def load_csv(csv_path: str | Path | Sequence[str | Path]) -> Dataset:
	"""
	Read a Dataset from CSV. A single path reads one file (see `load_single_csv`); a sequence of
	paths reads one file per output (in output-index order) and merges them (see
	`merge_multioutput_datasets`).

	Parameters
	----------
	csv_path
		Path (single-output dataset) or one path per output, in output-index order.

	Returns
	-------
	Dataset built from the CSV file(s).

	Raises
	------
	ValueError
		If `csv_path` is a sequence and the files don't all share the same set of `TaskID`s.
	"""
	if isinstance(csv_path, (str, Path)):
		return load_single_csv(csv_path)

	csv_paths = list(csv_path)
	if len(csv_paths) == 1:
		return load_single_csv(csv_paths[0])

	task_id_sets = [set(pl.read_csv(p, columns=["TaskID"])["TaskID"].to_list()) for p in csv_paths]
	if any(s != task_id_sets[0] for s in task_id_sets[1:]):
		raise ValueError("All csv_path files must share the exact same set of TaskIDs to be merged into a multi-output Dataset.")

	return merge_multioutput_datasets([load_single_csv(p) for p in csv_paths])