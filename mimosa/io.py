"""
Read and write Dataset objects to/from external file formats.
"""

from pathlib import Path
import numpy as np
import polars as pl
import jax.numpy as jnp

from mimosa.data_structures import Dataset


def save_csv(csv_path: str | Path, dataset: Dataset) -> None:
	"""
	Write a Dataset to a CSV file, in "pivoted" form: one row per point.

	Columns: "TaskID", "Input<suffix>" (one per input dim), "Output<suffix>" (one per output dim).
	Missing values (NaN in `dataset.outputs`) are written as "nan".

	Parameters
	----------
	csv_path
		Path to write the CSV file to.
	dataset
		Dataset to write.
	"""
	T, N, O = dataset.outputs.shape
	I = dataset.inputs.shape[-1]

	inputs = np.broadcast_to(np.asarray(dataset.inputs), (T, N, I)).reshape(T * N, I)
	outputs = np.asarray(dataset.outputs).reshape(T * N, O)
	task_ids = np.repeat(np.arange(T), N)

	columns = ["TaskID"] + [f"Input{i + 1}" for i in range(I)] + [f"Output{o + 1}" for o in range(O)]
	df = pl.DataFrame(np.column_stack([task_ids, inputs, outputs]), schema=columns)
	df.write_csv(csv_path)


def load_csv(csv_path: str | Path) -> Dataset:
	"""
	Read a Dataset from a CSV file, in "pivoted" form: one row per point.

	Columns: "TaskID", "Input<suffix>" (one per input dim), "Output<suffix>" (one per output dim).
	Suffixes are arbitrary; only the "Input"/"Output" prefix matters. Any other column is ignored.
	Missing values ("nan") are read as NaN.

	Parameters
	----------
	csv_path
		Path to read the CSV file from.

	Returns
	-------
	Dataset built from the CSV's "Input*"/"Output*" columns.

	Raises
	------
	ValueError
		If the CSV is missing a "TaskID" column, has no "Input*"/"Output*" column,
		or tasks don't all have the same number of points.
	"""
	df = pl.read_csv(csv_path)

	if "TaskID" not in df.columns:
		raise ValueError("CSV must contain a 'TaskID' column.")
	input_cols = [c for c in df.columns if c.startswith("Input")]
	output_cols = [c for c in df.columns if c.startswith("Output")]
	if not input_cols:
		raise ValueError("CSV must contain at least one 'Input*' column.")
	if not output_cols:
		raise ValueError("CSV must contain at least one 'Output*' column.")

	task_ids = df["TaskID"].to_numpy()
	_, counts = np.unique(task_ids, return_counts=True)
	if np.unique(counts).size != 1:
		raise ValueError("Every task must have the same number of points.")
	T, N = counts.size, counts[0]

	sort_idx = np.argsort(task_ids, kind="stable")
	inputs = df.select(input_cols).to_numpy()[sort_idx].reshape(T, N, len(input_cols))
	outputs = df.select(output_cols).to_numpy()[sort_idx].reshape(T, N, len(output_cols))

	return Dataset(inputs=jnp.asarray(inputs), outputs=jnp.asarray(outputs))
