"""
Linear algebra and grid-indexing primitives used throughout the package: batched Cholesky solves,
and lexicographic search over grid points.
"""

import jax.numpy as jnp
import jax.lax as jlx
from jax import Array, jit, vmap
from jax.lax import cond, while_loop

from mimosa import DEFAULT_JITTER


def cho_factor(cov: Array, jitter: Array = DEFAULT_JITTER) -> Array:
	"""
	Cholesky factor of a covariance matrix, with jitter added to the diagonal for numerical stability.

	Unlike `jax.scipy.linalg.cho_factor`, does not symmetrise the input.

	Parameters
	----------
	cov
		Covariance matrix to factor. Shape `(..., N, N)`.
	jitter
		Amount of jitter added to the diagonal before factoring.

	Returns
	-------
	Lower Cholesky factor of `cov`. Shape `(..., N, N)`.
	"""
	return jlx.linalg.cholesky(cov + jitter * jnp.eye(cov.shape[-1]), symmetrize_input=False)


def cho_solve(cov_l: Array, res: Array, left_side: bool = True, lower: bool = True) -> Array:
	"""
	Solve `cov @ x = res` for x (or `x @ cov = res` if `left_side` is False), given the lower Cholesky
	factor `cov_l` of `cov` (`cov = cov_l @ cov_l^T`), as returned by `cho_factor`.

	Equivalent to `jax.scipy.linalg.cho_solve`, but uses lower factorisation by default and avoids its
	deprecation warning on batched 1D right-hand sides.

	Parameters
	----------
	cov_l
		Lower Cholesky factor of the covariance matrix. Shape `(..., N, N)`.
	res
		Right-hand side to solve for. Shape `(..., N, M)`.
	left_side
		If True, solve `cov @ x = res`; if False, solve `x @ cov = res`.
	lower
		Whether `cov_l` is lower- or upper-triangular.

	Returns
	-------
	x. Shape `(..., N, M)`.
	"""
	if left_side:
		y = jlx.linalg.triangular_solve(cov_l, res, left_side=True, lower=lower, transpose_a=False)
		return jlx.linalg.triangular_solve(cov_l, y, left_side=True, lower=lower, transpose_a=True)
	y = jlx.linalg.triangular_solve(cov_l, res, left_side=False, lower=lower, transpose_a=True)
	return jlx.linalg.triangular_solve(cov_l, y, left_side=False, lower=lower, transpose_a=False)


def searchsorted_2d(vector: Array, matrix: Array) -> Array:
	"""
	Find the index of `vector` in `matrix`, along axis 0.

	`matrix`'s rows must be sorted lexicographically (see `lexicographic_sort`), e.g.:
	[[1, 1, 0],
	 [1, 2, 1],
	 [1, 2, 2],
	 [2, 1, 3],
	 [2, 2, 1]]

	Parameters
	----------
	vector
		Vector to search for.
	matrix
		Matrix to search in.

	Returns
	-------
	Index of `vector` in `matrix`, or `len(matrix)` if not found.
	"""

	@jit
	def compare_vectors(v1, v2):
		"""Compare two vectors lexicographically. Returns -1 if v1 < v2, 0 if equal, 1 if v1 > v2"""
		diff = v1 - v2
		# Find first non-zero element
		nonzero_mask = diff != 0
		# If all elements are zero, vectors are equal
		first_nonzero_idx = jnp.argmax(nonzero_mask)

		return cond(
			jnp.any(nonzero_mask),
			lambda: jnp.array(jnp.sign(diff[first_nonzero_idx])).astype(int),
			lambda: jnp.array(0).astype(int)
		)

	@jit
	def search_condition(state):
		start, end, found = state
		return (start < end) & (~found)

	@jit
	def search_step(state):
		start, end, found = state
		mid = (start + end) // 2

		comparison = compare_vectors(vector, matrix[mid])

		# If vectors are equal, we found it
		new_found = comparison == 0
		new_start = cond(comparison < 0, lambda: start, lambda: mid + 1)
		new_end = cond(comparison < 0, lambda: mid, lambda: end)

		# If found, return the index in start position
		final_start = cond(new_found, lambda: mid, lambda: new_start)

		return final_start, new_end, new_found

	# Initial state: (start, end, found)
	initial_state = (0, len(matrix), False)
	final_start, final_end, found = while_loop(search_condition, search_step, initial_state)

	# Return the found index or len(matrix) if not found
	return cond(found, lambda: final_start, lambda: len(matrix))


searchsorted_2d_vectorised = jit(vmap(searchsorted_2d, in_axes=(0, None)))


def lexicographic_sort(arr: Array) -> Array:
	"""
	Sort a 2D array lexicographically along its first dimension.

	Parameters
	----------
	arr
		2D array to sort.

	Returns
	-------
	`arr`, with rows sorted lexicographically.
	"""
	return arr[jnp.lexsort(arr.T[::-1])]


def compute_mapping(grid: Array, element: Array) -> Array:
	"""
	Find the index of `element` in `grid`.

	Parameters
	----------
	grid
		Sorted grid points, of shape `(N,)` or `(N, I)`. If 2D, rows must be sorted lexicographically
		(see `lexicographic_sort`).
	element
		Element(s) to search for, of shape matching `grid` minus its leading axis.

	Returns
	-------
	Index of `element` in `grid`.
	"""
	if grid.shape[-1] == 1:
		# We only have 1 input dimension, and we can use the fast jnp.searchsorted function
		return jnp.searchsorted(grid.squeeze(axis=-1), element.squeeze(axis=-1))
	# Multiple input dimensions requires our custom lexicographic search
	return searchsorted_2d_vectorised(element, grid)
