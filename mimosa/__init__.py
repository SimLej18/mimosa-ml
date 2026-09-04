"""
mimosa-ml: multi-task, multi-cluster Gaussian process regression with heterogeneous sampling.
"""
import importlib.metadata
import jax.numpy as jnp

DEFAULT_JITTER = jnp.asarray(1e-8)

# Index standing for "this input point is not on the grid". Out of bounds for any grid, so a scatter
# drops it and a gather clamps it. Weakly typed, so it adopts the dtype of the mappings it is
# written into. See `mimosa.mappings` for the full contract.
PAD_INDEX: int = int(jnp.iinfo(jnp.int32).max)

from mimosa.data_structures import (
	Dataset, Dimensions, ModelConfig, DataRemovalConfig, Parameters, ParameterPriors,
	Grid, Mixture, Hyperprior, Hyperposterior, MultivariateNormal,
)
from mimosa.io import save_csv, load_csv
from mimosa import laplace
from mimosa.models import BasicModel
from mimosa.mixture import KMeansMixtureInitialiser
from mimosa.synthetic import generate_data, RandomDataRemover, build_parameters, sample_parameters_from_priors

__all__ = [
	"PAD_INDEX",
	"DEFAULT_JITTER",
	"laplace",
	"BasicModel",
	"KMeansMixtureInitialiser",
	"Dataset",
	"Dimensions",
	"ModelConfig",
	"DataRemovalConfig",
	"Parameters",
	"ParameterPriors",
	"Grid",
	"Mixture",
	"Hyperprior",
	"Hyperposterior",
	"MultivariateNormal",
	"save_csv",
	"load_csv",
	"generate_data",
	"RandomDataRemover",
	"build_parameters",
	"sample_parameters_from_priors",
]

__version__ = importlib.metadata.version("mimosa-ml")
