"""
mimosa-ml: multi-task, multi-cluster Gaussian process regression with heterogeneous sampling.
"""
import importlib.metadata
import jax.numpy as jnp

DEFAULT_JITTER = jnp.asarray(1e-8)

from mimosa.data_structures import (
	Dataset, Dimensions, ModelConfig, DataRemovalConfig, Parameters, ParameterPriors,
	Grid, Mixture, Hyperprior, Hyperposterior, MultivariateNormal,
)
from mimosa.io import save_csv, load_csv
from mimosa.models import BasicModel
from mimosa.synthetic import generate_data, RandomDataRemover, build_parameters, sample_parameters_from_priors

__all__ = [
	"BasicModel",
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
