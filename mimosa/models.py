"""
Top-level models, combining grid construction, mixture initialisation/update, hyperposterior
computation, hyperparameter optimisation, and prediction into a single fit/predict interface.
"""

from abc import abstractmethod
import jax
from jax import Array
import equinox as eqx
import optimistix as optx

from mimosa.hyperpost import Hyperpost
from mimosa.nll import ClusterNLL, TaskNLL
from mimosa.optimisers import ClusterOptimiser, TaskOptimiser
from mimosa.mixture import KMeansMixtureInitialiser, MixtureInitialiser, MixtureUpdater
from mimosa.laplace import LaplaceApproximator, IdentityLaplaceApproximator
from mimosa.prediction import Predictor
from mimosa.data_structures import Dataset, Grid, Mixture, Parameters, MultivariateNormal
from mimosa import DEFAULT_JITTER


class AbstractModel(eqx.Module):
    """
    Base class for models exposing a `fit`/`predict` interface.
    """
    @abstractmethod
    def fit(self, *args, **kwargs):
        """
        Fit the model's parameters and mixture to a Dataset.
        """
        pass

    @abstractmethod
    def predict(self, *args, **kwargs):
        """
        Predict outputs at the grid points for a fitted model.
        """
        pass


class BasicModel(AbstractModel):
    """
    Default model pipeline: identity Laplace approximation, k-means mixture initialisation,
    LBFGS-optimised cluster and task hyperparameters.

    Unlike grid construction (see `mimosa.grid.GridBuilder`), every step here is jit-compatible, so
    `fit`/`predict` are jitted end-to-end. The `Grid` itself is a required argument rather than an
    attribute: building it (e.g. via `mimosa.grid.UnionGrid`) isn't always jit-compatible, so it must
    be computed by the caller outside of `fit`/`predict`.

    Attributes
    ----------
    laplace_approximation
        Wraps the dataset before fitting/prediction (identity by default).
    mixture_initialiser
        Initialises the tasks' mixture responsibilities.
    mixture_updater
        Updates the tasks' mixture responsibilities during fitting.
    hyperpost
        Computes the hyperposterior over each mean-process's values at the grid points.
    cluster_nll
        Negative log-likelihood used to optimise the cluster mean and kernel hyperparameters.
    task_nll
        Negative log-likelihood used to optimise the task and noise kernel hyperparameters.
    cluster_optimiser
        Optimiser for the cluster mean and kernel hyperparameters.
    task_optimiser
        Optimiser for the task and noise kernel hyperparameters.
    predictor
        Computes predictions from a fitted model.
    jitter
        Diagonal jitter added before Cholesky factorizations, for numerical stability.
    """
    laplace_approximation: LaplaceApproximator
    mixture_initialiser: MixtureInitialiser
    mixture_updater: MixtureUpdater
    hyperpost: Hyperpost
    cluster_nll: ClusterNLL
    task_nll: TaskNLL
    cluster_optimiser: ClusterOptimiser
    task_optimiser: TaskOptimiser
    predictor: Predictor
    jitter: Array

    def __init__(self, prng_key: Array, n_clusters: int, jitter: Array = DEFAULT_JITTER):
        """
        Parameters
        ----------
        prng_key
            `jax.random` PRNG key, used to initialise the mixture via k-means++.
        n_clusters
            Number of mean-processes in the mixture.
        jitter
            Diagonal jitter added before Cholesky factorizations, for numerical stability.
        """
        self.laplace_approximation = IdentityLaplaceApproximator()
        self.mixture_initialiser = KMeansMixtureInitialiser(prng_key, n_clusters)
        self.mixture_updater = MixtureUpdater()
        self.hyperpost = Hyperpost()
        self.cluster_nll = ClusterNLL()
        self.task_nll = TaskNLL()
        self.cluster_optimiser = ClusterOptimiser(
            solver=optx.LBFGS(atol=1e-3, rtol=1e-3),
            nll=self.cluster_nll,
        )
        self.task_optimiser = TaskOptimiser(
            solver=optx.LBFGS(atol=1e-3, rtol=1e-3),
            nll=self.task_nll,
        )
        self.predictor = Predictor()
        self.jitter = jitter

    @eqx.filter_jit
    def fit(self, dataset: Dataset, grid: Grid, mixture_proportions: Array, parameters: Parameters,
            n_iter: int = 50) -> tuple[Parameters, Mixture]:
        """
        Fit the model's cluster/task hyperparameters and mixture responsibilities to a Dataset.

        Alternates, for `n_iter` iterations: computing the hyperposterior, updating the mixture
        responsibilities, optimising the cluster hyperparameters, then the task hyperparameters
        (each by maximum-a-posteriori), and updating the mixture responsibilities again.

        Parameters
        ----------
        dataset
            Dataset to fit the model to.
        grid
            Grid of points and mappings of `dataset`'s inputs onto it, e.g. from
            `mimosa.grid.UnionGrid`.
        mixture_proportions
            Fixed mixture proportions of each mean-process.
        parameters
            Initial model parameters (mean, kernels).
        n_iter
            Number of fitting iterations.

        Returns
        -------
        parameters
            Fitted model parameters.
        mixture
            Fitted mixture, with `mixture_proportions` unchanged and updated responsibilities.
        """
        dataset = self.laplace_approximation.wrap(dataset)

        mixture = self.mixture_initialiser(dataset)
        mixture = Mixture(proportions=mixture_proportions, responsibilities=mixture.responsibilities)

        def step(i, args):
            parameters, mixture = args

            hyperposterior = self.hyperpost(
                dataset, grid, mixture, parameters, jitter=self.jitter)

            mixture = self.mixture_updater(
                dataset, grid, parameters.task_kernel + parameters.noise_kernel,
                hyperposterior, mixture, jitter=self.jitter)

            cluster_mean, cluster_kernel = self.cluster_optimiser(
                parameters.cluster_mean, parameters.cluster_kernel,
                hyperposterior, grid, jitter=self.jitter).value

            optim_task = self.task_optimiser(
                parameters.task_kernel + parameters.noise_kernel,
                dataset, grid, hyperposterior, mixture, jitter=self.jitter).value
            task_kernel, noise_kernel = optim_task.left, optim_task.right

            parameters = Parameters(
                cluster_mean=cluster_mean, cluster_kernel=cluster_kernel,
                task_kernel=task_kernel, noise_kernel=noise_kernel)

            # We do not update mixture at first iter to help convergence
            mixture = jax.lax.cond(
                i != 0,
                lambda m: self.mixture_updater(
                    dataset, grid, parameters.task_kernel + parameters.noise_kernel,
                    hyperposterior, m, jitter=self.jitter),
                lambda m: m,
                mixture)

            return parameters, mixture

        return jax.lax.fori_loop(0, n_iter, step, (parameters, mixture))


    @eqx.filter_jit
    def predict(self, dataset: Dataset, grid: Grid, mixture: Mixture, parameters: Parameters) -> MultivariateNormal:
        """
        Predict outputs at the grid points, for a fitted model.

        Parameters
        ----------
        dataset
            Dataset to condition predictions on.
        grid
            Grid of points and mappings of `dataset`'s inputs onto it, e.g. from
            `mimosa.grid.UnionGrid`.
        mixture
            Fitted mixture.
        parameters
            Fitted model parameters.

        Returns
        -------
        Predicted distribution over each task's outputs at the grid points, for every mean-process.
        """
        dataset = self.laplace_approximation.wrap(dataset)
        hyperposterior = self.hyperpost(dataset, grid, mixture, parameters, jitter=self.jitter)
        return self.predictor(dataset, grid, hyperposterior, parameters, jitter=self.jitter)
