"""
Soft k-means in JAX with k-means++ initialisation.

Stopping criterion is a fixed number of iterations (`lax.scan`), and output is "soft":
responsibilities (mixture weights) of shape `(N, k)`, each row summing to 1. Since softmax is
strictly positive, cluster weights never vanish, so there is no "empty cluster" problem
unlike hard k-means.
"""

from jax import Array, lax, nn, random
import jax.numpy as jnp


def _sq_dists(X: Array, C: Array) -> Array:
    """
    Pairwise squared Euclidean distances between `X` and `C`.

    Parameters
    ----------
    X
        Points. Shape `(N, D)`.
    C
        Centers. Shape `(k, D)`.

    Returns
    -------
    Squared distances. Shape `(N, k)`.
    """
    return jnp.sum((X[:, None, :] - C[None, :, :]) ** 2, axis=-1)


def kmeanspp_init(key: Array, X: Array, k: int) -> Array:
    """
    Sample `k` initial centers via k-means++ (probability proportional to squared distance).

    Requires `N >= k` and the points not all identical (otherwise the sampling distribution
    degenerates to zero everywhere).

    Parameters
    ----------
    key
        `jax.random` PRNG key.
    X
        Points to sample centers from. Shape `(N, D)`.
    k
        Number of centers to sample.

    Returns
    -------
    Sampled centers. Shape `(k, D)`.
    """
    n = X.shape[0]
    key, sub = random.split(key)
    c0 = X[random.randint(sub, (), 0, n)]
    d2 = jnp.sum((X - c0) ** 2, axis=1)  # D² to the nearest center

    def pick(carry, _):
        key, d2 = carry
        key, sub = random.split(key)
        i = random.choice(sub, n, p=d2 / d2.sum())
        d2 = jnp.minimum(d2, jnp.sum((X - X[i]) ** 2, axis=1))
        return (key, d2), X[i]

    _, rest = lax.scan(pick, (key, d2), None, length=k - 1)
    return jnp.concatenate([c0[None], rest], axis=0)  # (k, D)


def soft_kmeans(key: Array, X: Array, k: int, n_iters: int = 50, stiffness: float = 1.0) -> tuple[Array, Array]:
    """
    Soft k-means (isotropic EM) with k-means++ initialisation.

    Parameters
    ----------
    key
        `jax.random` PRNG key (k-means++ init).
    X
        Points to cluster. Shape `(N, D)`.
    k
        Number of clusters.
    n_iters
        Number of Lloyd iterations.
    stiffness
        Softmax inverse-temperature β. β -> +inf recovers hard k-means; β -> 0 gives uniform
        memberships. Tune relative to the scale of `X` (β acts on squared distances).

    Returns
    -------
    centers
        Cluster centers. Shape `(k, D)`.
    resp
        Responsibilities, each row summing to 1. Shape `(N, k)`.
    """
    C0 = kmeanspp_init(key, X, k)

    def lloyd(C, _):
        R = nn.softmax(-stiffness * _sq_dists(X, C), axis=1)  # E-step: (N, k)
        C = (R.T @ X) / R.sum(0)[:, None]                     # M-step: (k, D)
        return C, None

    C, _ = lax.scan(lloyd, C0, None, length=n_iters)
    R = nn.softmax(-stiffness * _sq_dists(X, C), axis=1)
    return C, R