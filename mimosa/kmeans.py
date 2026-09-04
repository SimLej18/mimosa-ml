"""
Soft k-means in JAX with k-means++ initialisation.

Stopping criterion is a fixed number of iterations (`lax.scan`), and output is "soft":
responsibilities (mixture weights) of shape `(N, k)`, each row summing to 1. Since softmax is
strictly positive, cluster weights never vanish, so there is no "empty cluster" problem
unlike hard k-means.
"""

from jax import Array, lax, nn, random, vmap
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


def soft_kmeans(key: Array, X: Array, k: int, n_iters: int = 50, stiffness: float = 1.0,
                n_restarts: int = 8) -> tuple[Array, Array]:
    """
    Soft k-means (isotropic EM) with k-means++ initialisation, best of `n_restarts` runs.

    Parameters
    ----------
    key
        `jax.random` PRNG key (split into one k-means++ init per restart).
    X
        Points to cluster. Shape `(N, D)`.
    k
        Number of clusters.
    n_iters
        Number of Lloyd iterations.
    stiffness
        Softmax inverse-temperature β. β -> +inf recovers hard k-means; β -> 0 gives uniform
        memberships. Tune relative to the scale of `X` (β acts on squared distances).
    n_restarts
        Number of independently initialised runs. All run in parallel under `vmap`, and the one
        reaching the lowest free energy is returned.

    Returns
    -------
    centers
        Cluster centers of the best restart. Shape `(k, D)`.
    resp
        Its responsibilities, each row summing to 1. Shape `(N, k)`.
    """
    def fit(key: Array) -> tuple[Array, Array, Array]:
        def lloyd(C, _):
            R = nn.softmax(-stiffness * _sq_dists(X, C), axis=1)  # E-step: (N, k)
            C = (R.T @ X) / R.sum(0)[:, None]                     # M-step: (k, D)
            return C, None

        C, _ = lax.scan(lloyd, kmeanspp_init(key, X, k), None, length=n_iters)
        logits = -stiffness * _sq_dists(X, C)
        # Free energy the E-step attains, up to the constant factor 1/β: the objective Lloyd
        # descends here, so restarts are ranked by what they actually optimise. Hard inertia
        # (Σ min d²) only agrees with it as β -> +inf.
        return C, nn.softmax(logits, axis=1), -nn.logsumexp(logits, axis=1).sum()

    C, R, free_energy = vmap(fit)(random.split(key, n_restarts))
    best = jnp.argmin(free_energy)
    return C[best], R[best]
