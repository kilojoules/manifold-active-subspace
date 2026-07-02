"""TwoNN intrinsic dimension estimator (Facco et al. 2017)."""
import numpy as np
from sklearn.neighbors import NearestNeighbors


def _two_nn_ratios(X):
    """Sorted ratios mu = r2/r1 of second- to first-NN distances (ties dropped)."""
    nbrs = NearestNeighbors(n_neighbors=3).fit(X)
    dists, _ = nbrs.kneighbors(X)
    r1, r2 = dists[:, 1], dists[:, 2]
    valid = (r1 > 0) & (r2 > r1)
    return np.sort(r2[valid] / r1[valid])


def twonn_id(X, discard_fraction=0.1):
    """Facco et al. (2017) TwoNN estimator.

    Follows the original procedure: compute mu_i = r2/r1, form the empirical
    CDF F(mu_(i)) = i/M, discard the `discard_fraction` largest ratios, and fit
    -log(1 - F(mu)) = d * log(mu) with a straight line through the origin.
    """
    X = np.asarray(X, dtype=np.float64)
    if len(X) < 10:
        return float("nan")
    mu = _two_nn_ratios(X)
    M = len(mu)
    if M < 10:
        return float("nan")
    F = np.arange(1, M + 1) / M
    keep = min(int(np.floor(M * (1.0 - discard_fraction))), M - 1)
    x = np.log(mu[:keep])
    y = -np.log(1.0 - F[:keep])
    denom = np.sum(x * x)
    if denom <= 0:
        return float("nan")
    return float(np.sum(x * y) / denom)


def twonn_id_mle(X):
    """Bias-corrected Pareto MLE variant: d = (M-1) / sum(log mu), no trimming.

    Kept for comparison with the CDF-fit estimator; both agree asymptotically.
    """
    X = np.asarray(X, dtype=np.float64)
    if len(X) < 10:
        return float("nan")
    mu = _two_nn_ratios(X)
    M = len(mu)
    if M < 10:
        return float("nan")
    s = np.sum(np.log(mu))
    return float((M - 1) / s) if s > 0 else float("nan")


def twonn_id_subsampled(X, n_rep=20, frac=0.8, random_state=0):
    """Mean and std of the Facco estimate over random subsamples (error bar)."""
    X = np.asarray(X, dtype=np.float64)
    rng = np.random.default_rng(random_state)
    N = len(X)
    m = max(10, int(N * frac))
    vals = []
    for _ in range(n_rep):
        idx = rng.choice(N, m, replace=False)
        v = twonn_id(X[idx])
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return float("nan"), float("nan")
    return float(np.mean(vals)), float(np.std(vals))
