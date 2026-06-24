"""TwoNN intrinsic dimension estimator (Facco et al. 2017)."""
import numpy as np
from sklearn.neighbors import NearestNeighbors


def twonn_id(X, n_subsample=None, random_state=42):
    """
    Estimate intrinsic dimension from the ratio of 1st and 2nd nearest-neighbor distances.

    X: (N, d) array of points
    Returns: float ID estimate, or nan if degenerate
    """
    rng = np.random.default_rng(random_state)
    N = len(X)

    if n_subsample is not None and n_subsample < N:
        idx = rng.choice(N, n_subsample, replace=False)
        X = X[idx]
        N = n_subsample

    if N < 10:
        return float("nan")

    nbrs = NearestNeighbors(n_neighbors=3, algorithm="auto").fit(X)
    dists, _ = nbrs.kneighbors(X)

    r1 = dists[:, 1]
    r2 = dists[:, 2]

    valid = (r1 > 0) & (r2 > r1)
    if valid.sum() < 10:
        return float("nan")

    mu = r2[valid] / r1[valid]
    # MLE for Pareto: d = (N-1) / sum(log(mu))
    d_hat = (valid.sum() - 1) / np.sum(np.log(mu))
    return float(d_hat)
