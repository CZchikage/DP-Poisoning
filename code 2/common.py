"""Shared routines for the fixed-design label-replacement experiments.

Model: ridge regression with public, standardized features, an unpenalized
intercept, and labels mapped to [0, 1]. Output and objective perturbation are
exact Gaussian mechanisms with a common covariance, so every certificate is a
statement about whitened mean shifts w_i (Section V-D).
"""
from pathlib import Path
import itertools

import numpy as np
import pandas as pd
from scipy.special import ndtr

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"

LAMBDA = 0.1          # ridge parameter
SEEDS = range(5)      # fixed 80/20 splits
TRAIN_FRACTION = 0.8
RESTRICTED_RADIUS = 0.1
DATASETS = ("wine_red", "cahousing")
MECHANISMS = ("output", "objective")
THREATS = ("full", "restricted")


def tv_gaussian(separation):
    """TV distance between two Gaussians with a common covariance and
    Mahalanobis separation `separation`."""
    return 2.0 * ndtr(np.asarray(separation) / 2.0) - 1.0


def load(name):
    """Return features, labels in [0, 1], and the label map.
    Label maps use public ranges only."""
    if name == "wine_red":
        raw = np.loadtxt(DATA / "winequality-red.csv", delimiter=";", skiprows=1)
        return raw[:, :-1], np.clip(raw[:, -1] / 10.0, 0.0, 1.0), "quality/10"
    if name == "cahousing":
        frame = pd.read_csv(DATA / "housing.csv").dropna()   # rows with missing features
        numeric = frame.drop(columns=["median_house_value", "ocean_proximity"]).to_numpy(float)
        indicators = pd.get_dummies(frame.ocean_proximity.astype(str), drop_first=True).to_numpy(float)
        labels = np.clip(frame.median_house_value.to_numpy() / 500001.0, 0.0, 1.0)
        return np.column_stack([numeric, indicators]), labels, "value/500001"
    raise ValueError(name)


def split_design(features, labels, seed):
    """Fixed split, standardization by training statistics, intercept column,
    and the ridge inverse Hessian K (intercept unpenalized)."""
    order = np.random.default_rng(seed).permutation(len(labels))
    n = int(TRAIN_FRACTION * len(labels))
    train, test = order[:n], order[n:]
    mean = features[train].mean(0)
    scale = features[train].std(0)
    scale = np.where(scale > 0, scale, 1.0)
    x_train = np.column_stack([(features[train] - mean) / scale, np.ones(n)])
    x_test = np.column_stack([(features[test] - mean) / scale, np.ones(len(test))])
    penalty = np.eye(x_train.shape[1])
    penalty[-1, -1] = 0.0
    K = np.linalg.inv(x_train.T @ x_train / n + LAMBDA * penalty)
    theta = K @ x_train.T @ labels[train] / n
    return dict(n=n, x=x_train, x_test=x_test, y=labels[train], y_test=labels[test],
                K=K, theta=theta)


def whitened(design, mechanism, mu=1.0):
    """Whitened record vectors w_i and the released covariance at label-GDP
    level mu on the label domain [0, 1], so that max_i ||w_i|| = mu."""
    x, K, n = design["x"], design["K"], design["n"]
    directions = x @ K if mechanism == "output" else x
    sigma = np.linalg.norm(directions, axis=1).max() / mu
    shape = np.eye(x.shape[1]) if mechanism == "output" else K @ K
    return directions / sigma, (sigma / n) ** 2 * shape


def replacement_range(labels, threat):
    """Admissible label displacements [t^-, t^+] for each record."""
    low, high = -labels, 1.0 - labels
    if threat == "restricted":
        low = np.maximum(low, -RESTRICTED_RADIUS)
        high = np.minimum(high, RESTRICTED_RADIUS)
    return low, high


def bounds(W, low, high, k, block=1500):
    """U0 >= U1 >= U2 >= U3 >= R_k of Eq. (68), with label-domain diameter 1.
    Row sums of the k largest entries of W_ij = b_i b_j |G_ij| are formed in
    blocks so that no n x n matrix is stored."""
    g = np.linalg.norm(W, axis=1)
    b = np.maximum(-low, high)
    n = len(g)

    def top(v):
        return np.partition(v, n - k)[-k:].sum()

    rows = np.empty(n)
    for start in range(0, n, block):
        A = np.abs(W[start:start + block] @ W.T) * b[start:start + block, None] * b[None, :]
        rows[start:start + block] = np.partition(A, n - k, axis=1)[:, -k:].sum(1)
    U0, U1, U2 = k * g.max(), top(g), top(b * g)
    return np.array([U0, U1, U2, min(U2, np.sqrt(top(rows)))])


def feasible_attack(W, low, high, k, seed):
    """Feasible attack with displacement L <= R_k, found by alternating
    direction and support updates. Returns L, changed indices, label shifts."""
    rng = np.random.default_rng(seed)
    d = W.shape[1]
    g = np.linalg.norm(W, axis=1)
    largest = W[np.argsort(g)[-8:]]
    starts = np.vstack([np.eye(d), -np.eye(d), largest, -largest, rng.normal(size=(16, d))])
    best, best_idx, best_t = 0.0, None, None
    for e in starts:
        e = e / max(np.linalg.norm(e), 1e-20)
        for _ in range(50):
            projection = W @ e
            t = np.where(projection >= 0, high, low)
            idx = np.argsort(t * projection)[-k:]
            shift = t[idx] @ W[idx]
            value = np.linalg.norm(shift)
            if value > best:
                best, best_idx, best_t = value, idx.copy(), t[idx].copy()
            if value <= 1e-20:
                break
            new = shift / value
            if np.linalg.norm(new - e) < 1e-10:
                break
            e = new
    assert len(best_idx) <= k
    assert np.all(best_t >= low[best_idx] - 1e-12) and np.all(best_t <= high[best_idx] + 1e-12)
    return float(best), best_idx.tolist(), best_t.tolist()


def exact_candidate_optimum(W, low, high, kmax):
    """Exact max over |S| <= k and endpoint choices on a small candidate set,
    for every k <= kmax; each record is unchanged or moved to an endpoint."""
    choice = np.array(list(itertools.product((0, 1, 2), repeat=len(W))))
    T = np.where(choice == 1, low, np.where(choice == 2, high, 0.0))
    value = np.linalg.norm(T @ W, axis=1)
    support = (choice > 0).sum(1)
    return {k: float(value[support <= k].max()) for k in range(1, kmax + 1)}
