#!/usr/bin/env python3
"""Controlled Gaussian-location paths for geometry-aware poisoning bounds.

For P_i = N(mu_i, sigma^2 I), mu_i = sum_{r<=i} v_r, this script computes
the exact endpoint Hellinger distance and TV, the exact square-root-density
Gram matrix, and several pathwise certificates.  All one-step shifts have the
same Euclidean norm.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import ndtr


SIGMA = 1.0
RATIOS = (0.10, 0.25, 0.50)
MAIN_RATIO = 0.25
K_VALUES = tuple(range(2, 17))
MAX_K = max(K_VALUES)
DIMENSION = MAX_K
CLUSTER_BLOCK_SIZE = 4
RANDOM_REPLICATES = 50
BASE_SEED = 20270927

PATTERNS = ("aligned", "alternating", "orthogonal", "clustered", "random")
PATTERN_LABELS = {
    "aligned": "Aligned",
    "alternating": "Alternating",
    "orthogonal": "Orthogonal",
    "clustered": "Block-aligned (b=4)",
    "random": "Random (mean)",
}
COLORS = {
    "aligned": "#0072B2",
    "alternating": "#D55E00",
    "orthogonal": "#009E73",
    "clustered": "#CC79A7",
    "random": "#E69F00",
}


def bc(mu_a: np.ndarray, mu_b: np.ndarray, sigma: float) -> float:
    """Bhattacharyya coefficient for equal isotropic covariance Gaussians."""
    sq = float(np.dot(mu_a - mu_b, mu_a - mu_b))
    return math.exp(-sq / (8.0 * sigma * sigma))


def hellinger_from_distance(distance: float, sigma: float) -> float:
    return math.sqrt(max(0.0, 1.0 - math.exp(-(distance**2) / (8.0 * sigma**2))))


def tv_from_distance(distance: float, sigma: float) -> float:
    return float(2.0 * ndtr(distance / (2.0 * sigma)) - 1.0)


def path_gram(means: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """Return one-step Hellinger magnitudes and their normalized Gram matrix."""
    k = means.shape[0] - 1
    bmat = np.empty((k + 1, k + 1), dtype=float)
    for i in range(k + 1):
        for j in range(k + 1):
            bmat[i, j] = bc(means[i], means[j], sigma)

    h2 = 1.0 - np.diag(bmat, k=1)
    h = np.sqrt(np.maximum(h2, 0.0))
    inner = np.empty((k, k), dtype=float)
    for i in range(k):
        for j in range(k):
            # <sqrt(p_{i+1})-sqrt(p_i), sqrt(p_{j+1})-sqrt(p_j)>
            inner[i, j] = (
                bmat[i + 1, j + 1]
                - bmat[i + 1, j]
                - bmat[i, j + 1]
                + bmat[i, j]
            )
    denom = 2.0 * np.outer(h, h)
    gram = inner / denom
    gram = 0.5 * (gram + gram.T)
    np.fill_diagonal(gram, 1.0)
    return h, gram


def exact_chi(cmat: np.ndarray) -> float:
    """Exact max_{z in {0,1}^k} z^T C z in O(k 2^k) work."""
    scores = np.zeros(1, dtype=float)
    bits = np.zeros((1, 0), dtype=float)
    for j in range(cmat.shape[0]):
        cross = bits @ cmat[j, :j] if j else np.zeros(1, dtype=float)
        added = scores + cmat[j, j] + 2.0 * cross
        scores = np.concatenate((scores, added))
        bits = np.vstack(
            (
                np.column_stack((bits, np.zeros(bits.shape[0]))),
                np.column_stack((bits, np.ones(bits.shape[0]))),
            )
        )
    return float(np.max(scores))


def make_directions(pattern: str, max_k: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    directions = np.zeros((max_k, dim), dtype=float)
    if pattern == "aligned":
        directions[:, 0] = 1.0
    elif pattern == "alternating":
        directions[:, 0] = np.where(np.arange(max_k) % 2 == 0, 1.0, -1.0)
    elif pattern == "orthogonal":
        directions[np.arange(max_k), np.arange(max_k)] = 1.0
    elif pattern == "clustered":
        directions[np.arange(max_k), np.arange(max_k) // CLUSTER_BLOCK_SIZE] = 1.0
    elif pattern == "random":
        directions = rng.normal(size=(max_k, dim))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    else:
        raise ValueError(f"Unknown pattern: {pattern}")
    return directions


def one_run(pattern: str, directions: np.ndarray, k: int, ratio: float, replicate: int) -> dict[str, float | int | str]:
    r = ratio * SIGMA
    steps = r * directions[:k]
    means = np.vstack((np.zeros(DIMENSION), np.cumsum(steps, axis=0)))
    h_steps, cmat = path_gram(means, SIGMA)
    endpoint_distance = float(np.linalg.norm(means[-1] - means[0]))
    endpoint_h = hellinger_from_distance(endpoint_distance, SIGMA)
    endpoint_tv = tv_from_distance(endpoint_distance, SIGMA)
    decomposition_h2 = float(h_steps @ cmat @ h_steps)
    h_step_theory = hellinger_from_distance(r, SIGMA)

    chi = exact_chi(cmat)
    upper = np.triu(cmat, k=1)
    b_plus = float(k + 2.0 * np.maximum(upper, 0.0).sum())
    lambda_max = float(np.linalg.eigvalsh(cmat)[-1])
    spectral = float(k * lambda_max)
    group_h = hellinger_from_distance(k * r, SIGMA)
    triangle_h = min(1.0, k * h_step_theory)

    def cert(factor: float) -> float:
        return min(1.0, h_step_theory * math.sqrt(max(0.0, factor)))

    return {
        "pattern": pattern,
        "replicate": replicate,
        "seed": BASE_SEED + replicate if pattern == "random" else BASE_SEED,
        "k": k,
        "dimension": DIMENSION,
        "sigma": SIGMA,
        "r": r,
        "r_over_sigma": ratio,
        "cluster_block_size": CLUSTER_BLOCK_SIZE,
        "endpoint_mean_distance": endpoint_distance,
        "step_h_theory": h_step_theory,
        "step_h_min": float(h_steps.min()),
        "step_h_max": float(h_steps.max()),
        "step_h_max_abs_error": float(np.max(np.abs(h_steps - h_step_theory))),
        "endpoint_h": endpoint_h,
        "endpoint_h2": endpoint_h**2,
        "endpoint_tv": endpoint_tv,
        "decomposition_h2": decomposition_h2,
        "decomposition_abs_error": abs(decomposition_h2 - endpoint_h**2),
        "chi": chi,
        "b_plus": b_plus,
        "spectral_factor": spectral,
        "lambda_max": lambda_max,
        "cert_exact_chi_h": cert(chi),
        "cert_b_plus_h": cert(b_plus),
        "cert_spectral_h": cert(spectral),
        "group_gaussian_h": group_h,
        "triangle_h": triangle_h,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict]) -> list[dict]:
    numeric = [key for key, value in rows[0].items() if isinstance(value, (int, float))]
    group_keys = ("pattern", "k", "r_over_sigma")
    grouped: dict[tuple, list[dict]] = {}
    for row in rows:
        grouped.setdefault(tuple(row[key] for key in group_keys), []).append(row)
    result = []
    for key, group in sorted(grouped.items()):
        out: dict[str, float | int | str] = dict(zip(group_keys, key))
        out["n_replicates"] = len(group)
        for field in numeric:
            vals = np.asarray([float(item[field]) for item in group])
            out[f"{field}_mean"] = float(vals.mean())
            out[f"{field}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        result.append(out)
    return result


def rows_for(summary: list[dict], pattern: str, ratio: float) -> list[dict]:
    return sorted(
        [r for r in summary if r["pattern"] == pattern and math.isclose(float(r["r_over_sigma"]), ratio)],
        key=lambda r: int(r["k"]),
    )


def style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 9.5,
            "legend.fontsize": 7.4,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 180,
        }
    )


def plot_main(summary: list[dict], out_stem: Path) -> None:
    style()
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.55), constrained_layout=True)
    ks = np.asarray(K_VALUES)

    ax = axes[0]
    for pattern in PATTERNS:
        data = rows_for(summary, pattern, MAIN_RATIO)
        y = np.asarray([r["endpoint_h_mean"] for r in data])
        sd = np.asarray([r["endpoint_h_std"] for r in data])
        ax.plot(ks, y, marker="o", ms=2.8, lw=1.5, color=COLORS[pattern], label=PATTERN_LABELS[pattern])
        if pattern == "random":
            ax.fill_between(ks, np.maximum(0, y - sd), np.minimum(1, y + sd), color=COLORS[pattern], alpha=0.18, linewidth=0)
    ref = rows_for(summary, "aligned", MAIN_RATIO)
    ax.plot(ks, [r["group_gaussian_h_mean"] for r in ref], "k--", lw=1.25, label="Gaussian group baseline")
    ax.plot(ks, [r["triangle_h_mean"] for r in ref], color="0.35", ls=":", lw=1.4, label="Triangle bound")
    ax.set(xlabel="Number of record changes, $k$", ylabel="Endpoint Hellinger distance", ylim=(-0.02, 1.02))
    ax.text(0.02, 0.97, "(a)", transform=ax.transAxes, fontweight="bold",
            va="top", ha="left", fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.85))
    ax.set_xticks((2, 4, 8, 12, 16))
    ax.legend(
        frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.03),
        fontsize=6.6, handlelength=1.4, columnspacing=0.9, handletextpad=0.4, borderaxespad=0.1,
    )

    ax = axes[1]
    fields = (
        ("endpoint_h_mean", "Endpoint", "#222222", "o"),
        ("cert_exact_chi_h_mean", r"$h\sqrt{\chi(C)}$", "#0072B2", "s"),
        ("cert_b_plus_h_mean", r"$h\sqrt{B_+(C)}$", "#009E73", "^"),
        ("cert_spectral_h_mean", "Spectral", "#CC79A7", "D"),
        ("group_gaussian_h_mean", "Gaussian group", "#D55E00", "v"),
        ("triangle_h_mean", "Triangle", "#777777", "P"),
    )
    x = np.arange(len(PATTERNS))
    offsets = np.linspace(-0.25, 0.25, len(fields))
    for offset, (field, label, color, marker) in zip(offsets, fields):
        vals = []
        for pattern in PATTERNS:
            row = [r for r in rows_for(summary, pattern, MAIN_RATIO) if int(r["k"]) == MAX_K][0]
            vals.append(row[field])
        ax.scatter(x + offset, vals, color=color, marker=marker, s=27, label=label, zorder=3)
    ax.set_xticks(x, ["Aligned", "Alternating", "Orthogonal", "Block-aligned", "Random"], rotation=25, ha="right")
    ax.set(ylabel="Hellinger distance / upper bound", ylim=(-0.02, 1.02))
    ax.text(0.02, 0.04, "(b)", transform=ax.transAxes, fontweight="bold",
            va="bottom", ha="left", fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.85))
    ax.grid(axis="y", color="0.9", linewidth=0.7)
    ax.legend(
        frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.03),
        fontsize=6.6, handlelength=1.4, columnspacing=0.9, handletextpad=0.4, borderaxespad=0.1,
    )

    ax = axes[2]
    orth = rows_for(summary, "orthogonal", MAIN_RATIO)
    box_h = np.minimum(1.0, np.asarray([r["cert_exact_chi_h_mean"] for r in orth]))
    general_tv = box_h * np.sqrt(2.0 - box_h**2)
    with np.errstate(divide="ignore"):
        gaussian_tv = np.where(box_h < 1.0, 2.0 * ndtr(np.sqrt(-2.0 * np.log1p(-box_h**2))) - 1.0, 1.0)
    group_tv = 2.0 * ndtr(ks * MAIN_RATIO / (2.0 * SIGMA)) - 1.0
    ax.plot(ks, [r["endpoint_tv_mean"] for r in orth], marker="o", ms=3.2, lw=1.5,
            color=COLORS["orthogonal"], label="Orthogonal TV")
    ax.plot(ks, gaussian_tv, ls="--", marker="o", ms=4.2, mfc="none", lw=1.0,
            color="#56B4E9", label="Gaussian box TV")
    ax.plot(ks, general_tv, marker="s", ms=2.8, lw=1.3, color="#D55E00", label="General box TV")
    ax.plot(ks, group_tv, "k--", lw=1.1, label="GDP group TV")
    ax.legend(
        frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.03),
        fontsize=6.6, handlelength=1.8, columnspacing=0.9, handletextpad=0.4, borderaxespad=0.1,
    )
    ax.set(xlabel="Number of record changes, $k$", ylabel="Total variation", ylim=(-0.02, 1.02))
    ax.text(0.02, 0.97, "(c)", transform=ax.transAxes, fontweight="bold",
            va="top", ha="left", fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.85))
    ax.set_xticks((2, 4, 8, 12, 16))
    ax.grid(axis="y", color="0.92", linewidth=0.6)

    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)

def plot_grams(out_stem: Path) -> None:
    style()
    k = 16
    fig, axes = plt.subplots(1, 5, figsize=(11.2, 2.35), constrained_layout=True)
    last_image = None
    for ax, pattern in zip(axes, PATTERNS):
        replicates = RANDOM_REPLICATES if pattern == "random" else 1
        cmats = []
        for rep in range(replicates):
            rng = np.random.default_rng(BASE_SEED + rep)
            directions = make_directions(pattern, MAX_K, DIMENSION, rng)
            steps = MAIN_RATIO * SIGMA * directions[:k]
            means = np.vstack((np.zeros(DIMENSION), np.cumsum(steps, axis=0)))
            _, cmat = path_gram(means, SIGMA)
            cmats.append(cmat)
        cmat = np.mean(cmats, axis=0)
        last_image = ax.imshow(cmat, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
        ax.set_title(PATTERN_LABELS[pattern])
        ax.set_xticks((0, 7, 15), (1, 8, 16))
        ax.set_yticks((0, 7, 15), (1, 8, 16))
        ax.set_xlabel("Step $j$")
    axes[0].set_ylabel("Step $i$")
    cbar = fig.colorbar(last_image, ax=axes, fraction=0.022, pad=0.018)
    cbar.set_label("Square-root-density alignment $c_{ij}$")
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)


def write_readme(path: Path, rows: list[dict], summary: list[dict]) -> None:
    max_decomp = max(float(r["decomposition_abs_error"]) for r in rows)
    max_step = max(float(r["step_h_max_abs_error"]) for r in rows)
    main = [r for r in summary if math.isclose(float(r["r_over_sigma"]), MAIN_RATIO) and int(r["k"]) == MAX_K]
    endpoint_lines = []
    for pattern in PATTERNS:
        row = [r for r in main if r["pattern"] == pattern][0]
        endpoint_lines.append(
            f"- {PATTERN_LABELS[pattern]}: H={row['endpoint_h_mean']:.4f}, "
            f"TV={row['endpoint_tv_mean']:.4f}, chi-certificate={row['cert_exact_chi_h_mean']:.4f}."
        )
    content = f"""# Controlled Gaussian-path experiment

This experiment uses `P_i = N(sum_{{r<=i}} v_r, sigma^2 I)` and keeps every
one-step shift norm fixed while varying only its direction.  Hellinger distance
uses `H^2(P,Q) = 1 - BC(P,Q)`.

## Parameters

- `sigma = {SIGMA}`
- `r/sigma in {RATIOS}`; the figures use `{MAIN_RATIO}`
- `k = {min(K_VALUES)},...,{max(K_VALUES)}`
- ambient dimension `{DIMENSION}`
- clustered paths use aligned blocks of size `{CLUSTER_BLOCK_SIZE}`, with orthogonal block axes
- random paths average `{RANDOM_REPLICATES}` independent draws with public seeds `{BASE_SEED},...,{BASE_SEED + RANDOM_REPLICATES - 1}`
- exact `chi(C)` is evaluated by exhaustive binary quadratic maximization

## Numerical checks

- Maximum error in `h^T C h = H^2(P_0,P_k)`: `{max_decomp:.3e}`
- Maximum deviation of a computed one-step Hellinger magnitude from its common theoretical value: `{max_step:.3e}`

## Main setting at k={MAX_K}, r/sigma={MAIN_RATIO}

{chr(10).join(endpoint_lines)}

The Gaussian/GDP group baseline is `sqrt(1-exp(-(k*r/sigma)^2/8))`; it is
attained by the fully aligned mean path.  The generic triangle bound is
`min(1,k*h)`.  `B_+(C) = k + 2 sum_{{i<j}} [c_ij]_+`, and the spectral factor is
`k lambda_max(C)`.

## Files

- `gaussian_path_results_raw.csv`: every deterministic run and all random replicates
- `gaussian_path_results_summary.csv`: mean and standard deviation by pattern, k, and ratio
- `gaussian_path_main.pdf/png`: endpoint separation and certificate comparison
- `gaussian_path_gram_matrices.pdf/png`: exact pathwise Gram matrices at the main setting

Reproduce from the project root with:

```bash
python3 simulations/gaussian_paths.py
```
"""
    path.write_text(content)


def main() -> None:
    output_dir = Path(__file__).resolve().parents[1] / "results" / "simulations" / "gaussian_paths"
    output_dir.mkdir(parents=True, exist_ok=True)

    deterministic = {}
    for pattern in PATTERNS[:-1]:
        deterministic[pattern] = make_directions(pattern, MAX_K, DIMENSION, np.random.default_rng(BASE_SEED))
    random_directions = [
        make_directions("random", MAX_K, DIMENSION, np.random.default_rng(BASE_SEED + rep))
        for rep in range(RANDOM_REPLICATES)
    ]

    rows: list[dict] = []
    for ratio in RATIOS:
        for k in K_VALUES:
            for pattern, directions in deterministic.items():
                rows.append(one_run(pattern, directions, k, ratio, 0))
            for rep, directions in enumerate(random_directions):
                rows.append(one_run("random", directions, k, ratio, rep))

    summary = summarize(rows)
    write_csv(output_dir / "gaussian_path_results_raw.csv", rows)
    write_csv(output_dir / "gaussian_path_results_summary.csv", summary)
    plot_main(summary, output_dir / "gaussian_path_main")
    plot_grams(output_dir / "gaussian_path_gram_matrices")
    write_readme(output_dir / "README.md", rows, summary)

    max_decomp = max(float(r["decomposition_abs_error"]) for r in rows)
    max_step = max(float(r["step_h_max_abs_error"]) for r in rows)
    if max_decomp > 1e-10 or max_step > 1e-10:
        raise RuntimeError(f"Sanity check failed: decomposition={max_decomp}, step={max_step}")
    print(f"Wrote {len(rows)} raw rows and {len(summary)} summary rows to {output_dir}")
    print(f"max decomposition error={max_decomp:.3e}; max step error={max_step:.3e}")


if __name__ == "__main__":
    main()
