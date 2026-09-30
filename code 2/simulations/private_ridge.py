#!/usr/bin/env python3
"""Controlled anisotropic-ridge reference paths for two DP perturbations.

We use fixed replacement-gradient directions d_i and the ridge Hessian
H_kappa = diag(1, 1/kappa).  Objective perturbation follows d_i, whereas
output perturbation follows q_i = -H_kappa^{-1} d_i.  Each direction is
normalized and assigned the same Gaussian signal-to-noise ratio, so all
one-step Hellinger magnitudes match exactly across mechanisms and kappa.
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


OUT = Path(__file__).resolve().parents[1] / "results" / "simulations" / "private_ridge"
SIGMA = 1.0
MAIN_RATIO = 0.20
RATIOS = (0.05, 0.10, 0.20, 0.40)
K_VALUES = tuple(range(2, 17, 2))
KAPPAS = (1, 2, 4, 8, 16, 32, 64)
MAIN_K = 16
MAIN_KAPPA = 32
BETA = 0.12
BASE_SEED = 20270927

MECHANISMS = ("objective", "output")
COLORS = {"objective": "#D55E00", "output": "#0072B2"}
LABELS = {"objective": "Objective perturbation", "output": "Output perturbation"}


def bc(mu_a: np.ndarray, mu_b: np.ndarray, sigma: float) -> float:
    delta = mu_a - mu_b
    return math.exp(-float(delta @ delta) / (8.0 * sigma * sigma))


def hellinger_from_distance(distance: float, sigma: float = SIGMA) -> float:
    return math.sqrt(max(0.0, 1.0 - math.exp(-(distance**2) / (8.0 * sigma**2))))


def tv_from_distance(distance: float, sigma: float = SIGMA) -> float:
    return float(2.0 * ndtr(distance / (2.0 * sigma)) - 1.0)


def path_gram(means: np.ndarray, sigma: float = SIGMA) -> tuple[np.ndarray, np.ndarray]:
    k = means.shape[0] - 1
    bmat = np.empty((k + 1, k + 1))
    for i in range(k + 1):
        for j in range(k + 1):
            bmat[i, j] = bc(means[i], means[j], sigma)
    h = np.sqrt(np.maximum(0.0, 1.0 - np.diag(bmat, k=1)))
    inner = np.empty((k, k))
    for i in range(k):
        for j in range(k):
            inner[i, j] = (
                bmat[i + 1, j + 1] - bmat[i + 1, j]
                - bmat[i, j + 1] + bmat[i, j]
            )
    gram = inner / (2.0 * np.outer(h, h))
    gram = 0.5 * (gram + gram.T)
    np.fill_diagonal(gram, 1.0)
    return h, gram


def exact_chi(cmat: np.ndarray) -> float:
    """Exact max over binary selectors, feasible here because k <= 16."""
    scores = np.zeros(1)
    bits = np.zeros((1, 0))
    for j in range(cmat.shape[0]):
        cross = bits @ cmat[j, :j] if j else np.zeros(1)
        added = scores + cmat[j, j] + 2.0 * cross
        scores = np.concatenate((scores, added))
        bits = np.vstack((
            np.column_stack((bits, np.zeros(bits.shape[0]))),
            np.column_stack((bits, np.ones(bits.shape[0]))),
        ))
    return float(scores.max())


def normalized_rows(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def gradient_directions(k: int) -> np.ndarray:
    signs = np.where(np.arange(k) % 2 == 0, 1.0, -1.0)
    alpha = math.sqrt(1.0 - BETA**2)
    return np.column_stack((alpha * signs, np.full(k, BETA)))


def mechanism_directions(mechanism: str, k: int, kappa: float) -> np.ndarray:
    d = gradient_directions(k)
    if mechanism == "objective":
        raw = d
    elif mechanism == "output":
        # Global minus sign does not affect within-path alignment.
        h_inv = np.diag((1.0, kappa))
        raw = -(d @ h_inv.T)
    else:
        raise ValueError(mechanism)
    return normalized_rows(raw)


def cert(step_h: float, factor: float) -> float:
    return min(1.0, step_h * math.sqrt(max(0.0, factor)))


def one_run(mechanism: str, k: int, kappa: float, ratio: float) -> tuple[dict, np.ndarray, np.ndarray]:
    dirs = mechanism_directions(mechanism, k, kappa)
    steps = ratio * SIGMA * dirs
    means = np.vstack((np.zeros(2), np.cumsum(steps, axis=0)))
    h_steps, c_exact = path_gram(means)
    c_cos = dirs @ dirs.T
    endpoint_distance = float(np.linalg.norm(means[-1]))
    endpoint_h = hellinger_from_distance(endpoint_distance)
    endpoint_tv = tv_from_distance(endpoint_distance)
    step_h = hellinger_from_distance(ratio * SIGMA)
    decomposition = float(h_steps @ c_exact @ h_steps)

    chi = exact_chi(c_exact)
    b_plus = float(k + 2.0 * np.maximum(np.triu(c_exact, 1), 0.0).sum())
    lambda_max = float(np.linalg.eigvalsh(c_exact)[-1])
    spectral = k * lambda_max
    chi_cos = exact_chi(c_cos)
    pair_mask = ~np.eye(k, dtype=bool)
    mean_cosine = float(c_cos[pair_mask].mean())
    mean_exact = float(c_exact[pair_mask].mean())

    row = {
        "mechanism": mechanism,
        "k": k,
        "kappa": kappa,
        "hessian_condition_number": kappa,
        "beta": BETA,
        "sigma": SIGMA,
        "r_over_sigma": ratio,
        "step_h_theory": step_h,
        "step_h_min": float(h_steps.min()),
        "step_h_max": float(h_steps.max()),
        "step_h_max_abs_error": float(np.max(np.abs(h_steps - step_h))),
        "endpoint_distance": endpoint_distance,
        "endpoint_h": endpoint_h,
        "endpoint_tv": endpoint_tv,
        "decomposition_h2": decomposition,
        "decomposition_abs_error": abs(decomposition - endpoint_h**2),
        "mean_offdiag_cosine": mean_cosine,
        "mean_offdiag_exact_gram": mean_exact,
        "gram_max_abs_cosine_error": float(np.max(np.abs(c_exact - c_cos))),
        "gram_normalized_fro_error": float(np.linalg.norm(c_exact - c_cos, "fro") / k),
        "chi": chi,
        "chi_cosine": chi_cos,
        "b_plus": b_plus,
        "lambda_max": lambda_max,
        "spectral_factor": spectral,
        "cert_exact_chi_h": cert(step_h, chi),
        "cert_cosine_chi_h": cert(step_h, chi_cos),
        "cert_b_plus_h": cert(step_h, b_plus),
        "cert_spectral_h": cert(step_h, spectral),
        "group_gaussian_h": hellinger_from_distance(k * ratio * SIGMA),
        "triangle_h": min(1.0, k * step_h),
    }
    return row, c_exact, c_cos


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def style() -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.4,
        "axes.labelsize": 9, "axes.titlesize": 9.4,
        "legend.fontsize": 7.3, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "figure.dpi": 180,
    })


def select(rows: list[dict], **kwargs) -> list[dict]:
    return [r for r in rows if all(r[key] == value for key, value in kwargs.items())]


def plot_main(rows: list[dict], out_stem: Path) -> None:
    style()
    # One row of four panels, sized for an IEEE two-column figure* (\textwidth).
    fig, axes = plt.subplots(1, 4, figsize=(11.6, 3.05), constrained_layout=True)

    def label(ax, text: str, loc: str = "top-left") -> None:
        x, y, va = (0.03, 0.96, "top") if loc == "top-left" else (0.03, 0.04, "bottom")
        ax.text(x, y, text, transform=ax.transAxes, fontweight="bold", fontsize=9.2,
                va=va, ha="left",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.85))

    ax = axes[0]
    for mech in MECHANISMS:
        data = sorted(select(rows, mechanism=mech, k=MAIN_K, r_over_sigma=MAIN_RATIO), key=lambda x: x["kappa"])
        ax.plot([x["kappa"] for x in data], [x["mean_offdiag_cosine"] for x in data],
                marker="o", lw=1.7, ms=3.2, color=COLORS[mech], label=LABELS[mech])
    ax.axhline(0, color="0.7", lw=0.8)
    ax.set_xscale("log", base=2)
    ax.set(xticks=KAPPAS, xlabel=r"Hessian condition number $\kappa_H$",
           ylabel="Mean off-diagonal cosine", ylim=(-1.05, 1.05))
    ax.set_xticklabels([str(x) for x in KAPPAS])
    label(ax, "(a)", "bottom-left")
    ax.legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.02),
              fontsize=7.0, handlelength=1.4, columnspacing=0.9, handletextpad=0.4, borderaxespad=0.1)

    ax = axes[1]
    for mech in MECHANISMS:
        data = sorted(select(rows, mechanism=mech, k=MAIN_K, r_over_sigma=MAIN_RATIO), key=lambda x: x["kappa"])
        ax.plot([x["kappa"] for x in data], [x["endpoint_h"] for x in data],
                marker="o", lw=1.7, ms=3.2, color=COLORS[mech], label=LABELS[mech])
    baseline = select(rows, mechanism="objective", k=MAIN_K, kappa=1, r_over_sigma=MAIN_RATIO)[0]
    ax.axhline(baseline["group_gaussian_h"], color="0.2", ls="--", lw=1.1, label="Gaussian group")
    ax.axhline(baseline["triangle_h"], color="0.45", ls=":", lw=1.2, label="Triangle")
    ax.set_xscale("log", base=2)
    ax.set(xticks=KAPPAS, xlabel=r"Hessian condition number $\kappa_H$",
           ylabel="Endpoint Hellinger distance", ylim=(-0.02, 1.05))
    ax.set_xticklabels([str(x) for x in KAPPAS])
    label(ax, "(b)", "top-left")
    ax.legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.02),
              fontsize=7.0, handlelength=1.4, columnspacing=0.9, handletextpad=0.4, borderaxespad=0.1)

    ax = axes[2]
    metrics = ("endpoint_h", "cert_exact_chi_h", "cert_b_plus_h", "cert_spectral_h", "group_gaussian_h", "triangle_h")
    names = ("Endpoint", r"$h\sqrt{\chi}$", r"$h\sqrt{B_+}$", "Spectral", "Group", "Triangle")
    x = np.arange(len(metrics))
    width = 0.36
    for offset, mech in ((-width / 2, "objective"), (width / 2, "output")):
        row = select(rows, mechanism=mech, k=MAIN_K, kappa=MAIN_KAPPA, r_over_sigma=MAIN_RATIO)[0]
        ax.bar(x + offset, [row[m] for m in metrics], width, color=COLORS[mech], alpha=0.88, label=LABELS[mech])
    ax.set(xticks=x, xticklabels=names, ylabel="Hellinger distance / bound", ylim=(0, 1.05))
    ax.tick_params(axis="x", labelsize=7)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    label(ax, "(c)", "top-left")
    ax.legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.02),
              fontsize=7.0, handlelength=1.4, columnspacing=0.9, handletextpad=0.4, borderaxespad=0.1)

    ax = axes[3]
    for mech in MECHANISMS:
        for kappa, ls in ((1, ":"), (MAIN_KAPPA, "-")):
            data = sorted(select(rows, mechanism=mech, k=MAIN_K, kappa=kappa), key=lambda x: x["r_over_sigma"])
            ax.plot([x["r_over_sigma"] for x in data], [x["gram_normalized_fro_error"] for x in data],
                    marker="o", lw=1.5, ms=3, ls=ls, color=COLORS[mech],
                    label=f"{LABELS[mech].split()[0]}, $\\kappa_H={kappa}$")
    ax.set(xlabel=r"One-step shift $\eta/\sigma$", ylabel=r"$\|C-C_{\rm cos}\|_F/k$", ylim=(-0.03, 1.05))
    label(ax, "(d)", "top-left")
    ax.legend(loc="lower right", bbox_to_anchor=(0.99, 0.16), ncol=1, fontsize=7.0,
              handlelength=1.6, frameon=True, framealpha=0.9, edgecolor="none")

    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".png"), bbox_inches="tight", dpi=260)
    plt.close(fig)

def plot_budget(rows: list[dict], out_stem: Path) -> None:
    style()
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.55), constrained_layout=True)
    panel_letters = ("(a)", "(b)")
    handles_labels = None
    for ax, mech, letter in zip(axes, MECHANISMS, panel_letters):
        data = sorted(select(rows, mechanism=mech, kappa=MAIN_KAPPA, r_over_sigma=MAIN_RATIO), key=lambda x: x["k"])
        ks = [x["k"] for x in data]
        curves = (
            ("endpoint_h", "Endpoint", "#222222", "o"),
            ("cert_exact_chi_h", r"$h\sqrt{\chi(C)}$", "#0072B2", "s"),
            ("cert_b_plus_h", r"$h\sqrt{B_+(C)}$", "#009E73", "^"),
            ("cert_spectral_h", "Spectral", "#CC79A7", "D"),
            ("group_gaussian_h", "Gaussian group", "#D55E00", "v"),
            ("triangle_h", "Triangle", "#777777", "P"),
        )
        for field, label, color, marker in curves:
            ax.plot(ks, [x[field] for x in data], label=label, color=color, marker=marker, lw=1.3, ms=3)
        ax.set(xlabel="Poisoning budget $k$", ylabel="Hellinger distance / bound",
               ylim=(-0.02, 1.03), xticks=K_VALUES)
        # Top-left is the one corner every curve stays clear of (all series
        # start near the bottom at small k), so it is a safe home for the
        # panel label without covering any line.
        ax.text(0.03, 0.97, f"{letter} {LABELS[mech]} ($\\kappa_H={MAIN_KAPPA}$)", transform=ax.transAxes,
                fontweight="bold", fontsize=8.4, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.85))
        if handles_labels is None:
            handles_labels = ax.get_legend_handles_labels()
    fig.legend(*handles_labels, frameon=False, ncol=6, loc="lower center",
               bbox_to_anchor=(0.5, 1.0), bbox_transform=fig.transFigure,
               fontsize=7.4, handlelength=1.4, columnspacing=1.0, handletextpad=0.4)
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".png"), bbox_inches="tight", dpi=260)
    plt.close(fig)


def plot_grams(grams: dict[tuple[str, int, float, float], tuple[np.ndarray, np.ndarray]], out_stem: Path) -> None:
    style()
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 6.0), constrained_layout=True)
    for col, mech in enumerate(MECHANISMS):
        exact, cosine = grams[(mech, MAIN_K, MAIN_KAPPA, MAIN_RATIO)]
        for row_idx, (mat, title) in enumerate(((exact, "Exact finite-shift Gram"), (cosine, "Cosine approximation"))):
            ax = axes[row_idx, col]
            im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
            ax.set(title=f"{LABELS[mech]}\n{title}", xlabel="Record index", ylabel="Record index")
    fig.colorbar(im, ax=axes, shrink=0.75, label="Gram entry")
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".png"), bbox_inches="tight", dpi=260)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    grams = {}
    for ratio in RATIOS:
        for kappa in KAPPAS:
            for k in K_VALUES:
                for mech in MECHANISMS:
                    row, c_exact, c_cos = one_run(mech, k, kappa, ratio)
                    rows.append(row)
                    grams[(mech, k, kappa, ratio)] = (c_exact, c_cos)
    write_csv(OUT / "private_ridge_results_raw.csv", rows)

    summary = [r for r in rows if r["r_over_sigma"] == MAIN_RATIO]
    write_csv(OUT / "private_ridge_results_summary.csv", summary)
    plot_main(rows, OUT / "private_ridge_main")
    plot_budget(rows, OUT / "private_ridge_budget_sweep")
    plot_grams(grams, OUT / "private_ridge_gram_matrices")

    checks = {
        "max_step_h_error": max(r["step_h_max_abs_error"] for r in rows),
        "max_decomposition_error": max(r["decomposition_abs_error"] for r in rows),
        "min_gram_eigenvalue": min(float(np.linalg.eigvalsh(x[0])[0]) for x in grams.values()),
    }
    key_rows = {
        mech: select(rows, mechanism=mech, k=MAIN_K, kappa=MAIN_KAPPA, r_over_sigma=MAIN_RATIO)[0]
        for mech in MECHANISMS
    }
    print("checks", checks)
    for mech, row in key_rows.items():
        print(mech, {key: row[key] for key in (
            "mean_offdiag_cosine", "endpoint_h", "endpoint_tv", "chi", "cert_exact_chi_h",
            "cert_b_plus_h", "cert_spectral_h", "group_gaussian_h", "triangle_h",
            "gram_normalized_fro_error")})


if __name__ == "__main__":
    main()
