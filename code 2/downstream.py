"""Per-instance downstream certificates (Section VI-C, Appendix B-B).

For each test point z, the event is z^T theta > c for a fixed public threshold
c on the transformed label scale. Both mechanisms release a Gaussian
prediction, so q(z) = Phi^{-1}(p(z)) = (z^T theta_hat - c) / sqrt(z^T Sigma z).
Under a-GDP, Phi(q - a) <= p'(z) <= Phi(q + a), so the majority decision is
certified when |q(z)| > a. We compare a = U0 (group privacy), a = U2, and
a = U3, and also report the TV conversion |p(z) - 1/2| > 2 Phi(a/2) - 1.
"""
import csv
import resource
import sys
import time

import numpy as np
from scipy.special import ndtr

import common as C

BUDGETS = (1, 2, 4, 8)
MU = 1.0
THRESHOLD = {"wine_red": 0.6, "cahousing": 200000 / 500001}


def main():
    C.RESULTS.mkdir(exist_ok=True)
    rows, timing = [], []
    for name in C.DATASETS:
        features, labels, _ = C.load(name)
        for seed in C.SEEDS:
            design = C.split_design(features, labels, seed)
            x_test = design["x_test"]
            margin_raw = x_test @ design["theta"] - THRESHOLD[name]
            correct = (margin_raw > 0) == (design["y_test"] > THRESHOLD[name])
            for mechanism in C.MECHANISMS:
                W, Sigma = C.whitened(design, mechanism, mu=MU)
                q = margin_raw / np.sqrt(np.einsum("ij,jk,ik->i", x_test, Sigma, x_test))
                p_margin = np.abs(ndtr(q) - 0.5)
                for threat in C.THREATS:
                    low, high = C.replacement_range(design["y"], threat)
                    step = 1.0 if threat == "full" else C.RESTRICTED_RADIUS
                    for k in BUDGETS:
                        start = time.perf_counter()
                        U = C.bounds(W, low, high, k)
                        elapsed = time.perf_counter() - start
                        caps = {"U0": k * MU * step, "U2": U[2], "U3": U[3]}
                        row = dict(dataset=name, seed=seed, mechanism=mechanism, threat=threat, k=k,
                                   accuracy=float(correct.mean()), **caps)
                        for tag, a in caps.items():
                            certified = np.abs(q) > a
                            row[f"certified_{tag}"] = float(certified.mean())
                            row[f"certified_correct_{tag}"] = float((certified & correct).mean())
                            row[f"tv_certified_{tag}"] = float((p_margin > C.tv_gaussian(a)).mean())
                        rows.append(row)
                        if name == "cahousing":
                            timing.append(dict(seed=seed, mechanism=mechanism, threat=threat, k=k,
                                               n=design["n"], d=design["x"].shape[1], seconds=elapsed))
            print(f"{name}: split {seed} done", flush=True)
    for fname, data in (("downstream.csv", rows), ("timing_cahousing.csv", timing)):
        with (C.RESULTS / fname).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_mb = peak / (1024 ** 2) if sys.platform == "darwin" else peak / 1024
    print(f"peak resident memory: {peak_mb:.0f} MB")


if __name__ == "__main__":
    main()
