"""Label-replacement certificates on the public designs (Section VI-C,
Appendix B-B).

For every dataset, split, mechanism, threat set, and budget k this script
computes U0, ..., U3, a feasible attack L, and exact optima on small candidate
sets, together with test MSE over a privacy grid. Displacements are computed
at mu = 1; they scale linearly with mu because w_i(mu) = mu * w_i(1).
"""
import csv
import json

import numpy as np

import common as C

BUDGETS = (1, 2, 4, 8, 16)
EXACT_BUDGETS = (2, 4, 8)
N_CANDIDATES = 12
MU_GRID = (0.25, 0.5, 1.0, 2.0)


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    C.RESULTS.mkdir(exist_ok=True)
    bound_rows, exact_rows, utility_rows, attacks, datasets = [], [], [], [], []
    for name in C.DATASETS:
        features, labels, label_map = C.load(name)
        for seed in C.SEEDS:
            design = C.split_design(features, labels, seed)
            x_test, y_test = design["x_test"], design["y_test"]
            clean_mse = float(np.mean((x_test @ design["theta"] - y_test) ** 2))
            constant_mse = float(np.mean((design["y"].mean() - y_test) ** 2))
            if seed == 0:
                datasets.append(dict(dataset=name, N=len(labels), n_train=design["n"],
                                     d=design["x"].shape[1],
                                     rank=int(np.linalg.matrix_rank(design["x"])),
                                     label_map=label_map))
            for mechanism in C.MECHANISMS:
                W, Sigma = C.whitened(design, mechanism, mu=1.0)
                noise = float(np.einsum("ij,jk,ik->", x_test, Sigma, x_test) / len(x_test))
                for mu in MU_GRID:
                    utility_rows.append(dict(dataset=name, seed=seed, mechanism=mechanism, mu=mu,
                                             clean_mse=clean_mse, constant_mse=constant_mse,
                                             private_mse=clean_mse + noise / mu ** 2))
                for threat in C.THREATS:
                    low, high = C.replacement_range(design["y"], threat)
                    step = 1.0 if threat == "full" else C.RESTRICTED_RADIUS
                    found = {}
                    for k in BUDGETS:
                        U = C.bounds(W, low, high, k)
                        L, idx, shift = C.feasible_attack(W, low, high, k, seed + 100)
                        assert L <= U[3] + 1e-10 and np.all(np.diff(U) <= 1e-10)
                        found[k] = L
                        U0 = k * step                   # group baseline under the matching adjacency
                        U1 = U[1] * step                # equals U2 for restricted replacements
                        total = U0 - U[3]
                        valid = total / U0 > 1e-3
                        bound_rows.append(dict(
                            dataset=name, seed=seed, mechanism=mechanism, threat=threat, k=k,
                            U0=U0, U1=U1, U2=U[2], U3=U[3], L=L, L_over_U3=L / U[3],
                            magnitude_share=(U0 - U1) / total if valid else np.nan,
                            reach_share=(U1 - U[2]) / total if valid else np.nan,
                            geometry_share=(U[2] - U[3]) / total if valid else np.nan))
                        attacks.append(dict(dataset=name, seed=seed, mechanism=mechanism, threat=threat,
                                            k=k, indices=[int(i) for i in idx],
                                            label_shift=[float(t) for t in shift], displacement=L))
                    b = np.maximum(-low, high)
                    cand = np.argsort(b * np.linalg.norm(W, axis=1))[-N_CANDIDATES:]
                    optimum = C.exact_candidate_optimum(W[cand], low[cand], high[cand], max(EXACT_BUDGETS))
                    for k in EXACT_BUDGETS:
                        U_cand = C.bounds(W[cand], low[cand], high[cand], k)
                        L_cand, _, _ = C.feasible_attack(W[cand], low[cand], high[cand], k, seed + 100)
                        exact_rows.append(dict(
                            dataset=name, seed=seed, mechanism=mechanism, threat=threat, k=k,
                            candidate_optimum=optimum[k], candidate_U3=U_cand[3],
                            U3_over_optimum=U_cand[3] / optimum[k],
                            candidate_search_gap=optimum[k] - L_cand, full_design_L=found[k]))
            print(f"{name}: split {seed} done", flush=True)
    write_csv(C.RESULTS / "bounds.csv", bound_rows)
    write_csv(C.RESULTS / "exact_candidates.csv", exact_rows)
    write_csv(C.RESULTS / "utility.csv", utility_rows)
    write_csv(C.RESULTS / "datasets.csv", datasets)
    (C.RESULTS / "feasible_attacks.json").write_text(json.dumps(attacks))
    assert all(r["full_design_L"] >= r["candidate_optimum"] - 1e-12 for r in exact_rows)


if __name__ == "__main__":
    main()
