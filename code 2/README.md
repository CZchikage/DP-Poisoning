# Cross-record geometry of multi-record poisoning robustness: experiments

Code and data for Figs. 1, 2, and 4-6 and Tables III-IX in the paper.
The finite-sample reference-path diagnostic in Fig. 3 is not included in this
artifact.

## Requirements

Python 3.10 or later with the packages in `requirements.txt`. No network access
is needed; both datasets are included in `data/`, with sources and SHA-256
hashes in `data/MANIFEST.json`.

## Reproduction

```sh
pip install -r requirements.txt
./run_all.sh
```

All outputs are written to `results/`. Runtime is about 15 minutes on one CPU
core, most of it spent on California housing.

| Script | Paper | Outputs |
|---|---|---|
| `simulations/gaussian_paths.py` | Sec. VI-A, Figs. 1 and 4 | `results/simulations/gaussian_paths/` |
| `simulations/private_ridge.py` | Sec. VI-B, Figs. 2, 5, and 6 | `results/simulations/private_ridge/` |
| `certificates.py` | Sec. VI-C, Appendix B-B | `bounds.csv`, `exact_candidates.csv`, `utility.csv`, `feasible_attacks.json`, `datasets.csv` |
| `downstream.py` | Sec. VI-C (downstream certificates), Appendix B-B | `downstream.csv`, `timing_cahousing.csv` |
| `tables.py` | LaTeX tables of Sec. VI-C and Appendix B-B | `results/tables/*.tex` |

Shared routines (data loading, design, whitening, bounds, attack search, exact
enumeration) are in `common.py`.

## Protocol for the real-design experiments

- **Data.** Red-wine subset of Wine Quality, and California housing after
  removing the 207 rows with a missing `total_bedrooms` value;
  `ocean_proximity` is one-hot encoded with `<1H OCEAN` as the reference level.
- **Labels.** Mapped to [0, 1] by fixed public rules: quality/10 and
  value/500001 (the census top-code).
- **Splits.** Seeds 0 to 4 of NumPy's default generator permute the rows; the
  first floor(0.8 N) rows form the training set.
- **Design.** Features standardized by training statistics, intercept
  appended, ridge parameter 0.1, intercept unpenalized.
- **Mechanisms.** Gaussian output and objective perturbation, both calibrated
  to exact label-level mu-GDP on the domain [0, 1]; objective perturbation
  uses no curvature correction. Displacements are computed at mu = 1 and
  scale linearly with mu.
- **Threat sets.** Full-domain replacements y' in [0, 1], and restricted
  replacements |y' - y| <= 0.1 under the same calibration.
- **Budgets.** k in {1, 2, 4, 8, 16} for certificates and {1, 2, 4, 8} for
  downstream certificates.

The feasible-attack search gives lower bounds only; it has no global
optimality guarantee on full designs. Exact optima are computed on
12-record candidate sets and are not full-design optima. Computations use
float64 arithmetic.
