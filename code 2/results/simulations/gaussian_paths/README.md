# Controlled Gaussian-path experiment

This experiment uses `P_i = N(sum_{r<=i} v_r, sigma^2 I)` and keeps every
one-step shift norm fixed while varying only its direction.  Hellinger distance
uses `H^2(P,Q) = 1 - BC(P,Q)`.

## Parameters

- `sigma = 1.0`
- `r/sigma in (0.1, 0.25, 0.5)`; the figures use `0.25`
- `k = 2,...,16`
- ambient dimension `16`
- clustered paths use aligned blocks of size `4`, with orthogonal block axes
- random paths average `50` independent draws with public seeds `20270927,...,20270976`
- exact `chi(C)` is evaluated by exhaustive binary quadratic maximization

## Numerical checks

- Maximum error in `h^T C h = H^2(P_0,P_k)`: `7.092e-15`
- Maximum deviation of a computed one-step Hellinger magnitude from its common theoretical value: `3.331e-16`

## Main setting at k=16, r/sigma=0.25

- Aligned: H=0.9299, TV=0.9545, chi-certificate=0.9299.
- Alternating: H=0.0000, TV=0.0000, chi-certificate=0.7057.
- Orthogonal: H=0.3428, TV=0.3829, chi-certificate=0.3428.
- Block-aligned (b=4): H=0.6273, TV=0.6827, chi-certificate=0.6273.
- Random (mean): H=0.3375, TV=0.3768, chi-certificate=0.3990.

The Gaussian/GDP group baseline is `sqrt(1-exp(-(k*r/sigma)^2/8))`; it is
attained by the fully aligned mean path.  The generic triangle bound is
`min(1,k*h)`.  `B_+(C) = k + 2 sum_{i<j} [c_ij]_+`, and the spectral factor is
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
