# Reproducibility audit

The reproducibility workflow was rerun from a clean output directory on 4 September 2026. The supplied driver regenerated the source numerical sweeps and the numerical results underlying the manuscript tables, figures, and compact validation files without errors.

## Environment used for the final audit

- Python 3.13.5
- NumPy 2.3.5
- pandas 2.2.3
- Matplotlib 3.10.8

The declared requirements remain intentionally broader (`numpy>=1.24`, `pandas>=2.0`, `matplotlib>=3.7`) because the code uses long-stable public APIs.

## Numerical checks

- Representative Shishkin finite-difference algebraic residual: `2.07e-13`.
- Representative Bakhvalov-S finite-difference algebraic residual: `8.42e-12`.
- Full-BVP Shishkin crossover: 42 strict wins + 6 resolution-floor ties for the target allocation in each norm, with 0 reversals.
- 800-case frozen balanced validation: 95.375% exact integer matches, 100% within one interval, mean regret 0.000485%, maximum regret 0.05461%.
- Bakhvalov-S balanced coefficient at `N=1024`: theory `0.5767861755`; computed coefficient range `0.5767872502`--`0.5767925218`.
- Bakhvalov-S selected `N=256` balanced exhaustive optima: the 2/3-power prediction is exact in every listed case.
- Parameter-uniform pilot at `N=64`: equal split `1.0986320818e-02`; square-root split `6.9791903381e-03`.

## Data files

The files under `data/` provide direct reviewer-readable support for the manuscript tables and principal validation summaries. Some are compact projections of larger deterministic sweeps. Running `reproduce_jsc_stype.py` regenerates the complete source sweeps in `output/data/`, together with the figure files in `output/figures/`.

## Independent consistency checks

The exact manufactured endpoint-layer functions satisfy the stated boundary data. The finite-difference solution was substituted back into the assembled nonuniform discrete equations. The Bakhvalov-S interpolation constant was also checked independently by directly interpolating the exact layer; at `N=1024` the numerical/theoretical coefficient ratio is about `1.0000018`.
