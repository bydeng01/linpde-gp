# Inequality-Constrained GP-PDE

This guide covers the inequality-constrained experiments in
[LinPDE-GP](../../README.md). Installation instructions are in the root README.
Paper: [arXiv:2610.13804](https://arxiv.org/abs/2610.13804).

The method conditions a Gaussian-process prior on two kinds of information in sequence. The
equality information *E* (a PDE residual, boundary data, noisy observations) is handled by
`linpde-gp`. The inequality information *C* (bounds, monotonicity, no-arbitrage conditions),
imposed at a finite set of virtual points, is handled by
[`gp_constr`](https://github.com/bydeng01/gp_constr). The result is the posterior of *u*
given both *E* and *C*.

## Repository map

```
src/linpde_gp_constrained/                # integration package (linpde-gp + gp_constr)
├── linpde_gp_constrained/                #   the importable package
├── tests/                                #   unit and integration tests
└── docs/upstream-issues.md               #   backend limitations and wrapper workarounds
gp_constr/                                # submodule: inequality-constrained GP backend
experiments/linpde_gp_constrained/        # everything that produces a paper number or figure
├── poisson_2d_weak_form/                 #   2D Poisson with partial boundary data
├── poisson_2d_ablation/                  #   equality-only vs inequality-only vs both
├── poisson_2d_data_assimilation/         #   noisy and sparse observations
├── black_scholes_surface/                #   arbitrage-free option-price surfaces
├── air_quality_adv_diff/                 #   nonnegative air-quality field
├── poisson_weak_form/                    #   1D weak-form Poisson example
└── conventions/                          #   check of gp_constr's kernel conventions
```

### Integration package

`src/linpde_gp_constrained` is a thin wrapper; it does not modify `linpde-gp` or `gp_constr`.

| Piece | Role |
| :--- | :--- |
| `make_pde_collocation_posterior` | equality posterior from a pointwise (collocation) PDE residual plus boundary data or point observations |
| `make_pde_equality_posterior` | equality posterior from a weak-form PDE residual (test-basis projection) plus boundary data |
| `InequalityConstraint` | one constraint `a <= L u(X_v) <= b` at virtual points `X_v`, with `L` the identity or a first-order partial derivative |
| `ConstrainedLinPDEPosterior` | conditions an equality posterior on a list of inequality constraints and returns constrained means and samples |
| `LinPDEKernelAdapter`, `PriorKernelAdapter` | expose a `linpde-gp` posterior, or the bare prior, to `gp_constr` |
| `transform_bounds_for_zero_mean` | folds the equality posterior mean into the inequality bounds |

### Experiments

| Folder | Paper | What it shows | Entry point |
| :--- | :--- | :--- | :--- |
| `poisson_2d_weak_form/` | §6.1 | 2D Poisson with boundary data on two edges only: monotonicity corrects the reconstruction near the unobserved corner | `run.py`, then `figure.py`, `figure_full_7panel.py`, `make_table.py` |
| `poisson_2d_ablation/` | §6.1 | the same problem with equality-only, inequality-only and combined conditioning | `reproduce.sh` |
| `poisson_2d_data_assimilation/` | §6.1 | noisy scattered observations: the gain from monotonicity grows as the data get noisier or sparser | `reproduce.sh` |
| `black_scholes_surface/` | §6.2, App. C.1 | call-price surfaces under no-arbitrage constraints, synthetic (Black–Scholes) and real (S&P 500) | `reproduce.sh`, `reproduce_spx.sh` |
| `air_quality_adv_diff/` | §6.3, App. C.2 | a nonnegative pollutant field from an advection–diffusion model and sparse monitors, synthetic and real (OpenAQ) | `reproduce.sh`, `sweep_real.py` |
| `poisson_weak_form/` | — | 1D Poisson with a weak-form PDE residual (not in the paper) | `run.py`, then `figure.py` |
| `conventions/` | — | numerical check of `gp_constr`'s derivative-kernel conventions (NumPy only, under a second) | `verify_gpconstr_conventions.py` |

Each folder also holds the outputs behind the paper: figure PDFs, `comparison.txt` and
`diagnostics.json` from the snapshot run, sweep CSVs, and `*_numbers.tex` LaTeX macros. The
figure scripts render from the saved run outputs (`fields.npz`, written by `run.py`) and the
sweep CSVs. The NPZ files are local outputs and are not committed: snapshot figures
require a prior solver run, while CSV-only summaries can be regenerated from a fresh
checkout. The `check_*.py` scripts are fast R-free sanity checks (closed-form kernel
dispatch, ground truth); the air-quality and Black–Scholes `reproduce.sh` scripts
run them first.

## Requirements

Use the Docker image from the root README. It includes `linpde-gp`, `probnum`, JAX,
`gp_constr`, R, `rpy2` and the R packages that `gp_constr` uses for truncated multivariate
normal computations (`mvtnorm`, `tmvtnorm`, `TruncatedNormal`, `truncnorm`). The image installs
the current CRAN versions; the paper used R 4.5.0, mvtnorm 1.3-6, tmvtnorm 1.7,
TruncatedNormal 2.3 and truncnorm 1.0-9.

Without R, the equality-only results and the R-free checks still run, but the constrained
results need the R backend. The air-quality scripts can instead fall back to an
experiment-local truncated-normal sampler.

## Data

- `air_quality_adv_diff/data/`: four OpenAQ snapshots and an offline sample with synthetic
  values; the transport wind comes from Open-Meteo. Sources, attribution and licences are in
  [`air_quality_adv_diff/data/README.md`](air_quality_adv_diff/data/README.md).
- `black_scholes_surface/data/`: the S&P 500 option chain is not redistributed. How to
  download your own snapshot is in
  [`black_scholes_surface/data/README.md`](black_scholes_surface/data/README.md).
- All other experiments are synthetic and need no data.

## Reproduce

Run everything from the repository root inside the container:

```bash
./docker-run.sh shell          # drop into /app, then e.g.:
bash experiments/linpde_gp_constrained/poisson_2d_ablation/reproduce.sh
```

The `reproduce.sh` scripts set `PYTHONPATH` themselves. To run a single script, set it first:

```bash
export PYTHONPATH=src:gp_constr:src/linpde_gp_constrained
python experiments/linpde_gp_constrained/poisson_2d_weak_form/run.py
python experiments/linpde_gp_constrained/poisson_2d_weak_form/figure.py
```

Notes per experiment:

- **Air quality, OpenAQ.** `reproduce.sh` covers the synthetic experiment and runs the
  offline sample through the `--data` path. The pooled real-data result comes from

  ```bash
  python experiments/linpde_gp_constrained/air_quality_adv_diff/sweep_real.py \
      --config experiments/linpde_gp_constrained/air_quality_adv_diff/snapshots.txt
  ```

  Add `--from-csv` to rebuild its figures and LaTeX macros from the committed sweep CSV
  without re-running the solver.
- **Black–Scholes, S&P 500.** Download a snapshot first (see the data README), then run
  `bash experiments/linpde_gp_constrained/black_scholes_surface/reproduce_spx.sh data/spx_chain_<date>.csv [NSEEDS]`.
  The full sweep takes about 13–22 hours for 15–25 seeds.
- **Black–Scholes, synthetic figure.** `run.py` and `run.py --data` write `fields.npz` and
  `diagnostics.json` to the same folder, so a real-data run overwrites the synthetic outputs.
  Run `run.py` without `--data` before `figure.py`; `figure.py` stops with an error if it finds
  real-data outputs.
- **Poisson with partial boundary data.** `figure.py` draws the four-panel main figure,
  `figure_full_7panel.py` the seven-panel supplementary figure and `make_table.py` the metrics
  table. `sweep.py` is an R-free tuning sweep.

Random seeds are fixed in every script. Figure scripts write a PDF, which is tracked, plus
SVG/TIFF/PNG copies, which git ignores.

## Tests

```bash
./docker-run.sh test
```

This runs the whole test suite, including the package tests in
`src/linpde_gp_constrained/tests/`. Tests that need the R backend are skipped when it is
unavailable.

## Citation

```bibtex
@misc{deng2026constrainedprobabilisticnumericsgaussian,
      title={Constrained Probabilistic Numerics: Gaussian Process Solvers for Linear PDEs under Linear Inequality Constraints}, 
      author={Boyuan Deng and Kshitiz Upadhyay and Michael Shields},
      year={2026},
      eprint={2610.13804},
      archivePrefix={arXiv},
      primaryClass={stat.ML},
      url={https://arxiv.org/abs/2610.13804}, 
}
```
