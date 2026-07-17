# Helmholtz Equation & Brain MRE

Part of [LinPDE-GP](../README.md). Install once via the root README, then use this guide to
find your way around.

## Repository map

```
src/linpde_gp/                          # the solver core (shared library)
experiments/                            # everything that produces a paper number or figure
├── inHomo_*Helmholtz_*.ipynb           #   synthetic GP-PDE experiments
├── helmholtz_equation_baselines/       #   synthetic baseline solvers: FDM + PINN
├── helmholtz_brain_forward_bvp.py      #   brain GP-PDE driver
├── *fd*brain*.py, fd_*.py              #   brain deterministic FD baseline + its figures
├── helmholtz_brain_outputs/            #   brain results (logs, figures) land here
├── reproduce_paper.sh                  #   reproduce the whole paper
└── reproduce_brain.sh                  #   reproduce just the brain
data/brain_experiment_data/mre_udel/    # brain data (external — see "Data" below)
```

### Solver core

The Helmholtz support added to LinPDE-GP:

| Piece | Path |
| :--- | :--- |
| Helmholtz differential operator | `src/linpde_gp/linfuncops/diffops/_helmholtz_operator.py` |
| Helmholtz BVP problem definition | `src/linpde_gp/problems/pde/_helmholtz.py` |
| Operator → kernel dispatch (how the operator hits the covariance) | `src/linpde_gp/randprocs/covfuncs/linfuncops/diffops/_registry.py` |

### Synthetic experiments

Each experiment is a self-contained notebook directly under `experiments/`. The `Npts` suffix
is the number of interior collocation points, so the three variants per row are a convergence
sweep.

| Dimension | GP-PDE solver notebook (real / complex `k`) |
| :--- | :--- |
| 1D | `inHomo_{real,complex}Helmholtz_dirichlet_1d_{3,5,7}pts.ipynb` |
| 2D | `inHomo_{real,complex}Helmholtz_dirichlet_2d_{64,100,144}pts.ipynb` |
| 3D (cube) | `inHomo_{real,complex}Helmholtz_dirichlet_3d_{729,1000,1331}pts.ipynb` |

### Baseline solvers

The reference solvers the GP-PDE method is compared against live in **one folder**:

```
experiments/helmholtz_equation_baselines/
├── fdm_inHomo_{real,complex}Helmholtz_dirichlet_{1,2,3}d.ipynb    # finite-difference method
└── pinn_inHomo_{real,complex}Helmholtz_dirichlet_{1,2,3}d.ipynb   # physics-informed neural network
```

Six FDM + six PINN notebooks (one per dimension × real/complex).

### Brain MRE experiment

| Piece | Path | Role |
| :--- | :--- | :--- |
| GP-PDE driver | `experiments/helmholtz_brain_forward_bvp.py` | the method applied to brain data (paper reproduction flags are shown in `reproduce_brain.sh`) |
| Deterministic baseline / paper figure | `experiments/fd_composite_fixed.py` | solves the variable-coefficient complex Helmholtz problem and inserts the FDM column into the paper figure |
| Auxiliary FD analyses | `experiments/helmholtz_brain_fd_baseline.py`, `fd_brain_resolution_sweep.py`, `fd_resolution_figure.py`, `fd_panels_3mm.py`, `helmholtz_brain_fd_figure.py` | legacy one-off baseline and resolution analyses; their `ROOT` / `OUT` paths must be configured before use |
| Results output directory | `experiments/helmholtz_brain_outputs/` | `reproduce_brain.sh` writes its logs and figures here |

## Data (brain, external)

The synthetic experiments need no data. The **brain MRE data is not included** in this
repository and must be obtained from the source:

> **Brain Biomechanics Imaging Resources (BBIR)** — <https://www.nitrc.org/projects/bbir>
> (NITRC; Johns Hopkins University & Washington University, NIH-NINDS U01 NS112120)

Download the **[Year 4 `U01_UDEL_v4d` release](https://www.nitrc.org/frs/?group_id=1390&release_id=4867)**
used by this code (a free NITRC account may be required) and place subjects 0001--0003 so the
layout matches what the driver reads:

```
data/brain_experiment_data/mre_udel/
└── <SUBJECT>_v4/                              # <SUBJECT> ∈ U01_UDEL_0001_01, 0002_01, 0003_01
    ├── <SUBJECT>_register_to_MRE/
    │   └── <SUBJECT>_MREreg_brainmask.nii     # brain mask
    └── <SUBJECT>_MRE_AP_<F>Hz/                # <F> ∈ 30, 50, 70
        ├── <SUBJECT>_MRE_AP_<F>Hz_props_shear_real.nii   # storage modulus G'
        ├── <SUBJECT>_MRE_AP_<F>Hz_props_shear_imag.nii   # loss modulus G''
        └── …                                             # displacement fields, etc.
```

See `U01_NITRC_UDEL_Description_v03a.pdf` in the same Year 4 release for the full field list.
If using the newer `v5a` release, adjust the directory mapping because the driver expects `_v4`
paths.
Use of the data is subject to NITRC's / the providers' terms.

## Reproduce

Both scripts run inside the project's Docker container (shared install in the root README)
and self-locate to the repo root, so they work from any directory.

**Whole paper:**

```bash
./docker-run.sh shell          # drop into /app, then:
bash experiments/reproduce_paper.sh
```

This executes the synthetic GP-PDE and baseline notebooks in place, then delegates
the brain results to `reproduce_brain.sh`. Open any single synthetic notebook
interactively with `./docker-run.sh jupyter`.

**Just the brain:**

```bash
bash experiments/reproduce_brain.sh
```

This runs the 27-case sweep underlying the seven-row Table 1, Table 2 (prior ablation at the
anchor case), the loss-tangent calculation, the GP reconstructions, and the final FDM--GP paper
figure. Rows are read from `experiments/helmholtz_brain_outputs/sweep_log.txt` and
`ablation_*.log`. The auxiliary FD resolution analyses are not invoked.

> **Note.** The archived SNR scatter is not reproduced because no upstream SNR generator exists.
> The brain scripts use tissue density ρ = 1040 kg/m³; regenerate any older cached metrics
> before quoting them (the loss tangent is density-independent).

## Citation

```bibtex
@misc{deng2026operatorinformedgaussianprocessescomplex,
      title={Operator-Informed Gaussian Processes for Complex Helmholtz Wavefields: From Synthetic Benchmarks to In Vivo Brain Elastography}, 
      author={Boyuan Deng and Kshitiz Upadhyay and Michael Shields},
      year={2026},
      eprint={2607.14193},
      archivePrefix={arXiv},
      primaryClass={stat.ML},
      url={https://arxiv.org/abs/2607.14193}, 
}
```
