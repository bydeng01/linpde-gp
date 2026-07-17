"""Helmholtz forward BVP on MRE brain data.

This script applies the variable-coefficient Helmholtz pipeline to a real
human-brain MRE dataset:

  k^2(x) = rho * omega^2 / (G'(x) + i G''(x))         (from NLI inversion)

The forward problem is

  (Delta + k^2(x)) u_c(x) = 0     on the brain interior
                  u_c(x)  = u_c^meas(x)  on the brain surface

solved component-by-component (c in {x, y, z}) on the bounding box of the
brain mask in MRE space.

What this script delivers
-------------------------
1. **Empirical model-mismatch diagnostic** (always computed; does not require
   GP-PDE regression).

   For the measured displacement field this is
       r_c(x) = (Delta u_c^meas)(x) + k^2(x) * u_c^meas(x)
   computed via a 7-point finite-difference Laplacian, and we report
   |r| / |k^2 * u| (matching ``data/.../scripts/helmholtz_residual_demo.py``).
   The magnitude of |r|/|k^2 u| measures *where* the simplifying assumption
   (locally homogeneous shear modulus, near-incompressibility) breaks down on
   this data.

2. **GP forward solve** (Re and Im separately), using Matern 5/2 in 3D
   via the closed-form Laplacian-Laplacian handlers in the variable-
   coefficient Helmholtz extension. We solve two scalar BVPs:

   * one for u_c^Re with operator L_R := Delta + Re(k^2), data Re(u^meas),
   * one for u_c^Im with operator L_R                  , data Im(u^meas),

   and recombine. The off-diagonal coupling Im(k^2) (the dissipation) is
   ignored in the operator; this is a controlled approximation that holds
   when |Im k^2| << |Re k^2|, which is roughly true for in-vivo brain tissue
   at MRE frequencies (loss tangent ~0.2-0.4). A fully coupled real-2
   representation would use ``HelmholtzReal2Operator.from_coefficient_field``;
   wiring up the 2-component GP is left as future work.

3. **Two figure types per spec**:

   * ``helmholtz_brain_<subject>_<f>Hz_comp<c>_main.png``: 3x3 grid
     (rows: Re/Im/|.|; cols: measured, predicted, posterior std).
   * ``helmholtz_brain_<subject>_<f>Hz_comp<c>_mismatch.png``: 1x3
     (posterior std, |r|/|k^2 u|, scatter of std vs |r|/|k^2 u|).

Usage
-----
    python experiments/helmholtz_brain_forward_bvp.py
    python experiments/helmholtz_brain_forward_bvp.py --sweep
"""

from __future__ import annotations

import argparse
import itertools
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplcache")

import matplotlib

matplotlib.use("Agg")
from jax import numpy as jnp
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import probnum as pn
from scipy.ndimage import binary_erosion

import linpde_gp
from linpde_gp.functions import JaxLambdaFunction
from linpde_gp.linfuncops.diffops import HelmholtzOperator, HelmholtzReal2Operator
from linpde_gp.randprocs.covfuncs import (
    CoregionalizedMultiOutputCovarianceFunction,
    IndependentMultiOutputCovarianceFunction,
)

RHO = 1040.0  # kg/m^3 (brain tissue; was 1000.0)
COMPONENT_NAME = {0: "x", 1: "y", 2: "z"}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_subject_frequency(
    root: Path, subject: str, freq_hz: int, observable: str = "disp"
):
    """Load the vector field selected by ``observable`` and the per-voxel k^2 inputs.

    ``observable`` selects which complex vector field is returned as ``data["u"]``:

    * ``"disp"``  — the measured displacement ``u_meas`` (microns), the legacy
      observable used in Phases 6-7. ``u = disp_re + i disp_im``.
    * ``"curl"``  — the curl-filtered field ``q = (∇ × u_meas)``, also of shape
      ``(X,Y,Z,3)``. Under the locally-homogeneous shear-modulus assumption the
      curl satisfies the *same* scalar Helmholtz equation as the displacement,
      ``(Δ + k²(x)) q_c = 0`` per component, with the same NLI-derived ``k²``
      we already build below. Curl-based inversion (DI/MDEV/LFE in classical
      MRE) removes the irrotational compressional part and any constant /
      rigid-body bias, at the cost of one additional spatial derivative on the
      raw displacement.

    The returned dict keeps the field under ``data["u"]`` regardless of choice
    so all downstream solver code is observable-agnostic; the chosen observable
    is recorded under ``data["observable"]``.
    """
    if observable not in ("disp", "curl"):
        raise ValueError(f"observable must be 'disp' or 'curl', got {observable!r}")

    freq_dir = root / f"{subject}_v4" / f"{subject}_MRE_AP_{freq_hz}Hz"
    reg_dir = root / f"{subject}_v4" / f"{subject}_register_to_MRE"

    stem = f"{subject}_MRE_AP_{freq_hz}Hz"
    field_re = np.asanyarray(
        nib.load(str(freq_dir / f"{stem}_{observable}_re.nii")).dataobj
    ).astype(np.float64)
    field_im = np.asanyarray(
        nib.load(str(freq_dir / f"{stem}_{observable}_im.nii")).dataobj
    ).astype(np.float64)
    G_re = np.asanyarray(
        nib.load(str(freq_dir / f"{stem}_props_shear_real.nii")).dataobj
    ).astype(np.float64)
    G_im = np.asanyarray(
        nib.load(str(freq_dir / f"{stem}_props_shear_imag.nii")).dataobj
    ).astype(np.float64)
    mask_img = nib.load(str(reg_dir / f"{subject}_MREreg_brainmask.nii"))
    mask = np.asanyarray(mask_img.dataobj).astype(bool)

    zooms = mask_img.header.get_zooms()[:3]  # in mm

    return {
        "u": field_re + 1j * field_im,  # microns, shape (X,Y,Z,3)
        "G": G_re + 1j * G_im,  # Pa, shape (X,Y,Z)
        "mask": mask,  # bool, shape (X,Y,Z)
        "zooms_mm": tuple(float(z) for z in zooms),
        "observable": observable,
    }


# ---------------------------------------------------------------------------
# Finite-difference Laplacian
# ---------------------------------------------------------------------------


def laplacian_3d(field: np.ndarray, dx: float, dy: float, dz: float) -> np.ndarray:
    """7-point stencil Laplacian with periodic-roll boundaries.

    The caller masks edges away from the boundary so the periodic-wrap
    artifact does not contaminate the diagnostic.
    """
    out = np.zeros_like(field)
    out += (np.roll(field, -1, axis=0) - 2 * field + np.roll(field, 1, axis=0)) / dx**2
    out += (np.roll(field, -1, axis=1) - 2 * field + np.roll(field, 1, axis=1)) / dy**2
    out += (np.roll(field, -1, axis=2) - 2 * field + np.roll(field, 1, axis=2)) / dz**2
    return out


def empirical_residual(
    u_meas: np.ndarray,  # complex, shape (X,Y,Z,3)
    k2: np.ndarray,  # complex, shape (X,Y,Z), NaN outside mask
    mask: np.ndarray,  # bool, shape (X,Y,Z)
    spacing_mm: tuple,
):
    """Return per-voxel residual r and forcing magnitude |k^2 u|.

    The spec asks for the relative diagnostic |r| / |k^2 u| (matching
    ``data/.../scripts/helmholtz_residual_demo.py``). The previous version of
    this script normalized by |u| instead; that disagrees with the demo by
    ~5 orders of magnitude when k^2 is in 1/m^2.
    """
    dx, dy, dz = (s * 1e-3 for s in spacing_mm)
    interior = mask.copy()
    for axis in range(3):
        interior &= np.roll(mask, 1, axis=axis) & np.roll(mask, -1, axis=axis)

    f_emp = np.empty_like(u_meas)
    k2u = np.empty_like(u_meas)
    for c in range(u_meas.shape[-1]):
        lap = laplacian_3d(u_meas[..., c], dx, dy, dz)
        k2u[..., c] = k2 * u_meas[..., c]
        f_emp[..., c] = lap + k2u[..., c]
    return f_emp, k2u, interior


# ---------------------------------------------------------------------------
# Variable-coefficient Helmholtz operator from the volumetric k^2 field
# ---------------------------------------------------------------------------


def build_real_k2_field_from_grid(
    k2_volume: np.ndarray,  # real-valued field (Re or Im of full k^2)
    origin_mm: np.ndarray,
    spacing_mm: np.ndarray,
    fill_value: float = 0.0,
) -> pn.functions.Function:
    """Wrap a 3D voxelized REAL k^2 volume as a continuous pn.functions.Function.

    Nearest-voxel lookup with a configurable fill value outside the brain mask.
    Coordinates are in METERS (consistent with the Laplacian dimensions in
    1/m^2).
    """
    spacing_m = spacing_mm * 1e-3
    origin_m = origin_mm * 1e-3
    nx, ny, nz = k2_volume.shape

    k2_np = np.where(np.isfinite(k2_volume), k2_volume, fill_value).astype(np.float64)
    k2_jax = jnp.asarray(k2_np.reshape(-1))

    def _lookup(x):
        ix = jnp.clip(
            ((x[0] - origin_m[0]) / spacing_m[0]).astype(jnp.int32), 0, nx - 1
        )
        iy = jnp.clip(
            ((x[1] - origin_m[1]) / spacing_m[1]).astype(jnp.int32), 0, ny - 1
        )
        iz = jnp.clip(
            ((x[2] - origin_m[2]) / spacing_m[2]).astype(jnp.int32), 0, nz - 1
        )
        idx = ix * (ny * nz) + iy * nz + iz
        return k2_jax[idx]

    return JaxLambdaFunction(_lookup, input_shape=(3,), output_shape=(), vectorize=True)


def build_complex_k2_field_from_grid(
    k2_volume: np.ndarray,  # complex-valued field
    origin_mm: np.ndarray,
    spacing_mm: np.ndarray,
    fill_value: complex = 0.0,
) -> pn.functions.Function:
    """Wrap a complex-valued 3D k^2 volume as a continuous pn.functions.Function.

    Used by the Phase-7 two-component pipeline: ``HelmholtzReal2Operator``
    consumes the *full* complex k^2(x), splitting it into alpha = Re k^2 and
    beta = Im k^2 internally.

    Nearest-voxel lookup; coordinates in METERS (matching the Laplacian
    dimensions in 1/m^2).
    """
    spacing_m = spacing_mm * 1e-3
    origin_m = origin_mm * 1e-3
    nx, ny, nz = k2_volume.shape

    k2_np = np.where(np.isfinite(k2_volume), k2_volume, fill_value).astype(
        np.complex128
    )
    k2_jax = jnp.asarray(k2_np.reshape(-1))

    def _lookup(x):
        ix = jnp.clip(
            ((x[0] - origin_m[0]) / spacing_m[0]).astype(jnp.int32), 0, nx - 1
        )
        iy = jnp.clip(
            ((x[1] - origin_m[1]) / spacing_m[1]).astype(jnp.int32), 0, ny - 1
        )
        iz = jnp.clip(
            ((x[2] - origin_m[2]) / spacing_m[2]).astype(jnp.int32), 0, nz - 1
        )
        idx = ix * (ny * nz) + iy * nz + iz
        return k2_jax[idx]

    return JaxLambdaFunction(_lookup, input_shape=(3,), output_shape=(), vectorize=True)


# ---------------------------------------------------------------------------
# Forward GP-PDE solve
# ---------------------------------------------------------------------------


def _build_index_sets(mask, n_shell, n_interior, rng):
    shell = mask & ~binary_erosion(mask, iterations=2)
    interior = binary_erosion(mask, iterations=3)

    shell_idx = np.argwhere(shell)
    interior_idx = np.argwhere(interior)

    if shell_idx.shape[0] > n_shell:
        sub = rng.choice(shell_idx.shape[0], size=n_shell, replace=False)
        shell_idx = shell_idx[sub]
    if interior_idx.shape[0] > n_interior:
        sub = rng.choice(interior_idx.shape[0], size=n_interior, replace=False)
        interior_idx = interior_idx[sub]
    return shell_idx, interior_idx


def _solve_real_bvp(
    *,
    helmholtz_op_real: HelmholtzOperator,
    cov_base,
    output_scale: float,
    X_bc: np.ndarray,
    Y_bc: np.ndarray,  # real-valued shell observations
    X_pde: np.ndarray,
    Y_pde: np.ndarray,  # real-valued PDE forcing (typically zeros)
    bc_noise: float,
    pde_noise_rel: float,  # PDE noise as a fraction of trace(K)/N
    cov_jitter_rel: float,  # jitter for PD-ness of conditional Gram
):
    """Run a single scalar BVP and return (post, info)."""
    prior = pn.randprocs.GaussianProcess(
        mean=linpde_gp.functions.Zero(input_shape=(3,)),
        cov=output_scale**2 * cov_base,
    )

    # BC observations
    bc_post = prior.condition_on_observations(
        Y_bc,
        X=X_bc,
        b=pn.randvars.Normal(
            np.zeros(X_bc.shape[0]),
            bc_noise * np.eye(X_bc.shape[0]),
        ),
    )

    # PDE observations (forcing = 0 for homogeneous Helmholtz BVP)
    # Set PDE noise scale from the BVP-conditioned prior trace at X_pde so
    # that pde_noise_rel is dimensionally meaningful. The jitter is added on
    # top to keep the conditional Gram PD.
    bvp_var = np.asarray(bc_post.var(X_pde)).flatten()
    # NOTE: the previous version of this line was
    #   trace_avg = max(float(np.mean(bvp_var)), 1.0)
    # which silently clamps pde_noise to pde_noise_rel * 1.0 whenever the
    # BC-conditioned variance is small. That is wrong when output_scale is
    # tuned away from 1.0 (e.g. for the curl observable, ~2e-4): the floor
    # makes pde_noise enormous relative to the prior variance, which kills
    # the PDE constraint. We use the actual mean variance and rely on
    # cov_jitter_rel for PD safety.
    trace_avg = float(np.mean(bvp_var))
    pde_noise = pde_noise_rel * trace_avg + cov_jitter_rel * trace_avg
    post = bc_post.condition_on_observations(
        Y_pde,
        X=X_pde,
        L=helmholtz_op_real,
        b=pn.randvars.Normal(
            np.zeros(X_pde.shape[0]),
            pde_noise * np.eye(X_pde.shape[0]),
        ),
    )
    return post, {"trace_avg": trace_avg, "pde_noise": pde_noise}


def gp_forward_solve(
    *,
    u_meas: np.ndarray,
    k2_volume: np.ndarray,  # complex, shape (X,Y,Z)
    mask: np.ndarray,
    spacing_mm: tuple,
    component: int,
    n_shell: int,
    n_interior: int,
    kernel: str = "matern",
    matern_nu: float = 2.5,
    lengthscale_mm: float = 20.0,
    output_scale: float = 1.0,
    bc_noise: float = 1e-6,
    pde_noise_rel: float = 1e-4,
    cov_jitter_rel: float = 1e-8,
    seed: int = 0,
):
    """Solve real and imaginary BVPs separately for one component.

    Returns a dict with mean_complex, std_complex (best-effort), index sets,
    timing.
    """
    rng = np.random.default_rng(seed)
    spacing_m = np.array(spacing_mm) * 1e-3

    shell_idx, interior_idx = _build_index_sets(mask, n_shell, n_interior, rng)
    print(f"  shell idx     : {shell_idx.shape[0]}")
    print(f"  interior idx  : {interior_idx.shape[0]}")

    def voxel_to_meters(idx_arr):
        return idx_arr.astype(np.float64) * spacing_m

    X_bc = voxel_to_meters(shell_idx)
    X_pde = voxel_to_meters(interior_idx)

    Y_bc_complex = u_meas[shell_idx[:, 0], shell_idx[:, 1], shell_idx[:, 2], component]

    # --- Build operator with the REAL part of k^2 (controlled approximation;
    #     see module docstring). Im k^2 is ignored in the operator; we only
    #     use it in the data-side residual diagnostic.
    k2_real = build_real_k2_field_from_grid(
        k2_volume.real,
        origin_mm=np.zeros(3),
        spacing_mm=np.array(spacing_mm),
        fill_value=0.0,
    )
    helmholtz_op_real = HelmholtzOperator.from_coefficient_field(
        domain_shape=(3,), k_squared_field=k2_real
    )

    # --- Kernel ---
    if kernel == "matern":
        cov_base = linpde_gp.randprocs.covfuncs.Matern(
            (3,), nu=matern_nu, lengthscales=lengthscale_mm * 1e-3
        )
    elif kernel == "expquad":
        cov_base = linpde_gp.randprocs.covfuncs.ExpQuad(
            (3,), lengthscales=lengthscale_mm * 1e-3
        )
    else:
        raise ValueError(f"Unknown kernel: {kernel}")

    Y_pde_zero = np.zeros(X_pde.shape[0], dtype=np.float64)

    # --- Solve Re ---
    t0 = time.time()
    try:
        post_R, info_R = _solve_real_bvp(
            helmholtz_op_real=helmholtz_op_real,
            cov_base=cov_base,
            output_scale=output_scale,
            X_bc=X_bc,
            Y_bc=Y_bc_complex.real,
            X_pde=X_pde,
            Y_pde=Y_pde_zero,
            bc_noise=bc_noise,
            pde_noise_rel=pde_noise_rel,
            cov_jitter_rel=cov_jitter_rel,
        )
        t_R = time.time() - t0
        print(f"  Re BVP OK in {t_R:.1f} s, pde_noise={info_R['pde_noise']:.3e}")
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        print(f"  Re BVP FAILED: {type(exc).__name__}: {exc}")
        return {
            "status": "failed",
            "phase": "real",
            "error": str(exc),
            "shell_idx": shell_idx,
            "interior_idx": interior_idx,
        }

    # --- Solve Im ---
    t1 = time.time()
    try:
        post_I, info_I = _solve_real_bvp(
            helmholtz_op_real=helmholtz_op_real,
            cov_base=cov_base,
            output_scale=output_scale,
            X_bc=X_bc,
            Y_bc=Y_bc_complex.imag,
            X_pde=X_pde,
            Y_pde=Y_pde_zero,
            bc_noise=bc_noise,
            pde_noise_rel=pde_noise_rel,
            cov_jitter_rel=cov_jitter_rel,
        )
        t_I = time.time() - t1
        print(f"  Im BVP OK in {t_I:.1f} s, pde_noise={info_I['pde_noise']:.3e}")
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        print(f"  Im BVP FAILED: {type(exc).__name__}: {exc}")
        return {
            "status": "failed",
            "phase": "imag",
            "error": str(exc),
            "shell_idx": shell_idx,
            "interior_idx": interior_idx,
        }

    return {
        "status": "ok",
        "mode": "real_only",
        "post_R": post_R,
        "post_I": post_I,
        "shell_idx": shell_idx,
        "interior_idx": interior_idx,
        "elapsed": time.time() - t0,
        "info_R": info_R,
        "info_I": info_I,
    }


# ---------------------------------------------------------------------------
# Phase-7: 2-component forward solve against the FULL complex Helmholtz block
# ---------------------------------------------------------------------------


def _build_real2_multi_cov(
    *,
    prior_kind: str,
    domain_shape,
    kernel: str,
    matern_nu: float,
    lengthscale_mm: float,
    output_scale: float,
    coreg_corr,
    lmc_lengthscales_mm,
):
    """Build the 2-component prior covariance for the real2 Helmholtz GP.

    Phase 8d. ``prior_kind`` selects the cross-channel (Re/Im) prior structure:

    * ``"iid"`` — ``output_scale**2 * IndependentMultiOutput(base, base)``.
      Reproduces the pre-Phase-8 path bit-for-bit (the default).
    * ``"icm"`` — ``CoregionalizedMultiOutput(base, B)`` with
      ``B = output_scale**2 * [[1, rho], [rho, 1]]``. ``output_scale**2`` is
      folded into ``B`` (per the spec, to avoid double-counting scale), so the
      marginal prior variance matches the IID path. ``rho`` defaults to 0,
      which makes ICM reduce *exactly* to IID (a built-in sanity check).
    * ``"lmc"`` — ``sum_q CoregionalizedMultiOutput(base_q, B_q)`` over base
      kernels with distinct lengthscales (``--lmc-lengthscales``), each sharing
      correlation ``rho``; ``output_scale**2`` is split evenly across the Q
      terms so the marginal variance again matches the IID/ICM scale.
    """

    def _base(ls_mm):
        if kernel == "matern":
            return linpde_gp.randprocs.covfuncs.Matern(
                domain_shape, nu=matern_nu, lengthscales=ls_mm * 1e-3
            )
        if kernel == "expquad":
            return linpde_gp.randprocs.covfuncs.ExpQuad(
                domain_shape, lengthscales=ls_mm * 1e-3
            )
        raise ValueError(f"Unknown kernel: {kernel}")

    if prior_kind == "iid":
        base = _base(lengthscale_mm)
        return output_scale**2 * IndependentMultiOutputCovarianceFunction(base, base)

    rho = 0.0 if coreg_corr is None else float(coreg_corr)
    if not -1.0 < rho < 1.0:
        raise ValueError(f"--coreg-corr must lie in (-1, 1) for a PSD B; got {rho}")
    corr = np.array([[1.0, rho], [rho, 1.0]])

    if prior_kind == "icm":
        base = _base(lengthscale_mm)
        return CoregionalizedMultiOutputCovarianceFunction(base, output_scale**2 * corr)

    if prior_kind == "lmc":
        if lmc_lengthscales_mm:
            ls_list = list(lmc_lengthscales_mm)
        else:
            # Default Q=2: one short, one long lengthscale around the base.
            ls_list = [lengthscale_mm * 0.5, lengthscale_mm * 2.0]
        n_terms = len(ls_list)
        B_q = (output_scale**2 / n_terms) * corr
        terms = [
            CoregionalizedMultiOutputCovarianceFunction(_base(ls), B_q)
            for ls in ls_list
        ]
        cov = terms[0]
        for term in terms[1:]:
            cov = cov + term  # JaxSumCovarianceFunction of ICM terms (LMC)
        return cov

    raise ValueError(f"Unknown prior_kind: {prior_kind!r}")


def _solve_real2_bvp(
    *,
    helmholtz_op_real2: HelmholtzReal2Operator,
    cov_base,
    output_scale: float,
    X_bc: np.ndarray,
    Y_bc_complex: np.ndarray,  # complex shell observations
    X_pde: np.ndarray,
    bc_noise: float,
    pde_noise_rel: float,
    cov_jitter_rel: float,
    multi_cov=None,
):
    """Condition a 2-component GP on Dirichlet BC + homogeneous Helmholtz PDE.

    The prior is

        u ~ GP(0, k(x, x') . I_2)

    with ``k = output_scale**2 * cov_base`` a scalar Matern (built outside).
    The operator is the full real2 block

        H = [ Delta + alpha   -beta;
                beta           Delta + alpha ]

    where alpha = Re k^2 and beta = Im k^2. ``Y_bc`` is provided as a complex
    1D array and unpacked into a real (N_bc, 2) layout. PDE forcing is zero
    in the homogeneous Helmholtz model.
    """
    # Default (multi_cov is None) reproduces the pre-Phase-8 IID prior exactly.
    # Phase 8d injects an ICM/LMC prior via ``multi_cov`` (output_scale**2 is
    # already folded into that object by ``_build_real2_multi_cov``).
    if multi_cov is None:
        multi_cov = output_scale**2 * IndependentMultiOutputCovarianceFunction(
            cov_base, cov_base
        )
    prior = pn.randprocs.GaussianProcess(
        mean=linpde_gp.functions.Zero(input_shape=(3,), output_shape=(2,)),
        cov=multi_cov,
    )

    # BC observations: stack (Re, Im) into the trailing axis (batch-last layout
    # accepted by _preprocess_observations, which transposes internally).
    Y_bc = np.stack([Y_bc_complex.real, Y_bc_complex.imag], axis=-1)
    n_bc = X_bc.shape[0]
    bc_b = pn.randvars.Normal(
        np.zeros((2, n_bc)),
        bc_noise * np.eye(2 * n_bc),
    )
    bc_post = prior.condition_on_observations(Y_bc, X=X_bc, b=bc_b)

    # Scale PDE noise to the BVP-conditioned variance so pde_noise_rel is
    # dimensionally meaningful, mirroring the scalar path.
    bvp_var = np.asarray(bc_post.var(X_pde))  # shape (N_pde, 2)
    # See _solve_real_bvp for the rationale: the previous `max(..., 1.0)`
    # floor silently kills the PDE constraint when output_scale is small,
    # because it pegs trace_avg above the actual post variance by orders of
    # magnitude. Drop the floor; rely on cov_jitter_rel for PD safety.
    trace_avg = float(np.mean(bvp_var))
    pde_noise = pde_noise_rel * trace_avg + cov_jitter_rel * trace_avg

    n_pde = X_pde.shape[0]
    Y_pde = np.zeros((n_pde, 2), dtype=np.float64)
    pde_b = pn.randvars.Normal(
        np.zeros((2, n_pde)),
        pde_noise * np.eye(2 * n_pde),
    )
    post = bc_post.condition_on_observations(
        Y_pde, X=X_pde, L=helmholtz_op_real2, b=pde_b
    )

    return post, {"trace_avg": trace_avg, "pde_noise": pde_noise}


def gp_forward_solve_real2(
    *,
    u_meas: np.ndarray,
    k2_volume_complex: np.ndarray,
    mask: np.ndarray,
    spacing_mm: tuple,
    component: int,
    n_shell: int,
    n_interior: int,
    kernel: str = "matern",
    matern_nu: float = 2.5,
    lengthscale_mm: float = 20.0,
    output_scale: float = 1.0,
    bc_noise: float = 1e-6,
    pde_noise_rel: float = 1e-4,
    cov_jitter_rel: float = 1e-8,
    seed: int = 0,
    prior_kind: str = "iid",
    coreg_corr=None,
    lmc_lengthscales_mm=None,
):
    """Forward solve using the full complex Helmholtz block and a 2-component GP.

    Returns a dict with ``post``, ``shell_idx``, ``interior_idx``, ``elapsed``,
    and ``info``. ``post.mean(X_eval)`` has shape ``(N, 2)`` with the Re/Im
    components stacked on the trailing axis.
    """
    rng = np.random.default_rng(seed)
    spacing_m = np.array(spacing_mm) * 1e-3

    shell_idx, interior_idx = _build_index_sets(mask, n_shell, n_interior, rng)
    print(f"  shell idx     : {shell_idx.shape[0]}")
    print(f"  interior idx  : {interior_idx.shape[0]}")

    def voxel_to_meters(idx_arr):
        return idx_arr.astype(np.float64) * spacing_m

    X_bc = voxel_to_meters(shell_idx)
    X_pde = voxel_to_meters(interior_idx)

    Y_bc_complex = u_meas[shell_idx[:, 0], shell_idx[:, 1], shell_idx[:, 2], component]

    # Operator with the FULL complex k^2 (Phase-7c innovation).
    k2_field_complex = build_complex_k2_field_from_grid(
        k2_volume_complex,
        origin_mm=np.zeros(3),
        spacing_mm=np.array(spacing_mm),
        fill_value=0.0 + 0.0j,
    )
    op_real2 = HelmholtzReal2Operator.from_coefficient_field(
        domain_shape=(3,), k_squared_field=k2_field_complex
    )

    if kernel == "matern":
        cov_base = linpde_gp.randprocs.covfuncs.Matern(
            (3,), nu=matern_nu, lengthscales=lengthscale_mm * 1e-3
        )
    elif kernel == "expquad":
        cov_base = linpde_gp.randprocs.covfuncs.ExpQuad(
            (3,), lengthscales=lengthscale_mm * 1e-3
        )
    else:
        raise ValueError(f"Unknown kernel: {kernel}")

    # Phase 8d: assemble the cross-channel (Re/Im) prior. Default "iid" gives
    # output_scale**2 * IndependentMultiOutput(base, base), bit-for-bit the
    # pre-Phase-8 behaviour; "icm"/"lmc" couple the channels.
    multi_cov = _build_real2_multi_cov(
        prior_kind=prior_kind,
        domain_shape=(3,),
        kernel=kernel,
        matern_nu=matern_nu,
        lengthscale_mm=lengthscale_mm,
        output_scale=output_scale,
        coreg_corr=coreg_corr,
        lmc_lengthscales_mm=lmc_lengthscales_mm,
    )
    print(
        f"  prior         : {prior_kind}"
        + (
            f" (rho={float(coreg_corr):+.2f})"
            if coreg_corr is not None and prior_kind in ("icm", "lmc")
            else ""
        )
    )

    t0 = time.time()
    try:
        post, info = _solve_real2_bvp(
            helmholtz_op_real2=op_real2,
            cov_base=cov_base,
            output_scale=output_scale,
            X_bc=X_bc,
            Y_bc_complex=Y_bc_complex,
            X_pde=X_pde,
            bc_noise=bc_noise,
            pde_noise_rel=pde_noise_rel,
            cov_jitter_rel=cov_jitter_rel,
            multi_cov=multi_cov,
        )
        t_solve = time.time() - t0
        print(f"  Real2 BVP OK in {t_solve:.1f} s, pde_noise={info['pde_noise']:.3e}")
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        print(f"  Real2 BVP FAILED: {type(exc).__name__}: {exc}")
        return {
            "status": "failed",
            "phase": "real2",
            "error": str(exc),
            "shell_idx": shell_idx,
            "interior_idx": interior_idx,
        }

    return {
        "status": "ok",
        "mode": "real2",
        "post": post,
        "shell_idx": shell_idx,
        "interior_idx": interior_idx,
        "elapsed": time.time() - t0,
        "info": info,
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _take_axial(field, z):
    return np.rot90(np.take(field, z, axis=2))


def _dense_slice_predictions(
    res: dict,
    mask: np.ndarray,
    spacing_m: np.ndarray,
    z_slice: int,
    std_chunk: int = 1500,
    std_subsample: int | None = None,
    seed: int = 0,
):
    """Evaluate the trained GP posterior densely on every brain-interior voxel
    of the displayed axial slice.

    Returns ``(mean_R_2d, mean_I_2d, std_R_2d, std_I_2d)``, each a 2D array the
    same shape as ``np.take(mask, z_slice, axis=2)`` (i.e., (X, Y)) with NaN
    outside the brain mask. The figure code applies ``np.rot90`` later to
    match the orientation of the existing measured-field panels. Both the Re
    and Im posterior s.d. are returned because the real2 posterior ``.std()``
    yields both channels in a single call (the Im channel is therefore free),
    and the two channels are not guaranteed identical once the real2 Helmholtz
    operator couples them.

    Mean is a single batched call (cheap, K_*X @ alpha with precomputed
    alpha) and is always evaluated. Std is the expensive piece — exact
    chunked back-substitutions cost ~70 s for M ~ 8 k voxels at N ~ 2 k
    training points.

    ``std_subsample``
        * ``None`` (default) — exact std at every in-mask slice voxel via
          chunks of ``std_chunk`` queries. ~70 s for an 8 k-voxel slice.
        * ``int`` — exact std at this many randomly chosen in-mask voxels,
          then linearly interpolated (scipy.griddata) to the full slice.
          The std field is smooth on the kernel lengthscale (~15 mm) within
          a slice extent of ~100 mm, so ~400 exact evals reconstruct the
          full slice to well under the colormap quantization. Cost ~3-5 s.
    """
    slice_mask = np.take(mask, z_slice, axis=2)  # (X, Y)
    slice_idx = np.argwhere(slice_mask)  # (M, 2) of (x, y) indices
    M = slice_idx.shape[0]

    nan2d = np.full(slice_mask.shape, np.nan, dtype=np.float64)
    if M == 0:
        return nan2d.copy(), nan2d.copy(), nan2d.copy(), nan2d.copy()

    # Build (M, 3) physical coordinates for the dense in-mask voxels.
    full_idx = np.column_stack([slice_idx, np.full(M, z_slice, dtype=slice_idx.dtype)])
    X_query = full_idx.astype(np.float64) * spacing_m

    # ---- mean (always full M) ----
    if res["mode"] == "real_only":
        post_R = res["post_R"]
        post_I = res["post_I"]
        mean_R_flat = np.asarray(post_R.mean(X_query)).flatten()
        mean_I_flat = np.asarray(post_I.mean(X_query)).flatten()
    elif res["mode"] == "real2":
        post = res["post"]
        mean_vec = np.asarray(post.mean(X_query))  # (M, 2)
        mean_R_flat = mean_vec[..., 0].flatten()
        mean_I_flat = mean_vec[..., 1].flatten()
    else:
        raise RuntimeError(f"Unknown solve mode: {res.get('mode')}")

    # ---- std (both Re and Im channels, shape (M', 2)) ----
    def _std_at(X):
        """Return per-channel posterior std (columns [Re, Im]) at the M' rows
        of X. On the real2 path ``post.std`` already returns both channels in
        one call, so the Im channel is free. On the real_only path Re and Im
        are separate scalar BVPs against the same operator and identical point
        sets, so we query both posteriors explicitly (their variance happens to
        coincide, but we do not hard-code that)."""
        if res["mode"] == "real_only":
            s_R = np.asarray(res["post_R"].std(X)).flatten()
            s_I = np.asarray(res["post_I"].std(X)).flatten()
            return np.column_stack([s_R, s_I])
        return np.asarray(res["post"].std(X)).reshape(-1, 2)

    try:
        if std_subsample is not None and std_subsample < M:
            # Cheap path: exact std on a random subsample, interpolate the
            # rest. The std field is smooth on the kernel lengthscale.
            from scipy.interpolate import griddata as _griddata

            rng_local = np.random.default_rng(seed)
            sub_sel = rng_local.choice(M, size=std_subsample, replace=False)
            std_sub = _std_at(X_query[sub_sel])  # (S, 2)
            std_flat = np.empty((M, 2), dtype=np.float64)
            for ch in range(2):
                col = _griddata(
                    slice_idx[sub_sel].astype(np.float64),
                    std_sub[:, ch],
                    slice_idx.astype(np.float64),
                    method="linear",
                    fill_value=float(np.nanmedian(std_sub[:, ch])),
                )
                # griddata("linear") returns NaN at strict-extrapolation
                # points; patch those with nearest-neighbor to keep the panel
                # filled.
                nan_pts = ~np.isfinite(col)
                if nan_pts.any():
                    col[nan_pts] = _griddata(
                        slice_idx[sub_sel].astype(np.float64),
                        std_sub[:, ch],
                        slice_idx[nan_pts].astype(np.float64),
                        method="nearest",
                    )
                std_flat[:, ch] = col
        else:
            # Exact path: full chunked back-substitution.
            std_flat = np.empty((M, 2), dtype=np.float64)
            for s in range(0, M, std_chunk):
                e = min(M, s + std_chunk)
                std_flat[s:e] = _std_at(X_query[s:e])
    except np.linalg.LinAlgError as exc:
        print(f"  dense std unavailable: {exc}")
        std_flat = np.full((M, 2), np.nan)

    mean_R_2d = nan2d.copy()
    mean_I_2d = nan2d.copy()
    std_R_2d = nan2d.copy()
    std_I_2d = nan2d.copy()
    mean_R_2d[slice_mask] = mean_R_flat
    mean_I_2d[slice_mask] = mean_I_flat
    std_R_2d[slice_mask] = std_flat[:, 0]
    std_I_2d[slice_mask] = std_flat[:, 1]
    return mean_R_2d, mean_I_2d, std_R_2d, std_I_2d


def _add_scale_bar(ax, n_cols, mm_per_col, bar_mm=20.0):
    """Draw a horizontal scale bar in the lower-left of an imshow panel.

    ``n_cols`` is the displayed array width (columns) and ``mm_per_col`` is the
    in-plane voxel size along that displayed axis (mm). After ``_take_axial``'s
    rot90 the displayed horizontal axis corresponds to data axis 0, so the
    caller passes ``zooms_mm[0]``.
    """
    if mm_per_col is None or not np.isfinite(mm_per_col) or mm_per_col <= 0:
        return
    n_pix = bar_mm / mm_per_col
    ylim = ax.get_ylim()  # imshow: y axis is inverted (top=0)
    n_rows = max(ylim) if ylim else n_cols
    x0 = n_cols * 0.06
    y0 = n_rows * 0.93
    ax.plot(
        [x0, x0 + n_pix],
        [y0, y0],
        color="k",
        lw=1.5,
        solid_capstyle="butt",
        clip_on=False,
    )
    ax.text(
        x0 + n_pix / 2.0,
        y0 - n_rows * 0.025,
        f"{bar_mm:.0f} mm",
        ha="center",
        va="bottom",
        fontsize=6,
        color="k",
    )


def save_main_figure(
    out_path: Path,
    u_meas_c: np.ndarray,  # complex, (X,Y,Z)
    u_pred_c: np.ndarray,  # complex, (X,Y,Z); legacy sparse-overlay path
    std_c: np.ndarray,  # real, (X,Y,Z); Re-channel std, legacy path
    interior: np.ndarray,
    z_slice: int,
    title: str,
    *,
    std_c_im: np.ndarray | None = None,  # real, (X,Y,Z); Im-channel std, legacy path
    dense_slice_pred_R: np.ndarray | None = None,
    dense_slice_pred_I: np.ndarray | None = None,
    dense_slice_std: np.ndarray | None = None,
    dense_slice_std_im: np.ndarray | None = None,
    spacing_mm: np.ndarray | None = None,
    cbar_label_signal: str = "",
    cbar_label_std: str = "s.d.",
    draw_suptitle: bool = True,
):
    """Publication-quality reconstruction figure.

    Layout: rows {Re, Im, |.|} x columns {measured, predicted}, each row
    sharing ONE colour scale across measured and predicted (set from the
    measured/reference field) so the GP posterior-mean amplitude shrinkage is
    shown honestly rather than hidden by per-panel autoscaling. The right of
    the grid carries the posterior s.d.: one panel for the Re channel (top row)
    and one for the Im channel (middle row), both on a single shared
    uncertainty scale with one colorbar. The magnitude row (|.|) has no native
    posterior s.d. — |u| is a nonlinear function of the two GP outputs, so its
    uncertainty is not produced by the solver — and that cell is left empty.
    Uncertainty uses its own scale, separate from the signal, since it is a
    different quantity.

    Column headers ("Measured" / "Predicted" / "Posterior s.d.") and left row
    labels (Re / Im / |.|) replace per-panel titles. Output is written as PNG
    (300 dpi) plus editable-text SVG and PDF for journal submission.

    If ``dense_slice_*`` arrays are provided (each shape ``(X, Y)``, NaN outside
    the brain mask, predictions evaluated densely on every interior voxel of the
    displayed slice), they replace the sparse subsample-overlay visualization.

    ``spacing_mm`` (the per-axis voxel size, e.g. ``data["zooms_mm"]``) enables a
    scale bar; ``cbar_label_signal`` / ``cbar_label_std`` set colorbar units;
    ``draw_suptitle=False`` suppresses the on-figure title for the paper version.
    """
    take = lambda f: _take_axial(f, z_slice)

    interior_slice = take(interior)
    u_meas_R = take(u_meas_c.real)
    u_meas_I = take(u_meas_c.imag)
    u_meas_M = take(np.abs(u_meas_c))

    if dense_slice_pred_R is not None and dense_slice_pred_I is not None:
        # _take_axial does np.take(., z) then np.rot90 — dense slice arrays
        # are already (X, Y) at this z, so we only need the rot90.
        u_pred_R = np.rot90(dense_slice_pred_R)
        u_pred_I = np.rot90(dense_slice_pred_I)
        u_pred_M = np.rot90(
            np.where(
                np.isnan(dense_slice_pred_R),
                np.nan,
                np.sqrt(
                    np.where(np.isnan(dense_slice_pred_R), 0, dense_slice_pred_R) ** 2
                    + np.where(np.isnan(dense_slice_pred_I), 0, dense_slice_pred_I) ** 2
                ),
            )
        )
    else:
        u_pred_R = take(u_pred_c.real)
        u_pred_I = take(u_pred_c.imag)
        u_pred_M = take(np.abs(u_pred_c))

    if dense_slice_std is not None:
        std_R = np.rot90(dense_slice_std)
    else:
        std_R = take(std_c)  # Re-channel posterior std

    # Im-channel posterior std. Falls back to the Re channel only if the caller
    # supplied no Im std (older call sites), so the panel is still populated.
    if dense_slice_std_im is not None:
        std_I = np.rot90(dense_slice_std_im)
    elif std_c_im is not None:
        std_I = take(std_c_im)
    else:
        std_I = std_R

    mm_per_col = None
    if spacing_mm is not None:
        try:
            mm_per_col = float(np.asarray(spacing_mm).ravel()[0])
        except (IndexError, TypeError, ValueError):
            mm_per_col = None

    rc = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",  # editable text in SVG
        "pdf.fonttype": 42,  # editable TrueType text in PDF
        "font.size": 8,
        "axes.linewidth": 0.6,
    }
    with matplotlib.rc_context(rc):
        fig = plt.figure(figsize=(9.5, 8.0))
        # Columns: [0] measured, [1] predicted, [2] per-row signal colorbar,
        # [3] spacer, [4] full-height posterior-s.d. map, [5] s.d. colorbar.
        # The spacer (col 3) reserves room for the signal colorbar's tick
        # labels + axis label; without it the vertically-centred s.d. image in
        # col 4 (created after the colorbars) overdraws the middle (Im) row's
        # colorbar labels — the Re/|.| rows survive only because the centred
        # image does not reach their height.
        gs = fig.add_gridspec(
            3,
            6,
            width_ratios=[1.0, 1.0, 0.06, 0.45, 1.0, 0.06],
            wspace=0.06,
            hspace=0.06,
        )

        rows = [
            ("Re", u_meas_R, u_pred_R, "RdBu_r"),
            ("Im", u_meas_I, u_pred_I, "RdBu_r"),
            (r"$|\cdot|$", u_meas_M, u_pred_M, "viridis"),
        ]
        for r, (label, meas, pred, cmap) in enumerate(rows):
            meas_v = np.where(interior_slice & np.isfinite(meas), meas, np.nan)
            pred_v = np.where(interior_slice & np.isfinite(pred), pred, np.nan)

            # One shared scale per row, anchored to the MEASURED (reference)
            # field so the predicted panel is shown on the same axis.
            if cmap == "RdBu_r":
                vmax = np.nanpercentile(np.abs(meas_v), 99)
                if not np.isfinite(vmax) or vmax == 0:
                    vmax = 1.0
                vmin = -vmax
            else:
                vmin = 0.0
                vmax = np.nanpercentile(meas_v, 99)
                if not np.isfinite(vmax) or vmax <= vmin:
                    vmax = vmin + 1e-12

            ax_m = fig.add_subplot(gs[r, 0])
            ax_p = fig.add_subplot(gs[r, 1])
            im = ax_m.imshow(meas_v, cmap=cmap, vmin=vmin, vmax=vmax)
            ax_p.imshow(pred_v, cmap=cmap, vmin=vmin, vmax=vmax)

            for ax in (ax_m, ax_p):
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)

            ax_m.set_ylabel(
                label,
                fontsize=11,
                fontweight="bold",
                rotation=0,
                labelpad=14,
                va="center",
            )
            if r == 0:
                ax_m.set_title("Measured", fontsize=10, fontweight="bold")
                ax_p.set_title("Predicted", fontsize=10, fontweight="bold")
                _add_scale_bar(ax_m, meas_v.shape[1], mm_per_col)

            cax = fig.add_subplot(gs[r, 2])
            cb = fig.colorbar(im, cax=cax)
            cb.ax.tick_params(labelsize=6)
            cb.outline.set_linewidth(0.4)
            if cbar_label_signal:
                cb.set_label(cbar_label_signal, fontsize=7)

        # Posterior s.d. on the right: one panel for the Re channel (top row)
        # and one for the Im channel (middle row), on a single shared
        # uncertainty scale with one colorbar. The magnitude row has no native
        # posterior s.d. (|u| is nonlinear in the two GP outputs, so the solver
        # does not produce one), so that cell carries a short note instead of a
        # misleading propagated estimate.
        std_R_v = np.where(interior_slice & np.isfinite(std_R), std_R, np.nan)
        std_I_v = np.where(interior_slice & np.isfinite(std_I), std_I, np.nan)
        finite_std = np.concatenate(
            [
                std_R_v[np.isfinite(std_R_v)],
                std_I_v[np.isfinite(std_I_v)],
            ]
        )

        if finite_std.size:
            s_vmax = max(float(np.nanpercentile(finite_std, 99)), 1e-12)

            sm = None
            for sd_row, sd_v in ((0, std_R_v), (1, std_I_v)):
                ax_sd = fig.add_subplot(gs[sd_row, 4])
                ax_sd.set_xticks([])
                ax_sd.set_yticks([])
                for spine in ax_sd.spines.values():
                    spine.set_visible(False)
                im_sd = ax_sd.imshow(sd_v, cmap="magma", vmin=0, vmax=s_vmax)
                if sm is None:
                    sm = im_sd

            # Column header anchored to the top of the gridspec cell (row 0) so
            # it aligns with the Measured / Predicted headers regardless of the
            # aspect-driven vertical centring of the images.
            cell = gs[0, 4].get_position(fig)
            fig.text(
                (cell.x0 + cell.x1) / 2.0,
                cell.y1 + 0.005,
                "Posterior s.d.",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

            # One shared colorbar spanning the Re+Im s.d. rows.
            cax_s = fig.add_subplot(gs[0:2, 5])
            cb_s = fig.colorbar(sm, cax=cax_s)
            cb_s.ax.tick_params(labelsize=6)
            cb_s.outline.set_linewidth(0.4)
            if cbar_label_std:
                cb_s.set_label(cbar_label_std, fontsize=7)

            # Magnitude row: |.| has no native posterior s.d. (nonlinear in the
            # GP outputs), so that s.d. cell is intentionally left empty.

        if draw_suptitle and title:
            fig.suptitle(title, fontsize=11)

        out_path = Path(out_path)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        fig.savefig(out_path.with_suffix(".svg"), bbox_inches="tight")
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)


def save_mismatch_figure(
    out_path: Path,
    std_c: np.ndarray,
    rel_residual: np.ndarray,  # |r|/|k^2 u|, (X,Y,Z)
    interior: np.ndarray,
    z_slice: int,
    title: str,
    *,
    dense_slice_std: np.ndarray | None = None,
):
    """1x3: posterior std map, |r|/|k^2 u| map, scatter of std vs |r|/|k^2 u|.

    If ``dense_slice_std`` (shape ``(X, Y)``, NaN outside the brain mask) is
    provided, it replaces the sparse subsample-overlay std map. The scatter
    plot in the third panel always uses the full 3D subsample (matched to
    where the residual is meaningful, namely the eroded interior).
    """
    take = lambda f: _take_axial(f, z_slice)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    interior_slice = take(interior)
    res_slice = take(rel_residual)

    if dense_slice_std is not None:
        std_v = np.rot90(dense_slice_std)
    else:
        std_slice = take(std_c)
        std_v = np.where(interior_slice & np.isfinite(std_slice), std_slice, np.nan)
    res_v = np.where(interior_slice & np.isfinite(res_slice), res_slice, np.nan)

    if np.any(np.isfinite(std_v)):
        v_hi = np.nanpercentile(std_v, 99)
        im = axes[0].imshow(std_v, cmap="magma", vmin=0, vmax=max(v_hi, 1e-12))
        axes[0].set_title("posterior std (Re)", fontsize=10)
        axes[0].axis("off")
        plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
    else:
        axes[0].set_visible(False)

    if np.any(np.isfinite(res_v)):
        v_hi = np.nanpercentile(res_v, 95)
        im = axes[1].imshow(res_v, cmap="magma", vmin=0, vmax=max(v_hi, 1e-12))
        axes[1].set_title(r"|r| / |k$^2$ u|", fontsize=10)
        axes[1].axis("off")
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    else:
        axes[1].set_visible(False)

    # Scatter on the interior 3D voxels, not just the slice
    in3d = interior & np.isfinite(std_c) & np.isfinite(rel_residual)
    if np.sum(in3d) > 50:
        s = std_c[in3d]
        r = rel_residual[in3d]
        # Subsample for plotting if too many points
        if s.size > 5000:
            sub = np.random.default_rng(0).choice(s.size, size=5000, replace=False)
            s, r = s[sub], r[sub]
        axes[2].scatter(s, r, s=3, alpha=0.4)
        # Pearson coefficient
        pearson = np.corrcoef(std_c[in3d], rel_residual[in3d])[0, 1]
        axes[2].set_xlabel("posterior std (Re)")
        axes[2].set_ylabel(r"|r| / |k$^2$ u|")
        axes[2].set_title(f"std vs residual  (Pearson={pearson:.3f})")
        axes[2].grid(True, alpha=0.3)
    else:
        axes[2].set_visible(False)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-case driver
# ---------------------------------------------------------------------------


def run_one_case(args, subject: str, freq: int, component: int):
    obs = getattr(args, "observable", "disp")
    obs_sym = "u" if obs == "disp" else "q"

    # Resolve BC noise: --bc-noise-rel (if set) takes precedence and scales
    # with output_scale^2 so the GP signal-to-noise ratio is preserved when
    # the user retunes --output-scale for a different observable / magnitude.
    bc_noise_abs = args.bc_noise
    if args.bc_noise_rel is not None:
        bc_noise_abs = args.bc_noise_rel * args.output_scale**2
    args._bc_noise_resolved = (
        bc_noise_abs  # consumed below; not modifying args.bc_noise
    )

    print(
        f"\n== Subject {subject} @ {freq} Hz, component "
        f"{COMPONENT_NAME[component]}, observable={obs} ({obs_sym}) =="
    )
    print(
        f"  scales: output_scale={args.output_scale:.3e}  "
        f"bc_noise={bc_noise_abs:.3e}"
        + (
            f" (=bc_noise_rel {args.bc_noise_rel:.3e} * output_scale^2)"
            if args.bc_noise_rel is not None
            else " (absolute)"
        )
    )
    data = load_subject_frequency(args.root, subject, freq, observable=obs)
    omega = 2.0 * np.pi * freq
    rho_omega2 = RHO * omega**2

    # k^2 on the full grid (NaN outside mask, 0 for the solver lookup)
    safe_G = np.where(np.abs(data["G"]) > 0, data["G"], np.nan)
    k2_full = rho_omega2 / safe_G
    k2_for_solver = np.where(data["mask"] & np.isfinite(k2_full), k2_full, 0.0)

    # ---- Empirical f^emp diagnostic (always) ----
    f_emp, k2u, interior_diag = empirical_residual(
        data["u"],
        np.where(data["mask"], k2_full, np.nan),
        data["mask"],
        data["zooms_mm"],
    )
    f_mag = np.sqrt(np.sum(np.abs(f_emp) ** 2, axis=-1))
    k2u_mag = np.sqrt(np.sum(np.abs(k2u) ** 2, axis=-1))
    rel_residual = np.where(k2u_mag > 0, f_mag / k2u_mag, np.nan)

    interior_finite = interior_diag & np.isfinite(rel_residual)
    rr_int = rel_residual[interior_finite]
    print(
        f"  diagnostic |r|/|k^2 {obs_sym}|  : "
        f"median={np.nanmedian(rr_int):.3f}, "
        f"p25={np.nanpercentile(rr_int, 25):.3f}, "
        f"p75={np.nanpercentile(rr_int, 75):.3f}, "
        f"frac<1: {np.mean(rr_int < 1.0) * 100:.1f}%"
    )

    if args.skip_gp:
        return {
            "subject": subject,
            "freq": freq,
            "component": component,
            "diag_rel_median": float(np.nanmedian(rr_int)),
            "status": "skipped_gp",
        }

    # ---- GP forward solve ----
    if args.legacy_real_only:
        res = gp_forward_solve(
            u_meas=data["u"],
            k2_volume=k2_for_solver,
            mask=data["mask"],
            spacing_mm=data["zooms_mm"],
            component=component,
            n_shell=args.n_shell,
            n_interior=args.n_interior,
            kernel=args.kernel,
            matern_nu=args.matern_nu,
            lengthscale_mm=args.lengthscale_mm,
            output_scale=args.output_scale,
            bc_noise=bc_noise_abs,
            pde_noise_rel=args.pde_noise_rel,
            cov_jitter_rel=args.cov_jitter_rel,
        )
    else:
        # Phase-7c default: full complex Helmholtz block on a 2-component GP.
        # Phase 8d: --prior selects the Re/Im cross-channel prior structure.
        res = gp_forward_solve_real2(
            u_meas=data["u"],
            k2_volume_complex=k2_for_solver,
            mask=data["mask"],
            spacing_mm=data["zooms_mm"],
            component=component,
            n_shell=args.n_shell,
            n_interior=args.n_interior,
            kernel=args.kernel,
            matern_nu=args.matern_nu,
            lengthscale_mm=args.lengthscale_mm,
            output_scale=args.output_scale,
            bc_noise=bc_noise_abs,
            pde_noise_rel=args.pde_noise_rel,
            cov_jitter_rel=args.cov_jitter_rel,
            prior_kind=args.prior,
            coreg_corr=args.coreg_corr,
            lmc_lengthscales_mm=args.lmc_lengthscales,
        )
    if res["status"] != "ok":
        print(f"  GP forward solve FAILED ({res.get('phase')})")
        return {
            "subject": subject,
            "freq": freq,
            "component": component,
            "diag_rel_median": float(np.nanmedian(rr_int)),
            "status": res["status"],
            "error": res.get("error", ""),
        }

    spacing_m = np.array(data["zooms_mm"]) * 1e-3
    interior_idx = res["interior_idx"]
    X_eval = interior_idx.astype(np.float64) * spacing_m

    if res["mode"] == "real_only":
        mean_R = np.asarray(res["post_R"].mean(X_eval)).flatten()
        mean_I = np.asarray(res["post_I"].mean(X_eval)).flatten()
        try:
            std_R = np.asarray(res["post_R"].std(X_eval)).flatten()
            std_I = np.asarray(res["post_I"].std(X_eval)).flatten()
        except np.linalg.LinAlgError as exc:
            print(f"  std_R/std_I unavailable: {exc}")
            std_R = np.full_like(mean_R, np.nan)
            std_I = std_R
    elif res["mode"] == "real2":
        mean_vec = np.asarray(res["post"].mean(X_eval))  # (N, 2)
        mean_R = mean_vec[..., 0].flatten()
        mean_I = mean_vec[..., 1].flatten()
        try:
            std_vec = np.asarray(res["post"].std(X_eval))  # (N, 2)
            # Keep both channels: the figure shows a posterior s.d. panel for
            # Re and for Im. Re-channel std also drives the `confident` metric
            # and the std-vs-residual diagnostic below.
            std_R = std_vec[..., 0].flatten()
            std_I = std_vec[..., 1].flatten()
        except np.linalg.LinAlgError as exc:
            print(f"  std unavailable: {exc}")
            std_R = np.full_like(mean_R, np.nan)
            std_I = std_R
    else:
        raise RuntimeError(f"Unknown solve mode: {res.get('mode')}")

    u_true_R = data["u"][..., component].real[
        interior_idx[:, 0], interior_idx[:, 1], interior_idx[:, 2]
    ]
    u_true_I = data["u"][..., component].imag[
        interior_idx[:, 0], interior_idx[:, 1], interior_idx[:, 2]
    ]
    mag_pred = np.sqrt(mean_R**2 + mean_I**2)
    mag_true = np.sqrt(u_true_R**2 + u_true_I**2)
    err_mag = mag_pred - mag_true
    rel_err = np.abs(err_mag) / np.maximum(mag_true, 1e-9)

    pearson = np.corrcoef(mag_pred, mag_true)[0, 1] if mag_pred.size > 1 else np.nan
    median_rel = float(np.nanmedian(rel_err))
    confident = np.isfinite(std_R) & (std_R < 0.25 * mag_true)
    if confident.sum() > 0:
        median_rel_conf = float(np.nanmedian(rel_err[confident]))
    else:
        median_rel_conf = float("nan")
    print(f"\n  GP forward solve metrics on n={mag_pred.size}:")
    print(f"    Pearson |{obs_sym}_pred| vs |{obs_sym}_meas|       = {pearson:.3f}")
    print(f"    median |err|/|{obs_sym}| (all)             = {median_rel:.3f}")
    print(
        f"    median |err|/|{obs_sym}| (confident std<25%) = {median_rel_conf:.3f}  "
        f"(n_confident={int(confident.sum())})"
    )

    # ---- Build full-grid maps for the figures ----
    u_pred_full = np.full(
        data["u"].shape[:3], np.nan + 1j * np.nan, dtype=np.complex128
    )
    std_full = np.full(data["u"].shape[:3], np.nan, dtype=np.float64)
    std_I_full = np.full(data["u"].shape[:3], np.nan, dtype=np.float64)
    u_pred_full[interior_idx[:, 0], interior_idx[:, 1], interior_idx[:, 2]] = (
        mean_R + 1j * mean_I
    )
    std_full[interior_idx[:, 0], interior_idx[:, 1], interior_idx[:, 2]] = std_R
    std_I_full[interior_idx[:, 0], interior_idx[:, 1], interior_idx[:, 2]] = std_I

    # std vs |f^emp|: voxel-wise Pearson over the subsample, in the interior
    rel_at_eval = rel_residual[
        interior_idx[:, 0], interior_idx[:, 1], interior_idx[:, 2]
    ]
    finite_pair = np.isfinite(std_R) & np.isfinite(rel_at_eval) & (rel_at_eval > 0)
    if finite_pair.sum() > 5:
        pearson_std_resid = float(
            np.corrcoef(std_R[finite_pair], rel_at_eval[finite_pair])[0, 1]
        )
    else:
        pearson_std_resid = float("nan")
    print(f"    Pearson(std, |r|/|k^2 {obs_sym}|) = {pearson_std_resid:.3f}")

    # ---- Figures ----
    z_slice = data["u"].shape[2] // 2
    cn = COMPONENT_NAME[component]
    obs_tag = "" if obs == "disp" else f"_{obs}"
    fig_main = (
        args.out_dir / f"helmholtz_brain_{subject}_{freq}Hz_comp{cn}{obs_tag}_main.png"
    )
    fig_mis = (
        args.out_dir
        / f"helmholtz_brain_{subject}_{freq}Hz_comp{cn}{obs_tag}_mismatch.png"
    )
    # Short, professional on-figure title (suppress entirely for the paper
    # version via --no-suptitle; full metadata belongs in the figure caption).
    title = f"GP wavefield reconstruction — {subject}, {freq} Hz, comp {cn} ({obs})"
    sig_label = "displacement (m)" if obs == "disp" else r"curl $\nabla\times u$"

    # Dense per-voxel posterior on the displayed slice.
    #
    # Mean is a single batched call against the precomputed `alpha` vector
    # (~1 s for 8 k voxels) — always rendered densely so the "predicted"
    # panels are filled (otherwise we get the sparse-dots artifact from the
    # 1000-point training subsample at this slice).
    #
    # Std is the expensive piece. Two modes:
    #   * --sparse-figures (default for sweeps): exact std on
    #     args.std_subsample (default 400) random in-slice voxels, linearly
    #     interpolated to the full slice. ~3-5 s. The std field is smooth on
    #     the kernel lengthscale (~15 mm) vs slice extent ~100 mm so
    #     interpolation error is well below the colormap quantization.
    #   * default: full chunked back-substitution, ~70 s.
    if args.sparse_figures:
        std_sub_eff = args.std_subsample
        std_mode_str = f"subsample {std_sub_eff}"
    else:
        std_sub_eff = None
        std_mode_str = "exact"
    t_dense = time.time()
    dense_R, dense_I, dense_std, dense_std_I = _dense_slice_predictions(
        res,
        mask=data["mask"],
        spacing_m=spacing_m,
        z_slice=z_slice,
        std_subsample=std_sub_eff,
    )
    M_slice = int(np.take(data["mask"], z_slice, axis=2).sum())
    print(
        f"  dense slice eval: {time.time() - t_dense:.1f} s "
        f"(M={M_slice} voxels, std={std_mode_str})"
    )

    save_main_figure(
        fig_main,
        u_meas_c=data["u"][..., component],
        u_pred_c=u_pred_full,
        std_c=std_full,
        std_c_im=std_I_full,
        interior=interior_diag,
        z_slice=z_slice,
        title=title,
        dense_slice_pred_R=dense_R,
        dense_slice_pred_I=dense_I,
        dense_slice_std=dense_std,
        dense_slice_std_im=dense_std_I,
        spacing_mm=data["zooms_mm"],
        cbar_label_signal=sig_label,
        cbar_label_std="s.d.",
        draw_suptitle=not args.no_suptitle,
    )
    save_mismatch_figure(
        fig_mis,
        std_c=std_full,
        rel_residual=rel_residual,
        interior=interior_diag,
        z_slice=z_slice,
        title=title + " — mismatch diagnostic",
        dense_slice_std=dense_std,
    )
    print(f"  saved {fig_main.name}")
    print(f"  saved {fig_mis.name}")

    return {
        "subject": subject,
        "freq": freq,
        "component": component,
        "observable": obs,
        "diag_rel_median": float(np.nanmedian(rr_int)),
        "pearson_mag": float(pearson),
        "median_rel_err": median_rel,
        "median_rel_err_confident": median_rel_conf,
        "pearson_std_resid": pearson_std_resid,
        "n_confident": int(confident.sum()),
        "status": "ok",
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / "data"
        / "brain_experiment_data"
        / "mre_udel",
    )
    ap.add_argument("--subject", default="U01_UDEL_0001_01")
    ap.add_argument("--freq", type=int, default=50)
    ap.add_argument(
        "--component", type=int, default=1, help="0=x, 1=y (AP-aligned in LPS), 2=z"
    )
    ap.add_argument(
        "--observable",
        default="disp",
        choices=["disp", "curl"],
        help="Which vector field to fit: 'disp' (legacy, the "
        "measured displacement u) or 'curl' (the curl-"
        "filtered field q = ∇×u). Both satisfy the same "
        "scalar Helmholtz equation (Δ + k^2(x)) f = 0 under "
        "local-homogeneity; curl removes the irrotational "
        "compressional part and any constant / rigid-body "
        "bias, classical MRE direct-inversion convention.",
    )
    ap.add_argument("--n-shell", type=int, default=2000)
    ap.add_argument("--n-interior", type=int, default=2000)
    ap.add_argument("--kernel", default="matern", choices=["matern", "expquad"])
    ap.add_argument("--matern-nu", type=float, default=2.5)
    ap.add_argument("--lengthscale-mm", type=float, default=15.0)
    ap.add_argument(
        "--output-scale",
        type=float,
        default=1.0,
        help="GP prior output scale (sigma). Prior variance scales "
        "as output_scale^2. Default 1.0 is implicitly tuned to "
        "the displacement magnitude (~20 microns). When using "
        "--observable curl the natural scale is ~median(|q|) "
        "over the mask (~2e-4 on Subject 0001/50 Hz), "
        "otherwise the prior is too loose and n_confident → 0.",
    )
    ap.add_argument(
        "--bc-noise",
        type=float,
        default=1e-6,
        help="Absolute BC observation noise *variance*. Default "
        "1e-6 is implicitly tuned to disp magnitude (~20 "
        "microns), giving noise/signal std ratio ~5e-5. "
        "Override via --bc-noise-rel for scale-aware tuning.",
    )
    ap.add_argument(
        "--bc-noise-rel",
        type=float,
        default=None,
        help="BC noise variance relative to output_scale^2. When "
        "set, overrides --bc-noise with "
        "bc_noise = bc_noise_rel * output_scale^2. Use this "
        "(e.g. 1e-6) when changing --output-scale so the BC "
        "noise-to-signal ratio stays in the same regime.",
    )
    ap.add_argument(
        "--pde-noise-rel",
        type=float,
        default=1e-4,
        help="PDE noise as a fraction of trace(K)/N",
    )
    ap.add_argument(
        "--cov-jitter-rel",
        type=float,
        default=1e-7,
        help="Diagonal jitter to keep conditional Gram PD",
    )
    ap.add_argument(
        "--skip-gp",
        action="store_true",
        help="Only run the empirical residual diagnostic.",
    )
    ap.add_argument(
        "--sparse-figures",
        action="store_true",
        help="Use the fast subsample-and-interpolate path for the "
        "posterior std (--std-subsample exact evals + linear "
        "griddata; ~3-5 s for an 8 k-voxel slice at N ~ 2 k) "
        "instead of the full chunked back-substitution "
        "(~70 s). Mean is rendered densely either way. Use "
        "this for sweeps; drop it for publication-quality "
        "headline figures where you want the exact std map.",
    )
    ap.add_argument(
        "--std-subsample",
        type=int,
        default=400,
        help="Number of exact std evaluations to do on the slice "
        "when --sparse-figures is on; the rest are linearly "
        "interpolated. Default 400 is sufficient for the "
        "kernel lengthscale ~15 mm vs slice extent ~100 mm. "
        "Ignored if --sparse-figures is not set.",
    )
    ap.add_argument(
        "--no-suptitle",
        action="store_true",
        help="Suppress the on-figure title in the main figure. Use "
        "for the paper version, where the subject id, "
        "frequency, component and observable belong in the "
        "figure caption rather than on the panel.",
    )
    ap.add_argument(
        "--prior",
        default="iid",
        choices=["iid", "icm", "lmc"],
        help="Cross-channel (Re/Im) prior for the real2 path. "
        "'iid' (default) reproduces the pre-Phase-8 "
        "IndependentMultiOutput prior bit-for-bit. 'icm' "
        "couples Re/Im via a single coregionalization matrix "
        "B = output_scale^2 * [[1,rho],[rho,1]] (see "
        "--coreg-corr). 'lmc' sums Q ICM terms with distinct "
        "lengthscales (see --lmc-lengthscales). Ignored on the "
        "--legacy-real-only path.",
    )
    ap.add_argument(
        "--coreg-corr",
        type=float,
        default=None,
        help="Re/Im prior correlation rho in (-1, 1) for "
        "--prior icm/lmc. Default None == rho=0, which makes "
        "ICM reduce exactly to the IID prior (a sanity check). "
        "Sweep e.g. 0.3, 0.6, 0.9 to couple the channels.",
    )
    ap.add_argument(
        "--lmc-lengthscales",
        type=lambda s: [float(v) for v in s.split(",")],
        default=None,
        help="Comma-separated base-kernel lengthscales (mm) for "
        "--prior lmc, e.g. '7.5,30'. Default None uses two "
        "terms at 0.5x and 2x --lengthscale-mm.",
    )
    ap.add_argument(
        "--legacy-real-only",
        action="store_true",
        help="Legacy Re-only path: solve Re and Im as two scalar "
        "BVPs against (Delta + Re k^2) and drop Im k^2. The "
        "default solves the FULL complex Helmholtz block via "
        "HelmholtzReal2Operator on a 2-component GP prior "
        "built from IndependentMultiOutputCovarianceFunction.",
    )
    ap.add_argument(
        "--sweep",
        action="store_true",
        help="Run the spec's 3x3x3 sweep (3 subjects, 3 freqs, 3 components). "
        "Single-case unless this is set.",
    )
    ap.add_argument(
        "--sweep-subjects",
        nargs="+",
        default=["U01_UDEL_0001_01", "U01_UDEL_0002_01", "U01_UDEL_0003_01"],
    )
    ap.add_argument("--sweep-freqs", nargs="+", type=int, default=[30, 50, 70])
    ap.add_argument("--sweep-components", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "helmholtz_brain_outputs",
    )
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.legacy_real_only and args.prior != "iid":
        print(
            f"  [warn] --prior {args.prior} has no effect on the "
            "--legacy-real-only path (scalar BVPs); it applies only to the "
            "default real2 2-component solve."
        )

    if args.sweep:
        results = []
        for subj, freq, comp in itertools.product(
            args.sweep_subjects, args.sweep_freqs, args.sweep_components
        ):
            try:
                results.append(run_one_case(args, subj, freq, comp))
            except Exception as exc:  # pylint: disable=broad-except
                print(f"  CASE FAILED: {type(exc).__name__}: {exc}")
                results.append(
                    {
                        "subject": subj,
                        "freq": freq,
                        "component": comp,
                        "status": "failed",
                        "error": str(exc),
                    }
                )
        # Summary table
        print(f"\n=== Sweep summary (observable={args.observable}) ===")
        obs_sym = "u" if args.observable == "disp" else "q"
        for r in results:
            if r.get("status") == "ok":
                print(
                    f"  {r['subject']} @ {r['freq']}Hz comp{COMPONENT_NAME[r['component']]} "
                    f"diag={r['diag_rel_median']:.2f}  "
                    f"Pearson(|{obs_sym}|)={r['pearson_mag']:+.2f}  "
                    f"median rel err={r['median_rel_err']:.2f}  "
                    f"Pearson(std,res)={r['pearson_std_resid']:+.2f}"
                )
            else:
                print(
                    f"  {r['subject']} @ {r['freq']}Hz comp{COMPONENT_NAME[r['component']]} "
                    f"-- {r.get('status')} {r.get('error','')[:80]}"
                )
        return 0

    res = run_one_case(args, args.subject, args.freq, args.component)
    return 0 if res.get("status") in ("ok", "skipped_gp") else 1


if __name__ == "__main__":
    sys.exit(main())
