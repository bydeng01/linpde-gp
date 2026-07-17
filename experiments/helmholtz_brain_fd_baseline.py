"""Deterministic finite-difference solve of the variable-coefficient complex
Helmholtz BVP on brain MRE data, for one config (subject 0003, 70 Hz, comp x).

Solves   (Delta + k^2(x)) q = 0   on interior D = erosion(mask, 2)
with      q = q_meas             on the shell band S = mask \\ D  (Dirichlet),

using the SAME k^2 = rho*omega^2/G and the SAME 7-point Laplacian (in metres)
as experiments/helmholtz_brain_forward_bvp.py. This is the deterministic
counterpart to the GP forward solve: identical inputs (NLI moduli + measured
boundary trace), interior held out. Correlations are reported on the GP eval
region E = erosion(mask, 3).
"""

import sys
import time

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_erosion
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import LinearOperator, bicgstab, spilu, splu

RHO = 1040.0  # kg/m^3 (brain tissue; was 1000.0)
ROOT = "/sessions/blissful-practical-mayer/mnt/linpde-gp/data/brain_experiment_data/mre_udel"
SUB, FREQ, COMP = "U01_UDEL_0003_01", 70, 0  # comp x
OUT = "/sessions/blissful-practical-mayer/mnt/outputs"


def load():
    fd = f"{ROOT}/{SUB}_v4/{SUB}_MRE_AP_{FREQ}Hz"
    reg = f"{ROOT}/{SUB}_v4/{SUB}_register_to_MRE"
    stem = f"{SUB}_MRE_AP_{FREQ}Hz"
    cre = np.asanyarray(nib.load(f"{fd}/{stem}_curl_re.nii").dataobj).astype(np.float64)
    cim = np.asanyarray(nib.load(f"{fd}/{stem}_curl_im.nii").dataobj).astype(np.float64)
    Gre = np.asanyarray(nib.load(f"{fd}/{stem}_props_shear_real.nii").dataobj).astype(
        np.float64
    )
    Gim = np.asanyarray(nib.load(f"{fd}/{stem}_props_shear_imag.nii").dataobj).astype(
        np.float64
    )
    m_img = nib.load(f"{reg}/{SUB}_MREreg_brainmask.nii")
    mask = np.asanyarray(m_img.dataobj).astype(bool)
    zooms = np.array(m_img.header.get_zooms()[:3], dtype=np.float64)  # mm
    q = (cre + 1j * cim)[..., COMP]  # complex curl, component x
    G = Gre + 1j * Gim
    return q, G, mask, zooms


def laplacian_3d(field, dx, dy, dz):
    out = np.zeros_like(field)
    out += (np.roll(field, -1, 0) - 2 * field + np.roll(field, 1, 0)) / dx**2
    out += (np.roll(field, -1, 1) - 2 * field + np.roll(field, 1, 1)) / dy**2
    out += (np.roll(field, -1, 2) - 2 * field + np.roll(field, 1, 2)) / dz**2
    return out


def main():
    q, G, mask, zooms = load()
    omega = 2 * np.pi * FREQ
    with np.errstate(divide="ignore", invalid="ignore"):
        k2 = RHO * omega**2 / G
    k2 = np.where(np.isfinite(k2), k2, 0.0)
    dx, dy, dz = zooms * 1e-3
    print(f"grid {mask.shape} spacing(mm) {zooms} mask {int(mask.sum())}")

    # --- sanity: empirical residual of MEASURED field on eval region (paper ~3.11)
    E = binary_erosion(mask, iterations=3)
    lap_meas = laplacian_3d(q, dx, dy, dz)
    r_meas = lap_meas + k2 * q
    rel = np.abs(r_meas) / np.maximum(np.abs(k2 * q), 1e-30)
    print(
        f"[check] measured |r|/|k^2 q| median on E = {np.median(rel[E]):.3f}  (paper 3.11)"
    )

    # --- BVP index sets
    D = binary_erosion(mask, iterations=2)  # unknown interior (PDE rows)
    S = mask & ~D  # Dirichlet shell band
    idx = np.full(mask.shape, -1, np.int64)
    coords = np.argwhere(D)
    nD = coords.shape[0]
    idx[D] = np.arange(nD)
    print(f"unknowns |D|={nD}  shell |S|={int(S.sum())}  eval |E|={int(E.sum())}")

    cx, cy, cz = coords[:, 0], coords[:, 1], coords[:, 2]
    diag = (-2 * (1 / dx**2 + 1 / dy**2 + 1 / dz**2) + k2[cx, cy, cz]).astype(
        np.complex128
    )
    rows = [np.arange(nD)]
    cols = [np.arange(nD)]
    vals = [diag]
    b = np.zeros(nD, np.complex128)
    for axis, h in enumerate((dx, dy, dz)):
        coeff = 1.0 / h**2
        for off in (-1, +1):
            nb = coords.copy()
            nb[:, axis] += off
            inb = (nb[:, axis] >= 0) & (nb[:, axis] < mask.shape[axis])
            nbx, nby, nbz = nb[:, 0], nb[:, 1], nb[:, 2]
            in_D = np.zeros(nD, bool)
            in_D[inb] = D[nbx[inb], nby[inb], nbz[inb]]
            # neighbors inside D -> matrix entry
            r_in = np.where(in_D)[0]
            rows.append(r_in)
            cols.append(idx[nbx[r_in], nby[r_in], nbz[r_in]])
            vals.append(np.full(r_in.size, coeff, np.complex128))
            # neighbors not in D -> Dirichlet (= measured trace) -> RHS
            r_out = np.where(~in_D)[0]
            qd = np.zeros(r_out.size, np.complex128)
            ok = inb[r_out]
            qd[ok] = q[nbx[r_out][ok], nby[r_out][ok], nbz[r_out][ok]]
            b[r_out] -= coeff * qd
    A = csc_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(nD, nD),
    )
    print(f"matrix {A.shape} nnz={A.nnz}", flush=True)
    t = time.time()
    ilu = spilu(A.tocsc(), drop_tol=1e-4, fill_factor=15)
    print(f"ILU built in {time.time()-t:.1f}s", flush=True)
    M = LinearOperator(A.shape, ilu.solve)
    it = {"n": 0}

    def cb(xk):
        it["n"] += 1

    t = time.time()
    x, info = bicgstab(A, b, rtol=1e-8, maxiter=2000, M=M, callback=cb)
    relres = np.linalg.norm(A @ x - b) / np.linalg.norm(b)
    print(
        f"BiCGStab info={info} iters={it['n']} in {time.time()-t:.1f}s  ||Ax-b||/||b||={relres:.2e}",
        flush=True,
    )
    if relres > 1e-5:  # fallback to direct if iterative struggled
        print("iterative residual high; falling back to splu...", flush=True)
        x = splu(A).solve(b)
        print(
            f"[check] direct ||Ax-b||/||b|| = {np.linalg.norm(A@x-b)/np.linalg.norm(b):.2e}",
            flush=True,
        )

    qFD = np.full(mask.shape, np.nan, np.complex128)
    qFD[D] = x
    qFD[S] = q[S]
    np.savez(
        f"{OUT}/fd_brain_0003_70Hz_compx.npz",
        qFD=qFD,
        q=q,
        mask=mask,
        D=D,
        S=S,
        E=E,
        k2=k2,
        zooms=zooms,
    )

    # --- stability + correlation on E
    magFD = np.abs(qFD[E])
    magM = np.abs(q[E])
    print(
        f"[stability] max|qFD|/max|qmeas| on interior = {np.nanmax(np.abs(qFD[D]))/np.max(np.abs(q[D])):.2f}"
    )
    print(
        f"[result] corr(|qFD|,|qmeas|) on E = {np.corrcoef(magFD,magM)[0,1]:.3f}  (n={E.sum()})"
    )
    magFD_D = np.abs(qFD[D])
    magM_D = np.abs(q[D])
    print(
        f"[result] corr(|qFD|,|qmeas|) on D = {np.corrcoef(magFD_D,magM_D)[0,1]:.3f}  (n={D.sum()})"
    )
    # relative magnitude error like the paper
    rel_err = np.abs(magFD - magM) / np.maximum(magM, 1e-9)
    print(f"[result] median |err|/|q| on E = {np.median(rel_err):.3f}")


if __name__ == "__main__":
    main()
