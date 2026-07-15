"""Coarsen the FD grid and find the resolution at which the deterministic
solver matches the GP (Pearson ~0.753) on subject 0003 / 70 Hz / comp x.

At stride s the grid spacing is s*1.5 mm, which reduces both the interior
DOF and the number of Dirichlet boundary points, bringing the deterministic
solver to a constraint budget comparable to the GP's 2000 sparse boundary
observations. We solve the coarse BVP, interpolate back to the GP's exact
2000-voxel eval set, and report Pearson(|q|) vs the measurement.
"""
import numpy as np, time
from scipy.ndimage import binary_erosion
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spilu, bicgstab, LinearOperator
from scipy.interpolate import RegularGridInterpolator

OUT = "/sessions/blissful-practical-mayer/mnt/outputs"
d = np.load(f"{OUT}/fd_brain_0003_70Hz_compx.npz")
q, mask, k2, zooms = d["q"], d["mask"], d["k2"], d["zooms"]

# GP's exact 2000-voxel eval set (seed 0: shell draws first, then interior)
rng = np.random.default_rng(0)
shell = mask & ~binary_erosion(mask, iterations=2)
interior = binary_erosion(mask, iterations=3)
si, ii = np.argwhere(shell), np.argwhere(interior)
rng.choice(si.shape[0], size=2000, replace=False)
ii = ii[rng.choice(ii.shape[0], size=2000, replace=False)]
mag_meas = np.abs(q[ii[:, 0], ii[:, 1], ii[:, 2]])

def solve_fd(mask_c, k2_c, q_c, spacing_mm, erode=1):
    dx, dy, dz = np.array(spacing_mm) * 1e-3
    D = binary_erosion(mask_c, iterations=erode)
    S = mask_c & ~D
    idx = np.full(mask_c.shape, -1, np.int64)
    coords = np.argwhere(D); nD = coords.shape[0]
    idx[D] = np.arange(nD)
    cx, cy, cz = coords.T
    diag = (-2 * (1/dx**2 + 1/dy**2 + 1/dz**2) + k2_c[cx, cy, cz]).astype(np.complex128)
    rows, cols, vals = [np.arange(nD)], [np.arange(nD)], [diag]
    b = np.zeros(nD, np.complex128)
    for axis, h in enumerate((dx, dy, dz)):
        coeff = 1.0/h**2
        for off in (-1, 1):
            nb = coords.copy(); nb[:, axis] += off
            inb = (nb[:, axis] >= 0) & (nb[:, axis] < mask_c.shape[axis])
            nbx, nby, nbz = nb.T
            in_D = np.zeros(nD, bool)
            in_D[inb] = D[nbx[inb], nby[inb], nbz[inb]]
            r_in = np.where(in_D)[0]
            rows.append(r_in); cols.append(idx[nbx[r_in], nby[r_in], nbz[r_in]])
            vals.append(np.full(r_in.size, coeff, np.complex128))
            r_out = np.where(~in_D)[0]; ok = inb[r_out]
            qd = np.zeros(r_out.size, np.complex128)
            qd[ok] = q_c[nbx[r_out][ok], nby[r_out][ok], nbz[r_out][ok]]
            b[r_out] -= coeff*qd
    A = csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(nD, nD))
    ilu = spilu(A, drop_tol=1e-4, fill_factor=15)
    x, info = bicgstab(A, b, rtol=1e-8, maxiter=3000, M=LinearOperator(A.shape, ilu.solve))
    qc = np.full(mask_c.shape, np.nan, np.complex128); qc[D] = x; qc[S] = q_c[S]
    return qc, int(S.sum()), nD

print(f"{'stride':>6} {'spacing_mm':>10} {'gridDOF':>8} {'bdryPts':>8} {'Pearson':>8} {'relerr':>7} {'nEval':>6}")
print(f"{1:>6d} {zooms[0]:>10.1f} {'307518':>8} {'88650':>8} {0.855:>8.3f} {0.296:>7.3f} {2000:>6d}  (full res, from prior run)")
for s in (2, 3, 4, 5, 6, 8):
    mask_c = mask[::s, ::s, ::s]; k2_c = k2[::s, ::s, ::s]; q_c = q[::s, ::s, ::s]
    qc, nb, nD = solve_fd(mask_c, k2_c, q_c, zooms*s, erode=1)
    # interpolate coarse solution to the fine eval voxels (coarse idx = fine/s)
    axes = tuple(np.arange(n) for n in mask_c.shape)
    pts = ii.astype(np.float64) / s
    fr = RegularGridInterpolator(axes, qc.real, method="linear", bounds_error=False, fill_value=np.nan)(pts)
    fi = RegularGridInterpolator(axes, qc.imag, method="linear", bounds_error=False, fill_value=np.nan)(pts)
    mag = np.sqrt(fr**2 + fi**2)
    ok = np.isfinite(mag)
    p = np.corrcoef(mag[ok], mag_meas[ok])[0, 1]
    rel = np.median(np.abs(mag[ok]-mag_meas[ok])/np.maximum(mag_meas[ok], 1e-9))
    print(f"{s:>6d} {zooms[0]*s:>10.1f} {nD:>8d} {nb:>8d} {p:>8.3f} {rel:>7.3f} {int(ok.sum()):>6d}")
