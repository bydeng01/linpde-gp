"""Solve FDM at 3.0 mm (stride 2), slice it at the figure's central axial
plane, and render the 3 FDM panels (Re/Im/|.|) to exact 215x221 px PNGs
matching the provided figure's colormaps and per-row colour scaling
(99th percentile of the measured field, shared with Measured/Predicted)."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import NearestNDInterpolator, RegularGridInterpolator
from scipy.ndimage import binary_erosion
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import LinearOperator, bicgstab, spilu

mpl.use("Agg")

OUT = "/sessions/blissful-practical-mayer/mnt/outputs"
d = np.load(f"{OUT}/fd_brain_0003_70Hz_compx.npz")
q, mask, k2, zooms = d["q"], d["mask"], d["k2"], d["zooms"]
S_STRIDE = 2
Z = mask.shape[2] // 2  # central axial slice, = 40


def solve_fd(mask_c, k2_c, q_c, spacing_mm, erode=1):
    dx, dy, dz = np.array(spacing_mm) * 1e-3
    D = binary_erosion(mask_c, iterations=erode)
    Sb = mask_c & ~D
    idx = np.full(mask_c.shape, -1, np.int64)
    coords = np.argwhere(D)
    nD = coords.shape[0]
    idx[D] = np.arange(nD)
    cx, cy, cz = coords.T
    diag = (-2 * (1 / dx**2 + 1 / dy**2 + 1 / dz**2) + k2_c[cx, cy, cz]).astype(
        np.complex128
    )
    rows, cols, vals = [np.arange(nD)], [np.arange(nD)], [diag]
    b = np.zeros(nD, np.complex128)
    for axis, h in enumerate((dx, dy, dz)):
        coeff = 1.0 / h**2
        for off in (-1, 1):
            nb = coords.copy()
            nb[:, axis] += off
            inb = (nb[:, axis] >= 0) & (nb[:, axis] < mask_c.shape[axis])
            nbx, nby, nbz = nb.T
            in_D = np.zeros(nD, bool)
            in_D[inb] = D[nbx[inb], nby[inb], nbz[inb]]
            r_in = np.where(in_D)[0]
            rows.append(r_in)
            cols.append(idx[nbx[r_in], nby[r_in], nbz[r_in]])
            vals.append(np.full(r_in.size, coeff, np.complex128))
            r_out = np.where(~in_D)[0]
            ok = inb[r_out]
            qd = np.zeros(r_out.size, np.complex128)
            qd[ok] = q_c[nbx[r_out][ok], nby[r_out][ok], nbz[r_out][ok]]
            b[r_out] -= coeff * qd
    A = csc_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(nD, nD),
    )
    ilu = spilu(A, drop_tol=1e-4, fill_factor=15)
    x, info = bicgstab(
        A, b, rtol=1e-8, maxiter=3000, M=LinearOperator(A.shape, ilu.solve)
    )
    qc = np.full(mask_c.shape, np.nan, np.complex128)
    qc[D] = x
    qc[Sb] = q_c[Sb]
    return qc


s = S_STRIDE
mask_c, k2_c, q_c = mask[::s, ::s, ::s], k2[::s, ::s, ::s], q[::s, ::s, ::s]
qc = solve_fd(mask_c, k2_c, q_c, zooms * s, erode=1)

# interpolate coarse solution onto the FINE central-axial slice (z=Z plane)
axes = tuple(np.arange(n) for n in mask_c.shape)
fri = RegularGridInterpolator(
    axes, qc.real, method="linear", bounds_error=False, fill_value=np.nan
)
fii = RegularGridInterpolator(
    axes, qc.imag, method="linear", bounds_error=False, fill_value=np.nan
)
nx, ny = mask.shape[:2]
gx, gy = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
pts = np.stack([gx.ravel() / s, gy.ravel() / s, np.full(gx.size, Z / s)], axis=1)
qfdm = (fri(pts) + 1j * fii(pts)).reshape(nx, ny)

# Display region = the figure's interior_diag (1-voxel-eroded interior from
# empirical_residual): mask AND all six face-neighbours in mask.
interior_diag = mask.copy()
for ax in (0, 1, 2):
    interior_diag &= np.roll(mask, 1, ax) & np.roll(mask, -1, ax)
disp_sl = interior_diag[:, :, Z]
# Fill every display voxel (interior_diag) left NaN by the linear interpolation
# with the nearest VALID coarse value via a KD-tree, so the FDM panel covers the
# full interior_diag extent (no holes / no severed thin structures) and matches
# the Measured/Predicted brain outline. RegularGridInterpolator-nearest fails
# here because the nearest grid CELL can itself be outside the decimated mask.
fin = np.argwhere(np.isfinite(qc))
nnd_r = NearestNDInterpolator(fin, qc.real[fin[:, 0], fin[:, 1], fin[:, 2]])
nnd_i = NearestNDInterpolator(fin, qc.imag[fin[:, 0], fin[:, 1], fin[:, 2]])
fillm = disp_sl & ~np.isfinite(qfdm)
if fillm.any():
    fp = np.stack(
        [gx[fillm] / s, gy[fillm] / s, np.full(int(fillm.sum()), Z / s)], axis=1
    )
    qfdm[fillm] = nnd_r(fp) + 1j * nnd_i(fp)
meas_sl = q[:, :, Z]
CNAN = complex(
    np.nan, np.nan
)  # complex NaN: masks BOTH Re and Im (nan+0j would leave Im=0)
take = lambda f: np.rot90(np.where(disp_sl, f, CNAN))
# The original figure's panels are tight-cropped to the brain bounding box (so
# the brain fills the panel). Match that: crop to the mask bbox (+ small margin)
# then let imshow stretch it to fill, exactly as Measured/Predicted are shown.
mslice = np.rot90(mask[:, :, Z])
rr = np.where(mslice.any(1))[0]
cc = np.where(mslice.any(0))[0]
PAD = 5
r0, r1 = max(0, rr[0] - PAD), min(mslice.shape[0] - 1, rr[-1] + PAD)
c0, c1 = max(0, cc[0] - PAD), min(mslice.shape[1] - 1, cc[-1] + PAD)
crop = lambda a: a[r0 : r1 + 1, c0 : c1 + 1]
meas_d, fdm_d = crop(take(meas_sl)), crop(take(qfdm))


# per-row colour scale from the MEASURED field (99th pct), shared across columns
def vlim(part):
    if part == "abs":
        return 0.0, np.nanpercentile(np.abs(meas_d), 99)
    v = np.nanpercentile(np.abs(getattr(meas_d, part)), 99)
    return -v, v


rows = [("real", "RdBu_r"), ("imag", "RdBu_r"), ("abs", "viridis")]
print("per-row vlim (compare to figure colorbars Re~0.0010, Im~0.0006, |.|~0.0014):")
for part, _ in rows:
    print(f"  {part}: {vlim(part)}")

for r, (part, cmap) in enumerate(rows):
    vmn, vmx = vlim(part)
    val = getattr(fdm_d, part) if part != "abs" else np.abs(fdm_d)
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad("white")
    fig = plt.figure(figsize=(215 / 100, 221 / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.imshow(np.ma.masked_invalid(val), cmap=cm, vmin=vmn, vmax=vmx, aspect="auto")
    fig.savefig(f"{OUT}/fdm_panel_{part}.png", dpi=100, facecolor="white")
    plt.close(fig)
print("saved fdm_panel_real/imag/abs.png (215x221)")
