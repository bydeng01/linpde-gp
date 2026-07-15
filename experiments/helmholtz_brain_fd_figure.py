"""Comparison figure: measured curl vs deterministic FD reconstruction.
Same central axial slice, orientation, and per-row colour scaling as the
provided GP figure (helmholtz_brain_U01_UDEL_0003_01_70Hz_compx_curl_main.pdf).
"""
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

OUT = "/sessions/blissful-practical-mayer/mnt/outputs"
d = np.load(f"{OUT}/fd_brain_0003_70Hz_compx.npz")
qFD, q, mask = d["qFD"], d["q"], d["mask"]
zooms = d["zooms"]

z = mask.shape[2] // 2
take = lambda f: np.rot90(np.take(f, z, axis=2))
msk = take(mask).astype(bool)

def chan(field, part):
    v = take(getattr(field, part) if part in ("real", "imag") else np.abs(field))
    return np.where(msk, v, np.nan)

rows = [("real", "Re", "RdBu_r"), ("imag", "Im", "RdBu_r"), ("abs", "|q|", "viridis")]
fig, axes = plt.subplots(3, 2, figsize=(6.2, 8.6))
col_titles = ["Measured", "FDM (deterministic)"]

for r, (part, lab, cmap) in enumerate(rows):
    meas = chan(q, part); pred = chan(qFD, part)
    if part == "abs":
        vmax = np.nanpercentile(meas, 99); vmin = 0.0
    else:
        vmax = np.nanpercentile(np.abs(meas), 99); vmin = -vmax
    for c, img in enumerate([meas, pred]):
        ax = axes[r, c]
        im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks([]); ax.set_yticks([])
        if r == 0:
            ax.set_title(col_titles[c], fontsize=12)
        if c == 0:
            ax.set_ylabel(lab, fontsize=13, rotation=0, labelpad=18, va="center")
    cb = fig.colorbar(im, ax=axes[r, :].tolist(), fraction=0.046, pad=0.02)
    cb.ax.tick_params(labelsize=7)

fig.suptitle(
    "Subject 0003, 70 Hz, curl component x\n"
    "Deterministic FD solve of $(\\Delta+\\kappa^2(\\mathbf{x}))q=0$, same NLI moduli + measured boundary trace",
    fontsize=10.5, y=0.98)
fig.text(0.5, 0.005,
    "Pearson(|q|): FDM 0.855  vs  GP 0.753   |   median rel. err: FDM 0.296  vs  GP 0.400   "
    "(same 2000-voxel eval set)", ha="center", fontsize=9)
fig.tight_layout(rect=[0, 0.02, 1, 0.95])
for ext in ("pdf", "png"):
    fig.savefig(f"{OUT}/fd_vs_measured_0003_70Hz_compx.{ext}", dpi=200, bbox_inches="tight")
print("saved fd_vs_measured_0003_70Hz_compx.pdf/.png")
