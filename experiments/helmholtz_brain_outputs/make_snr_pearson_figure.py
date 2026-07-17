"""SNR vs Pearson master figure (Nature-style, Python/matplotlib).

Claim: at fixed drive frequency, curl SNR predicts GP-PDE reconstruction
accuracy (Pearson); low-frequency cells fall off that trend because of
Helmholtz model breakdown, not noise.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
})

# --- data: (SNR_dB, Pearson, label, dx, dy, ha, va) ---
# 70 Hz cells (fixed physics validity) -> define the trend
hz70 = [
    (6.072, 0.766, "0001·x"),
    (3.931, 0.591, "0001·y"),
    (5.335, 0.625, "0001·z"),
    (5.289, 0.679, "0002·x"),
    (6.768, 0.753, "0003·x"),
]
# low-freq cells (same comp-x / subj 0001) -> off-trend
lowf = [
    (8.925, 0.735, "50 Hz"),
    (11.588, 0.596, "30 Hz"),
]
label_off = {  # per-point label offsets (data units)
    "0001·x": (0.10, 0.011, "left", "bottom"),
    "0001·y": (0.12, -0.004, "left", "top"),
    "0001·z": (0.12, -0.010, "left", "top"),
    "0002·x": (-0.12, 0.012, "right", "bottom"),
    "0003·x": (0.12, 0.000, "left", "center"),
}

COL70 = "#2F6DB5"   # in-regime trend (neutral-signal blue)
COL50 = "#E8943A"   # off-trend accent (amber)
COL30 = "#C7382E"   # off-trend accent (red = strongest deviation)
GREY = "#5A5A5A"

x70 = np.array([p[0] for p in hz70]); y70 = np.array([p[1] for p in hz70])
lr = stats.linregress(x70, y70)
r, p = lr.rvalue, lr.pvalue

fig, ax = plt.subplots(figsize=(3.5, 3.05))

# spec threshold
ax.axhline(0.75, color=GREY, lw=0.6, ls=(0, (4, 3)), zorder=1)
ax.text(11.9, 0.756, "spec 0.75", fontsize=5.8, color=GREY, ha="right", va="bottom")

# 70 Hz fit: solid over data range, dashed extrapolation beyond
xs_fit = np.linspace(x70.min(), x70.max(), 50)
ax.plot(xs_fit, lr.intercept + lr.slope * xs_fit, color=COL70, lw=1.3, zorder=3)
xs_ext = np.linspace(x70.max(), 12.2, 50)
ax.plot(xs_ext, lr.intercept + lr.slope * xs_ext, color=COL70, lw=1.0,
        ls=(0, (5, 3)), alpha=0.7, zorder=2)

# 70 Hz points
for sx, sy, lab in hz70:
    ax.scatter(sx, sy, s=34, color=COL70, edgecolor="white", linewidth=0.6, zorder=5)
    dx, dy, ha, va = label_off[lab]
    ax.annotate(lab, (sx, sy), (sx + dx, sy + dy), fontsize=6,
                color="#1f1f1f", ha=ha, va=va)

# frequency trajectory: SAME field (0001 comp-x) tracked across drive freq
traj = [(6.072, 0.766), (8.925, 0.735), (11.588, 0.596)]
for (ax0, ay0), (ax1, ay1) in zip(traj[:-1], traj[1:]):
    ax.annotate("", xy=(ax1, ay1), xytext=(ax0, ay0),
                arrowprops=dict(arrowstyle="-|>", color=GREY, lw=0.8, alpha=0.55,
                                shrinkA=6, shrinkB=6))
# low-freq points
for (sx, sy, lab), c, mk in [(lowf[0], COL50, "D"), (lowf[1], COL30, "s")]:
    ax.scatter(sx, sy, s=40, color=c, marker=mk, edgecolor="white",
               linewidth=0.6, zorder=6)
    ax.annotate(lab, (sx, sy), (sx, sy - 0.017), fontsize=6.4, color=c,
                ha="center", va="top", fontweight="bold")
ax.text(9.5, 0.690, "decreasing\ndrive frequency", fontsize=5.6, color=GREY,
        ha="left", va="center", style="italic")
ax.text(12.3, 0.828,
        "lower freq → higher SNR yet lower accuracy\n"
        "(Helmholtz residual ↑, spatial contrast ↓)",
        fontsize=5.8, color="#1f1f1f", ha="right", va="top")

# trend stats box
ax.text(3.75, 0.815,
        f"70 Hz cells:\nr = {r:.2f}, p = {p:.3f}\n(component + subject)",
        fontsize=6, color=COL70, ha="left", va="top")

ax.set_xlabel("Curl SNR (dB)")
ax.set_ylabel(r"Reconstruction accuracy, Pearson $|\hat{q}|$ vs $|q|$")
ax.set_xlim(3.3, 12.4)
ax.set_ylim(0.55, 0.83)
ax.set_xticks([4, 6, 8, 10, 12])
ax.tick_params(width=0.8, length=3)

fig.tight_layout()
for ext, kw in [("svg", {}), ("pdf", {}), ("tiff", {"dpi": 600}), ("png", {"dpi": 300})]:
    fig.savefig(f"snr_pearson_master.{ext}", bbox_inches="tight", **kw)
print(f"70Hz trend: r={r:.3f} p={p:.4f} slope={lr.slope:.4f} intercept={lr.intercept:.4f}")
print("saved snr_pearson_master.{svg,pdf,tiff,png}")
