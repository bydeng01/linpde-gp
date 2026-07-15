"""Pearson(|q|) vs FD grid spacing, with the GP level marked. Shows the
deterministic solver matching the GP at ~9 mm effective resolution."""
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

OUT = "/sessions/blissful-practical-mayer/mnt/outputs"
spacing = np.array([1.5, 3.0, 4.5, 6.0, 7.5, 9.0, 12.0])
pearson = np.array([0.855, 0.807, 0.822, 0.804, 0.813, 0.742, 0.722])
GP = 0.753

fig, ax = plt.subplots(figsize=(6.2, 4.3))
ax.plot(spacing, pearson, "o-", color="#1f77b4", lw=2, ms=7, label="FD (deterministic)")
ax.axhline(GP, ls="--", color="#d62728", lw=1.8, label=f"GP (sparse, 2000 bdry pts) = {GP:.3f}")
# crossover near 9 mm
ax.axvline(9.0, ls=":", color="gray", lw=1.2)
ax.annotate("FD matches GP\n$\\approx$9 mm grid\n($\\sim$800 bdry pts)",
            xy=(9.0, 0.742), xytext=(9.6, 0.78), fontsize=9,
            arrowprops=dict(arrowstyle="->", color="gray"))
ax.set_xlabel("FD grid spacing (mm)", fontsize=11)
ax.set_ylabel("Pearson(|q|) vs measured", fontsize=11)
ax.set_title("Subject 0003, 70 Hz, comp $x$: FD accuracy vs resolution", fontsize=11)
ax.set_ylim(0.70, 0.88); ax.grid(alpha=0.3); ax.legend(fontsize=9, loc="lower left")
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(f"{OUT}/fd_resolution_vs_gp_0003_70Hz_compx.{ext}", dpi=200, bbox_inches="tight")
print("saved fd_resolution_vs_gp_0003_70Hz_compx.pdf/.png")
