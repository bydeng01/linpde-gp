"""Insert a geometry-matched FDM column into the GP curl figure.

Fixes the size/placement bug in the previous splice: instead of pasting
fixed 215x221 panels at hardcoded positions (which were smaller than and
offset from the true 217x273 brain panels), this:
  * detects the real Measured/Predicted brain rectangles from the raster,
  * renders each FDM panel cropped to the SAME interior_diag brain bbox and
    masked to the SAME silhouette the GP figure uses (lines 950/1024 of
    helmholtz_brain_forward_bvp.py), so the FDM brain is pixel-aligned in
    size, aspect and outline with Measured/Predicted.
Self-contained: re-solves the 3 mm FDM, reproducing fd_panels_3mm.py.
"""
import os, numpy as np, nibabel as nib, fitz
from PIL import Image, ImageDraw, ImageFont
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.ndimage import binary_erosion
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spilu, splu, bicgstab, LinearOperator
from scipy.interpolate import RegularGridInterpolator, NearestNDInterpolator

_here = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("REPO", os.path.dirname(_here))      # <repo>/experiments/.. == repo root
ROOT = f"{REPO}/data/brain_experiment_data/mre_udel"
OUTDIR = os.environ.get("OUTDIR", f"{REPO}/experiments/helmholtz_brain_outputs")
ORIG_PDF = os.environ.get("ORIG_PDF",
    f"{OUTDIR}/helmholtz_brain_U01_UDEL_0003_01_70Hz_compx_curl_main.pdf")
os.makedirs(OUTDIR, exist_ok=True)
RHO = 1040.0  # kg/m^3 (brain tissue; was 1000.0)
SUB, FREQ, COMP = "U01_UDEL_0003_01", 70, 0  # subject 0003, 70 Hz, comp x (curl)
RASTER_W = 1703

# ----------------------------------------------------------------- FDM 3 mm solve
fd = f"{ROOT}/{SUB}_v4/{SUB}_MRE_AP_{FREQ}Hz"; reg = f"{ROOT}/{SUB}_v4/{SUB}_register_to_MRE"
stem = f"{SUB}_MRE_AP_{FREQ}Hz"
cre = np.asanyarray(nib.load(f"{fd}/{stem}_curl_re.nii").dataobj).astype(np.float64)
cim = np.asanyarray(nib.load(f"{fd}/{stem}_curl_im.nii").dataobj).astype(np.float64)
Gre = np.asanyarray(nib.load(f"{fd}/{stem}_props_shear_real.nii").dataobj).astype(np.float64)
Gim = np.asanyarray(nib.load(f"{fd}/{stem}_props_shear_imag.nii").dataobj).astype(np.float64)
m_img = nib.load(f"{reg}/{SUB}_MREreg_brainmask.nii")
mask = np.asanyarray(m_img.dataobj).astype(bool)
zooms = np.array(m_img.header.get_zooms()[:3], np.float64)
q = (cre + 1j*cim)[..., COMP]; G = Gre + 1j*Gim
omega = 2*np.pi*FREQ
with np.errstate(divide="ignore", invalid="ignore"):
    k2 = RHO*omega**2 / G
k2 = np.where(np.isfinite(k2), k2, 0.0)

def solve_fd(mask_c, k2_c, q_c, spacing_mm, erode=1):
    dx, dy, dz = np.array(spacing_mm)*1e-3
    D = binary_erosion(mask_c, iterations=erode); Sb = mask_c & ~D
    idx = np.full(mask_c.shape, -1, np.int64); coords = np.argwhere(D); nD = coords.shape[0]; idx[D] = np.arange(nD)
    cx, cy, cz = coords.T
    diag = (-2*(1/dx**2+1/dy**2+1/dz**2)+k2_c[cx, cy, cz]).astype(np.complex128)
    rows, cols, vals = [np.arange(nD)], [np.arange(nD)], [diag]; b = np.zeros(nD, np.complex128)
    for axis, h in enumerate((dx, dy, dz)):
        coeff = 1.0/h**2
        for off in (-1, 1):
            nb = coords.copy(); nb[:, axis] += off
            inb = (nb[:, axis] >= 0) & (nb[:, axis] < mask_c.shape[axis]); nbx, nby, nbz = nb.T
            in_D = np.zeros(nD, bool); in_D[inb] = D[nbx[inb], nby[inb], nbz[inb]]
            r_in = np.where(in_D)[0]
            rows.append(r_in); cols.append(idx[nbx[r_in], nby[r_in], nbz[r_in]]); vals.append(np.full(r_in.size, coeff, np.complex128))
            r_out = np.where(~in_D)[0]; ok = inb[r_out]; qd = np.zeros(r_out.size, np.complex128)
            qd[ok] = q_c[nbx[r_out][ok], nby[r_out][ok], nbz[r_out][ok]]; b[r_out] -= coeff*qd
    A = csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(nD, nD))
    ilu = spilu(A, drop_tol=1e-4, fill_factor=15)
    x, _ = bicgstab(A, b, rtol=1e-8, maxiter=3000, M=LinearOperator(A.shape, ilu.solve))
    if np.linalg.norm(A @ x - b) > 1e-5 * np.linalg.norm(b):   # direct fallback if iterative struggles
        x = splu(A.tocsc()).solve(b)
    qc = np.full(mask_c.shape, np.nan, np.complex128); qc[D] = x; qc[Sb] = q_c[Sb]; return qc

Z = mask.shape[2]//2
interior_diag = mask.copy()
for ax in (0, 1, 2): interior_diag &= np.roll(mask, 1, ax) & np.roll(mask, -1, ax)
disp_sl = interior_diag[:, :, Z]
MODE = os.environ.get("FD_DISPLAY", "full")   # "full" (1.5 mm, == the Pearson-0.855 baseline) or "3mm"
if MODE == "3mm":
    s = 2
    qc = solve_fd(mask[::s, ::s, ::s], k2[::s, ::s, ::s], q[::s, ::s, ::s], zooms*s, erode=1)
    axc = tuple(np.arange(n) for n in qc.shape)
    fri = RegularGridInterpolator(axc, qc.real, method="linear", bounds_error=False, fill_value=np.nan)
    fii = RegularGridInterpolator(axc, qc.imag, method="linear", bounds_error=False, fill_value=np.nan)
    nx, ny = mask.shape[:2]; gx, gy = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    pts = np.stack([gx.ravel()/s, gy.ravel()/s, np.full(gx.size, Z/s)], axis=1)
    qfdm = (fri(pts)+1j*fii(pts)).reshape(nx, ny)
    fin = np.argwhere(np.isfinite(qc))
    nnd_r = NearestNDInterpolator(fin, qc.real[fin[:, 0], fin[:, 1], fin[:, 2]])
    nnd_i = NearestNDInterpolator(fin, qc.imag[fin[:, 0], fin[:, 1], fin[:, 2]])
    fillm = disp_sl & ~np.isfinite(qfdm)
    if fillm.any():
        fp = np.stack([gx[fillm]/s, gy[fillm]/s, np.full(int(fillm.sum()), Z/s)], axis=1)
        qfdm[fillm] = nnd_r(fp)+1j*nnd_i(fp)
else:
    # full-resolution deterministic baseline (interior D = erosion(mask,2), measured
    # Dirichlet shell) -> the DISPLAYED solve is the one whose Pearson is 0.855.
    qFD = solve_fd(mask, k2, q, zooms, erode=2)
    qfdm = qFD[:, :, Z]
    E = binary_erosion(mask, iterations=3)
    print(f"[fdm] full-res corr(|qFD|,|qmeas|) on E = "
          f"{np.corrcoef(np.abs(qFD[E]), np.abs(q[E]))[0,1]:.3f}  (2000-set baseline = 0.855)")

# display-frame slices (rot90, masked to interior_diag like the GP figure)
CNAN = complex(np.nan, np.nan)
take = lambda f: np.rot90(np.where(disp_sl, f, CNAN))
meas_d = take(q[:, :, Z]); fdm_d = take(qfdm)
sil = np.rot90(disp_sl)                                    # brain silhouette (display frame)
r0, r1 = np.where(sil.any(1))[0][[0, -1]]; c0, c1 = np.where(sil.any(0))[0][[0, -1]]   # tight bbox
crop = lambda a: a[r0:r1+1, c0:c1+1]
sil_c = crop(sil)
# per-row colour scale = 99th pct of |measured| on the slice (shared with GP cols)
def vlim(part):
    if part == "abs": return 0.0, float(np.nanpercentile(np.abs(meas_d), 99))
    v = float(np.nanpercentile(np.abs(getattr(meas_d, part)), 99)); return -v, v

# ------------------------------------------------------ detect panel geometry
doc = fitz.open(ORIG_PDF); page = doc[0]; zoom = RASTER_W/page.rect.width
pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
# header baseline/size from the real "Measured" label so "FDM" matches exactly
hdr_top, hdr_size = 139.0, 28.0
for _b in page.get_text("dict")["blocks"]:
    for _l in _b.get("lines", []):
        for _sp in _l["spans"]:
            if _sp["text"].strip() == "Measured":
                hdr_top = _sp["bbox"][1]*zoom; hdr_size = _sp["size"]*zoom
arr = np.asarray(img); nw = ~np.all(arr > 245, axis=2); H, W = nw.shape
def bb(x0, x1, y0, y1):
    sub = nw[y0:y1, x0:x1]; ys, xs = np.where(sub)
    return x0+xs.min(), y0+ys.min(), x0+xs.max(), y0+ys.max()
# RASTER_W fixes the WIDTH (zoom = 1703/page_width), so the column x-positions are
# stable across re-renders; only the figure HEIGHT/aspect drifts (bbox_inches=
# "tight" depends on tick labels / value ranges). So keep fixed x-windows for the
# columns but AUTO-detect the 3 brain rows by y-projection -- hardcoded y-windows
# silently mis-fire whenever the base figure is re-rendered (e.g. at a new rho).
# Coarse column windows are used ONLY to locate the three brain rows. A full-
# height bb() over the Measured window also catches the "20 mm" scale-bar
# annotation in the left margin (its label reaches x~150, in the Re/Im row gap);
# that dragged mx0 left and inflated brain_w, stretching the FDM panel ~9% wider
# than Measured/Predicted (and making the column spacing uneven). So detect the
# rows first, then re-measure the exact brain x-extent inside those rows (and
# clipped to x>=130), where the scale-bar annotation cannot leak in.
mx0, _, mx1, _ = bb(150, 430, int(hdr_top) + 40, H - 1)   # coarse Measured x-window
px0, _, _,   _ = bb(540, 820, int(hdr_top) + 40, H - 1)   # Predicted brain left x
occ = nw[:, mx0:mx1 + 1].any(axis=1).copy()               # rows with Measured ink
occ[:int(hdr_top + 1.5 * hdr_size)] = False               # drop the header band
bands, _y = [], 0
while _y < H:
    if occ[_y]:
        _y0 = _y
        while _y < H and occ[_y]:
            _y += 1
        bands.append((_y0, _y - 1))
    else:
        _y += 1
bands = [b for b in bands if b[1] - b[0] >= 20]            # ignore specks
bands = sorted(sorted(bands, key=lambda b: b[1] - b[0], reverse=True)[:3])
if len(bands) == 3:
    row_tops = [int(b[0]) for b in bands]
    brain_h = int(min(b[1] - b[0] + 1 for b in bands))     # common height, no overhang
    # exact brain x-extent measured ONLY inside the brain rows, so the scale bar
    # in the top margin can't inflate brain_w (which would stretch FDM wider).
    row_mask = np.zeros(H, bool)
    for _b in bands: row_mask[_b[0]:_b[1] + 1] = True
    _bx = np.where(nw[row_mask][:, 130:430].any(axis=0))[0]
    mx0 = 130 + int(_bx.min()); mx1 = 130 + int(_bx.max())
else:                                                      # fallback: equal thirds
    _, _y0a, _, _y1a = bb(mx0, mx1 + 1, int(hdr_top) + 40, H - 1)
    _step = (_y1a - _y0a + 1) // 3
    row_tops = [int(_y0a), int(_y0a + _step), int(_y0a + 2 * _step)]
    brain_h = int(_step)
brain_w = mx1 - mx0 + 1
pitch = px0 - mx0
print(f"[geometry] brain {brain_w}x{brain_h}  meas_x0={mx0} pred_x0={px0} "
      f"pitch={pitch} row_tops={row_tops}  raster={W}x{H}  nbands={len(bands)}")

# ------------------------------------------------------ render FDM brains (217x273)
def render_brain(part, cmap):
    vmn, vmx = vlim(part)
    val = crop(getattr(fdm_d, part) if part != "abs" else np.abs(fdm_d)).copy()
    val[~sil_c] = np.nan                              # mask to exact silhouette
    cm = plt.get_cmap(cmap).copy(); cm.set_bad("white")
    fig = plt.figure(figsize=(brain_w/100, brain_h/100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    ax.imshow(np.ma.masked_invalid(val), cmap=cm, vmin=vmn, vmax=vmx, aspect="auto")
    fig.canvas.draw()
    out = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy(); plt.close(fig)
    return Image.fromarray(out).resize((brain_w, brain_h))
panels = {p: render_brain(p, cm) for p, cm in [("real", "RdBu_r"), ("imag", "RdBu_r"), ("abs", "viridis")]}

# ------------------------------------------------------ build FDM column + splice
col = Image.new("RGB", (pitch, H), (255, 255, 255))
local_x = mx0 - (px0 - pitch)        # brain offset within its column (== 0: column-left, like Measured)
for part, top in zip(("real", "imag", "abs"), row_tops):
    col.paste(panels[part], (local_x, top))
# FDM header centred over the brain x-range, matched to the Measured header baseline/size
draw = ImageDraw.Draw(col)
font = ImageFont.truetype(fm.findfont(fm.FontProperties(weight="bold")), int(round(hdr_size)))
tb = draw.textbbox((0, 0), "FDM", font=font)
draw.text((local_x + (brain_w-(tb[2]-tb[0]))/2, hdr_top - tb[1]), "FDM", fill=(0, 0, 0), font=font)

out = Image.new("RGB", (W + pitch, H), (255, 255, 255))
out.paste(img.crop((0, 0, px0, H)), (0, 0))                  # labels + Measured + gap
out.paste(col, (px0, 0))                                     # FDM column
out.paste(img.crop((px0, 0, W, H)), (px0 + pitch, 0))        # Predicted + cbars + s.d.
# no suptitle: trim only the whitespace/suptitle band ABOVE the auto-detected
# column headers (hdr_top). The previous hardcoded 130/105 assumed the headers
# began at ~y139; once the base figure is re-rendered with the headers near the
# top (no suptitle), those constants erased the headers AND painted white over
# the top of the Re-row brains. Deriving the trim from hdr_top fixes both.
top_margin = int(round(0.6 * hdr_size))                 # whitespace kept above the header text
top_keep = max(0, int(round(hdr_top)) - top_margin)
ImageDraw.Draw(out).rectangle((0, 0, out.size[0], top_keep), fill=(255, 255, 255))
out = out.crop((0, top_keep, out.size[0], out.size[1]))
out.save(f"{OUTDIR}/fig_with_fdm_0003_70Hz_compx.png")
out.save(f"{OUTDIR}/fig_with_fdm_0003_70Hz_compx.pdf", "PDF", resolution=200.0)

# ------------------------------------------------------ verify silhouette alignment
fsl = np.asarray(panels["abs"]); fnw = ~np.all(fsl > 245, axis=2)
om = nw[row_tops[2]:row_tops[2]+brain_h, mx0:mx0+brain_w]
hh = min(fnw.shape[0], om.shape[0]); ww = min(fnw.shape[1], om.shape[1])
iou = (fnw[:hh, :ww] & om[:hh, :ww]).sum() / max(1, (fnw[:hh, :ww] | om[:hh, :ww]).sum())
print(f"[verify] FDM vs Measured |.| silhouette IoU = {iou:.3f}  (1.0 = identical size/outline)")
print(f"saved {OUTDIR}/fig_with_fdm_0003_70Hz_compx.pdf  size={out.size}")
