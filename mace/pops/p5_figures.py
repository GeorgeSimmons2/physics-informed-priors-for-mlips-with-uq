"""
p5. Figures from eval_data.npz (p4), each its own PDF and PNG, built at half the MLST text width
(255 pt) with 11–13 pt text so they drop in two to a row without rescaling.

  parity_{E,F}_{constrained,unconstrained}       reference model vs DFT, ±σ of its committee, RMSE printed
  cij_spread                                     a0, C11, C12, C44 of every member, both reference models marked
Run: python3 p5_figures.py             from $OUT/mace/pops/eval_data.npz, figures beside it
     python3 p5_figures.py --replot    from published/mace/pops/eval_data.npz, figures to $OUT/mace/pops
"""
import sys
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPLOT = "--replot" in sys.argv
ROOT = os.environ.get("OUT", os.path.join(os.path.dirname(os.path.dirname(HERE)), "repro", "saved"))   # the route root
OUT = os.path.join(ROOT, "mace", "pops"); os.makedirs(OUT, exist_ok=True)
SRC = os.path.join(os.environ.get("REPLOT_FROM", os.path.join(os.path.dirname(os.path.dirname(HERE)), "published", "mace")), "pops") if REPLOT else OUT   # REPLOT_FROM: a route's mace/
PREFIX = ""
d = np.load(os.path.join(SRC, "eval_data.npz"))
W = 255 / 72                                    # half the MLST text width (510 pt), in inches
plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12, "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11})
COL = {"constrained": "C0", "unconstrained": "C1"}
LABEL = {"constrained": "constrained", "unconstrained": "MACE"}
UNIT = {"E": ("eV/atom", "meV/atom"), "F": ("eV/Å", "meV/Å")}
SYM = {"E": "E", "F": "F"}


def save(fig, name):
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{PREFIX}{name}.{ext}"), dpi=300)
    plt.close(fig)
    print(f"  {PREFIX}{name}.pdf/.png")


def get(name, q):
    return d[f"{name}_{q}_y"], d[f"{name}_{q}_f"], d[f"{name}_{q}_sd"]


for name in COL:
    for q in ("E", "F"):
        y, f, sd = get(name, q)
        rmse = 1e3 * np.sqrt(np.mean((f - y) ** 2))
        # parity: reference model, ±σ of its committee
        fig, ax = plt.subplots(figsize=(W, W))
        ax.errorbar(y, f, yerr=sd, fmt=".", ms=2, lw=0.5, elinewidth=0.5, color=COL[name], alpha=0.5, rasterized=True)
        lo, hi = min(y.min(), f.min()), max(y.max(), f.max())
        pad = 0.04 * (hi - lo)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k-", lw=0.8)
        ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(lo - pad, hi + pad); ax.set_aspect("equal")
        ax.set_xlabel(f"${SYM[q]}_\\mathrm{{DFT}}$ ({UNIT[q][0]})"); ax.set_ylabel(f"${SYM[q]}$, {LABEL[name]} ({UNIT[q][0]})")
        ax.text(0.05, 0.95, f"RMSE {rmse:.1f} {UNIT[q][1]}", transform=ax.transAxes, va="top")
        save(fig, f"parity_{q}_{name}")

cij = d["cij"]
TOL = 1e-3
fig, axes = plt.subplots(2, 2, figsize=(W, W))
rng = np.random.default_rng(0)
for ax, (n, key, ref) in zip(axes.flat, ((2, "$a_0$ (Å)", 4.032), (3, "$C_{11}$ (GPa)", 116.3),
                                       (4, "$C_{12}$ (GPa)", 64.8), (5, "$C_{44}$ (GPa)", 30.9))):
    for j, name in enumerate(COL):
        m = (cij[:, 0] == j) & (cij[:, 1] >= 0)
        r0 = cij[(cij[:, 0] == j) & (cij[:, 1] < 0)][0]
        ax.plot(np.full(m.sum(), j) + rng.uniform(-0.18, 0.18, m.sum()), cij[m, n], ".", ms=2.5, alpha=0.6, color=COL[name])
        ax.plot([j], [r0[n]], marker="*", ms=11, color=COL[name], mec="k", mew=0.6, ls="none")
    if n > 2:
        ax.axhspan(ref * (1 - TOL), ref * (1 + TOL), color="k", alpha=0.25, lw=0)
    ax.axhline(ref, color="k", lw=0.7)
    ax.set_xticks([0, 1], ["constr.", "MACE"]); ax.set_xlim(-0.6, 1.6)
    ax.set_title(key)
fig.tight_layout(rect=(0, 0.03, 1, 1))
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"{PREFIX}cij_spread.{ext}"), dpi=300)
print(f"  {PREFIX}cij_spread.pdf/.png")
