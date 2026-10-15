"""
p5. Figures from eval_data.npz (p4), each its own PDF and PNG, built at half the MLST text width
(3.45 in ≈ 250 pt) with 10–12 pt text so they drop in two to a row without rescaling.

  parity_{E,F}_{constrained,unconstrained}       reference model vs DFT, ±σ of its committee, RMSE printed
  uncertainty_vs_error_{E,F}_{constrained,unconstrained}   |error| of the reference model vs committee σ,
                                                  with |error| = σ and 2σ, and the binned RMS error
  calibration_{E,F}                              observed vs expected fraction with |error| ≤ zσ, both committees
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
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
REPLOT = "--replot" in sys.argv
ROOT = os.environ.get("OUT", os.path.join(os.path.dirname(os.path.dirname(HERE)), "repro", "saved"))   # the route root
OUT = os.path.join(ROOT, "mace", "pops"); os.makedirs(OUT, exist_ok=True)
SRC = os.path.join(os.path.dirname(os.path.dirname(HERE)), "published", "mace", "pops") if REPLOT else OUT
PREFIX = ""
d = np.load(os.path.join(SRC, "eval_data.npz"))
W = 3.45
plt.rcParams.update({"font.size": 11, "axes.labelsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10,
                     "legend.fontsize": 9, "axes.titlesize": 11})
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
        # uncertainty vs error, log–log, in meV
        e, s = 1e3 * np.abs(f - y), 1e3 * sd
        fig, ax = plt.subplots(figsize=(W, W))
        ax.loglog(s, np.maximum(e, 1e-3), ".", ms=2, alpha=0.3, color=COL[name], rasterized=True)
        lo, hi = np.percentile(s, 0.1) / 3, max(np.percentile(s, 99.9), np.percentile(e, 99.9)) * 3
        xs = np.geomspace(lo, hi, 50)
        ax.loglog(xs, xs, "k-", lw=0.8, label="|error| = σ")
        ax.loglog(xs, 2 * xs, "k--", lw=0.8, label="|error| = 2σ")
        edges = np.quantile(s, np.linspace(0, 1, 16))
        idx = np.clip(np.digitize(s, edges[1:-1]), 0, 14)
        bs = [s[idx == b].mean() for b in range(15)]
        be = [np.sqrt(np.mean(e[idx == b] ** 2)) for b in range(15)]
        ax.loglog(bs, be, "o-", color="k", ms=3, lw=1.2, mfc=COL[name], label="binned RMS error")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
        ax.set_xlabel(f"committee σ ({UNIT[q][1]})"); ax.set_ylabel(f"|error| ({UNIT[q][1]})")
        ax.legend(frameon=False, loc="upper left")
        save(fig, f"uncertainty_vs_error_{q}_{name}")

zs = np.linspace(0, 3.5, 71)
for q in ("E", "F"):
    fig, ax = plt.subplots(figsize=(W, W))
    for name in COL:
        y, f, sd = get(name, q)
        z = np.abs(f - y) / np.maximum(sd, 1e-12)
        ax.plot(2 * norm.cdf(zs) - 1, [np.mean(z <= zz) for zz in zs], color=COL[name], lw=2, label=LABEL[name])
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_xlabel("expected fraction"); ax.set_ylabel(f"observed fraction ({SYM[q]})")
    ax.legend(frameon=False, loc="upper left")
    save(fig, f"calibration_{q}")

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
    ax.set_title(key, fontsize=10)
    ax.tick_params(labelsize=9)
fig.tight_layout(rect=(0, 0.03, 1, 1))
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"{PREFIX}cij_spread.{ext}"), dpi=300)
print(f"  {PREFIX}cij_spread.pdf/.png")
