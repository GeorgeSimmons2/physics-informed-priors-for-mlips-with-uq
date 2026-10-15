"""
p6. The reference model's |error| on the held-out test set against the committee's largest deviation from that
reference model toward the DFT value, on linear axes (θ_ref = θc for the constrained committee, θ0, MACE as trained,
for the unconstrained one): max_m (a·θ_m − a·θ_ref) where DFT lies above the reference prediction, max_m
(a·θ_ref − a·θ_m) where it lies below, 0 if no member goes that way. A point above the line deviation = |error| is an
error some member reaches: a DFT value inside the committee's min–max range (p4's envelope coverage).
The uncertainty here is that deviation, not the committee σ of p4 and p5 (both left unchanged).

  maxdev_test.txt                                   coverage (fraction with |error| ≤ deviation toward DFT) per committee
  maxdev_vs_error_{E,F}_{constrained,unconstrained}  one PDF and PNG each, 3.45 in square; both axes run to the
                                                    99.5th percentile, and the points beyond are counted on the panel
  maxdev_data.npz                                   y, reference prediction and deviation toward DFT, for --replot
In:  $OUT/mace/pops/rows/rows_test.npz (p1), $OUT/mace/pops/committee_*.csv (p3), $OUT/mace/results/theta*.csv
Run: python3 p6_maxdev_vs_error.py             $OUT/mace/pops in and out
     python3 p6_maxdev_vs_error.py --replot    from published/mace/pops/maxdev_data.npz, figures to $OUT/mace/pops
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPLOT = "--replot" in sys.argv
ROOT = os.environ.get("OUT", os.path.join(os.path.dirname(os.path.dirname(HERE)), "repro", "saved"))   # the route root
OUT = os.path.join(ROOT, "mace", "pops"); os.makedirs(OUT, exist_ok=True)
PREFIX = ""
NAMES = ("constrained", "unconstrained")

if REPLOT:
    d = dict(np.load(os.path.join(os.path.dirname(os.path.dirname(HERE)), "published", "mace", "pops", "maxdev_data.npz")))
else:
    load = lambda f: np.loadtxt(f, delimiter=",")
    MAIN = os.path.join(ROOT, "mace", "results")
    com = {n: load(f"{OUT}/committee_{n}.csv") for n in NAMES}
    ref = {"constrained": load(f"{MAIN}/theta_constrained_elastic.csv").ravel(), "unconstrained": load(f"{MAIN}/theta0.csv").ravel()}
    t = np.load(os.path.join(ROOT, "mace", "pops", "rows", "rows_test.npz"))
    d, lines = {}, [f"largest deviation of the 100 members from each committee's reference model toward DFT, on {len(t['yE'])} test "
                    f"frames and {len(t['yF'])} force components (energies meV/atom, forces meV/Å)",
                    f"{'':14s} {'q':2s} {'RMSE ref':>9s} {'median dev':>11s} {'|err| ≤ max dev':>16s}"]
    for name in NAMES:
        for q, X, y in (("E", t["XE"], t["yE"]), ("F", t["XF"], t["yF"])):
            f = X @ ref[name]
            D = X @ com[name].T - f[:, None]                            # member − reference, points × members
            dev = np.where(y >= f, D.max(1), -D.min(1)).clip(min=0)     # largest deviation toward the DFT value
            d[f"{name}_{q}_y"], d[f"{name}_{q}_f"], d[f"{name}_{q}_dev"] = y, f, dev
            lines.append(f"{name:14s} {q:2s} {1e3 * np.sqrt(np.mean((f - y) ** 2)):9.2f} {1e3 * np.median(dev):11.2f} "
                         f"{np.mean(np.abs(f - y) <= dev):16.3f}")
    txt = "\n".join(lines)
    print(txt)
    open(os.path.join(OUT, "maxdev_test.txt"), "w").write(txt + "\n")
    np.savez(os.path.join(OUT, "maxdev_data.npz"), **d)

W = 3.45
plt.rcParams.update({"font.size": 11, "axes.labelsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10,
                     "legend.fontsize": 9, "axes.titlesize": 11})
COL = {"constrained": "C0", "unconstrained": "C1"}
UNIT = {"E": "meV/atom", "F": "meV/Å"}
BOX = dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5)
for name in NAMES:
    for q in ("E", "F"):
        e = 1e3 * np.abs(d[f"{name}_{q}_f"] - d[f"{name}_{q}_y"])
        m = 1e3 * d[f"{name}_{q}_dev"]
        lim = 1.05 * np.percentile(np.concatenate([e, m]), 99.5)
        fig, ax = plt.subplots(figsize=(W, W))
        ax.fill_between([0, lim], [0, lim], lim, color="0.5", alpha=0.08, lw=0)      # max deviation ≥ |error|: covered
        ax.plot(e, m, ".", ms=2, alpha=0.3, color=COL[name], rasterized=True)
        ax.plot([0, lim], [0, lim], "k-", lw=0.8)
        ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_aspect("equal")
        ax.set_xlabel(f"|error| ({UNIT[q]})"); ax.set_ylabel(f"deviation toward DFT ({UNIT[q]})")
        ax.text(0.05, 0.95, f"covered: {100 * np.mean(m >= e):.0f}%", transform=ax.transAxes, va="top", bbox=BOX)
        ax.text(0.05, 0.86, f"{np.sum((e > lim) | (m > lim))} of {len(e)} beyond the axes", transform=ax.transAxes,
                va="top", fontsize=8, color="0.4", bbox=BOX)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(OUT, f"{PREFIX}maxdev_vs_error_{q}_{name}.{ext}"), dpi=300)
        plt.close(fig)
        print(f"  {PREFIX}maxdev_vs_error_{q}_{name}.pdf/.png")
