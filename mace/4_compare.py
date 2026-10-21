"""
4. Before/after: accuracy and elastic constants of the trained MACE model and of its elastic-constrained
readout refit, on the held-out test sample.

  (a, b) per-atom energy parity, MACE / constrained          (c) energy-error distributions, overlaid
  (d, e) force-component parity, MACE / constrained          (f) C11, C12, C44, B vs target
All errors are test-set RMSEs.  Energies are exact from the design files (E_constrained = E_frozen + D·θ);
forces come from both patched models; C_ij from MACE's analytic stress (±0.5% strain) at each model's own
a0 — independent of the finite-difference rows used in the fit.  Also asserts that the patched model's
energy equals E_frozen + D·θ on a strained, rattled cell.

  python3 4_compare.py            compute from $OUT/mace/results (θ, design files, model) → compare_data.npz + figure
  python3 4_compare.py --replot   only the figure, from published/mace/results/compare_data.npz (no model or data)
Out ($OUT/mace/results): compare_elastic.{pdf,png,txt}; the computing run also saves the constrained model as an
ordinary MACE checkpoint, Al_small_stagetwo_elastic.model.   Env: OUT, N_TEST (3000)
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPLOT = "--replot" in sys.argv
C_BEFORE, C_AFTER = "0.45", "C0"
KEYS = ["C11", "C12", "C44", "B"]
TARGET = {"C11": 116.3, "C12": 64.8, "C44": 30.9, "B": (116.3 + 2 * 64.8) / 3}   # = lib.C_TARGET, lib.B_TARGET

if REPLOT:
    ROOT = os.environ.get("OUT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "repro", "saved"))
    DIR, PREFIX = os.path.join(ROOT, "mace", "results"), ""
    os.makedirs(DIR, exist_ok=True)
    z = np.load(os.path.join(os.environ.get("REPLOT_FROM", os.path.join(os.path.dirname(HERE), "published", "mace")), "results", "compare_data.npz"))   # REPLOT_FROM: another route's mace/
else:
    import lib
    DIR, PREFIX = lib.OUT, ""
    N_TEST = int(os.environ.get("N_TEST", 3000))
    theta = np.loadtxt(os.path.join(DIR, "theta_constrained_elastic.csv"), delimiter=",").ravel()
    before, after = lib.load_calc(), lib.patched_calc(theta)
    import torch
    torch.save(after.models[0], os.path.join(DIR, "Al_small_stagetwo_elastic.model"))   # usable on its own
    # patch check: E_after − E_before = D·(θ − θ0) away from equilibrium
    basis = lib.Basis(lib.load_calc())
    x = lib.strained(lib.fcc(4.0), e1=0.02, e4=0.03); x.rattle(0.05, seed=1)
    Ex = []
    for c in (before, after):
        y = x.copy(); y.calc = c; Ex.append(y.get_potential_energy())
    err = abs(Ex[1] - Ex[0] - basis.D(x) @ (theta - basis.theta0))
    print(f"patch check |ΔE − D·δθ| = {err:.1e} eV"); assert err < 1e-8

    D = np.loadtxt(os.path.join(DIR, "D_test.csv"), delimiter=",")
    E = np.loadtxt(os.path.join(DIR, "E_test.csv"), delimiter=",")
    test = lib.test_frames(N_TEST)
    assert np.allclose([a.info["dft_energy"] for a in test], E[:, 0]), "test frames differ from 2_design.py"

    def forces(calc):
        out = []
        for at in test:
            y = at.copy(); y.calc = calc; out.append(y.get_forces().ravel())
        return np.concatenate(out)
    n = E[:, 3]
    C = {m: lib.stress_strain_cij(c) for m, c in (("before", before), ("after", after))}
    z = dict(e_dft=E[:, 0] / n, e_before=E[:, 1] / n, e_after=(E[:, 2] + D @ theta) / n,
             f_dft=np.concatenate([a.arrays["dft_forces"].ravel() for a in test]),
             f_before=forces(before), f_after=forces(after),
             **{f"{m}_{k}": C[m][k] for m in C for k in ["a0", *KEYS]})
    np.savez(os.path.join(DIR, "compare_data.npz"), **z)

rmse = lambda a, b: 1e3 * np.sqrt(np.mean((a - b) ** 2))
R = {q: (rmse(z[f"{q}_before"], z[f"{q}_dft"]), rmse(z[f"{q}_after"], z[f"{q}_dft"])) for q in ("e", "f")}
C = {m: {k: float(z[f"{m}_{k}"]) for k in ["a0", *KEYS]} for m in ("before", "after")}

lines = [f"MACE (before) vs elastic-constrained readout refit (after); test set {len(z['e_dft'])} frames, "
         f"{len(z['f_dft'])} force components",
         f"{'':8s} {'E RMSE (meV/atom)':>18s} {'F RMSE (meV/Å)':>15s}",
         f"{'before':8s} {R['e'][0]:18.2f} {R['f'][0]:15.2f}",
         f"{'after':8s} {R['e'][1]:18.2f} {R['f'][1]:15.2f}",
         f"{'':8s} {'a0 (Å)':>8s} " + " ".join(f"{k:>7s}" for k in KEYS) + "   (GPa, stress–strain at own a0)"]
for m in ("before", "after"):
    lines.append(f"{m:8s} {C[m]['a0']:8.4f} " + " ".join(f"{C[m][k]:7.1f}" for k in KEYS))
lines.append(f"{'target':8s} {4.032:8.4f} " + " ".join(f"{TARGET[k]:7.1f}" for k in KEYS))
txt = "\n".join(lines)
print(txt)
open(os.path.join(DIR, f"{PREFIX}compare_elastic.txt"), "w").write(txt + "\n")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12, "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11})
fig, axes = plt.subplots(2, 3, figsize=(7.08, 5.2))            # 7.08 in ≈ 510 pt, the MLST text width


def parity(ax, x, y, col, title, unit, r):
    lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
    pad = 0.04 * (hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k-", lw=0.8)
    ax.plot(x, y, ".", ms=2, alpha=0.4, color=col, rasterized=True)
    ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(lo - pad, hi + pad)
    ax.set_title(title, loc="left")
    ax.text(0.05, 0.95, f"RMSE\n{r:.1f} {unit}", transform=ax.transAxes, va="top")


for ax, q, col, t in ((axes[0, 0], "e_before", C_BEFORE, "(a) MACE"), (axes[0, 1], "e_after", C_AFTER, "(b) constrained")):
    parity(ax, z["e_dft"], z[q], col, t, "meV/atom", rmse(z[q], z["e_dft"]))
    ax.set_xlabel(r"$E_\mathrm{DFT}$ (eV/atom)"); ax.set_ylabel(r"$E$ (eV/atom)")
for ax, q, col, t in ((axes[1, 0], "f_before", C_BEFORE, "(d) MACE"), (axes[1, 1], "f_after", C_AFTER, "(e) constrained")):
    parity(ax, z["f_dft"], z[q], col, t, "meV/Å", rmse(z[q], z["f_dft"]))
    ax.set_xlabel(r"$F_\mathrm{DFT}$ (eV/Å)"); ax.set_ylabel(r"$F$ (eV/Å)")

ax = axes[0, 2]
err_b, err_a = 1e3 * (z["e_before"] - z["e_dft"]), 1e3 * (z["e_after"] - z["e_dft"])
lim = np.percentile(np.abs(np.concatenate([err_b, err_a])), 99.5)
bins = np.linspace(-lim, lim, 51)
nb, _, _ = ax.hist(err_b, bins=bins, color=C_BEFORE, alpha=0.55, label="MACE")
ax.hist(err_a, bins=bins, histtype="step", lw=1.8, color=C_AFTER, label="constrained")
ax.set_xlabel(r"$\Delta E$ (meV/atom)"); ax.set_ylabel("frames")
ax.set_ylim(0, 1.6 * max(nb.max(), np.histogram(err_a, bins=bins)[0].max()))   # headroom for the legend
ax.set_title("(c) energy errors", loc="left")
ax.legend(frameon=False, loc="upper left", handlelength=1.2)

ax = axes[1, 2]
x, w = np.arange(len(KEYS)), 0.38
ax.bar(x - w / 2, [C["before"][k] for k in KEYS], w, color=C_BEFORE, label="MACE")
ax.bar(x + w / 2, [C["after"][k] for k in KEYS], w, color=C_AFTER, label="constrained")
ax.plot(x, [TARGET[k] for k in KEYS], "k_", ms=14, mew=2.2, ls="none", label="target")
ax.set_ylim(0, 1.65 * max(max(C[m][k] for m in C) for k in KEYS))
ax.set_xticks(x, ["$C_{11}$", "$C_{12}$", "$C_{44}$", "$B$"]); ax.set_ylabel("GPa")
ax.set_title("(f) $C_{ij}$ and $B$", loc="left")
ax.legend(frameon=False, loc="upper right", handlelength=1.0, borderaxespad=0.2)
for a in axes.flat:
    a.set_box_aspect(1)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(DIR, f"{PREFIX}compare_elastic.{ext}"), dpi=200)
print(f"→ {os.path.join(DIR, PREFIX + 'compare_elastic')}.pdf/.png")
