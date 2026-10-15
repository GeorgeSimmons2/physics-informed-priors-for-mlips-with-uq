"""
p4. Evaluate the constrained and unconstrained POPS committees of p3, each against its reference model:
the elastic-constrained readout θc ($OUT/mace/results/theta_constrained_elastic.csv) and MACE's trained readout θ0
($OUT/mace/results/theta0.csv).

  Test set (3,000 held-out frames, all force components, from the rows of p1), per committee and quantity:
  the reference model's RMSE, the committee spread σ, the fraction of DFT values inside the committee's
  min–max envelope, and the calibration of the reference model's error against σ (|error| ≤ zσ).
  Elastic constants: relaxed a0 and stress–strain C11, C12, C44 of every member and of both reference models.
Out ($OUT/mace/pops): uq_test.txt, cij_members.csv, eval_data.npz (everything p5_figures.py plots)
Run: python3 p4_evaluate.py      Env: OUT, THREADS
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import lib  # noqa: E402

OUT, MAIN = lib.POPS, lib.OUT
load = lambda f: np.loadtxt(f, delimiter=",")
NAMES = ("constrained", "unconstrained")
com = {n: load(f"{OUT}/committee_{n}.csv") for n in NAMES}
ref = {"constrained": load(f"{MAIN}/theta_constrained_elastic.csv").ravel(), "unconstrained": load(f"{MAIN}/theta0.csv").ravel()}
t = np.load(os.path.join(lib.ROWS, "rows_test.npz"))

# ── test-set UQ, against each committee's reference model ──────────────────
data, lines = {}, [f"POPS committees on {len(t['yE'])} held-out test frames, {len(t['yF'])} force components "
                   f"({len(com['constrained'])} members each), errors of each committee's reference model "
                   "(constrained: θc; unconstrained: MACE as trained); energies meV/atom, forces meV/Å",
                   f"{'':14s} {'q':2s} {'RMSE ref':>9s} {'mean σ':>8s} {'RMSE/σ':>7s} {'in envelope':>11s} {'|err|≤σ':>8s} {'|err|≤2σ':>9s}"]
for name in NAMES:
    for q, X, y in (("E", t["XE"], t["yE"]), ("F", t["XF"], t["yF"])):
        P = X @ com[name].T                              # points × members
        sd, f = P.std(1), X @ ref[name]
        err = f - y
        inside = np.mean((y >= P.min(1)) & (y <= P.max(1)))
        z = np.abs(err) / np.maximum(sd, 1e-12)
        data[f"{name}_{q}_y"], data[f"{name}_{q}_f"], data[f"{name}_{q}_sd"] = y, f, sd
        rmse = 1e3 * np.sqrt(np.mean(err**2))
        lines.append(f"{name:14s} {q:2s} {rmse:9.2f} {1e3 * sd.mean():8.2f} {rmse / (1e3 * sd.mean()):7.2f} "
                     f"{inside:11.3f} {np.mean(z <= 1):8.3f} {np.mean(z <= 2):9.3f}")
lines.append("(Gaussian expectation: |err|≤σ 0.683, |err|≤2σ 0.954; 'in envelope' = DFT inside the committee's min–max)")
txt = "\n".join(lines)
print(txt)
open(os.path.join(OUT, "uq_test.txt"), "w").write(txt + "\n")

# ── C_ij and a0 of every member and of both reference models ──────────────
calc = lib.load_calc()


def set_theta(theta):
    """Write θ into calc's readout weights (as lib.patched_calc) and drop ASE's result cache."""
    import torch
    m = calc.models[0]
    ro = m.readouts
    n0, n1 = ro[0].linear.irreps_in.count("0e"), ro[1].linear_2.irreps_in.dim
    scale = float(m.scale_shift.scale.reshape(-1)[0])
    lib._set_lin(ro[0].linear, n0, theta[:n0] / scale)
    lib._set_lin(ro[1].linear_2, n1, theta[n0:n0 + n1] / scale)
    with torch.no_grad():
        m.scale_shift.shift.fill_(float(theta[-1]))
    calc.results, calc.atoms = {}, None


rows = []
for j, name in enumerate(NAMES):
    for i, theta in [(-1, ref[name])] + list(enumerate(com[name])):          # member −1 = reference model
        set_theta(theta)
        r = lib.stress_strain_cij(calc)
        rows.append([j, i, r["a0"], r["C11"], r["C12"], r["C44"], r["B"]])
rows = np.array(rows)
np.savetxt(os.path.join(OUT, "cij_members.csv"), rows, delimiter=",",
           header="committee (0 constrained, 1 unconstrained),member (-1 = reference model),a0 (A),C11,C12,C44,B (GPa)")
for j, name in enumerate(NAMES):
    r, r0 = rows[(rows[:, 0] == j) & (rows[:, 1] >= 0)], rows[(rows[:, 0] == j) & (rows[:, 1] < 0)][0]
    print(f"{name:14s} reference a0 {r0[2]:.4f} C11 {r0[3]:.1f} C12 {r0[4]:.1f} C44 {r0[5]:.1f} | members a0 "
          f"{r[:, 2].min():.4f}–{r[:, 2].max():.4f}  C11 {r[:, 3].min():.1f}–{r[:, 3].max():.1f}  "
          f"C12 {r[:, 4].min():.1f}–{r[:, 4].max():.1f}  C44 {r[:, 5].min():.1f}–{r[:, 5].max():.1f} GPa")
np.savez(os.path.join(OUT, "eval_data.npz"), cij=rows, **data)
print(f"→ {OUT}/eval_data.npz")
