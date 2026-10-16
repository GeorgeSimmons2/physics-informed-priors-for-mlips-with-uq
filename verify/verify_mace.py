"""
verify/verify_mace.py — how closely a route reproduces the published MACE results at a0 = 4.032 Å.

saved route (the published model): θ0 must match exactly, θc and θ_LS to a relative 1e-6 (one QP,
pinned OSQP schedule), the test RMSEs, a0 and C_ij to the published digits, the POPS committees as
distributions (z-scores, as verify/lib_verify.jl), and their test coverage to 1 percentage point.
refit route (MACE retrained from the DFT data): the physics is judged — a0 and C_ij on target, test RMSEs
within 15% of the published ones, coverage within 3 points — and the retrained weights are compared with the
published model's.  Training is deterministic on fixed hardware and threads, so they can be identical; when the
retrained readout θ0 equals the published one, the POPS committees are compared as on the saved route.

Run: .venv/bin/python verify/verify_mace.py saved|refit       (markdown table to stdout)
"""
import os
import re
import sys

import numpy as np

route = sys.argv[1] if len(sys.argv) > 1 else "saved"
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
O, P = os.path.join(R, "repro", route, "mace"), os.path.join(R, "published", "mace")
rows = []
Z_TOL = 4.5


def add(label, tier, ok, detail):
    rows.append((label, tier, "info" if tier == "info" else "  — " if ok is None else ("PASS" if ok else "FAIL"), detail))


def have(*paths):
    return all(os.path.isfile(p) for p in paths)


load = lambda p: np.loadtxt(p, delimiter=",", ndmin=1)
nums = lambda p: [float(x) for x in re.findall(r"[-+]?\d+\.\d+(?:e[-+]?\d+)?", open(p).read())]


def lines_with(p, key):
    return [l for l in open(p) if l.startswith(key)]


# ── mean model ───────────────────────────────────────────────────────────────
res, pub = os.path.join(O, "results"), os.path.join(P, "results")
if route == "saved":
    for f, tol, tier in (("theta0.csv", 0.0, "exact"), ("theta_constrained_elastic.csv", 1e-6, "converged"),
                         ("theta_ls_elastic.csv", 1e-6, "converged")):
        a, b = os.path.join(pub, f), os.path.join(res, f)
        if not have(a, b):
            add(f, tier, None, f"not produced yet: {b}"); continue
        x, y = load(a).ravel(), load(b).ravel()
        rel = np.linalg.norm(x - y) / np.linalg.norm(x)
        add(f, tier, rel <= tol, f"‖Δθ‖/‖θ‖ = {rel:.2e}, max|Δ| = {np.abs(x - y).max():.2e} (tol {tol:.0e})")
    for f, key in (("compare_elastic.txt", None), ("elastic_matscipy.txt", None)):
        a, b = os.path.join(pub, f), os.path.join(res, f)
        if not have(a, b):
            add(f, "converged", None, f"not produced yet: {b}"); continue
        x, y = np.array(nums(a)), np.array(nums(b))
        ok = len(x) == len(y) and np.abs(x - y).max() <= 0.1
        add(f + ": RMSEs, a0, C_ij", "converged", ok,
            f"{len(x)} numbers, max|Δ| = {np.abs(x - y).max() if len(x) == len(y) else float('nan'):.3g} (tol 0.1 in the printed units)")
else:
    b = os.path.join(res, "compare_elastic.txt")
    if not have(b):
        add("compare_elastic.txt", "physics", None, f"not produced yet: {b}")
    else:
        after = [l for l in open(b) if l.startswith("after")]
        rm_pub = [float(v) for v in lines_with(os.path.join(pub, "compare_elastic.txt"), "after")[0].split()[1:3]]
        rm = [float(v) for v in after[0].split()[1:3]]
        cij = [float(v) for v in after[1].split()[1:6]]
        target = [4.032, 116.3, 64.8, 30.9, 82.0]
        ok_c = all(abs(c - t) / t <= 2e-3 for c, t in zip(cij, target))
        ok_r = all(r <= 1.15 * p for r, p in zip(rm, rm_pub))
        add("retrained + constrained: a0 and C_ij on target", "physics", ok_c,
            "a0 {:.4f} Å, C11/C12/C44/B {:.1f}/{:.1f}/{:.1f}/{:.1f} GPa (target 4.032, 116.3/64.8/30.9/82.0)".format(*cij))
        add("retrained + constrained: test RMSE", "physics", ok_r,
            f"E {rm[0]:.2f} meV/atom, F {rm[1]:.1f} meV/Å (published {rm_pub[0]:.2f}, {rm_pub[1]:.1f}; tol +15%)")
    b = os.path.join(res, "elastic_matscipy.txt")
    if have(b):
        c = [l for l in open(b) if l.startswith("constrained")][0]
        v = [float(x) for x in re.findall(r"\d+\.\d+", c)][:4]
        add("retrained + constrained: matscipy a0, C_ij", "physics", abs(v[0] - 4.032) < 5e-4 and abs(v[1] - 116.3) < 1 and abs(v[2] - 64.8) < 1 and abs(v[3] - 30.9) < 1,
            f"a0 {v[0]:.4f} Å, C11/C12/C44 {v[1]:.1f}/{v[2]:.1f}/{v[3]:.1f} GPa")

# ── the retrained network (refit route) ─────────────────────────────────────
same_network = route == "saved"
if route == "refit":
    mr = os.path.join(R, "repro", route, "mace", "models", "Al_small_stagetwo.model")
    if have(mr):
        import torch
        sa = torch.load(os.path.join(R, "mace", "models", "Al_small_stagetwo.model"), map_location="cpu", weights_only=False).state_dict()
        sb = torch.load(mr, map_location="cpu", weights_only=False).state_dict()
        d = max((sa[k].double() - sb[k].double()).abs().max().item() for k in sa if sa[k].numel()) if sa.keys() == sb.keys() else float("inf")
        add("retrained network: weights vs the published model", "info", None,
            f"{sum(v.numel() for v in sa.values())} weights, max|Δ| = {d:.3e}" + (" — bit-identical" if d == 0 else ""))
    a, b = os.path.join(pub, "theta0.csv"), os.path.join(res, "theta0.csv")
    same_network = have(a, b) and np.array_equal(load(a), load(b))

# ── POPS committees ──────────────────────────────────────────────────────────
pops, ppub = os.path.join(O, "pops"), os.path.join(P, "pops")
for name in ("constrained", "unconstrained"):
    a, b = os.path.join(ppub, f"committee_{name}.csv"), os.path.join(pops, f"committee_{name}.csv")
    if not have(a, b):
        add(f"POPS {name} committee", "statistical", None, f"not produced yet: {b}"); continue
    if not same_network:
        add(f"POPS {name} committee", "statistical", None, "retrained network differs: θ not comparable; see coverage and C_ij spread"); continue
    A, B = load(a), load(b)
    n = A.shape[0]
    mA, mB, sA, sB = A.mean(0), B.mean(0), A.std(0, ddof=1), B.std(0, ddof=1)
    sb = np.maximum((sA + sB) / 2, np.finfo(float).eps)
    zm = np.max(np.abs(mA - mB) / (sb * np.sqrt(2 / n)))
    zs = np.max(np.abs(sA - sB) / sb * np.sqrt(2 * (n - 1)))
    add(f"POPS {name} committee", "statistical", max(zm, zs) <= Z_TOL,
        f"n={n}  worst z(mean) = {zm:.2f}, worst z(std) = {zs:.2f}  (max|Δθ| = {np.abs(A - B).max():.2e})")
a, b = os.path.join(ppub, "maxdev_test.txt"), os.path.join(pops, "maxdev_test.txt")
if have(a, b):
    tol = 1.0 if route == "saved" else 3.0
    for line_a, line_b in zip([l for l in open(a) if l.startswith(("constrained", "unconstrained"))],
                              [l for l in open(b) if l.startswith(("constrained", "unconstrained"))]):
        ca, cb = 100 * float(line_a.split()[-1]), 100 * float(line_b.split()[-1])
        add(f"POPS coverage, {' '.join(line_a.split()[:2])} (DFT inside the committee range)", "statistical",
            abs(ca - cb) <= tol, f"{ca:.1f} % → {cb:.1f} % (tol {tol:.0f} points)")
else:
    add("POPS coverage", "statistical", None, f"not produced yet: {b}")
a, b = os.path.join(ppub, "cij_members.csv"), os.path.join(pops, "cij_members.csv")
if have(a, b):
    A, B = load(a), load(b)
    for j, name in enumerate(("constrained", "unconstrained")):
        sa = A[(A[:, 0] == j) & (A[:, 1] >= 0)][:, 3].std(); sb_ = B[(B[:, 0] == j) & (B[:, 1] >= 0)][:, 3].std()
        ok = (sb_ < 0.1) if name == "constrained" else (0.3 * sa <= sb_ <= 3 * sa)
        add(f"POPS {name}: spread of C11 over members", "statistical", ok, f"{sa:.3g} → {sb_:.3g} GPa")
else:
    add("POPS C_ij spread", "statistical", None, f"not produced yet: {b}")

print(f"## MACE: route `{route}` against published/\n")
print("| quantity | tier | verdict | detail |\n|---|---|---|---|")
for r in rows:
    print(f"| {r[0]} | `{r[1]}` | **{r[2]}** | {r[3]} |")
print(f"\n{sum(r[2] == 'PASS' for r in rows)} passed, {sum(r[2] == 'FAIL' for r in rows)} failed, "
      f"{sum(r[2].strip() == '—' for r in rows)} not yet produced or not comparable, {sum(r[2] == 'info' for r in rows)} reported for information.")
