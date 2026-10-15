"""
1. Constraint rows: derivatives of the readout basis D for FCC Al (4-atom cubic cell) at the target a0.

Central differences (H_A in a, H_E in strain; cell' = cell·(I + ε)ᵀ, Voigt ε₄ = 2ε_yz):
    dD_da, d2D_da2, d2D_de11, d2D_de12, d2D_de44
so that for any θ:  C_ij = (160.2/V) d2D_deij·θ (GPa),  P ∝ dD_da·θ,  B = (160.2/9a) d2D_da2·θ.
E_frozen = E_MACE − D·θ0 = Σ E0 is constant, so its derivatives vanish (checked; written to frozen.csv).
Checks printed: E_frozen constant; rows·θ0 reproduce MACE's own C_ij (compare ASE stress–strain at A0).
Out: $OUT/{theta0,D0,dD_da,d2D_da2,d2D_de11,d2D_de12,d2D_de44,frozen}.csv, meta.txt
"""
import os

import numpy as np
from ase.units import GPa

import lib

H_A, H_E = 1e-3, 1e-4
calc = lib.load_calc()
basis = lib.Basis(calc)
a_model, _ = lib.relaxed_a0(calc)
c0 = lib.fcc(lib.A0)
V = c0.get_volume()
print(f"model {lib.MODEL_FILE}\nmodel relaxed a0 = {a_model:.5f} Å; rows at A0 = {lib.A0:.5f} Å; {basis.theta0.size} columns")


def E_frozen(at):
    x = at.copy(); x.calc = calc
    return x.get_potential_energy() - basis.D(at) @ basis.theta0


def derivs(f):
    f0, fp, fm = f(c0), f(lib.fcc(lib.A0 + H_A)), f(lib.fcc(lib.A0 - H_A))
    h = H_E
    return [(fp - fm) / (2 * H_A),
            (fp - 2 * f0 + fm) / H_A**2,
            (f(lib.strained(c0, e1=h)) - 2 * f0 + f(lib.strained(c0, e1=-h))) / h**2,
            (f(lib.strained(c0, e1=h, e2=h)) - f(lib.strained(c0, e1=h, e2=-h))
             - f(lib.strained(c0, e1=-h, e2=h)) + f(lib.strained(c0, e1=-h, e2=-h))) / (4 * h**2),
            (f(lib.strained(c0, e4=h)) - 2 * f0 + f(lib.strained(c0, e4=-h))) / h**2]


for a in (lib.A0, lib.A0 + 0.05):
    print(f"  E_frozen at a = {a:.3f}: {E_frozen(lib.fcc(a)):.10f} eV   (must be the same constant)")
rows = derivs(basis.D)
frozen = np.array(derivs(E_frozen))
k = lib.EV_A3_TO_GPA / V
tot = frozen + np.array([r @ basis.theta0 for r in rows])
print(f"  MACE from rows at A0: P {lib.EV_A3_TO_GPA * tot[0] / (3 * lib.A0**2):+.3f} GPa  C11 {k * tot[2]:.1f}  "
      f"C12 {k * tot[3]:.1f}  C44 {k * tot[4]:.1f}  B {lib.EV_A3_TO_GPA * tot[1] / (9 * lib.A0):.1f} GPa")
h, s = 5e-3, {}
for lab, kw in (("p1", dict(e1=h)), ("m1", dict(e1=-h)), ("p4", dict(e4=h)), ("m4", dict(e4=-h))):
    x = lib.strained(c0, **kw); x.calc = calc; s[lab] = x.get_stress()
s1, s4 = (s["p1"] - s["m1"]) / (2 * h), (s["p4"] - s["m4"]) / (2 * h)
print(f"  ASE stress–strain at A0: C11 {s1[0] / GPa:.1f}  C12 {s1[1] / GPa:.1f}  C44 {s4[3] / GPa:.1f} GPa"
      "   (differs from the rows by O(P) while P(A0) ≠ 0)")

for name, arr in zip(["theta0", "D0", "dD_da", "d2D_da2", "d2D_de11", "d2D_de12", "d2D_de44", "frozen"],
                     [basis.theta0, basis.D(c0), *rows, frozen]):
    np.savetxt(os.path.join(lib.OUT, f"{name}.csv"), np.asarray(arr)[None, :], delimiter=",")
with open(os.path.join(lib.OUT, "meta.txt"), "w") as f:
    f.write(f"a0 {lib.A0:.10f}\nV {V:.10f}\na_model {a_model:.10f}\nn_atoms {len(c0)}\nH_A {H_A}\nH_E {H_E}\n")
print(f"→ {lib.OUT}")
