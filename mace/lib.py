"""
Shared by the mace/ scripts: the trained Al MACE model, its linear readout basis, patching, and the paths.

The readout basis writes MACE's energy exactly as a linear function of the last-layer readout weights:

    E = Σ_atoms E0  +  D·θ,      D = Σ_atoms [ h⁰[0e] (64), σ(W₁h¹) (16), 1 ]      (81 columns)

h⁰ are the layer-0 scalars, σ(W₁h¹) the hidden units of the layer-1 MLP readout, and
θ = [scale·w₀, scale·w₂, shift] are the trained readout weights (θ0).  Everything up to D is frozen.
Changing θ is a change of MACE's own weights: patched_calc(θ) writes θ back into the model.

Paths: OUT is the route root (default repro/saved); the mean-model results go to $OUT/mace/results, POPS to
$OUT/mace/pops (rows in $OUT/mace/pops/rows).  The published outputs are under published/mace/.
Env: OUT, MODEL_FILE (default mace/models/Al_small_stagetwo.model, the published model), A0 (4.032 Å), THREADS
"""
import os

import numpy as np
import torch
from ase.build import bulk

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data", "Al")
MODEL_FILE = os.environ.get("MODEL_FILE", os.path.join(HERE, "models", "Al_small_stagetwo.model"))
ROOT = os.environ.get("OUT", os.path.join(os.path.dirname(HERE), "repro", "saved"))
OUT = os.path.join(ROOT, "mace", "results")
POPS = os.path.join(ROOT, "mace", "pops")
ROWS = os.path.join(POPS, "rows")
PUBLISHED = os.path.join(os.path.dirname(HERE), "published", "mace")
for _d in (OUT, POPS, ROWS):
    os.makedirs(_d, exist_ok=True)
torch.set_num_threads(int(os.environ.get("THREADS", os.cpu_count())))
EV_A3_TO_GPA = 160.21766208

# Targets: the paper's Al elastic constants (GPa), and the lattice constant (Å) at which they apply.
C_TARGET = {"C11": 116.3, "C12": 64.8, "C44": 30.9}
B_TARGET = (C_TARGET["C11"] + 2 * C_TARGET["C12"]) / 3
# A0: the experimental lattice constant of Al near 0 K, 4.03185 Å extrapolated to 0 K (Straumanis & Woodard,
# Acta Cryst. A27, 549 (1971); tabulated in Nakashima, arXiv:2002.01562, Appendix A).
A0 = float(os.environ.get("A0", 4.032))


def load_calc():
    from mace.calculators import MACECalculator
    return MACECalculator(model_paths=MODEL_FILE, device="cpu", default_dtype="float64")


def probe(lin, n):
    """Effective weights of an e3nn 0e → 0e Linear (unit-vector probe)."""
    X = torch.zeros(n, lin.irreps_in.dim, dtype=torch.float64)
    X[:, :n] = torch.eye(n, dtype=torch.float64)
    with torch.no_grad():
        return lin(X)[:, 0].numpy()


def _set_lin(lin, n, w_eff):
    with torch.no_grad():
        lin.weight.fill_(1.0)
    alpha = probe(lin, n)
    with torch.no_grad():
        lin.weight.copy_(torch.as_tensor(w_eff / alpha, dtype=lin.weight.dtype).reshape(lin.weight.shape))


class Basis:
    """Hooks a calculator's model; D and E (and F, G = −∂D/∂r) from one forward pass."""

    def __init__(self, calc):
        self.calc = calc
        m = calc.models[0]
        ro = m.readouts
        assert len(ro) == 2 and hasattr(ro[1], "linear_2"), "expected Linear + NonLinear readouts"
        self.n0, self.n1 = ro[0].linear.irreps_in.count("0e"), ro[1].linear_2.irreps_in.dim
        scale = float(m.scale_shift.scale.reshape(-1)[0])
        self.theta0 = np.concatenate([scale * probe(ro[0].linear, self.n0), scale * probe(ro[1].linear_2, self.n1),
                                      [float(m.scale_shift.shift.reshape(-1)[0])]])
        self.buf, self.keep_graph = {}, False
        grab = lambda key: (lambda mod, a: self.buf.__setitem__(key, a[0] if self.keep_graph else a[0].detach()))
        ro[0].register_forward_pre_hook(grab("h0"))
        ro[1].linear_2.register_forward_pre_hook(grab("x1"))

    def _D(self):
        return torch.cat([self.buf["h0"][:, :self.n0].sum(0), self.buf["x1"].sum(0)])

    def row(self, atoms):
        """(D, E_MACE) from one energy-only forward pass (no ASE result cache)."""
        batch = self.calc._atoms_to_batch(atoms)
        with torch.no_grad():
            out = self.calc.models[0](batch.to_dict(), training=False, compute_force=False, compute_stress=False)
        return np.concatenate([self._D().numpy(), [len(atoms)]]), float(out["energy"].reshape(-1)[0])

    def D(self, atoms):
        return self.row(atoms)[0]

    def row_forces(self, atoms):
        """(D, E_MACE, F_MACE (N×3), G (3N × 81) = −∂D/∂r, rows atom-major as F.ravel())."""
        data = self.calc._atoms_to_batch(atoms).to_dict()
        pos = data["positions"]
        self.keep_graph = True
        with torch.enable_grad():
            pos.requires_grad_(True)
            out = self.calc.models[0](data, training=False, compute_force=False, compute_stress=False)
            E, Dv = out["energy"].reshape(-1)[0], self._D()
            (gE,) = torch.autograd.grad(E, pos, retain_graph=True)
            m = Dv.numel()
            try:
                (J,) = torch.autograd.grad(Dv, pos, grad_outputs=torch.eye(m, dtype=Dv.dtype), is_grads_batched=True)
            except Exception:
                J = torch.stack([torch.autograd.grad(Dv[j], pos, retain_graph=True)[0] for j in range(m)])
        self.keep_graph = False
        n = len(atoms)
        G = np.zeros((3 * n, m + 1))
        G[:, :m] = -J.detach().reshape(m, 3 * n).T.numpy()
        return np.concatenate([Dv.detach().numpy(), [n]]), float(E.detach()), -gE.detach().numpy(), G


def patched_calc(theta):
    """A fresh calculator with θ written into the readout weights (energy = Σ E0 + D·θ)."""
    calc = load_calc()
    m = calc.models[0]
    ro = m.readouts
    n0, n1 = ro[0].linear.irreps_in.count("0e"), ro[1].linear_2.irreps_in.dim
    assert theta.size == n0 + n1 + 1, (theta.size, n0 + n1 + 1)
    scale = float(m.scale_shift.scale.reshape(-1)[0])
    _set_lin(ro[0].linear, n0, theta[:n0] / scale)
    _set_lin(ro[1].linear_2, n1, theta[n0:n0 + n1] / scale)
    with torch.no_grad():
        m.scale_shift.shift.fill_(float(theta[-1]))
    return calc


def fcc(a):
    return bulk("Al", "fcc", a=a, cubic=True)


def strained(atoms, e1=0.0, e2=0.0, e4=0.0):
    """cell' = cell·(I + ε)ᵀ, Voigt ε₄ = 2ε_yz."""
    eps = np.array([[e1, 0, 0], [0, e2, e4 / 2], [0, e4 / 2, 0]])
    at = atoms.copy()
    at.set_cell(at.cell.array @ (np.eye(3) + eps).T, scale_atoms=True)
    return at


def relaxed_a0(calc):
    from scipy.optimize import minimize_scalar

    def E(a):
        x = fcc(a)
        x.calc = calc
        return x.get_potential_energy() / len(x)
    r = minimize_scalar(E, bounds=(3.8, 4.3), method="bounded", options={"xatol": 1e-8})
    return float(r.x), float(r.fun)


def stress_strain_cij(calc):
    """a0 and C11, C12, C44, B (GPa) from MACE's analytic stress under ±0.5% strain at the model's own a0."""
    from ase.units import GPa
    a, _ = relaxed_a0(calc)
    c0, h, st = fcc(a), 5e-3, {}
    for lab, kw in (("p1", dict(e1=h)), ("m1", dict(e1=-h)), ("p4", dict(e4=h)), ("m4", dict(e4=-h))):
        y = strained(c0, **kw); y.calc = calc; st[lab] = y.get_stress()
    s1, s4 = (st["p1"] - st["m1"]) / (2 * h), (st["p4"] - st["m4"]) / (2 * h)
    C11, C12, C44 = s1[0] / GPa, s1[1] / GPa, s4[3] / GPa
    return {"a0": a, "C11": C11, "C12": C12, "C44": C44, "B": (C11 + 2 * C12) / 3}


def test_frames(n_test):
    """The held-out test sample: n_test random frames of manual_df_test_Al.xyz (seed 0, sorted)."""
    from ase.io import read
    test = read(os.path.join(DATA, "manual_df_test_Al.xyz"), ":")
    return [test[i] for i in np.sort(np.random.default_rng(0).choice(len(test), n_test, replace=False))]
