"""mace/lib.py — the readout basis D (E = Σ E0 + D·θ), its forces G, patching θ back into MACE, structures, C_ij."""
import copy
import unittest

import numpy as np
import torch
from ase.io import read

import fixtures as fx
import lib
from harness import tested

L = "mace/lib.py"


class TestReadout(unittest.TestCase):
    def test_load_calc(self):
        tested(L, "load_calc")
        E, F = fx.mace_E_F(lib.fcc(lib.A0))
        self.assertTrue(np.isfinite(E))
        self.assertEqual(F.shape, (4, 3))
        self.assertLess(np.abs(F).max(), 1e-8)                       # perfect fcc: no forces by symmetry
        self.assertEqual(next(fx.calc().models[0].parameters()).dtype, torch.float64)

    def test_probe(self):
        tested(L, "probe")
        lin = fx.calc().models[0].readouts[0].linear
        n = lin.irreps_in.count("0e")
        w = lib.probe(lin, n)
        self.assertEqual(w.shape, (n,))
        X = torch.zeros(5, lin.irreps_in.dim, dtype=torch.float64)
        X[:, :n] = torch.randn(5, n, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
        with torch.no_grad():                                         # the layer on scalars is X·w
            np.testing.assert_allclose(lin(X)[:, 0].numpy(), X[:, :n].numpy() @ w, rtol=1e-12)

    def test_set_lin(self):
        tested(L, "_set_lin")
        lin = copy.deepcopy(fx.calc().models[0].readouts[1].linear_2)
        n = lin.irreps_in.dim
        target = np.random.default_rng(1).standard_normal(n)
        lib._set_lin(lin, n, target)
        np.testing.assert_allclose(lib.probe(lin, n), target, rtol=1e-12)   # _set_lin then probe is the identity

    def test_basis_init(self):
        tested(L, "Basis.__init__")
        b = fx.basis(); m = b.calc.models[0]
        self.assertEqual((b.n0, b.n1, b.theta0.size), (64, 16, 81))
        scale = float(m.scale_shift.scale.reshape(-1)[0])
        np.testing.assert_allclose(b.theta0[:64], scale * lib.probe(m.readouts[0].linear, 64))
        np.testing.assert_allclose(b.theta0[64:80], scale * lib.probe(m.readouts[1].linear_2, 16))
        self.assertEqual(b.theta0[-1], float(m.scale_shift.shift.reshape(-1)[0]))

    def test_row_D(self):
        tested(L, "Basis.row", "Basis.D", "Basis._D")
        b = fx.basis()
        e0 = []
        for at in (lib.fcc(lib.A0), lib.fcc(4.2), fx.rattled()):
            D, E = b.row(at)
            self.assertEqual(D.size, 81)
            self.assertEqual(D[-1], len(at))                          # the shift column counts atoms
            np.testing.assert_allclose(b._D().numpy(), D[:-1])        # the hooked features of that pass
            self.assertAlmostEqual(E, fx.mace_E_F(at)[0], delta=1e-9 * abs(E))
            np.testing.assert_array_equal(b.D(at), D)
            e0.append((E - D @ b.theta0) / len(at))
        # E = Σ E0 + D·θ0: what is left after the readout is the same energy per atom for every structure
        np.testing.assert_allclose(e0, e0[0], rtol=1e-10)
        ae = getattr(b.calc.models[0], "atomic_energies_fn", None)
        if ae is not None:
            self.assertAlmostEqual(e0[0], float(ae.atomic_energies.reshape(-1)[0]), delta=1e-8)

    def test_row_forces(self):
        tested(L, "Basis.row_forces")
        b = fx.basis()
        at = fx.rattled()
        D, E, F, G = b.row_forces(at)
        Dr, Er = b.row(at)
        np.testing.assert_allclose(D, Dr, rtol=1e-12); self.assertAlmostEqual(E, Er, delta=1e-9 * abs(E))
        np.testing.assert_allclose(F, fx.mace_E_F(at)[1], atol=1e-9)
        self.assertEqual(G.shape, (3 * len(at), 81))
        np.testing.assert_allclose((G @ b.theta0).reshape(-1, 3), F, atol=1e-9)     # F = G·θ0
        np.testing.assert_array_equal(G[:, -1], 0)                   # the shift column has no gradient
        j, h = 7, 1e-4                                                # G = −∂D/∂r, by central differences
        p, m = at.copy(), at.copy()
        p.positions[j // 3, j % 3] += h; m.positions[j // 3, j % 3] -= h
        np.testing.assert_allclose(-(b.D(p) - b.D(m)) / (2 * h), G[j], rtol=1e-5, atol=1e-8)

    def test_patched_calc(self):
        tested(L, "patched_calc")
        b = fx.basis()
        at = fx.rattled(seed=3)
        D, E, F, G = b.row_forces(at)
        e0 = (E - D @ b.theta0) / len(at)
        Ep, Fp = fx.mace_E_F(at, lib.patched_calc(b.theta0))          # θ0 is the trained model
        self.assertAlmostEqual(Ep, E, delta=1e-9 * abs(E)); np.testing.assert_allclose(Fp, F, atol=1e-9)
        th = b.theta0 * (1 + 0.1 * np.random.default_rng(2).standard_normal(81))
        Et, Ft = fx.mace_E_F(at, lib.patched_calc(th))                 # any θ: E = Σ E0 + D·θ, F = G·θ
        self.assertAlmostEqual(Et, len(at) * e0 + D @ th, delta=1e-9 * abs(Et))
        np.testing.assert_allclose(Ft, (G @ th).reshape(-1, 3), atol=1e-8)
        with self.assertRaises(AssertionError):
            lib.patched_calc(th[:-1])


class TestStructures(unittest.TestCase):
    def test_fcc(self):
        tested(L, "fcc")
        at = lib.fcc(4.0)
        self.assertEqual(len(at), 4)
        np.testing.assert_allclose(at.cell.array, 4.0 * np.eye(3))

    def test_strained(self):
        tested(L, "strained")
        at = lib.fcc(4.0)
        s = lib.strained(at, e1=0.01, e2=-0.02, e4=0.03)
        eps = np.array([[0.01, 0, 0], [0, -0.02, 0.015], [0, 0.015, 0]])   # Voigt e4 = 2ε_yz
        np.testing.assert_allclose(s.cell.array, at.cell.array @ (np.eye(3) + eps).T)
        np.testing.assert_allclose(s.get_scaled_positions(), at.get_scaled_positions(), atol=1e-14)
        self.assertIsNot(s, at)
        np.testing.assert_allclose(at.cell.array, 4.0 * np.eye(3))          # the input is untouched

    def test_relaxed_a0(self):
        tested(L, "relaxed_a0")
        a, e = lib.relaxed_a0(fx.calc())
        self.assertTrue(3.9 < a < 4.2)
        E = lambda x: fx.mace_E_F(lib.fcc(x))[0] / 4
        self.assertAlmostEqual(e, E(a), delta=1e-10)
        self.assertLess(E(a), E(a - 2e-3)); self.assertLess(E(a), E(a + 2e-3))

    def test_stress_strain_cij(self):
        tested(L, "stress_strain_cij")
        c = lib.stress_strain_cij(fx.calc())
        self.assertAlmostEqual(c["B"], (c["C11"] + 2 * c["C12"]) / 3, places=10)
        self.assertAlmostEqual(c["a0"], lib.relaxed_a0(fx.calc())[0], places=10)
        # the analytic-stress C_ij against second derivatives of the energy at the same a0
        x0, h = lib.fcc(c["a0"]), 5e-3
        V = x0.get_volume()
        E = lambda **kw: fx.mace_E_F(lib.strained(x0, **kw))[0]
        from ase.units import GPa
        C11 = (E(e1=h) - 2 * E() + E(e1=-h)) / h**2 / V / GPa
        C12 = (E(e1=h, e2=h) - E(e1=h, e2=-h) - E(e1=-h, e2=h) + E(e1=-h, e2=-h)) / (4 * h**2) / V / GPa
        C44 = (E(e4=h) - 2 * E() + E(e4=-h)) / h**2 / V / GPa
        for k, v in (("C11", C11), ("C12", C12), ("C44", C44)):
            self.assertAlmostEqual(c[k], v, delta=0.01 * abs(v), msg=k)

    def test_test_frames(self):
        tested(L, "test_frames")
        f1, f2 = lib.test_frames(4), lib.test_frames(4)
        self.assertEqual(len(f1), 4)
        all_ = read(lib.os.path.join(lib.DATA, "manual_df_test_Al.xyz"), ":")
        idx = np.sort(np.random.default_rng(0).choice(len(all_), 4, replace=False))
        for a, b, i in zip(f1, f2, idx):                              # seed 0, sorted, reproducible
            np.testing.assert_array_equal(a.positions, b.positions)
            np.testing.assert_array_equal(a.positions, all_[i].positions)


if __name__ == "__main__":
    unittest.main()
