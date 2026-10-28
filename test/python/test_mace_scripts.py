"""Functions the mace/ scripts define: frozen energies and FD derivatives (1), the design (2), the parity panel (4),
POPS rows (p1), writing θ into a calculator (p4), and the figure helpers (p5)."""
import os
import tempfile
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import fixtures as fx  # noqa: E402
import lib  # noqa: E402
from harness import script_namespace, tested  # noqa: E402


def e0_per_atom():
    b = fx.basis(); D, E = b.row(lib.fcc(lib.A0))
    return (E - D @ b.theta0) / 4


class TestConstraints(unittest.TestCase):
    F = "mace/1_constraints.py"

    def test_E_frozen(self):
        tested(self.F, "E_frozen")
        ns = script_namespace(self.F, inject=dict(calc=fx.calc(), basis=fx.basis()), only=["E_frozen"])
        for at in (lib.fcc(4.0), fx.rattled()):                       # what the readout does not touch: Σ E0
            self.assertAlmostEqual(ns["E_frozen"](at), len(at) * e0_per_atom(), delta=1e-8 * len(at))

    def test_derivs(self):
        tested(self.F, "derivs")
        c0 = lib.fcc(lib.A0)
        ns = script_namespace(self.F, inject=dict(c0=c0, H_A=1e-3, H_E=1e-3), only=["derivs"])
        a, V = lib.A0, lib.A0 ** 3
        d = ns["derivs"](lambda at: at.get_volume())                  # V(a) = a³, V(ε) = V0·det(I + ε)
        np.testing.assert_allclose(d[0], 3 * a ** 2, rtol=1e-6)
        np.testing.assert_allclose(d[1], 6 * a, rtol=1e-6)
        np.testing.assert_allclose(d[2], 0, atol=1e-6 * V)            # linear in e1
        np.testing.assert_allclose(d[3], V, rtol=1e-8)                # ∂²/∂e1∂e2 (1+e1)(1+e2)
        np.testing.assert_allclose(d[4], -V / 2, rtol=1e-8)           # 1 − e4²/4


class TestDesign(unittest.TestCase):
    F = "mace/2_design.py"

    def test_build(self):
        tested(self.F, "build")
        b = fx.basis()
        frames = lib.test_frames(3)
        ns = script_namespace(self.F, inject=dict(basis=b), only=["build"])
        ns["build"]("unit", frames)
        rd = lambda k: np.loadtxt(os.path.join(lib.OUT, f"{k}_unit.csv"), delimiter=",", ndmin=2)
        D, E, GtG, Gty, fst = rd("D"), rd("E"), rd("GtG"), rd("Gty").ravel(), rd("fstats").ravel()
        self.assertEqual(D.shape, (3, 81)); self.assertEqual(E.shape, (3, 4))
        n = np.array([len(at) for at in frames])
        np.testing.assert_array_equal(E[:, 3], n)
        np.testing.assert_allclose(E[:, 0], [at.info["dft_energy"] for at in frames])
        # the frozen part is Σ E0, and X·θ0 − y is MACE's per-atom energy error
        np.testing.assert_allclose(E[:, 2], n * e0_per_atom(), rtol=1e-10)
        Emace = np.array([fx.mace_E_F(at)[0] for at in frames])
        np.testing.assert_allclose(D / n[:, None] @ b.theta0 - (E[:, 0] - E[:, 2]) / n, (Emace - E[:, 0]) / n, atol=1e-10)
        # the Grams give MACE's force RMSE exactly
        res = np.concatenate([(fx.mace_E_F(at)[1] - at.arrays["dft_forces"]).ravel() for at in frames])
        yty, nF = fst
        self.assertEqual(nF, res.size)
        rms = np.sqrt((yty - 2 * b.theta0 @ Gty + b.theta0 @ GtG @ b.theta0) / nF)
        self.assertAlmostEqual(rms, np.sqrt(np.mean(res ** 2)), delta=1e-8)


class TestCompare(unittest.TestCase):
    def test_parity(self):
        F = "mace/4_compare.py"
        tested(F, "parity")
        ns = script_namespace(F, only=["parity"])
        fig, ax = plt.subplots()
        x, y = np.array([0.0, 1.0, 2.0]), np.array([0.1, 0.9, 2.5])
        ns["parity"](ax, x, y, "C0", "title", "meV", 12.34)
        lo, hi, pad = 0.0, 2.5, 0.04 * 2.5
        self.assertEqual(ax.get_xlim(), ax.get_ylim())                # square, shared limits
        np.testing.assert_allclose(ax.get_xlim(), (lo - pad, hi + pad))
        self.assertEqual(ax.get_title(loc="left"), "title")
        self.assertIn("RMSE\n12.3 meV", [t.get_text() for t in ax.texts])
        plt.close(fig)


class TestPopsRows(unittest.TestCase):
    def test_rows(self):
        F = "mace/pops/p1_rows.py"
        tested(F, "rows")
        b = fx.basis(); frames = lib.test_frames(2)
        with tempfile.TemporaryDirectory() as out:
            ns = script_namespace(F, inject=dict(basis=b, OUT=out), only=["rows"])
            ns["rows"]("unit", frames)
            z = np.load(os.path.join(out, "rows_unit.npz"))
            n = np.array([len(at) for at in frames])
            np.testing.assert_array_equal(z["nat"], n)
            self.assertEqual(z["XE"].shape, (2, 81)); self.assertEqual(z["XF"].shape, (3 * n.sum(), 81))
            Emace = np.array([fx.mace_E_F(at)[0] for at in frames])
            np.testing.assert_allclose(z["XE"] @ b.theta0 - z["yE"], (Emace - [at.info["dft_energy"] for at in frames]) / n,
                                       atol=1e-10)                    # a row's residual at θ0 is MACE's error
            Fm = np.concatenate([fx.mace_E_F(at)[1].ravel() for at in frames])
            Fd = np.concatenate([at.arrays["dft_forces"].ravel() for at in frames])
            np.testing.assert_allclose(z["XF"] @ b.theta0 - z["yF"], Fm - Fd, atol=1e-8)
            np.testing.assert_array_equal(z["frameF"], np.repeat([0, 1], 3 * n))


class TestEvaluate(unittest.TestCase):
    def test_set_theta(self):
        F = "mace/pops/p4_evaluate.py"
        tested(F, "set_theta")
        b = fx.basis(); at = fx.rattled(seed=4)
        D, E, Fr, G = b.row_forces(at)
        c = lib.load_calc()
        ns = script_namespace(F, inject=dict(calc=c), only=["set_theta"])
        x = at.copy(); x.calc = c; x.get_potential_energy()
        th = b.theta0 * (1 + 0.1 * np.random.default_rng(5).standard_normal(81))
        ns["set_theta"](th)
        self.assertEqual(c.results, {})                               # ASE's cache is dropped
        self.assertAlmostEqual(x.get_potential_energy(), len(at) * e0_per_atom() + D @ th, delta=1e-8 * abs(E))
        np.testing.assert_allclose(x.get_forces(), (G @ th).reshape(-1, 3), atol=1e-8)


class TestFigures(unittest.TestCase):
    F = "mace/pops/p5_figures.py"

    def test_save(self):
        tested(self.F, "save")
        with tempfile.TemporaryDirectory() as out:
            ns = script_namespace(self.F, inject=dict(OUT=out, PREFIX="t_"), only=["save"])
            fig, ax = plt.subplots(); ax.plot([0, 1], [0, 1])
            ns["save"](fig, "x")
            for ext in ("pdf", "png"):
                self.assertGreater(os.path.getsize(os.path.join(out, f"t_x.{ext}")), 500)
            self.assertFalse(plt.fignum_exists(fig.number))           # closed after saving

    def test_get(self):
        tested(self.F, "get")
        d = {f"E_con_{s}": np.arange(3) + i for i, s in enumerate(("y", "f", "sd"))}
        ns = script_namespace(self.F, inject=dict(d=d), only=["get"])
        y, f, sd = ns["get"]("E", "con")
        np.testing.assert_array_equal(y, [0, 1, 2]); np.testing.assert_array_equal(sd, [2, 3, 4])


if __name__ == "__main__":
    unittest.main()
