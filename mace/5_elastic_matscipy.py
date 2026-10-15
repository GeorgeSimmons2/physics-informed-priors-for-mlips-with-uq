"""
5. An independent check: the elastic constants of both models from matscipy's stress–strain fit on ASE Atoms
(ASE itself has no elasticity module).  Uses nothing from lib.py or the constraint rows — only the two
MACE checkpoints, an ASE cell relaxation, and matscipy.

  $MODEL_FILE (default mace/models/Al_small_stagetwo.model)   MACE as trained
  $OUT/mace/results/Al_small_stagetwo_elastic.model          the refit readout weights written in (4_compare.py)
Run: python3 5_elastic_matscipy.py    → $OUT/mace/results/elastic_matscipy.txt      Env: OUT, MODEL_FILE
"""
import os

from ase.build import bulk
from ase.filters import FrechetCellFilter
from ase.optimize import BFGS
from ase.units import GPa
from mace.calculators import MACECalculator
from matscipy.elasticity import fit_elastic_constants

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("OUT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "repro", "saved"))
out = os.path.join(ROOT, "mace", "results")
trained = os.environ.get("MODEL_FILE", os.path.join(HERE, "models", "Al_small_stagetwo.model"))
constrained = os.path.join(out, "Al_small_stagetwo_elastic.model")

lines = []
for name, path in (("MACE", trained), ("constrained", constrained)):
    atoms = bulk("Al", "fcc", a=4.05, cubic=True)
    atoms.calc = MACECalculator(model_paths=path, device="cpu", default_dtype="float64")
    BFGS(FrechetCellFilter(atoms, hydrostatic_strain=True), logfile=None).run(fmax=1e-6, steps=200)  # relax a0
    C, C_err = fit_elastic_constants(atoms, symmetry="cubic", N_steps=5, delta=1e-3, verbose=False)
    lines.append(f"{name:12s} a0 {atoms.cell[0, 0]:.4f} Å   C11 {C[0, 0] / GPa:6.1f}   C12 {C[0, 1] / GPa:6.1f}   "
                 f"C44 {C[3, 3] / GPa:6.1f} GPa   (fit error ≤ {C_err.max() / GPa:.2f} GPa)")
lines.append(f"{'target':12s} a0 4.0320 Å   C11  116.3   C12   64.8   C44   30.9 GPa")
print("\n".join(lines))
open(os.path.join(out, "elastic_matscipy.txt"), "w").write("\n".join(lines) + "\n")
