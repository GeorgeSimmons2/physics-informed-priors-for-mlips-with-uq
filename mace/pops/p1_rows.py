"""
p1. Per-observation rows for POPS: every energy and every force component of the training set
(data/Al/subset_20_percent.extxyz) and of the 3,000-frame test sample, in the readout basis of ../lib.py.

The mean model's design (../2_design.py) keeps only the force normal equations; POPS needs each row.
  energy row  a = D/N,  y = (E_DFT − E_frozen)/N        (per atom, as in the mean model's loss)
  force rows  a = G = −∂D/∂r (one per component),  y = F_DFT − F_frozen
Out: $OUT/mace/pops/rows/rows_{train,test}.npz  —  XE, yE (energies), XF, yF (force components), frameF (frame of each
force row), nat (atoms per frame).  Cost: ~20 min train, ~35 min test (16 CPU threads).
"""
import os
import sys
import time

import numpy as np
from ase.io import read

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import lib  # noqa: E402

OUT = lib.ROWS
basis = lib.Basis(lib.load_calc())


def rows(name, frames):
    t = time.time()
    XE, yE, nat, XF, yF, fr = [], [], [], [], [], []
    for i, at in enumerate(frames):
        D, Em, Fm, G = basis.row_forces(at)
        n = len(at)
        E_frozen = Em - D @ basis.theta0
        XE.append(D / n); yE.append((at.info["dft_energy"] - E_frozen) / n); nat.append(n)
        XF.append(G); yF.append((at.arrays["dft_forces"] - Fm).ravel() + G @ basis.theta0); fr.append(np.full(3 * n, i))
        if i % 500 == 0 or i == len(frames) - 1:
            print(f"  {name} {i + 1}/{len(frames)}  {time.time() - t:.0f}s", flush=True)
    z = dict(XE=np.array(XE), yE=np.array(yE), nat=np.array(nat),
             XF=np.vstack(XF), yF=np.concatenate(yF), frameF=np.concatenate(fr))
    np.savez(os.path.join(OUT, f"rows_{name}.npz"), **z)
    print(f"{name}: {len(z['yE'])} energy rows, {len(z['yF'])} force rows → {OUT}", flush=True)


rows("train", read(os.path.join(lib.DATA, "subset_20_percent.extxyz"), ":"))
rows("test", lib.test_frames(3000))
