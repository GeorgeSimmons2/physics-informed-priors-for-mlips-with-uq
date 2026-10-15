"""
2. Design: the readout basis on the whole training set (every frame of ../data/Al/subset_20_percent.extxyz,
the data the model was trained on) and on the held-out test sample (N_TEST random frames, seed 0, of
../data/Al/manual_df_test_Al.xyz).

Energies:  D rows and E columns (E_DFT, E_MACE, E_frozen = E_MACE − D·θ0, n_atoms);  y = (E_DFT − E_frozen)/N.
Forces:    G = −∂D/∂r per frame; only the normal equations are kept,
           GtG = Σ GᵀG,  Gty = Σ Gᵀ y_F,  yty = Σ y_Fᵀ y_F,  nF = Σ 3N        (y_F = F_DFT − F_frozen)
           so every force RMSE is exact:  Σ‖y_F − Gθ‖² = yty − 2θ·Gty + θᵀ GtG θ.
Out: $OUT/{D,E,GtG,Gty,fstats}_{train,test}.csv     Env: N_TEST (3000)
Cost: ~20 min for the training set, ~30 min for 3000 test frames (16 CPU threads).
"""
import os
import time

import numpy as np
from ase.io import read

import lib

N_TEST = int(os.environ.get("N_TEST", 3000))
basis = lib.Basis(lib.load_calc())


def build(name, frames):
    t = time.time()
    p = basis.theta0.size
    Ds, Es = [], []
    GtG, Gty, yty, nF = np.zeros((p, p)), np.zeros(p), 0.0, 0
    for i, at in enumerate(frames):
        D, Em, Fm, G = basis.row_forces(at)
        Ds.append(D)
        Es.append([at.info["dft_energy"], Em, Em - D @ basis.theta0, len(at)])
        yF = (at.arrays["dft_forces"] - Fm).ravel() + G @ basis.theta0      # F_DFT − F_frozen
        GtG += G.T @ G; Gty += G.T @ yF; yty += yF @ yF; nF += yF.size
        if i % 500 == 0 or i == len(frames) - 1:
            print(f"  {name} {i + 1}/{len(frames)}  {time.time() - t:.0f}s", flush=True)
    Ds, Es = np.array(Ds), np.array(Es)
    for key, arr in (("D", Ds), ("E", Es), ("GtG", GtG), ("Gty", Gty[None, :]), ("fstats", [[yty, nF]])):
        np.savetxt(os.path.join(lib.OUT, f"{key}_{name}.csv"), arr, delimiter=",")
    r = (Es[:, 0] - Es[:, 1]) / Es[:, 3]
    fr = np.sqrt(max(yty - 2 * basis.theta0 @ Gty + basis.theta0 @ GtG @ basis.theta0, 0) / nF)
    print(f"{name}: {len(Ds)} frames, {nF} force components; MACE E RMSE {1e3 * np.sqrt((r**2).mean()):.2f} meV/atom, "
          f"F RMSE {1e3 * fr:.1f} meV/Å", flush=True)


build("train", read(os.path.join(lib.DATA, "subset_20_percent.extxyz"), ":"))
build("test", lib.test_frames(N_TEST))
print(f"→ {lib.OUT}")
