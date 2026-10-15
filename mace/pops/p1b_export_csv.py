"""p1b. The training rows of p1 as CSV for p2_cloud.jl (the Julia environment has no .npz reader)."""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("OUT", os.path.join(os.path.dirname(os.path.dirname(HERE)), "repro", "saved"))   # the route root
ROWS = os.path.join(ROOT, "mace", "pops", "rows")
z = np.load(os.path.join(ROWS, "rows_train.npz"))
for k in ("XE", "yE", "XF", "yF"):
    np.savetxt(os.path.join(ROWS, f"{k}_train.csv"), z[k], delimiter=",")
print(f"wrote {', '.join(f'{k}_train.csv' for k in ('XE', 'yE', 'XF', 'yF'))} → {ROWS}")
