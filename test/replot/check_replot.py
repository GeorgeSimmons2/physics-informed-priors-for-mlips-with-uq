"""
Replot regression: the figures `run.sh replot` draws from published/ are the published ones.

    REPLOT_OUT=<dir> bash run.sh replot && .venv/bin/python test/replot/check_replot.py <dir>

Every PNG must be byte-identical to published/ (else the fraction of differing pixels is reported, and fewer than
0.1% passes as a rendering difference); the data the figures are drawn from must be identical (metadata.csv apart
from its relative-path column); every PDF must be a valid one-page PDF (PDFs embed their creation date, so their
bytes always differ).
"""
import csv
import os
import re
import sys
import zlib

import numpy as np
from PIL import Image

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PUB = os.path.join(REPO, "published")
FIGS = ["al12/hypercube/parity_fullcloud", "al12/hypercube/calibration_fullcloud",
        "al12/hypercube/parity_forest30", "al12/hypercube/calibration_forest30",
        "four_panel/bands_four_panel", "rls_npt/fig_200K/fig_200K_rls_vs_constrained",
        "mace/results/compare_elastic", "mace/pops/cij_spread",
        "mace/pops/parity_E_constrained", "mace/pops/parity_E_unconstrained",
        "mace/pops/parity_F_constrained", "mace/pops/parity_F_unconstrained"]
DATA = ["rls_npt/fig_200K/band_ticks.csv", "rls_npt/fig_200K/bands_rls.csv", "rls_npt/fig_200K/bands_constrained.csv",
        "rls_npt/fig_200K/rdf_rls.csv", "rls_npt/fig_200K/rdf_constrained.csv", "mace/results/compare_elastic.txt"]


def same_bytes(a, b):
    with open(a, "rb") as fa, open(b, "rb") as fb:
        return fa.read() == fb.read()


def png(rel, out):
    new, pub = os.path.join(out, rel), os.path.join(PUB, rel)
    if same_bytes(new, pub):
        return True, "identical"
    x, y = np.asarray(Image.open(new).convert("RGBA")), np.asarray(Image.open(pub).convert("RGBA"))
    if x.shape != y.shape:
        return False, f"size {x.shape[:2]} vs published {y.shape[:2]}"
    frac = np.mean(np.any(x != y, axis=2))
    return frac < 1e-3, f"{100 * frac:.3f}% of pixels differ (max {np.abs(x.astype(int) - y).max()}/255)"


def pdf(rel, out):
    b = open(os.path.join(out, rel), "rb").read()
    text = [b]
    for s in re.findall(rb"stream\r?\n(.*?)\r?\nendstream", b, re.S):   # Cairo keeps its page objects in Flate
        try:                                                             # compressed object streams
            text.append(zlib.decompress(s))
        except zlib.error:
            pass
    pages = sum(len(re.findall(rb"/Type\s*/Page(?![s\w])", t)) for t in text)
    return b.startswith(b"%PDF") and pages == 1, f"{len(b)} bytes, {pages} page(s)"


def metadata(out):
    rd = lambda d: [r[:-1] for r in csv.reader(open(os.path.join(d, "rls_npt/fig_200K/metadata.csv")))]
    return rd(out) == rd(PUB), "identical but for the path column"


def main(out):
    checks = []
    for f in FIGS:
        checks.append((f + ".png",) + png(f + ".png", out))
        checks.append((f + ".pdf",) + pdf(f + ".pdf", out))
    for f in DATA:
        checks.append((f, same_bytes(os.path.join(out, f), os.path.join(PUB, f)), "identical"))
    checks.append(("rls_npt/fig_200K/metadata.csv",) + metadata(out))
    for name, ok, msg in checks:
        print(f"{'ok  ' if ok else 'FAIL'} {name:58s} {msg if ok else msg + '  ← differs from published/'}")
    bad = [c for c in checks if not c[1]]
    print(f"\nreplot regression: {len(checks) - len(bad)} of {len(checks)} checks pass")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "repro", "replot"))))
