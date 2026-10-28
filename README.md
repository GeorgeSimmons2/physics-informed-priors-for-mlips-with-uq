# constrained-pops-repro

Reproduces the figures of the paper on physically constrained POPS uncertainty quantification for
aluminium, from a fresh environment, with one script.

## The figures

| figure | made by | output (under `published/` or `repro/<route>/`) |
|---|---|---|
| Phonon bands of unconstrained vs constrained committees, Al_12 (top) and Al_16 (bottom) | `four_panel/plot_four_panel.jl` | `four_panel/bands_four_panel.pdf` |
| RLS vs constrained RLS at 200 K: phonons at a(200 K) and RDFs | `rls_npt/04_fig_200K.jl` | `rls_npt/fig_200K/fig_200K_rls_vs_constrained.pdf` |
| Al_12 test-set parity and calibration of the constrained committee | `al12/03_hypercube_rejection.jl` | `al12/hypercube/{parity,calibration}_fullcloud.pdf` |
| MACE before/after: accuracy and C_ij of the trained model and its constrained readout | `mace/4_compare.py` | `mace/results/compare_elastic.pdf` |
| C_ij of every MACE POPS committee member, constrained vs unconstrained | `mace/pops/p5_figures.py` | `mace/pops/cij_spread.pdf` |

What is computed:
- **Al_12_4_6A_2_ (91 parameters).** Each member of the POPS cloud (the top-50%-leverage observations,
  73,479 members) is refitted with the lattice constant pinned, Born stability, and cutting planes until
  every phonon on the band path is above 0.15 THz at 0 K (`al12/01`, `al12/02`). A hypercube around the
  constrained cloud is then rejection-sampled on the same check (`al12/03`). `al12/04` draws the
  unconstrained ensemble, and `four_panel/build_bands_cache_Al_12.jl` relaxes every member and computes its bands.
- **Al_16_4_6A_3_ (684 parameters).** The cloud is pinned to the lattice constant in closed form and the
  hypercube is rejection-sampled on phonon stability (`al16/01`, `al16/02`).
- **RLS vs constrained RLS.** Al_12 is refitted with the lattice constant, Born stability and phonons
  above 0.01 THz at every volume from 1.00 to 1.10 a_eq (`rls_npt/01`). Both models run 200 K NPT with the
  same seed (`02`); the RLS model leaves FCC, the constrained one does not (`03`, `04`).
- **MACE.** The readout of the 64-channel MACE model is refitted under a0 = 4.032 Å and
  C11/C12/C44 = 116.3/64.8/30.9 GPa (`mace/1`–`3`) and compared with the trained model (`4`). Constrained and
  unconstrained POPS committees of the readout are built and evaluated in `mace/pops/p1`–`p4`.

## Running it

```bash
bash run.sh setup     # .venv/ and a repo-local Julia depot .julia/                       ~40 min, once
bash run.sh data      # the tier-B inputs from ../physics-informed-priors-repro, SHA-256 checked
bash run.sh replot    # every figure from published/ → repro/replot/                      minutes
bash run.sh saved     # recompute everything from the published fitted models → repro/saved/   SLURM
bash run.sh test      # unit tests (src/, lib/, every function the scripts define) + replot regression  ~25 min
```

Needs `julia` 1.11, `python3` 3.9 and SLURM (`PARTITION`, default `compute`). Every job fits on one
48-core node with about 170 GB of memory.

`published/` holds the last full rebuild from scratch (15–16 Oct 2026; constrained RLS 25 Oct). Reruns
reproduce its figures within numerical tolerance.

## Pipeline

```
Al_12   al12/01 ─► al12/02 ─► al12/03 ─► al12/04 ─► build_bands_cache_Al_12 ─┐
Al_16   al16/01 ─► al16/02 ───────────────────────────────────────────────────┴─► plot_four_panel
RLS     rls_npt/01 ─► 02 (constrained) ─┐
                      02 (RLS) ─────────┴─► 03 + 04 (fig_200K)
MACE    1 → 2 → 3_qp ─► 4 → 5
        p1 → p1b ─(+3_qp)─► p2 → p3 → p4 → p5
```

## Layout

```
run.sh                       setup | data | replot | saved   (extras: refit | all | figures)
src/                         ACEWorkflow: model build/load, POPS samplers, elastic Hessian, phonon band path
lib/bandpath.jl              per-basis Hessian (cached beside each model), band path, stability helpers
lib/parity_calibration.jl    test-set parity and calibration figures
al12/01_committee.jl         constrained mean model and the 30-member repaired forest
al12/02_cutting_plane_cloud.jl  constrain the whole cloud with cutting planes
al12/03_hypercube_rejection.jl  hypercube around the cloud, rejection sampling, parity/calibration
al12/03_replot_parity.jl     redraw al12/03's figures from its saved predictions
al12/04_naive_ensemble.jl    the unconstrained ensemble
al16/01_pinned_ensembles.jl  pinned hypercube: unconstrained and rejection-sampled ensembles
al16/02_bands.jl             relaxation and bands of every Al_16 member
four_panel/                  Al_12 band cache, then the 2×2 figure
rls_npt/                     01 constrained RLS, 02 NPT, 03 RDF, 04 the figure
mace/                        1–3 constrained readout, 4 before/after, 5 matscipy check; pops/ p1–p5 committees
ace_models/, mace/train.sh   refit both ACE models / retrain MACE from the DFT data (refit route)
models/, mace/models/        the published fitted models
published/                   the published results, read by replot
```

Scripts are copies of the ones that produced the paper's results; changes are marked `[PORT]` in each.
Paper figures are drawn at their printed width (510 pt full, 255 pt half) with 11–13 pt text.

## Extras

- `bash run.sh refit` refits both ACE models and retrains MACE (~5.5 h) from the DFT data, then runs the
  same chain into `repro/refit/`. `bash run.sh all` runs both routes.
- `bash run.sh figures ROUTE` redraws a finished route's figures from its results.
- `mace/5_elastic_matscipy.py` checks the MACE C_ij independently with matscipy.
- Further outputs: MACE POPS parity plots and test-set numbers (`mace/pops/parity_*`, `uq_test.txt`),
  the constrained RLS bands at each volume (`rls_npt/constrained/`), the Al_12 cloud diagnostics
  (`al12/cloud/member_diagnostics.csv`).

## Data

- **In git:** code, the published fitted models and `published/`.
- **Not in git** (`data/tierB.sha256`): the DFT training and test sets and the two ACE design matrices
  `models/*/A.csv`. `bash run.sh data` hardlinks or copies them in and checks every SHA-256.
