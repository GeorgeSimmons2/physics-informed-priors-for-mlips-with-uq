# constrained-pops-repro

Everything needed to rerun three results on physically constrained POPS uncertainty quantification
for aluminium, from a fresh environment, with one script.

1. **The four-panel phonon figure.** Unconstrained (left) vs constrained (right) POPS committees, with Al_12 on top and Al_16 below.
   - **Al_12_4_6A_2_ (91 parameters).** The whole top-50%-leverage POPS cloud is cut with
     cutting planes until every member is dynamically stable at 0 K with the lattice constant
     pinned. The hypercube around that cloud is then rejection-sampled.
   - **Al_16_4_6A_3_ (684 parameters).** The cloud is pinned to the lattice constant in closed
     form, and the hypercube is rejection-sampled on phonon stability.
2. **RLS vs constrained RLS at 200 K.** Al_12 is refitted with the lattice constant, Born
   stability and phonon stability constrained at every volume from 1.00 to 1.10 × a_eq. Both
   models then run 200 K NPT with the same seed. The RLS model leaves FCC; the constrained one
   does not.
3. **MACE with constrained readout.** This is the 64-channel MACE model (64x0e + 64x1o).
   - Its readout weights are refitted under the experimental lattice constant
     a0 = 4.032 Å and the elastic constants C11/C12/C44 = 116.3/64.8/30.9 GPa.
   - Constrained and unconstrained POPS committees of that readout are compared on a held-out
     test set.

## Quick start

```bash
bash run.sh setup     # .venv/ (requirements.txt) and Julia (Manifest.toml) in a repo-local depot .julia/   ~40 min, once
bash run.sh data      # the tier-B inputs from ../physics-informed-priors-repro (or: bash run.sh data <dir>), SHA-256 checked
bash run.sh replot    # every figure from the committed results in published/ → repro/replot/                ~5 min
bash run.sh all       # SLURM: the saved and refit routes below → repro/saved/, repro/refit/
bash run.sh verify saved && bash run.sh verify refit     # → repro/<route>/VERIFY.md
```

You need `julia` 1.11 on PATH (the Manifest was resolved with 1.11.8), `python3` 3.9 and SLURM.
The partition is `PARTITION` (default `compute`). Every job fits on one 48-core node with
about 170 GB of memory.

## Two routes

| route | starts from | what it tests |
|---|---|---|
| `saved` | the published fitted models in `models/` and `mace/models/` | every downstream stage: constraints, clouds, sampling, phonons, NPT, MACE readout QP, POPS |
| `refit` | the DFT data only | the same chain, after refitting both ACE models (`ace_models/fit_models.jl`) and retraining MACE (`mace/train.sh`, ~5.5 h) |

A route writes only under `repro/<route>/`. The exception is the per-model Hessian cache,
`models/<name>/cache/`, which is kept beside the models that route uses. `published/` is never
written. A route that already exists is refused unless `FORCE=1`.

## How closely a rerun must agree (`verify/`)

Each number is judged at the standard its computation can actually meet:

| tier | applies to | criterion |
|---|---|---|
| `exact` | seeded draws and closed-form corrections, with no QP on the path; the refit design matrices (SHA-256) | bit for bit |
| `converged` | one well-posed QP or a deterministic fit: θ_mean, constrained RLS θ, refitted ACE coefficients, member min ω and relaxed a | max\|Δ\| ≤ 1e-2 |
| `statistical` | cutting-plane cascades, rejection-sampled committees, MD | ensemble mean and spread within 4.5σ of sampling noise; coverage within 2–3 points; a(200 K) within 3 fluctuation σ and the same FCC verdict |

- **Why some stages are only statistical.**
  - OSQP's ρ schedule is pinned (`adaptive_rho_interval = 25`), so a single QP is reproducible.
  - A cutting-plane cascade is different: it can still accept a different but equally valid
    member on another machine or thread count.
  - So member identities are not reproducible, and the committee statistics are.
- **Retrained MACE.** Its weights depend on floating-point reduction order. On the refit route
  only the physics is checked: a0 and C_ij on target, test RMSE within 15% of the published
  values, and coverage within 3 points.

## Layout

```
run.sh                       setup | data | replot | saved | refit | all | verify
Project.toml, Manifest.toml  the Julia environment (21 direct dependencies)
requirements.txt             the Python environment (torch 2.8.0, mace-torch 0.3.16, e3nn 0.4.4, ase, matscipy)
src/                         ACEWorkflow: model build/load, POPS samplers, elastic Hessian, phonon band path
lib/bandpath.jl              undotted per-basis Hessian (cached), band path, stability helpers
lib/parity_calibration.jl    test-set parity and calibration figures of a committee
ace_models/fit_models.jl     refit Al_12_4_6A_2_ and Al_16_4_6A_3_ from the DFT data           (refit route)
al12/01_committee.jl         constrained mean θ_mean and the 30-member repaired forest
al12/02_cutting_plane_cloud.jl  cut the whole top-leverage cloud (73,479 members) until stable
al12/03_hypercube_rejection.jl  hypercube around the cloud, rejection sampling, parity/calibration
al12/04_naive_ensemble.jl    the unconstrained ensemble; band curves of both → four-panel cache
al16/01_pinned_ensembles.jl  pinned hypercube: unconstrained and rejection-sampled ensembles
al16/02_bands.jl             native relaxation + Hessian per member → four-panel cache
four_panel/plot_four_panel.jl   the 2×2 figure from the two caches
rls_npt/01_constrained_rls.jl   RLS refit with multi-volume phonon constraints
rls_npt/02_npt.jl            200 K NPT of one θ (Molly, 256 atoms, 30 000 steps)
rls_npt/03_rdf.jl            RDF and coordination of the production frames
rls_npt/04_fig_200K.jl       bands at a(200 K) + RDFs of both models
mace/train.sh                train the MACE model exactly as published                        (refit route)
mace/1_constraints.py … 5_elastic_matscipy.py   readout basis, constraint rows, QP, before/after, matscipy check
mace/pops/p1_rows.py … p6_maxdev_vs_error.py    POPS rows, clouds, committees, evaluation, figures
verify/                      verify.jl (ACE), verify_mace.py (MACE), lib_verify.jl (the tiers)
published/                   the published results every route is compared with, and replot reads
models/, mace/models/        the published fitted models (the saved route's starting point)
data/tierB.sha256            checksums of the inputs not committed (below)
```

## Pipeline

```
Al_12   al12/01 ─► al12/02 ─► al12/03 ─► al12/04 ─┐
Al_16   al16/01 ─► al16/02 ───────────────────────┴─► four_panel
RLS     rls_npt/01 ─► 02 (constrained) ─┐
                      02 (RLS) ─────────┴─► 03 + 04 (fig_200K)
MACE    1 → 2 → 3_qp ─► 4 → 5
        p1 → p1b ─(+3_qp)─► p2 → p3 → p4 → p5, p6
refit:  fit_models.jl precedes every ACE chain; train.sh precedes both MACE chains
```

## Data tiers

- **Tier A (committed, ~41 MB).**
  - Code.
  - The published fitted models: ACE `json`/`lin_params`/`P`/`W`/`Y`, and the MACE checkpoint
    with its training config and log.
  - Every published result that verify or replot reads (`published/`).
- **Tier B (not committed, `data/tierB.sha256`).**
  - The three DFT data sets: training, test and the 20% MACE training subset.
  - The two ACE design matrices `models/*/A.csv`. These are only read on the saved route; the
    refit route rebuilds them and checks the SHA-256.
  - `run.sh data` hardlinks the files when they are on the same filesystem and copies them
    otherwise, then checks every checksum.
