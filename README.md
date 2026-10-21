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
bash run.sh figures saved   # redraw a finished route's paper figures from its saved results (no recomputation) ~5 min
bash run.sh verify saved && bash run.sh verify refit     # → repro/<route>/VERIFY.md   (last full run: VERIFICATION.md)
```

You need `julia` 1.11 on PATH (the Manifest was resolved with 1.11.8), `python3` 3.9 and SLURM.
The partition is `PARTITION` (default `compute`). Every job fits on one 48-core node with
about 170 GB of memory.

## Figures

The paper figures are built at the width they are placed at in the manuscript, so no scaling
happens in LaTeX: full width 510 pt (`FIGW`), half width 255 pt. Text sizes are the same in all of
them: titles and panel letters 13 pt, axis labels 12 pt, ticks, legends and in-panel text 11 pt.
Those figures are the four-panel, fig_200K, the Al_12 parity/calibration plots and the MACE
before/after and POPS figures.

## Two routes

| route | starts from | what it tests |
|---|---|---|
| `saved` | the published fitted models in `models/` and `mace/models/` | every downstream stage: constraints, clouds, sampling, phonons, NPT, MACE readout QP, POPS |
| `refit` | the DFT data only | the same chain, after refitting both ACE models (`ace_models/fit_models.jl`) and retraining MACE (`mace/train.sh`, ~5.5 h) |

A route writes only under `repro/<route>/`. The exception is the per-model undotted-Hessian cache,
`models/<name>/results/`, which is kept beside the models that route uses (`repro/refit/models/` on the
refit route). `published/` is never written. A route that already exists is refused unless `FORCE=1`.

## How closely a rerun must agree (`verify/`)

Each number is judged at the standard its computation can actually meet. Where that standard is a
precision of the method, it is measured, not assumed:

| tier | applies to | criterion |
|---|---|---|
| `exact` | the refit's design matrices (SHA-256), Y, P, W; MACE's θ0 | bit for bit |
| `converged` | deterministic fits and QPs: refitted ACE coefficients, the MACE readout QP; θ_mean and the constrained RLS θ judged on what they predict | coefficients max\|Δ\| ≤ 1e-2 (MACE θ: 1e-6 relative); θ_mean within the published solution's own distance from the exact optimum; RLS θ predictions within 1e-2 of the fit residual |
| `statistical` | cutting-plane cascades, every hypercube draw (constrained or not), MD | committees: mean and spread within 4.5σ of sampling noise. Four-panel: no soft member in a constrained panel, the same unconstrained soft fraction (two-proportion test). Coverage: within 4.5·√2 draw-to-draw sd, measured. Cloud: robust quantiles within 1%. 200 K: RLS leaves FCC with a negative min ω; constrained RLS stays FCC, a(200 K) within 3σ |
| `info` | member-by-member comparisons the mechanism cannot reproduce | reported, not judged |

- **θ_mean is one strictly convex QP, but OSQP leaves it undetermined.**
  - OSQP stops at eps 1e-6 near the optimum in objective but arbitrarily along the fit's flat
    directions.
  - The published θ_mean is itself ~80% (in θ) from the exact KKT optimum: 10% of the fit residual
    in its training predictions, and up to 1.7 THz in its bands.
  - So `verify/mean_qp.jl` requires a rerun to be at least that close to the published θ_mean, in
    predictions and bands, with every constraint holding. Raw ‖Δθ‖ says nothing here.
  - The exact optimum fits 1% better and is also stable.
- **Cutting-plane cascades.**
  - OSQP's ρ schedule is pinned (`adaptive_rho_interval = 25`), so one QP follows one path.
  - A cascade can still end on a different but equally valid member. So member identities are not
    reproducible; the committee statistics are.
- **Hypercube draws.**
  - A draw has no QP in it, but it goes through the PCA of a numerically rank-deficient cloud
    (439 of 684 directions kept for Al_16).
  - It is bit-reproducible at a fixed BLAS thread count, even across nodes. A different BLAS thread
    count (node size) rotates the near-degenerate eigenvectors, and with them every member.
  - That is why the published Al_16 members are not recovered. The published run was most likely on
    a 64-core node; the earlier physics-informed-priors-repro verification did not recover them
    either.
  - What reproduces is the claim: the constrained panels have no soft member, the unconstrained
    ones do.
- **Coverage of one 30-member committee moves from draw to draw.**
  - The `al12_03_noise` job draws 600 members from the same box. By the prefix property its first 30
    are the route's committee.
  - `verify/coverage_noise.jl` scores 20 disjoint committees. The sd is 1.0 point (E) and 0.4 point
    (F) for the full-cloud hypercube.
- **The refit route's Al_12 cloud diagnostics are not comparable.**
  - The members' minimum frequency sits at the first q-point past the Γ cut, where ω² ≈ 1e-4 is
    a sum of large per-basis terms of mixed sign.
  - The refit's coefficients (1e-11 from published) relax to an a_eq 3e-9 Å away. The same θ
    scored on the two Hessians then differs by a median 0.042 THz (`verify/bp_sensitivity.jl`).
  - That changes which members fall under the 0.15 THz margin. The committees built from the cloud
    are still judged.
- **Retrained MACE.** Training reproduces the published log epoch for epoch on these nodes. Its
  downstream results are still judged on physics: a0 and C_ij on target, test RMSE within 15%,
  coverage within 3 points.

## Where the code comes from

Every ACE stage script, the four-panel scripts and `lib/` are verbatim copies of the
physics-informed-priors-repro files that produced the published results (commit 457087f). Each names
its source in a `[PORT]` header, and they differ only where marked `[PORT]`:
- **paths:** outputs under the route root `$OUT`, and models through `model_dir`, so the refit route's
  `ACE_MODELS` is honoured;
- **two memory fixes**, below.

Functions nothing calls were left out of `src/` and `lib/`. The MACE scripts are copies of
`mace_al_elastic_a4032/` with paths changed.

Both fixes leave results unchanged. Both were hidden before by 625 GB nodes, and by prebuilt
Hessian caches.
- **The Bumper leak.**
  - `evaluate_basis` in ACEpotentials bump-allocates its temporaries (`@withalloc`) into the
    task's Bumper buffer and never rewinds it.
  - Unscoped, every site Hessian of the undotted build leaks them as malloc'd slabs the GC cannot
    see: more than 160 GB for one 4×4×4 Al_12 build on 20 threads.
  - `lib/bandpath.jl` and `al16/01` now wrap each site in `Bumper.@no_escape`. That takes the
    build to 28 GB, and H matches the original code's cache to 1e-15 relative.
- **The n×n leverage product.**
  - `corrections()` in `src/POPS` took the leverages as `diag(X * A)`, forming the full
    146,958² product (173 GB).
  - Row-wise dot products give the same leverages (to 7e-18 on a test problem) in O(n·p).

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
                             (run again with 600 members by the al12_03_noise job, for verify)
al12/03_replot_parity.jl     redraw al12/03's parity/calibration figures from its saved predictions
al12/04_naive_ensemble.jl    the unconstrained (naive) ensemble; min ω and relaxed a of both ensembles
al16/01_pinned_ensembles.jl  pinned hypercube: unconstrained and rejection-sampled ensembles
al16/02_bands.jl             native relaxation + Hessian per member → Al_16 four-panel cache
four_panel/build_bands_cache_Al_12.jl  31 native Hessians, each member at its own geometry → Al_12 cache
four_panel/plot_four_panel.jl   the 2×2 figure from the two caches
rls_npt/01_constrained_rls.jl   RLS refit with multi-volume phonon constraints
rls_npt/02_npt.jl            200 K NPT of one θ (Molly, 256 atoms, 30 000 steps)
rls_npt/03_rdf.jl            RDF and coordination of the production frames
rls_npt/04_fig_200K.jl       bands at a(200 K) + RDFs of both models
mace/train.sh                train the MACE model exactly as published                        (refit route)
mace/1_constraints.py … 5_elastic_matscipy.py   readout basis, constraint rows, QP, before/after, matscipy check
mace/pops/p1_rows.py … p6_maxdev_vs_error.py    POPS rows, clouds, committees, evaluation, figures
verify/                      verify.jl (ACE), verify_mace.py (MACE), lib_verify.jl (the tiers), mean_qp.jl,
                             coverage_noise.jl, bp_sensitivity.jl (the measured yardsticks above)
published/                   the published results every route is compared with, and replot reads
models/, mace/models/        the published fitted models (the saved route's starting point)
data/tierB.sha256            checksums of the inputs not committed (below)
```

## Pipeline

```
Al_12   al12/01 ─► al12/02 ─► al12/03 ─► al12/04 ─► build_bands_cache_Al_12 ─┐
Al_16   al16/01 ─► al16/02 ───────────────────────────────────────────────────┴─► plot_four_panel
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
