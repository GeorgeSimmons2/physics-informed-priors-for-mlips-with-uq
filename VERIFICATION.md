# Verification: a fresh rebuild of everything, 15–16 Oct 2026

## Update, 25 Oct 2026: constrained RLS at OSQP eps 1e-8

`rls_npt/01_constrained_rls.jl` now solves its QPs at eps 1e-8 (was 1e-6). The published constrained RLS θ,
its 200 K NPT and `fig_200K` were regenerated from it on the saved route; the RLS model's NPT is unchanged.
The constrained-RLS rows in the 15–16 Oct record below compare against the previous θ.

| constrained RLS | eps 1e-6 (previous) | eps 1e-8 (published now) |
|---|---|---|
| weighted RMSE | 2.43300 | 2.43288 |
| min ω at a₀ (THz) | +0.406 | +0.423 |
| a(200 K) (Å) | 4.08421 ± 0.00481 | 4.08161 ± 0.00476 |
| expansion | +0.971% | +0.907% |
| ⟨coord⟩, still FCC | 12.00, yes | 12.00, yes |
| min ω at a(200 K), dense path (THz) | +0.235 | +0.248 |

Produced by `bash run.sh setup && bash run.sh data && bash run.sh all`, then `bash run.sh verify saved|refit`.
- **Code:** commit `2017593` of this repository (this file was added after the runs).
- **Environment:** a fresh `.venv/` and a fresh repo-local Julia depot.
- **Cluster:** SLURM `compute` partition (48-core Xeon Gold 6248R nodes).

How each number is judged, and why: README.md, "How closely a rerun must agree".

## Summary

| route | ACE | MACE |
|---|---|---|
| `saved` (published fitted models → everything downstream) | 13 passed, 0 failed, 3 info | 13 passed, 0 failed |
| `refit` (DFT data → refitted ACE models, retrained MACE → everything downstream) | 21 passed, 0 failed, 5 info | 11 passed, 0 failed, 1 info |

### Reproduced bit for bit
- **ACE refit from the DFT data:** design matrices (SHA-256), Y, P and W identical; coefficients
  to 9e-12 (Al_12) and 2e-8 (Al_16).
- **MACE retrained from the DFT data:** all 1,015,225 weights identical to the published model.
- **MACE downstream, on both routes:** the readout QP θ, both 100-member POPS committees, and every
  number and figure array.
- **200 K NPT of the RLS model, saved route:** the same trajectory statistics to every printed digit.

### Reproduced within the method's own precision
- **θ_mean.** Saved route: predictions 1.9e-2 of the fit residual, bands 0.26 THz. The published
  θ_mean is itself 1.0e-1 and 1.68 THz from the exact optimum. Refit route: 7e-6.
- **Committees.** Every Al_12 committee agrees in distribution (z ≤ 3.7 against 4.5), and coverage
  agrees within the measured draw-to-draw noise.
- **Four-panel.**
  - Al_12 unconstrained soft members 29 → 27 of 30; Al_16 4 → 2 (saved) and 3 (refit) of 20.
  - The constrained panels have no soft member on any route.
- **200 K NPT.**
  - RLS leaves FCC on both routes.
  - Constrained RLS stays FCC with a(200 K) = 4.08386 / 4.08397 Å against the published 4.08420 Å
    (σ 0.0048).

## Findings worth knowing
1. **The published θ_mean is an under-converged OSQP solution** of its QP: about 80% from the
   exact KKT optimum in θ, and 3.7e-5 short in objective. The exact optimum fits 1% better and is
   still phonon-stable.
2. **The stability margin is cancellation-limited near Γ.** It is checked at the first q-point
   past Γ, where ω² ≈ 1e-4 is formed from large per-basis terms. A 3e-9 Å change in a_eq moves it by
   about 0.04 THz, which is why the refit route's Al_12 cloud diagnostics are reported, not judged.
3. **Hypercube draws depend on the BLAS thread count.** They are bit-reproducible at a fixed count,
   even across nodes. The published Al_16 members came from a different node size and are not
   recovered; the claims they support are.
4. **Two memory bugs were fixed without changing results.** Both had been hidden by 625 GB nodes and
   prebuilt caches:
   - the Bumper leak in `evaluate_basis` (Hessian build >160 GB → 28 GB);
   - the n×n leverage product in `corrections()` (173 GB).
   - A third stage, `al12_04`, was resubmitted after the second fix; everything else ran once from
     `run.sh all`.

---

# Route `saved`

  loaded H_basis cache undotted_Hbasis_4x4x4_a4.04494.jls
## ACE: route `saved` against published/

| quantity | tier | verdict | detail |
|---|---|---|---|
| Al_12 constrained mean θ_mean (what it predicts) | `converged` | **PASS** | rerun vs published: predictions 1.91e-02, bands max|Δω| 0.257 THz — within the published solution's own distance from the exact optimum (1.03e-01, 1.678 THz); constraints hold; raw ‖Δθ‖/‖θ‖ = 5.14e-02 |
| Al_12 30-member repaired forest | `statistical` | **PASS** | n=30  worst z(mean) = 0.37, worst z(std) = 0.39  (threshold 4.5 σ of sampling noise) |
| Al_12 cloud: min ω per member (THz) | `statistical` | **PASS** | n=73479  quantiles 1/50/99% 0.15/0.1669/0.2715→0.15/0.1666/0.2716 THz (max rel Δ over 1–99% 0.0009, tol 0.010); members identical 66.6%, max|Δ| 0.116 |
| Al_12 cloud: cuts per member | `statistical` | **PASS** | n=73479  quantiles 1/50/99% 0.0/0.0/49.0→0.0/0.0/49.0  (max rel Δ over 1–99% 0.0000, tol 0.010); members identical 96.0%, max|Δ| 112 |
| Al_12 constrained committee (full-cloud hypercube) | `statistical` | **PASS** | n=30  worst z(mean) = 2.90, worst z(std) = 3.69  (threshold 4.5 σ of sampling noise) |
| Al_12 hypercube summary (full_cloud) | `statistical` | **PASS** | directions 63→63, acceptance 10.489→7.177 %, coverage E 90.1→92.4 %, F 97.5→98.5 %; coverage z = 1.58 (E, draw-to-draw sd 1.02), 1.88 (F, sd 0.36) |
| Al_12 hypercube summary (forest_30) | `statistical` | **PASS** | directions 30→30, acceptance 28.571→32.258 %, coverage E 52.4→52.4 %, F 71.7→71.2 %; coverage z = 0.00 (E, draw-to-draw sd 1.93), 0.26 (F, sd 1.38) |
| Al_12 naive ensemble (30 draws) | `statistical` | **PASS** | n=30  worst z(mean) = 0.52, worst z(std) = 2.15  (threshold 4.5 σ of sampling noise) |
| Al_12 naive members: min ω, relaxed a (member by member) | `info` | **info** | max|Δ| = 9.187e+00  median|Δ| = 9.144e-02  ‖Δ‖/‖θ‖ = 4.088e-01  (tol 1.0e-02) |
| Al_12 four-panel row: soft members (unconstrained, constrained) | `statistical` | **PASS** | published (29, 0) → reproduced (27, 0) of 30; unconstrained soft fraction z = 1.04; the constrained panel must have none |
| Al_16 four-panel row: soft members (unconstrained, constrained) | `statistical` | **PASS** | published (4, 0) → reproduced (2, 0) of 20; unconstrained soft fraction z = 0.89; the constrained panel must have none |
| Al_16 unconstrained ensemble: member, relaxed a, min ω (member by member) | `info` | **info** | max|Δ| = 2.555e+00  median|Δ| = 1.271e-07  ‖Δ‖/‖θ‖ = 6.383e-02  (tol 1.0e-02) |
| Al_16 constrained ensemble: member, relaxed a, min ω (member by member) | `info` | **info** | max|Δ| = 2.083e-01  median|Δ| = 3.436e-07  ‖Δ‖/‖θ‖ = 9.775e-03  (tol 1.0e-02) |
| constrained RLS θ (what it predicts) | `converged` | **PASS** | training predictions differ by 5.74e-04 of the published fit's residual (tol 1e-02); weighted RMSE 2.43300→2.43298; raw ‖Δθ‖/‖θ‖ = 1.06e-02 |
| 200 K NPT, RLS: leaves FCC | `statistical` | **PASS** | still FCC false→false, ⟨coord⟩ 9.46→9.46, min ω -4.075→-4.075 THz, a(200 K) 4.24253→4.24253 Å (fluctuation σ 7.0e-03; not compared, the structure has broken down) |
| 200 K NPT, constrained RLS: stays FCC, a(200 K) | `statistical` | **PASS** | still FCC true→true, ⟨coord⟩ 12.00→12.00, min ω +0.353→+0.354 THz, a(200 K) 4.08420→4.08386 Å (fluctuation σ 4.8e-03, |Δ| = 0.1σ) |

13 passed, 0 failed, 0 not yet produced, 3 reported for information.

## MACE: route `saved` against published/

| quantity | tier | verdict | detail |
|---|---|---|---|
| theta0.csv | `exact` | **PASS** | ‖Δθ‖/‖θ‖ = 0.00e+00, max|Δ| = 0.00e+00 (tol 0e+00) |
| theta_constrained_elastic.csv | `converged` | **PASS** | ‖Δθ‖/‖θ‖ = 0.00e+00, max|Δ| = 0.00e+00 (tol 1e-06) |
| theta_ls_elastic.csv | `converged` | **PASS** | ‖Δθ‖/‖θ‖ = 0.00e+00, max|Δ| = 0.00e+00 (tol 1e-06) |
| compare_elastic.txt: RMSEs, a0, C_ij | `converged` | **PASS** | 19 numbers, max|Δ| = 0 (tol 0.1 in the printed units) |
| elastic_matscipy.txt: RMSEs, a0, C_ij | `converged` | **PASS** | 14 numbers, max|Δ| = 0 (tol 0.1 in the printed units) |
| POPS constrained committee | `statistical` | **PASS** | n=100  worst z(mean) = 0.00, worst z(std) = 0.00  (max|Δθ| = 0.00e+00) |
| POPS unconstrained committee | `statistical` | **PASS** | n=100  worst z(mean) = 0.00, worst z(std) = 0.00  (max|Δθ| = 0.00e+00) |
| POPS coverage, constrained E (DFT inside the committee range) | `statistical` | **PASS** | 92.9 % → 92.9 % (tol 1 points) |
| POPS coverage, constrained F (DFT inside the committee range) | `statistical` | **PASS** | 87.5 % → 87.5 % (tol 1 points) |
| POPS coverage, unconstrained E (DFT inside the committee range) | `statistical` | **PASS** | 96.6 % → 96.6 % (tol 1 points) |
| POPS coverage, unconstrained F (DFT inside the committee range) | `statistical` | **PASS** | 89.1 % → 89.1 % (tol 1 points) |
| POPS constrained: spread of C11 over members | `statistical` | **PASS** | 0.0112 → 0.0112 GPa |
| POPS unconstrained: spread of C11 over members | `statistical` | **PASS** | 9.26 → 9.26 GPa |

13 passed, 0 failed, 0 not yet produced or not comparable, 0 reported for information.

---

# Route `refit`

  loaded H_basis cache undotted_Hbasis_4x4x4_a4.04494.jls
## ACE: route `refit` against published/

| quantity | tier | verdict | detail |
|---|---|---|---|
| Al_12_4_6A_2_ fitted coefficients | `converged` | **PASS** | max|Δ| = 9.344e-12  median|Δ| = 6.495e-15  ‖Δ‖/‖θ‖ = 2.608e-12  (tol 1.0e-02) |
| Al_12_4_6A_2_ Y.csv | `exact` | **PASS** | max|Δ| = 0.000e+00  median|Δ| = 0.000e+00  ‖Δ‖/‖θ‖ = 0.000e+00  (tol 0.0e+00) |
| Al_12_4_6A_2_ P.csv | `exact` | **PASS** | max|Δ| = 0.000e+00  median|Δ| = 0.000e+00  ‖Δ‖/‖θ‖ = 0.000e+00  (tol 0.0e+00) |
| Al_12_4_6A_2_ W.csv | `exact` | **PASS** | max|Δ| = 0.000e+00  median|Δ| = 0.000e+00  ‖Δ‖/‖θ‖ = 0.000e+00  (tol 0.0e+00) |
| Al_12_4_6A_2_ design matrix A.csv | `exact` | **PASS** | SHA-256 against the published design matrix |
| Al_16_4_6A_3_ fitted coefficients | `converged` | **PASS** | max|Δ| = 2.111e-08  median|Δ| = 3.916e-13  ‖Δ‖/‖θ‖ = 6.164e-09  (tol 1.0e-02) |
| Al_16_4_6A_3_ Y.csv | `exact` | **PASS** | max|Δ| = 0.000e+00  median|Δ| = 0.000e+00  ‖Δ‖/‖θ‖ = 0.000e+00  (tol 0.0e+00) |
| Al_16_4_6A_3_ P.csv | `exact` | **PASS** | max|Δ| = 0.000e+00  median|Δ| = 0.000e+00  ‖Δ‖/‖θ‖ = 0.000e+00  (tol 0.0e+00) |
| Al_16_4_6A_3_ W.csv | `exact` | **PASS** | max|Δ| = 0.000e+00  median|Δ| = 0.000e+00  ‖Δ‖/‖θ‖ = 0.000e+00  (tol 0.0e+00) |
| Al_16_4_6A_3_ design matrix A.csv | `exact` | **PASS** | SHA-256 against the published design matrix |
| Al_12 constrained mean θ_mean (what it predicts) | `converged` | **PASS** | rerun vs published: predictions 7.04e-06, bands max|Δω| 0.000 THz — within the published solution's own distance from the exact optimum (1.03e-01, 1.678 THz); constraints hold; raw ‖Δθ‖/‖θ‖ = 1.09e-05 |
| Al_12 30-member repaired forest | `statistical` | **PASS** | n=30  worst z(mean) = 1.22, worst z(std) = 1.72  (threshold 4.5 σ of sampling noise) |
| Al_12 cloud: min ω per member (THz) | `info` | **info** | n=73479  quantiles 1/50/99% 0.15/0.1669/0.2715→0.15/0.2105/0.3035 THz (max rel Δ over 1–99% 0.1104, tol 0.010); members identical 1.4%, max|Δ| 0.204; not comparable on the refit route (a_eq 3e-9 Å away; near-Γ ω is cancellation-limited) |
| Al_12 cloud: cuts per member | `info` | **info** | n=73479  quantiles 1/50/99% 0.0/0.0/49.0→0.0/0.0/45.0  (max rel Δ over 1–99% 0.0109, tol 0.010); members identical 76.9%, max|Δ| 112; not comparable on the refit route (a_eq 3e-9 Å away; near-Γ ω is cancellation-limited) |
| Al_12 constrained committee (full-cloud hypercube) | `statistical` | **PASS** | n=30  worst z(mean) = 2.36, worst z(std) = 3.32  (threshold 4.5 σ of sampling noise) |
| Al_12 hypercube summary (full_cloud) | `statistical` | **PASS** | directions 63→65, acceptance 10.489→13.216 %, coverage E 90.1→90.6 %, F 97.5→97.6 %; coverage z = 0.31 (E, draw-to-draw sd 1.04), 0.07 (F, sd 0.49) |
| Al_12 hypercube summary (forest_30) | `statistical` | **PASS** | directions 30→30, acceptance 28.571→36.145 %, coverage E 52.4→54.7 %, F 71.7→75.3 %; coverage z = 0.92 (E, draw-to-draw sd 1.74), 2.05 (F, sd 1.25) |
| Al_12 naive ensemble (30 draws) | `statistical` | **PASS** | n=30  worst z(mean) = 0.52, worst z(std) = 2.15  (threshold 4.5 σ of sampling noise) |
| Al_12 naive members: min ω, relaxed a (member by member) | `info` | **info** | max|Δ| = 9.187e+00  median|Δ| = 9.647e-02  ‖Δ‖/‖θ‖ = 4.088e-01  (tol 1.0e-02) |
| Al_12 four-panel row: soft members (unconstrained, constrained) | `statistical` | **PASS** | published (29, 0) → reproduced (27, 0) of 30; unconstrained soft fraction z = 1.04; the constrained panel must have none |
| Al_16 four-panel row: soft members (unconstrained, constrained) | `statistical` | **PASS** | published (4, 0) → reproduced (3, 0) of 20; unconstrained soft fraction z = 0.42; the constrained panel must have none |
| Al_16 unconstrained ensemble: member, relaxed a, min ω (member by member) | `info` | **info** | max|Δ| = 2.473e+00  median|Δ| = 1.639e-07  ‖Δ‖/‖θ‖ = 5.399e-02  (tol 1.0e-02) |
| Al_16 constrained ensemble: member, relaxed a, min ω (member by member) | `info` | **info** | max|Δ| = 1.752e-01  median|Δ| = 5.316e-07  ‖Δ‖/‖θ‖ = 8.980e-03  (tol 1.0e-02) |
| constrained RLS θ (what it predicts) | `converged` | **PASS** | training predictions differ by 1.99e-04 of the published fit's residual (tol 1e-02); weighted RMSE 2.43300→2.43300; raw ‖Δθ‖/‖θ‖ = 2.44e-02 |
| 200 K NPT, RLS: leaves FCC | `statistical` | **PASS** | still FCC false→false, ⟨coord⟩ 9.46→9.49, min ω -4.075→-2.590 THz, a(200 K) 4.24253→4.24489 Å (fluctuation σ 7.0e-03; not compared, the structure has broken down) |
| 200 K NPT, constrained RLS: stays FCC, a(200 K) | `statistical` | **PASS** | still FCC true→true, ⟨coord⟩ 12.00→12.00, min ω +0.353→+0.353 THz, a(200 K) 4.08420→4.08397 Å (fluctuation σ 4.8e-03, |Δ| = 0.0σ) |

21 passed, 0 failed, 0 not yet produced, 5 reported for information.

## MACE: route `refit` against published/

| quantity | tier | verdict | detail |
|---|---|---|---|
| retrained + constrained: a0 and C_ij on target | `physics` | **PASS** | a0 4.0320 Å, C11/C12/C44/B 116.4/64.7/30.9/82.0 GPa (target 4.032, 116.3/64.8/30.9/82.0) |
| retrained + constrained: test RMSE | `physics` | **PASS** | E 13.65 meV/atom, F 52.8 meV/Å (published 13.65, 52.8; tol +15%) |
| retrained + constrained: matscipy a0, C_ij | `physics` | **PASS** | a0 4.0320 Å, C11/C12/C44 116.4/64.7/30.9 GPa |
| retrained network: weights vs the published model | `info` | **info** | 1015225 weights, max|Δ| = 0.000e+00 — bit-identical |
| POPS constrained committee | `statistical` | **PASS** | n=100  worst z(mean) = 0.00, worst z(std) = 0.00  (max|Δθ| = 0.00e+00) |
| POPS unconstrained committee | `statistical` | **PASS** | n=100  worst z(mean) = 0.00, worst z(std) = 0.00  (max|Δθ| = 0.00e+00) |
| POPS coverage, constrained E (DFT inside the committee range) | `statistical` | **PASS** | 92.9 % → 92.9 % (tol 3 points) |
| POPS coverage, constrained F (DFT inside the committee range) | `statistical` | **PASS** | 87.5 % → 87.5 % (tol 3 points) |
| POPS coverage, unconstrained E (DFT inside the committee range) | `statistical` | **PASS** | 96.6 % → 96.6 % (tol 3 points) |
| POPS coverage, unconstrained F (DFT inside the committee range) | `statistical` | **PASS** | 89.1 % → 89.1 % (tol 3 points) |
| POPS constrained: spread of C11 over members | `statistical` | **PASS** | 0.0112 → 0.0112 GPa |
| POPS unconstrained: spread of C11 over members | `statistical` | **PASS** | 9.26 → 9.26 GPa |

11 passed, 0 failed, 0 not yet produced or not comparable, 1 reported for information.
