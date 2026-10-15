# al12/03_hypercube_rejection.jl — the constrained Al_12 committee (top-right of the four-panel figure).
#
# Builds the POPS hypercube around stage 02's cutting-plane-constrained cloud (and, for comparison,
# around stage 01's 30-member forest), rejection-samples each with the same predicate the members
# were built to (Born, a_eq within 10% of b″, every band-path ω ≥ cut margin), and writes the
# full-cloud committee.  Test-set parity and calibration of both committees are drawn at the
# final MLST display width (FIGW pt; fonts 13/12/11 pt — do not rescale in LaTeX).
#
# Ported from constrained_optimization/al12/03_hypercube_rejection.jl of the
# physics-informed-priors repository; its two band figures (superseded by four_panel/) and a
# printed dispersion diagnostic are dropped.  No sampler, predicate or plotting change.
#
# In:  $OUT/al12/committee/{theta_mean,committee_repaired}.csv, $OUT/al12/cloud/committee_stable.jls,
#      data/Al/manual_df_test_Al.xyz
# Out: $OUT/al12/hypercube/{committee_rejection_full_cloud.csv, hypercube_summary.csv,
#      parity_{fullcloud,forest30}, calibration_{fullcloud,forest30}, parity_calibration_predictions.jls}
# Run: julia --project -t 8 al12/03_hypercube_rejection.jl [n_members] [stride]    Env: OUT, ACE_MODELS, FIGW (540)

include(joinpath(@__DIR__, "..", "lib", "bandpath.jl"))
include(joinpath(@__DIR__, "..", "lib", "parity_calibration.jl"))
using SparseArrays, LinearAlgebra, Serialization, Random
Random.seed!(1234)

element, dataset = :Al, ""
N_cell_fc, N_per_seg = 4, [20, 20, 20, 20, 60]
cut_margin_THz, qΓtol = 0.15, 5e-2
n_members  = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 30
max_attempts = 5_000_000

result = load_model(element, 12, 4, 6, 2; dataset_name=dataset)
model  = result.model; lin_params = result.lin_params; n_params = length(lin_params)
P = result.P
OUT    = get(ENV, "OUT", joinpath(@__DIR__, "..", "repro", "saved"))
COMMITTEE_DIR, CLOUD_DIR = "$OUT/al12/committee", "$OUT/al12/cloud"
outdir = "$OUT/al12/hypercube"; mkpath(outdir)

# ── geometry, Born rows, band path (same as everywhere else) ─────────────────
a_eq = ACEWorkflow.relax_lattice_constant(model, element)
sys0 = ACEWorkflow.Elasticity.reference_system(element; a=a_eq)
L0   = ustrip.(ACEWorkflow.Elasticity.lattice_matrix(sys0.cell.cell_vectors))
eV_to_GPa = 160.2176621/abs(det(L0))
H_el = elastic_hessian_basis(model; element=element, a=a_eq)
c11_0 = reshape(H_el,36,n_params)[1,:]; c12_0 = reshape(H_el,36,n_params)[7,:]; c44_0 = reshape(H_el,36,n_params)[22,:]
lattice_basis(a) = ustrip.(u"eV", ACEpotentials.Models.potential_energy_basis(ACEWorkflow.Elasticity.reference_system(element; a=a), model))
b_prime        = ForwardDiff.derivative(lattice_basis, a_eq)
b_double_prime = ForwardDiff.derivative(a -> ForwardDiff.derivative(lattice_basis, a), a_eq)
born_rows  = vcat(c44_0', (c11_0.-c12_0)', (c11_0.+2 .*c12_0)', b_double_prime')
born_lower = [0.1, 1.0, 0.1, 1e-9]

bp = bandpath_Dk(result, model, element, a_eq, N_cell_fc; N_per_seg=N_per_seg)
ω2_cut = (cut_margin_THz / FREQ_THz)^2
@printf("a_eq = %.5f Å; band path %d q-points\n", a_eq, length(bp.Bq)); flush(stdout)

# ── the two clouds ───────────────────────────────────────────────────────────
θ_mean = vec(readdlm("$COMMITTEE_DIR/theta_mean.csv", ','))
full   = deserialize("$CLOUD_DIR/committee_stable.jls")
Θ_full = full.Θ                                            # 91 × N, stable only
Θ_30   = Matrix(readdlm("$COMMITTEE_DIR/committee_repaired.csv", ',')')
@printf("full cloud %d members | 30-member forest %d members\n", size(Θ_full,2), size(Θ_30,2))
flush(stdout)

function build_and_sample(Θ, label)
    deltas = Matrix(Θ' .- θ_mean')                          # N × K
    heig, hb = hypercube(deltas)
    widths = hb[2,:] .- hb[1,:]
    @printf("\n[%s] hypercube: %d directions retained, width mean %.4g max %.4g\n",
            label, size(heig,2), mean(widths), maximum(widths)); flush(stdout)
    K_ref = dot(θ_mean, b_double_prime)
    nck = Ref(0); nborn = Ref(0); naeq = Ref(0)
    pred = θ -> begin
        nck[] += 1
        all(born_lower .<= born_rows*θ) || return false
        nborn[] += 1
        abs(dot(b_prime, θ .- θ_mean)/K_ref) <= 0.1 || return false
        naeq[] += 1
        min_freq_stable(θ, bp; qΓtol=qΓtol) >= cut_margin_THz - 1e-6
    end
    t = @elapsed (mat, _) = rejection_sample_hypercube(heig, hb, θ_mean, pred;
                              number_of_committee_members=n_members,
                              max_attempts=max_attempts)
    @printf("[%s] funnel: %d drawn → %d Born → %d a_eq → %d accepted (%.3f%%)  [%.1f s]\n",
            label, nck[], nborn[], naeq[], n_members, 100n_members/max(nck[],1), t)
    members = [mat[:, i] for i in 1:size(mat,2)]
    mf = [min_freq_stable(θ, bp; qΓtol=qΓtol) for θ in members]
    @printf("[%s] sampled min ω ∈ [%.4f, %.4f] THz\n", label, minimum(mf), maximum(mf))
    flush(stdout)
    return members, heig, hb, 100n_members/max(nck[],1)
end

mem_full, heig_f, hb_f, acc_f = build_and_sample(Θ_full, "full cloud")
mem_30,   heig_3, hb_3, acc_3 = build_and_sample(Θ_30,   "30-member")

# FIGW is the width each figure will be DISPLAYED at, in points; fonts stay 13/12/11 pt.
FIGW = parse(Float64, get(ENV, "FIGW", "540"))

writedlm("$outdir/committee_rejection_full_cloud.csv", reduce(hcat, mem_full)', ',')

# ── test-set parity + calibration ────────────────────────────────────────────
TEST   = joinpath(@__DIR__, "..", "data", "Al", "manual_df_test_Al.xyz")
stride = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 20
@printf("\n── test-set predictions (stride %d) ──\n", stride); flush(stdout)
pr_f = committee_predictions_fast(model, mem_full, TEST; stride=stride, point_params=θ_mean,
                             per_atom=true)
pr_3 = committee_predictions_fast(model, mem_30,   TEST; stride=stride, point_params=θ_mean,
                             per_atom=true)
@printf("  %d test configurations\n", pr_f.n); flush(stdout)

cover(t, lo, hi) = 100*(1 - mean((t .< lo) .| (t .> hi)))
rmse(p, t)       = sqrt(mean((p .- t).^2))

# One figure per hypercube type, each with (a) energy and (b) force panels (lib/parity_calibration.jl).
clouds = ((pr_f, "hypercube from $(size(Θ_full,2)) deltas", "fullcloud"),
          (pr_3, "hypercube from $(size(Θ_30,2)) deltas",   "forest30"))
for (pr, ttl, tag) in clouds
    parity_figure(pr, ttl, tag, outdir; FIGW=FIGW)
    calibration_figure(pr, ttl, tag, outdir; FIGW=FIGW)
end
flush(stdout)

# the raw predictions, so the figures can be rebuilt (e.g. at another FIGW) without re-predicting
serialize("$outdir/parity_calibration_predictions.jls",
          (; pr_f, pr_3, stride, n_members,
             label_full = "hypercube from $(size(Θ_full,2)) deltas",
             label_30   = "hypercube from $(size(Θ_30,2)) deltas"))
println("predictions → $outdir/parity_calibration_predictions.jls")

println("\n══ COVERAGE ═══════════════════════════════════════════════════")
@printf("%-14s %10s %10s %12s %12s\n", "cloud", "E cov %", "F cov %", "E RMSE eV", "F RMSE eV/Å")
for (nm, pr) in (("full cloud", pr_f), ("forest (30)", pr_3))
    @printf("%-14s %10.1f %10.1f %12.4g %12.4g\n", nm,
            cover(pr.tE, pr.loE, pr.hiE), cover(pr.tF, pr.loF, pr.hiF),
            rmse(pr.pE, pr.tE), rmse(pr.pF, pr.tF))
end

open("$outdir/hypercube_summary.csv", "w") do io
    println(io, "cloud,n_members,n_directions,width_mean,width_max,acceptance_pct,E_coverage_pct,F_coverage_pct,E_rmse_eV_per_atom,F_rmse_eV_per_A")
    for (nm, Θ, he, hb, ac, pr) in (("full_cloud", Θ_full, heig_f, hb_f, acc_f, pr_f),
                                    ("forest_30",  Θ_30,   heig_3, hb_3, acc_3, pr_3))
        @printf(io, "%s,%d,%d,%.6g,%.6g,%.4f,%.3f,%.3f,%.6g,%.6g\n", nm, size(Θ,2), size(he,2),
                mean(hb[2,:].-hb[1,:]), maximum(hb[2,:].-hb[1,:]), ac,
                cover(pr.tE, pr.loE, pr.hiE), cover(pr.tF, pr.loF, pr.hiF),
                rmse(pr.pE, pr.tE), rmse(pr.pF, pr.tF))
    end
end
println("summary → $outdir/hypercube_summary.csv")
