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
RES    = "$(result.dir)/results"

ROUT   = get(ENV, "OUT", normpath(joinpath(@__DIR__, "..", "repro", "saved")))
COMMITTEE_DIR = get(ENV, "COMMITTEE_DIR", "$ROUT/al12/committee")  
CLOUD_DIR     = get(ENV, "CLOUD_DIR",     "$ROUT/al12/cloud")       
outdir = get(ENV, "OUTDIR", "$ROUT/al12/hypercube"); mkpath(outdir)  

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
Θ_full = full.Θ                               
Θ_30   = Matrix(readdlm("$COMMITTEE_DIR/committee_repaired.csv", ',')')       
@printf("full cloud %d members | legacy forest %d members\n", size(Θ_full,2), size(Θ_30,2))
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

function band_span(members)
    hi = Float64[]; lo = Float64[]
    for θ in members
        F = bands(θ, bp); push!(hi, maximum(F)); push!(lo, minimum(F))
    end
    return lo, hi
end
rng_idx  = randperm(size(Θ_full,2))[1:400]
lo_a, hi_a = band_span([Θ_full[:, j] for j in rng_idx])       # actual constrained members
lo_s, hi_s = band_span(mem_full)                              # hypercube-sampled
lo_3, hi_3 = band_span([Θ_30[:, j] for j in 1:size(Θ_30,2)])
println("\n── dispersion: actual members vs hypercube draws ──")
@printf("  actual full-cloud members (n=400) : ω_max ∈ [%.2f, %.2f] THz, median %.2f\n",
        minimum(hi_a), maximum(hi_a), median(hi_a))
@printf("  hypercube draws from that cloud   : ω_max ∈ [%.2f, %.2f] THz, median %.2f\n",
        minimum(hi_s), maximum(hi_s), median(hi_s))
@printf("  actual forest members (n=%d)       : ω_max ∈ [%.2f, %.2f] THz, median %.2f\n",
        length(hi_3), minimum(hi_3), maximum(hi_3), median(hi_3))
@printf("  → sampled/actual median ω_max ratio = %.2f\n", median(hi_s)/median(hi_a))
flush(stdout)

FIGW = parse(Float64, get(ENV, "FIGW", "540"))
BLU = RGBf(0.0, 0.447, 0.698); GRY = RGBAf(0.45, 0.45, 0.45, 0.35)
TITLE, LAB, TICK = 13, 12, 11

F_all = vcat([bands(θ, bp) for θ in mem_full], [bands(θ, bp) for θ in mem_30])
lo = minimum(minimum.(F_all)); hi = maximum(maximum.(F_all))
pad = 0.06*(hi-lo); ylim = (min(lo-pad, -0.4), hi+pad)

function bands_figure(members, ttl, acc, tag)
    fig = Figure(size=(FIGW, 0.62FIGW), figure_padding=(6, 10, 4, 6))
    Label(fig[1, 1],
          "constrained POPS deltas → hypercube → rejection sample ($(length(members)) drawn)";
          fontsize=TICK, padding=(0, 0, 0, 2))
   
    ax = Axis(fig[2, 1]; xlabel="Wave vector", ylabel="Frequency (THz)",
              title="$ttl  —  accept $(round(acc; digits=2))%",
              titlesize=TITLE, xlabelsize=LAB, ylabelsize=LAB,
              xticklabelsize=TICK, yticklabelsize=TICK,
              xticks=(bp.x_ticks, bp.labels), xgridvisible=false, ygridvisible=false,
              xtickalign=1, ytickalign=1, xticksize=4, yticksize=4)
    for θ in members
        Fθ = bands(θ, bp)
        for b in 1:3bp.Np; lines!(ax, bp.x_vals, Fθ[b,:]; color=GRY, linewidth=0.8); end
    end
    Fm = bands(θ_mean, bp)
    for b in 1:3bp.Np; lines!(ax, bp.x_vals, Fm[b,:]; color=BLU, linewidth=1.8); end
    hlines!(ax, [0.0]; color=:black, linestyle=:dash, linewidth=1.0)
    vlines!(ax, bp.x_ticks; color=(:black, 0.22), linewidth=0.7)
    xlims!(ax, first(bp.x_vals), last(bp.x_vals)); ylims!(ax, ylim...)
    rowgap!(fig.layout, 1, 2)
    save("$outdir/bands_$(tag).pdf", fig)
    save("$outdir/bands_$(tag).png", fig; px_per_unit=4)
    @printf("bands   [%-10s] → %s/bands_%s.{pdf,png}\n", tag, outdir, tag)
end

bands_figure(mem_full, "hypercube from $(size(Θ_full,2)) deltas", acc_f, "fullcloud")
bands_figure(mem_30,   "hypercube from $(size(Θ_30,2)) deltas",   acc_3, "forest30")
flush(stdout)

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

# ── SPLIT FIGURES ────────────────────────────────────────────────────────────
clouds = ((pr_f, "hypercube from $(size(Θ_full,2)) deltas", "fullcloud"),
          (pr_3, "hypercube from $(size(Θ_30,2)) deltas",   "forest30"))
for (pr, ttl, tag) in clouds
    parity_figure(pr, ttl, tag, outdir; FIGW=FIGW)
    calibration_figure(pr, ttl, tag, outdir; FIGW=FIGW)
end
flush(stdout)
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
