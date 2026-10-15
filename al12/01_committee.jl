# al12/01_committee.jl — the constrained mean model θ_mean and the 30-member repaired forest.
#
# Al_12_4_6A_2_, phonons on the UNDOTTED per-basis Hessian at a_eq (4×4×4 cell, dense Γ→K):
#   1. constrain a_eq: the mean fit with b′·θ = 0, Born stability, and cutting-plane rows until
#      every band-path frequency is ≥ cut_margin (the mean must obey the committee's constraints)
#   2. a 30-member POPS forest (5 top-leverage, 10 top-residual, 15 random observations), each
#      member re-solved with its interpolation equality, b′·θ = 0 and Born, then cut until stable
# Stages 03 (the 30-member hypercube) and 04 (the constrained centre) read both outputs.
#
# Ported from constrained_optimization/al12/01_committee_ncell4_densek.jl of the
# physics-informed-priors repository; the rejection-sampled 30-member committee, its band plots
# and test-set coverage, which nothing downstream reads, are dropped.
#
# Out: $OUT/al12/committee/{theta_mean.csv, committee_repaired.csv}
# Run: julia --project -t 20 al12/01_committee.jl        Env: OUT, ACE_MODELS, RHO_INTERVAL (25)

include(joinpath(@__DIR__, "..", "lib", "bandpath.jl"))
using SparseArrays, OSQP, Random
# OSQP's rho schedule is the only timing-dependent input on the QP path.  At its default
# (adaptive_rho_interval = 0) OSQP derives the iterations between rho updates from measured
# wall-clock time, so two runs take different iterate paths and, after a cutting-plane cascade,
# accept different vectors.  Pinning it makes reruns follow one path.
RHO_INTERVAL = parse(Int, get(ENV, "RHO_INTERVAL", "25"))

Random.seed!(1234)

element        = :Al
dataset        = ""                        # "" → models/Al_12_4_6A_2_ (the full training set)
N_cell_fc      = 4                         # 4×4×4: half-box 8.08 Å, clear of the 6 Å cutoff
N_per_seg      = [20, 20, 20, 20, 60]      # dense Γ→K, where the soft modes sit
cut_margin_THz = 0.15
n_lev, n_res, n_rand = 5, 10, 15
max_cuts       = 40

result     = load_model(element, 12, 4, 6, 2; dataset_name=dataset)
model      = result.model; lin_params = result.lin_params; n_params = length(lin_params)
P = result.P; Ap = Diagonal(result.W)*result.A/P; Yw = result.W.*result.Y; λ = 1.0/size(Ap,1)
OUT    = get(ENV, "OUT", joinpath(@__DIR__, "..", "repro", "saved"))
outdir = "$OUT/al12/committee"; mkpath(outdir)
@printf("Model %s: %d params, %d threads.  Outputs → %s\n", result.name, n_params, Threads.nthreads(), outdir)

# ── reference geometry, Born + a_eq rows ─────────────────────────────────────
a_eq = ACEWorkflow.relax_lattice_constant(model, element)
@printf("a_eq = %.5f Å\n", a_eq)
sys0 = ACEWorkflow.Elasticity.reference_system(element; a=a_eq)
L0   = ustrip.(ACEWorkflow.Elasticity.lattice_matrix(sys0.cell.cell_vectors)); eV_to_GPa = 160.2176621/abs(det(L0))
H_el = elastic_hessian_basis(model; element=element, a=a_eq)
c11_0 = reshape(H_el,36,n_params)[1,:]; c12_0 = reshape(H_el,36,n_params)[7,:]; c44_0 = reshape(H_el,36,n_params)[22,:]
born(θ) = (dot(c11_0,θ)*eV_to_GPa, dot(c12_0,θ)*eV_to_GPa, dot(c44_0,θ)*eV_to_GPa)
lattice_basis(a) = ustrip.(u"eV", ACEpotentials.Models.potential_energy_basis(ACEWorkflow.Elasticity.reference_system(element; a=a), model))
b_prime        = ForwardDiff.derivative(lattice_basis, a_eq)
b_double_prime = ForwardDiff.derivative(a -> ForwardDiff.derivative(lattice_basis, a), a_eq)
born_rows  = vcat(c44_0', (c11_0.-c12_0)', (c11_0.+2 .*c12_0)', b_double_prime')
born_lower = [0.1, 1.0, 0.1, 1e-9]

# ── undotted Hessian (once) → band-path D_k(q) over the full path ────────────
bp = bandpath_Dk(result, model, element, a_eq, N_cell_fc; N_per_seg=N_per_seg)
ω2_cut = (cut_margin_THz / FREQ_THz)^2
let fm = bands(lin_params, bp)
    @printf("Undotted band-path, mean model: ω ∈ [%.3f, %.3f] THz\n", minimum(fm), maximum(fm))
end

# ── constrain a_eq (mean fit) + leverage/residual forest ─────────────────────
C = Symmetric(Ap'*Ap .+ λ.*(P'*P)); Cf = cholesky(C)
AtX = Cf\Matrix(Ap'); θ̃ = Cf\(Ap'*Yw)
leverage = vec(sum(Ap'.*AtX; dims=1)); residual = Yw .- Ap*θ̃
Hqp = sparse(Ap'*Ap .+ λ.*(P'*P)); qqp = -(Ap'*Yw); osqp = OSQP.Model()
function constrain_member(i, extra_rows, extra_lower)
    rows = vcat(born_rows, extra_rows); lowers = vcat(born_lower, extra_lower)
    A_full = vcat(sparse(Ap[i,:]'), sparse(b_prime'/P), sparse(rows/P))
    l = vcat([Yw[i]],[0.0],lowers); u = vcat([Yw[i]],[0.0],fill(Inf,length(lowers)))
    OSQP.setup!(osqp; P=Hqp, q=qqp, A=A_full, l=l, u=u, max_iter=4_000_000, check_termination=25, verbose=false, eps_abs=1e-6, eps_rel=1e-6, adaptive_rho_interval=RHO_INTERVAL)
    r = OSQP.solve!(osqp); return P \ r.x, r.info.status
end
function mean_fit(extra_rows=zeros(0,n_params), extra_lower=Float64[])   # b′=0, Born + extra
    rows = vcat(born_rows, extra_rows); lowers = vcat(born_lower, extra_lower)
    A_full = vcat(sparse(b_prime'/P), sparse(rows/P))
    l = vcat([0.0],lowers); u = vcat([0.0],fill(Inf,length(lowers)))
    OSQP.setup!(osqp; P=Hqp, q=qqp, A=A_full, l=l, u=u, max_iter=4_000_000, check_termination=25, verbose=false, eps_abs=1e-6, eps_rel=1e-6, adaptive_rho_interval=RHO_INTERVAL)
    return P \ OSQP.solve!(osqp).x
end
# the MEAN must obey the same phonon constraints as the committee — sweep & cut it too
θ_mean = mean_fit(); mean_er = zeros(0,n_params); mean_el = Float64[]; mean_cuts = 0
for it in 0:max_cuts
    global θ_mean, mean_er, mean_el, mean_cuts
    soft = soft_modes(θ_mean, bp, ω2_cut); isempty(soft) && break
    for (iq,e) in soft; mean_er = vcat(mean_er, cut_row(iq,e,bp)'); push!(mean_el, ω2_cut); end
    mean_cuts += length(soft); θ_mean = mean_fit(mean_er, mean_el)
end
min_freq_stable(θ_mean, bp) >= cut_margin_THz - 1e-6 || @warn "mean model still soft after $mean_cuts cuts"
@printf("Constrained mean (%d phonon cuts): C11=%.1f C12=%.1f C44=%.1f GPa, b′·θ=%.1e, min ω=%.3f THz\n",
        mean_cuts, born(θ_mean)..., dot(b_prime, θ_mean), min_freq_stable(θ_mean, bp))

lev_idx = sortperm(leverage; rev=true)[1:n_lev]
res_idx = Int[]; for i in sortperm(abs.(residual); rev=true); i in lev_idx && continue; push!(res_idx,i); length(res_idx)==n_res && break; end
taken = Set(vcat(lev_idx,res_idx)); rand_idx = Int[]
while length(rand_idx) < n_rand; i = rand(1:length(Yw)); (i in taken) && continue; push!(rand_idx,i); push!(taken,i); end
selected = vcat(lev_idx, res_idx, rand_idx)

# ── a_eq-constrain each forest member, phonon-check, cut the soft ones ───────
println("\n── a_eq-constrain + phonon-repair the proposal forest ───────")
committee = Vector{Vector{Float64}}(undef, length(selected)); n_cuts = zeros(Int, length(selected))
for (k, i) in enumerate(selected)
    extra_rows = zeros(0, n_params); extra_lower = Float64[]
    θ, _ = constrain_member(i, extra_rows, extra_lower)
    for it in 0:max_cuts
        soft = soft_modes(θ, bp, ω2_cut)
        isempty(soft) && break
        it == max_cuts && (@warn "obs $i hit max_cuts (min ω=$(round(min_freq_stable(θ,bp);digits=3)))"; break)
        for (iq, e) in soft; extra_rows = vcat(extra_rows, cut_row(iq, e, bp)'); push!(extra_lower, ω2_cut); end
        n_cuts[k] += length(soft)
        θ, _ = constrain_member(i, extra_rows, extra_lower)
    end
    committee[k] = θ
end
minf = [min_freq_stable(θ, bp) for θ in committee]
@printf("Repaired (≥1 cut): %d / %d;  min band ω ∈ [%.3f, %.3f] THz\n",
        count(>(0), n_cuts), length(selected), minimum(minf), maximum(minf))

writedlm("$outdir/committee_repaired.csv", reduce(hcat, committee)', ',')
writedlm("$outdir/theta_mean.csv", θ_mean, ',')          # phonon-repaired constrained mean
println("→ $outdir/{theta_mean,committee_repaired}.csv")
