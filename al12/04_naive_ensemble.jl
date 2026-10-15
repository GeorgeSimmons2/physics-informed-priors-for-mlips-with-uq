# al12/04_naive_ensemble.jl — the unconstrained (naive) Al_12 ensemble and the Al_12 band curves.
#
# Draws the naive POPS ensemble from the same top-50%-leverage cloud stage 02 constrained (no
# constraints at all), and evaluates both Al_12 ensembles at the geometry their members hold:
#
#   NAIVE        native 4×4×4 Hessian per member, each relaxed to its OWN lattice constant —
#                nothing pins these, and evaluating them at a_eq would fold residual stress into
#                the phonons, so a soft mode would be unattributable between bad parameters and
#                the wrong volume.
#   CONSTRAINED  stage 03's members on the prebuilt undotted Hessian at a_eq — exact for them
#                because b′·θ = 0 puts their equilibrium there (checked below, not assumed).
#
# The naive arm is drawn with `sample_hypercube`, whose member i is row i of rand(n_members, d), so
# the ensemble depends on n_members (30) as well as the seed.  Each native build gets its own model
# copy: set_linear_parameters! mutates, so a shared model would be a silent data race.
#
# Ported from constrained_optimization/al12/04_naive_ensemble.jl of the physics-informed-priors
# repository.  It now writes the band curves it computes straight into the four-panel cache
# (previously rebuilt by a separate bands_four_panel/build_bands_cache_Al_12.jl, 31 more native
# Hessians), and its own figures, superseded by four_panel/, are dropped.
#
# In:  $OUT/al12/committee/theta_mean.csv, $OUT/al12/hypercube/committee_rejection_full_cloud.csv
# Out: $OUT/al12/naive/{samples_naive.csv, min_freq_naive.csv, min_freq_constrained.csv,
#      naive_vs_constrained.jls, bands_four_panel_Al_12.jls}
# Run: julia --project -t 40 al12/04_naive_ensemble.jl [n_members]    Env: OUT, ACE_MODELS, HESS_THREADS

include(joinpath(@__DIR__, "..", "lib", "bandpath.jl"))
using Random, Serialization
Random.seed!(1234)

element, dataset  = :Al, ""
N_cell_fc         = 4                          # clean cell: half-box 8.04 Å vs 6 Å cutoff
N_per_seg         = [20, 20, 20, 20, 60]       # identical to the constrained study
lev_percentile    = 0.5                        # the cloud the constrained study started from
qΓtol             = 5e-2
unstable_tol      = -0.05
n_members         = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 30
HESS_THREADS      = parse(Int, get(ENV, "HESS_THREADS", string(Threads.nthreads())))

result = load_model(element, 12, 4, 6, 2; dataset_name=dataset)
model  = result.model; lin_params = result.lin_params; n_params = length(lin_params)
P = result.P
Ap = Diagonal(result.W) * result.A / P
Yw = result.W .* result.Y
OUT    = get(ENV, "OUT", joinpath(@__DIR__, "..", "repro", "saved"))
SRC, COMMITTEE_DIR = "$OUT/al12/hypercube", "$OUT/al12/committee"
outdir = "$OUT/al12/naive"; mkpath(outdir)
structure = AtomsBuilder.Chemistry.symmetry(element)
@printf("Model %s: %d params, %d observations, %d Julia threads (%d for native Hessians)\n",
        result.name, n_params, size(Ap,1), Threads.nthreads(), HESS_THREADS); flush(stdout)

# ── geometry + the prebuilt undotted band path (the CONSTRAINED evaluator) ───
ACEpotentials.Models.set_linear_parameters!(model, lin_params)
a_eq = ACEWorkflow.relax_lattice_constant(model, element)
@printf("a_eq = %.5f Å; %d×%d×%d cell, half-box %.2f Å vs 6 Å cutoff\n",
        a_eq, N_cell_fc, N_cell_fc, N_cell_fc, N_cell_fc*a_eq/2); flush(stdout)
t_bp = @elapsed bp = bandpath_Dk(result, model, element, a_eq, N_cell_fc; N_per_seg=N_per_seg)
@printf("undotted band path: %d q-points, %d branches  [%.1f min]\n",
        length(bp.Bq), 3bp.Np, t_bp/60); flush(stdout)

lattice_basis(a) = ustrip.(u"eV", ACEpotentials.Models.potential_energy_basis(
                       ACEWorkflow.Elasticity.reference_system(element; a=a), model))
b_prime        = ForwardDiff.derivative(lattice_basis, a_eq)
b_double_prime = ForwardDiff.derivative(a -> ForwardDiff.derivative(lattice_basis, a), a_eq)

# ── evaluators, both returning the same shape so the plotter is shared ───────
function undotted_bands(θ)
    F = bands(θ, bp)
    return (F=F, x_vals=bp.x_vals, x_ticks=bp.x_ticks, labels=bp.labels, Np=bp.Np,
            minω=minimum(F[:, bp.qnorm .> qΓtol]), a=a_eq)
end

# `m` is a per-task model copy — never pass the shared one, it gets mutated.
function native_bands_one(m, θ, a_ref)
    ACEpotentials.Models.set_linear_parameters!(m, θ)
    a = try
        aa = ACEWorkflow.relax_lattice_constant(m, element)
        (0.9a_ref < aa < 1.1a_ref) ? aa :
            (@warn "relaxed a = $aa Å implausible; falling back to a_ref"; a_ref)
    catch e
        @warn "relax failed ($e); falling back to a_ref"; a_ref
    end
    sp, ss = bulk_prim_super(element; a=a, N_cell=N_cell_fc)
    fc = precompute_force_constants(sp, ss, m)
    ql, xv, xt, lb, _ = _band_path(structure, fc.L; N_per_seg=N_per_seg)
    Np = fc.Np; qn = norm.(ql)
    F = Matrix{Float64}(undef, 3Np, length(ql))
    for (iq, q) in enumerate(ql)
        ev = eigvals(Hermitian(dynamical_matrix_from_fc(fc, q)))
        F[:, iq] = sign.(ev) .* sqrt.(abs.(ev)) .* FREQ_THz
    end
    return (F=F, x_vals=xv, x_ticks=xt, labels=lb, Np=Np,
            minω=minimum(F[:, qn .> qΓtol]), a=a)
end

function native_bands_many(θs, a_ref; label="")
    n = length(θs)
    out = Vector{Any}(undef, n)
    nt  = clamp(HESS_THREADS, 1, min(Threads.nthreads(), n))
    done = Threads.Atomic{Int}(0)
    @printf("  [%s] %d native builds on %d tasks, %d deep …\n", label, n, nt, cld(n, nt))
    flush(stdout)
    t = @elapsed @sync for k in 1:nt
        Threads.@spawn begin
            m = deepcopy(model)                       # private to this task
            for i in k:nt:n
                out[i] = native_bands_one(m, θs[i], a_ref)
                d = Threads.atomic_add!(done, 1) + 1
                d % max(1, n ÷ 10) == 0 && (@printf("    %d/%d\n", d, n); flush(stdout))
            end
        end
    end
    @printf("  [%s] done in %.1f min (%.1f s/member)\n", label, t/60, t/n); flush(stdout)
    return Vector{typeof(out[1])}(out)
end

# ── ensemble A: NAIVE.  Same top-50%-leverage cloud, no constraints at all ───
# `corrections` is the library's own POPS routine; leverage_percentile = 0.5 selects
# exactly the population that al12/02 constrained.
println("\n── naive: unconstrained POPS cloud → hypercube → sample ──"); flush(stdout)
naive_deltas = corrections(Ap, Yw, P; leverage_percentile=lev_percentile)
@printf("  cloud: %d unconstrained deltas, %d params\n", size(naive_deltas)...)
eig_n, bd_n = hypercube(naive_deltas)
@printf("  hypercube: %d directions retained\n", size(eig_n, 2)); flush(stdout)
mat_n, _ = sample_hypercube(eig_n, bd_n, lin_params; number_of_committee_members=n_members)
mem_n = [mat_n[:, i] for i in 1:size(mat_n, 2)]

# ── ensemble B: CONSTRAINED.  Stage 03's members, not a re-draw ─────────────
REJ = "$SRC/committee_rejection_full_cloud.csv"
isfile(REJ) || error("""
    missing $REJ
    Run al12/03_hypercube_rejection.jl first — this script reuses its committee, so the
    constrained panel is the same ensemble, not an independent re-draw.""")
rej = readdlm(REJ, ',')
size(rej, 2) == n_params || error("$REJ is $(size(rej,2)) wide, model has $n_params params")
mem_c  = [collect(Float64, rej[i, :]) for i in 1:min(n_members, size(rej,1))]
θ_mean = vec(readdlm("$COMMITTEE_DIR/theta_mean.csv", ','))
@printf("\n── constrained: %d members reused from %s ──\n", length(mem_c), basename(REJ))
length(mem_c) == n_members ||
    @warn "constrained committee has only $(length(mem_c)) members; naive has $n_members — panels are not member-matched"

# ── IS THE PREBUILT a_eq HESSIAN ACTUALLY VALID FOR THESE MEMBERS? ──────────
# The whole justification for not rebuilding on the constrained side is b′·θ = 0.
# Δa ≈ −b′·θ / b″·θ is the Newton step from a_eq to the member's own equilibrium; if
# that is not ~0 the prebuilt operator is being used at the wrong geometry and the
# constrained panel is wrong.  Checked, not assumed.
Δa_c = [-dot(b_prime, θ)/dot(b_double_prime, θ) for θ in mem_c]
@printf("\nPIN CHECK (constrained): |Δa| ≤ %.3e Å (%.4f%% of a_eq)  %s\n",
        maximum(abs.(Δa_c)), 100*maximum(abs.(Δa_c))/a_eq,
        maximum(abs.(Δa_c))/a_eq < 1e-3 ? "← pinned; prebuilt a_eq Hessian is exact" :
                                          "← NOT pinned; rebuild these natively too")
maximum(abs.(Δa_c))/a_eq < 1e-3 ||
    @warn "constrained members are not at a_eq — the prebuilt Hessian is the wrong operator for them"
flush(stdout)

# ── evaluate ────────────────────────────────────────────────────────────────
println("\n── native Hessians for the naive ensemble ──"); flush(stdout)
nat_n = native_bands_many(mem_n, a_eq; label="naive")
cen_n = native_bands_one(deepcopy(model), lin_params, a_eq)
ACEpotentials.Models.set_linear_parameters!(model, lin_params)   # restore the shared model

nat_c = [undotted_bands(θ) for θ in mem_c]
cen_c = undotted_bands(θ_mean)

minf_n = [m.minω for m in nat_n]; a_n = [m.a for m in nat_n]
minf_c = [m.minω for m in nat_c]

@printf("\nmean model (lin_params)  : min ω = %+.4f THz at its own a = %.5f Å\n",
        cen_n.minω, cen_n.a)
@printf("constrained mean (θ_mean): min ω = %+.4f THz at a_eq\n", cen_c.minω)
@printf("NAIVE       : min ω ∈ [%+.4f, %+.4f], median %+.4f — %d/%d unstable (< %.2f THz)\n",
        minimum(minf_n), maximum(minf_n), median(minf_n),
        count(<(unstable_tol), minf_n), length(minf_n), unstable_tol)
@printf("CONSTRAINED : min ω ∈ [%+.4f, %+.4f], median %+.4f — %d/%d unstable\n",
        minimum(minf_c), maximum(minf_c), median(minf_c),
        count(<(unstable_tol), minf_c), length(minf_c))

@printf("\nrelaxed lattice constant, naive ensemble (a_eq = %.5f Å):\n", a_eq)
@printf("  a ∈ [%.5f, %.5f] Å, spread %.4f Å (%.2f%% of a_eq)\n",
        minimum(a_n), maximum(a_n), maximum(a_n)-minimum(a_n),
        100*(maximum(a_n)-minimum(a_n))/a_eq)
println("  → this spread is why the naive side is rebuilt natively; the constrained")
println("    side sits at a_eq by construction, so one prebuilt Hessian covers it.")
flush(stdout)

# ── save: CSVs, the summary, and the band curves in the four-panel cache format ───
# One y-range over BOTH ensembles and both centres, so the row compares like with like.
allF = vcat([m.F for m in nat_n], [m.F for m in nat_c], [cen_n.F, cen_c.F])
ylo = minimum(minimum.(allF)); yhi = maximum(maximum.(allF)); pad = 0.06*(yhi-ylo)
ylim = (min(ylo-pad, -0.4), yhi+pad)
@printf("\nrow frequency axis: [%.2f, %.2f] THz\n", ylim...)

writedlm("$outdir/min_freq_naive.csv",       hcat(minf_n, a_n), ',')
writedlm("$outdir/min_freq_constrained.csv", hcat(minf_c, Δa_c), ',')
writedlm("$outdir/samples_naive.csv", mat_n', ',')
serialize("$outdir/naive_vs_constrained.jls",
          (; minf_n, minf_c, a_n, Δa_c, ylim, a_eq, N_cell_fc, n_members,
             cen_n_min=cen_n.minω, cen_c_min=cen_c.minω,
             unstable_tol, lev_percentile, seed=1234))
# The two Al_12 panels are centred on different models — lin_params at its own geometry on the
# left, θ_mean on the right — hence `cen`; field names otherwise match al16/02's cache.
serialize("$outdir/bands_four_panel_Al_12.jls",
          (; res = [nat_n, nat_c], cen = [cen_n, cen_c], mean_b = cen_c,
             tags = ["unconstrained", "constrained"], ylim,
             N_CELL = N_cell_fc, N_PER_SEG = N_per_seg, UNSTABLE = unstable_tol, n_params, a_eq,
             model_name = result.name))
println("data  → $outdir/{samples_naive,min_freq_naive,min_freq_constrained}.csv, naive_vs_constrained.jls")
println("bands → $outdir/bands_four_panel_Al_12.jls")
