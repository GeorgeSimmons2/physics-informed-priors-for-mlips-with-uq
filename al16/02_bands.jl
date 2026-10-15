# al16/02_bands.jl — the Al_16 band curves of both ensembles (bottom row of the four-panel figure).
#
# Every member gets an INDEPENDENT geometry optimisation and its OWN native Hessian.
# Stage 1 screened proposals with a single undotted Hessian at a_eq, which is exact
# only because the members are pinned there — but that is the screen, not the result.
# Rebuilding natively at each member's own relaxed lattice constant means the figure
# does not inherit the screening assumption, and the two can be compared:
# stage 1's min ω is reported alongside this one as a CONTROL.  They should agree
# closely for the constrained ensemble; a gap means the pin drifted.
#
# ── WHY BOTH PANELS SHARE A FREQUENCY AXIS ──────────────────────────────────
# The comparison is how much WIDER the unconstrained ensemble is.  Independent axes
# would autoscale each to its own spread and destroy exactly that.
#
# ── PARALLELISM ─────────────────────────────────────────────────────────────
# One force-constant build per member, each task with its OWN deep copy of the model —
# set_linear_parameters! mutates, so a shared model is a data race giving quietly wrong
# bands rather than a crash.  Chunked @spawn, not @threads with threadid().
#
# Ported from constrained_optimization/al16/02_bands_two_ensembles.jl of the physics-informed-
# priors repository; its own figure (superseded by four_panel/) is dropped, and its comparison
# with the published members moved to verify.jl.
#
# In:  $OUT/al16/ensembles/{ensemble_unconstrained,ensemble_constrained}.csv, pinned_ensembles.jls
# Out: $OUT/al16/bands/{bands_two_ensembles.jls, bands_unconstrained.csv, bands_constrained.csv}
# Run: julia --project -t 40 al16/02_bands.jl     Env: OUT, ACE_MODELS, N_CELL (4), N_PER_SEG, UNSTABLE (-0.05), QOI_THREADS

include(joinpath(@__DIR__, "..", "lib", "bandpath.jl"))
using Serialization

element     = :Al
N_CELL      = parse(Int,     get(ENV, "N_CELL", "4"))
# MUST match stage 1's sampling, or the control below compares different q-grids and
# reports a difference that has nothing to do with the Hessians.
N_PER_SEG   = haskey(ENV, "N_PER_SEG") ? parse(Int, ENV["N_PER_SEG"]) :
                                         [20, 20, 20, 20, 60]
UNSTABLE    = parse(Float64, get(ENV, "UNSTABLE", "-0.05"))
qΓtol       = 5e-2
QOI_THREADS = parse(Int,     get(ENV, "QOI_THREADS", string(Threads.nthreads())))

OUT      = get(ENV, "OUT", joinpath(@__DIR__, "..", "repro", "saved"))
MODELDIR = model_dir("Al_16_4_6A_3_")                   # honours ACE_MODELS (the refit route)
SRC      = "$OUT/al16/ensembles"
outdir   = "$OUT/al16/bands"; mkpath(outdir)

# ── model (fast load: nothing here needs the 1.9 GB A.csv) ──────────────────
model, _ = ACEpotentials.load_model("$MODELDIR/Al_16_4_6A_3.json")
lin_params = vec(readdlm("$MODELDIR/lin_params.csv", ','))
ACEpotentials.Models.set_linear_parameters!(model, lin_params)
n_params = length(lin_params)
@printf("fast load: %d parameters (A.csv skipped)\n", n_params)

function read_ens(tag)
    f = "$SRC/ensemble_$(tag).csv"
    isfile(f) || error("missing $f — run al16/01_pinned_ensembles.jl first")
    M = readdlm(f, ',')
    size(M, 2) == n_params || error("$f is $(size(M,2)) wide, model has $n_params")
    return [collect(Float64, M[i, :]) for i in 1:size(M, 1)]
end
ens = [(tag = "unconstrained", label = "unconstrained (pinned, no predicate)", mem = read_ens("unconstrained")),
       (tag = "constrained",   label = "constrained (phonon-positive)",        mem = read_ens("constrained"))]
for e in ens; @printf("%-14s %d members\n", e.tag, length(e.mem)); end
structure = AtomsBuilder.Chemistry.symmetry(element)
flush(stdout)

# ── one member: relax, build its own Hessian, band structure ────────────────
# `m` is a per-task model copy.  Never pass the shared one.
function bands_native(m, θ)
    ACEpotentials.Models.set_linear_parameters!(m, θ)
    a = ACEWorkflow.relax_lattice_constant(m, element)
    sp, ss = bulk_prim_super(element; a=a, N_cell=N_CELL)
    fc = precompute_force_constants(sp, ss, m)
    ql, xv, xt, lb, _ = _band_path(structure, fc.L; N_per_seg=N_PER_SEG)
    Np = fc.Np; qn = norm.(ql)
    F = Matrix{Float64}(undef, 3Np, length(ql))
    for (iq, q) in enumerate(ql)
        ev = eigvals(Hermitian(dynamical_matrix_from_fc(fc, q)))
        F[:, iq] = sign.(ev) .* sqrt.(abs.(ev)) .* FREQ_THz
    end
    return (; F, x_vals=xv, x_ticks=xt, labels=lb, Np, a,
              minω = minimum(F[:, qn .> qΓtol]))
end

function bands_many(θs; label="")
    n = length(θs); out = Vector{Any}(undef, n)
    nt = clamp(QOI_THREADS, 1, min(Threads.nthreads(), n))
    @printf("[%s] %d independent relaxations + Hessians on %d tasks …\n", label, n, nt)
    flush(stdout)
    t = @elapsed @sync for k in 1:nt
        Threads.@spawn begin
            m = deepcopy(model)
            for i in k:nt:n; out[i] = bands_native(m, θs[i]); end
        end
    end
    @printf("[%s] %.1f min (%.1f s/member)\n", label, t/60, t/n); flush(stdout)
    return out
end

mean_b = bands_native(deepcopy(model), lin_params)
@printf("\nmean model: a = %.5f Å, min ω = %+.4f THz\n", mean_b.a, mean_b.minω)
res = [bands_many(e.mem; label=e.tag) for e in ens]

# ── control: does the native rebuild agree with stage 1's undotted screen? ──
JLS = "$SRC/pinned_ensembles.jls"
if isfile(JLS)
    d = deserialize(JLS)
    for (k, e) in enumerate(ens)
        scr = k == 1 ? d.wu : d.wc
        nat = [b.minω for b in res[k]]
        @printf("CONTROL [%-13s] native vs stage-1 undotted screen: max |Δ| = %.3e THz\n",
                e.tag, maximum(abs.(nat .- scr)))
    end
    println("  (should be ~1e-3 THz or below: members are pinned to a_eq to microangstroms,")
    println("   so the screen's single Hessian is the right operator.  A gap of ~0.1 THz")
    println("   means the two stages sampled DIFFERENT q-points, not that the pin drifted —")
    println("   check N_PER_SEG matches stage 1.)")
end

# ── statistics ──────────────────────────────────────────────────────────────
println("\n══ PHONONS, NATIVE PER-MEMBER HESSIANS ══════════════════════════")
for (k, e) in enumerate(ens)
    w = [b.minω for b in res[k]]; a = [b.a for b in res[k]]
    @printf("%-14s min ω ∈ [%+.4f, %+.4f], median %+.4f | %d/%d soft | a ∈ [%.5f, %.5f]\n",
            e.tag, minimum(w), maximum(w), median(w),
            count(<(UNSTABLE), w), length(w), minimum(a), maximum(a))
end
let wu = [b.minω for b in res[1]], wc = [b.minω for b in res[2]]
    @printf("\nspread ratio σ_constrained/σ_unconstrained = %.4f\n", std(wc)/std(wu))
    @printf("soft members: %d → %d out of %d\n",
            count(<(UNSTABLE), wu), count(<(UNSTABLE), wc), length(wu))
end
flush(stdout)

# ── one shared frequency axis over both ensembles (the four-panel row's y-range) ──
allF = vcat([b.F for r in res for b in r], [mean_b.F])
lo = minimum(minimum.(allF)); hi = maximum(maximum.(allF)); pad = 0.06*(hi-lo)
ylim = (min(lo-pad, -0.5), hi+pad)
for (k, e) in enumerate(ens)
    writedlm("$outdir/bands_$(e.tag).csv",
             hcat(1:length(res[k]), [b.a for b in res[k]], [b.minω for b in res[k]]), ',')
end
serialize("$outdir/bands_two_ensembles.jls",
          (; res, mean_b, ylim, N_CELL, N_PER_SEG, UNSTABLE,
             tags = [e.tag for e in ens]))
println("\ndata → $outdir/bands_{unconstrained,constrained}.csv  (member, relaxed a, min ω), bands_two_ensembles.jls")
