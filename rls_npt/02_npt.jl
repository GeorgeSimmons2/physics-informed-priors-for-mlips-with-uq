# ─────────────────────────────────────────────────────────────────────────────
# rls_npt/02_npt.jl — NPT of one ACE parameter vector: a(T), structure, and the trajectory.
#
# 4×4×4 FCC (256 atoms), 0 Pa, Langevin + Monte Carlo barostat, dt 1 fs, friction 0.01 fs⁻¹,
# N_EQUIL + N_PROD steps (10 000 + 20 000), logged every 50 steps with each frame's own box.
# a(T) = mean ∛V/4 over the production frames; ⟨coord⟩ (neighbours within 3.3 Å) and the median
# nearest-neighbour distance say whether the cell is still FCC (⟨coord⟩ ≥ 11).  min ω at a(T)
# comes from a NATIVE 4×4×4 Hessian of this θ (Γ excluded), so the acoustic zero does not count.
#
# SEED_MODE=same seeds every temperature with MD_SEED (common random numbers across models);
# per_T seeds MD_SEED + T.  Each temperature writes T<T>K/{summary_row.csv, md_trajectory.extxyz}.
#
# Ported from npt_al16/npt_member_Al_16.jl of the physics-informed-priors repository: the MD
# protocol, seeding and the summary row are unchanged; its diagnostic figures (RDF, MSD,
# convergence, FCC survival, clusters) are dropped — 03_rdf.jl makes the RDF the figure uses.
#
# Env:  THETA_FILE (required)  MODELDIR (default model_dir("Al_12_4_6A_2_"), honours ACE_MODELS)
#       TAG  OUTDIR  TEMPS ("200" or "100,200,300")  MD_SEED (1234)  SEED_MODE (per_T)  N_EQUIL  N_PROD
# Run:  julia --project -t 4 rls_npt/02_npt.jl
# ─────────────────────────────────────────────────────────────────────────────

include(joinpath(@__DIR__, "..", "lib", "bandpath.jl"))
using Molly, Random
using AtomsBuilder: bulk
import AtomsBuilder

@async while true; flush(stdout); flush(stderr); sleep(5); end   # SLURM block-buffers stdout

element        = :Al
MODELDIR       = normpath(get(ENV, "MODELDIR", model_dir("Al_12_4_6A_2_")))   # any <models>/<name>_ dir
MODELNAME      = basename(rstrip(MODELDIR, '/'))
N_CELL         = 4
N_PER_SEG      = 20
qΓtol          = 5e-2
structure      = AtomsBuilder.Chemistry.symmetry(element)

THETA_FILE     = get(ENV, "THETA_FILE", "")
isempty(THETA_FILE) && error("set THETA_FILE to the member's parameter vector")
TAG            = get(ENV, "TAG", splitext(basename(THETA_FILE))[1])
outdir         = get(ENV, "OUTDIR", joinpath(@__DIR__, "..", "repro", "saved", "rls_npt", "npt", TAG)); mkpath(outdir)
temperatures_K = parse.(Float64, split(get(ENV, "TEMPS", "300"), ','))
MD_SEED        = parse(Int, get(ENV, "MD_SEED", "1234"))
# SEED_MODE=same : every temperature (and, if the caller passes the same MD_SEED, every
#                  member) starts from the identical random stream — common random numbers,
#                  so member-to-member differences reflect θ rather than trajectory noise.
# SEED_MODE=per_T: seed MD_SEED+T (the original single-member runs).
SEED_MODE      = get(ENV, "SEED_MODE", "per_T")
SEED_MODE in ("same", "per_T") || error("SEED_MODE must be same or per_T")

supercell      = (4, 4, 4)
pressure       = 0.0u"GPa"
dt             = 1.0u"fs"
friction       = 0.01u"fs^-1"
n_equil        = parse(Int, get(ENV, "N_EQUIL", "10000"))
n_prod         = parse(Int, get(ENV, "N_PROD",  "20000"))
log_every      = 50
fcc_coord_tol  = 1.0
nn_cutoff      = 3.3

model, _ = ACEpotentials.load_model("$MODELDIR/$(rstrip(MODELNAME, '_')).json")
θ = vec(readdlm(THETA_FILE, ','))
n_params = length(vec(readdlm("$MODELDIR/lin_params.csv", ',')))
length(θ) == n_params || error("θ has $(length(θ)) entries, model has $n_params")
writedlm("$outdir/theta_used.csv", θ, ',')
@printf("%s (%d params), member %s, %d threads\n  θ ← %s\n  out → %s\n  T = %s K, seed %d (%s)\n",
        MODELNAME, n_params, TAG, Threads.nthreads(), THETA_FILE, outdir, join(Int.(temperatures_K), ", "), MD_SEED, SEED_MODE)

"min non-acoustic ω of θ at lattice constant a, native 4×4×4 Hessian, Γ excluded"
function native_minω(m, θ, a)
    ACEpotentials.Models.set_linear_parameters!(m, θ)
    sp, ss = bulk_prim_super(element; a=a, N_cell=N_CELL)
    fc = precompute_force_constants(sp, ss, m)
    ql, _ = _band_path(structure, fc.L; N_per_seg=N_PER_SEG)
    w = Inf
    for q in ql
        norm(q) > qΓtol || continue
        ev = eigvals(Hermitian(dynamical_matrix_from_fc(fc, q)))
        w = min(w, minimum(sign.(ev) .* sqrt.(abs.(ev)) .* FREQ_THz))
    end
    return w
end

ACEpotentials.Models.set_linear_parameters!(model, θ)
a0 = ACEWorkflow.relax_lattice_constant(model, element)
minω_a0 = native_minω(model, θ, a0)
@printf("0 K: relaxed a₀ = %.6f Å, min ω = %+.4f THz\n", a0, minω_a0)

# ── per-T MD analysis: the trajectory, a(T) and the FCC structure checks ─────────
function analyze_md(sys_md, dir, T_K; log_every, equil_frames, N_super)
    mkpath(dir)
    coords_hist = sys_md.loggers.coords.history
    vol_hist    = ustrip.(u"Å^3", sys_md.loggers.volume.history)
    temps_hist  = ustrip.(sys_md.loggers.temp.history)
    ener_hist   = ustrip.(sys_md.loggers.energy.history)
    n_frames    = length(coords_hist); n_atoms = length(coords_hist[1])
    side(f)     = cbrt(vol_hist[f])
    prod        = (equil_frames+1):n_frames

    a_prod   = cbrt.(vol_hist[prod]) ./ N_super
    a_T      = mean(a_prod); a_T_std = std(a_prod)

    species = [string(Molly.atomic_symbol(sys_md, i)) for i in 1:n_atoms]
    frames  = Dict{String,Any}[]
    for (f, fc) in enumerate(coords_hist)
        L = side(f)
        push!(frames, Dict{String,Any}(
            "N_atoms" => n_atoms,
            "info"    => Dict{String,Any}(
                "Lattice"     => "$L 0.0 0.0 0.0 $L 0.0 0.0 0.0 $L",
                "Properties"  => "species:S:1:pos:R:3",
                "energy"      => ener_hist[f], "temperature" => temps_hist[f],
                "step"        => (f-1)*log_every),
            "arrays"  => Dict{String,Any}(
                "species" => species,
                "pos"     => reduce(hcat, [ustrip.(u"Å", c) for c in fc]))))
    end
    ExtXYZ.write_frames("$dir/md_trajectory.extxyz", frames)

    mean_coord = Float64[]; med_nn = Float64[]
    for f in prod
        L = side(f); pos = [ustrip.(u"Å", c) for c in coords_hist[f]]; n = length(pos)
        coord = zeros(Int, n); nn = fill(Inf, n)
        for i in 1:n, j in i+1:n
            d = pos[i] .- pos[j]; d = d .- L .* round.(d ./ L); r = norm(d)
            r < nn[i] && (nn[i] = r); r < nn[j] && (nn[j] = r)
            r < nn_cutoff && (coord[i]+=1; coord[j]+=1)
        end
        push!(mean_coord, mean(coord)); push!(med_nn, median(nn))
    end
    coord_prod = mean(mean_coord); nn_prod = median(med_nn)
    still_fcc  = coord_prod >= 12.0 - fcc_coord_tol

    return (a_T=a_T, a_T_std=a_T_std, mean_T=mean(temps_hist[prod]),
            mean_coord=coord_prod, med_nn=nn_prod, still_fcc=still_fcc)
end

# ── NPT, one temperature at a time ───────────────────────────────────────────
equil_frames = div(n_equil, log_every)
N_super      = supercell[1]
open("$outdir/a0.csv", "w") do io
    println(io, "a0_Ang,min_omega_a0_THz"); @printf(io, "%.6f,%.4f\n", a0, minω_a0)
end

for T_K in temperatures_K
    Random.seed!(SEED_MODE == "same" ? MD_SEED : MD_SEED + round(Int, T_K))
    T = T_K * u"K"
    ACEpotentials.Models.set_linear_parameters!(model, θ)
    sys    = bulk(element, a=a0*u"Å", cubic=true) * supercell
    sys_md = Molly.System(sys; force_units=u"eV/Å", energy_units=u"eV")
    sys_md = Molly.System(sys_md;
        general_inters = (model,),
        velocities = Molly.random_velocities(sys_md, T),
        loggers = (temp   = Molly.TemperatureLogger(log_every),
                   coords = Molly.CoordinatesLogger(log_every),
                   volume = Molly.VolumeLogger(log_every),
                   energy = Molly.PotentialEnergyLogger(typeof(1.0u"eV"), log_every)))
    sim = Molly.Langevin(dt=dt, temperature=T, friction=friction,
                         coupling=Molly.MonteCarloBarostat(pressure, T, sys_md.boundary))
    @printf("\n  T = %4.0f K: %d equil + %d prod steps …\n", T_K, n_equil, n_prod)
    el = @elapsed Molly.simulate!(sys_md, sim, n_equil + n_prod)

    dir = "$outdir/T$(round(Int,T_K))K"
    ana = analyze_md(sys_md, dir, T_K; log_every=log_every, equil_frames=equil_frames, N_super=N_super)
    mω  = native_minω(model, θ, ana.a_T)
    @printf("    a(%.0f K) = %.5f ± %.5f Å  (Δa/a₀ = %+.3f%%),  ⟨T⟩ = %.0f K,  min ω(a_T) = %+.3f THz  [%.1f min MD]\n",
            T_K, ana.a_T, ana.a_T_std, 100*(ana.a_T-a0)/a0, ana.mean_T, mω, el/60)
    @printf("    structure: ⟨coord⟩ = %.2f (FCC 12), median NN = %.3f Å  →  %s\n",
            ana.mean_coord, ana.med_nn,
            ana.still_fcc ? "STILL FCC ✓" : "*** LEFT FCC — a(T) is NOT thermal expansion ***")
    open("$dir/summary_row.csv", "w") do io
        println(io, "T_K,a_Ang,a_std_Ang,mean_T_K,min_omega_THz,mean_coord,median_nn_Ang,still_fcc,md_minutes")
        @printf(io, "%.0f,%.6f,%.6f,%.2f,%.4f,%.4f,%.4f,%s,%.1f\n",
                T_K, ana.a_T, ana.a_T_std, ana.mean_T, mω, ana.mean_coord, ana.med_nn, ana.still_fcc, el/60)
    end
end
println("\ndone → $outdir")
