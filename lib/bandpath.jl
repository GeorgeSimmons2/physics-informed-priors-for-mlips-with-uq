# bandpath.jl — the band-path phonon machinery every ACE stage shares.
#
# Everything is built on the UNDOTTED per-basis Hessian H_k = ∂²B_k/∂r² (which depends only on
# geometry), so at a fixed lattice constant it is built ONCE and every parameter vector's phonon
# check / band structure is Σ_k θ_k D_k(q) — a matvec plus a tiny eigen per q.
#
# Loaded by the stage scripts:  include(joinpath(@__DIR__, "..", "lib", "bandpath.jl"))

using LinearAlgebra, DelimitedFiles, Statistics, Printf, Serialization
using StaticArrays, Unitful, ForwardDiff, CairoMakie, StatsBase
using ACEpotentials, ACEWorkflow
import ACEpotentials.Models: evaluate_basis
using AtomsBuilder
using ExtXYZ
import AtomsCalculatorsUtilities.SitePotentials: PairList, get_neighbours, cutoff_radius
import ACEWorkflow: precompute_force_constants, dynamical_matrix_from_fc, fcc_band_path, bulk_prim_super

# the high-symmetry path for the crystal structure (only FCC Al is used here)
_band_path(structure, L; N_per_seg) =
    structure == :fcc ? fcc_band_path(L; N_per_seg) : error("only :fcc is supported, got $structure")

# ── Undotted per-basis Hessian (parallel over atoms) ─────────────────────────
function _site_basis_hessian(model, Rs, Zs, z0, ps, st)
    nR = length(Rs)
    x0 = collect(Float64, reinterpret(Float64, Rs))
    tovec(x) = [SVector{3,eltype(x)}(x[3i-2], x[3i-1], x[3i]) for i in 1:nR]
    Bfun(x) = evaluate_basis(model, tovec(x), Zs, z0, ps, st)
    gradflat(x) = vec(ForwardDiff.jacobian(Bfun, x))
    Hflat = ForwardDiff.jacobian(gradflat, x0)
    return reshape(Hflat, length(Bfun(x0)), 3nR, 3nR)
end

function undotted_hessian(sys, V)
    nlist = PairList(sys, cutoff_radius(V)); Nat = length(sys); D = 3; ps = V.ps; st = V.st; m = V.model
    _, Rs0, Zs0, z00 = get_neighbours(sys, V, nlist, 1)
    N_basis = length(evaluate_basis(m, Rs0, Zs0, z00, ps, st))
    Ht = [zeros(D*Nat, D*Nat, N_basis) for _ in 1:Threads.nthreads()]
    Threads.@threads :static for i in 1:Nat
        H = Ht[Threads.threadid()]
        Js, Rs, Zs, z0 = get_neighbours(sys, V, nlist, i)
        Hi = _site_basis_hessian(m, Rs, Zs, z0, ps, st)
        Ji = (i-1)*D .+ (1:D)
        for (α1, j1) in enumerate(Js), (α2, j2) in enumerate(Js)
            A1 = (α1-1)*D .+ (1:D); A2 = (α2-1)*D .+ (1:D)
            J1 = (j1-1)*D .+ (1:D); J2 = (j2-1)*D .+ (1:D)
            @views for k in 1:N_basis
                blk = Hi[k, A1, A2]
                H[J1,J2,k] .+= blk; H[J1,Ji,k] .-= blk; H[Ji,J2,k] .-= blk; H[Ji,Ji,k] .+= blk
            end
        end
    end
    return sum(Ht)
end

# get (or build+cache) the undotted per-basis Hessian at lattice constant `a`
function undotted_Hbasis(result, model, element, a, N_cell; tag="")
    # cached beside the model, so each route's models keep their own; the file name rounds a to
    # 5 digits but the cache is BUILT at the exact a, which the loader checks (cache.a_eq)
    mkpath("$(result.dir)/cache")
    f = "$(result.dir)/cache/undotted_Hbasis_$(tag)$(N_cell)x$(N_cell)x$(N_cell)_a$(round(a;digits=5)).jls"
    if isfile(f)
        c = deserialize(f)
        if abs(c.a_eq - a) < 1e-4; println("  loaded H_basis cache $(basename(f))"); return c.H_basis; end
    end
    _, sys = bulk_prim_super(element; a=a, N_cell=N_cell)
    @printf("  building undotted H_basis at a=%.5f (%d atoms, %d threads) …\n", a, length(sys), Threads.nthreads())
    t = @elapsed Hb = undotted_hessian(sys, model)
    @printf("    done in %.1f min\n", t/60)
    tmp = "$f.tmp$(getpid())"                  # write-then-rename: a concurrent reader never sees half a file
    serialize(tmp, (H_basis=Hb, a_eq=a, N_cell=N_cell)); mv(tmp, f; force=true); return Hb
end

# band-path per-basis D_k(q) over the FULL path (Γ INCLUDED — for plotting AND checks)
function bandpath_Dk(result, model, element, a, N_cell; N_per_seg=20, tag="")
    sys_prim, sys_super = bulk_prim_super(element; a=a, N_cell=N_cell)
    ACEpotentials.Models.set_linear_parameters!(model, result.lin_params)
    fc0 = precompute_force_constants(sys_prim, sys_super, model); Np = fc0.Np; N3 = 3*length(sys_super)
    Hb = undotted_Hbasis(result, model, element, a, N_cell; tag=tag)
    n_params = length(result.lin_params)
    structure = AtomsBuilder.Chemistry.symmetry(element)
    q_list, x_vals, x_ticks, labels, _ = _band_path(structure, fc0.L; N_per_seg=N_per_seg)
    Bq = Vector{Matrix{ComplexF64}}(undef, length(q_list))
    for (iq, q) in enumerate(q_list)
        M = Matrix{ComplexF64}(undef, (3Np)^2, n_params)
        for k in 1:n_params
            M[:, k] = vec(dynamical_matrix_from_fc(merge(fc0, (H = reshape(Hb[:, :, k], N3, N3),)), q))
        end
        Bq[iq] = M
    end
    return (; Bq, q_list, x_vals, x_ticks, labels, Np, qnorm = norm.(q_list))
end

# frequencies (THz), 3Np × n_q, over the full path
function bands(θ, bp)
    F = Matrix{Float64}(undef, 3bp.Np, length(bp.Bq))
    for iq in eachindex(bp.Bq)
        ev = eigvals(Hermitian(reshape(bp.Bq[iq]*θ, 3bp.Np, 3bp.Np)))
        F[:, iq] = sign.(ev) .* sqrt.(abs.(ev)) .* FREQ_THz
    end
    return F
end

# minimum non-acoustic frequency (skips q near Γ where acoustic → 0) — the stability metric
function min_freq_stable(θ, bp; qΓtol=5e-2)
    m = Inf
    for iq in eachindex(bp.Bq)
        bp.qnorm[iq] < qΓtol && continue
        ev = eigvals(Hermitian(reshape(bp.Bq[iq]*θ, 3bp.Np, 3bp.Np)))
        m = min(m, minimum(sign.(ev).*sqrt.(abs.(ev)).*FREQ_THz))
    end
    return m
end

# soft (iq, eigenvector) modes below margin, skipping near-Γ — for cutting-plane
function soft_modes(θ, bp, ω2_cut; qΓtol=5e-2)
    soft = Tuple{Int,Vector{ComplexF64}}[]
    for iq in eachindex(bp.Bq)
        bp.qnorm[iq] < qΓtol && continue
        Fe = eigen(Hermitian(reshape(bp.Bq[iq]*θ, 3bp.Np, 3bp.Np)))
        for ν in 1:3bp.Np
            Fe.values[ν] < ω2_cut && push!(soft, (iq, Fe.vectors[:, ν]))
        end
    end
    return soft
end
cut_row(iq, e, bp) = (c = vec(conj(e)*transpose(e)); real(vec(transpose(c) * bp.Bq[iq])))
