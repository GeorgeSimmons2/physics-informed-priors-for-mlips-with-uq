# lib/bandpath.jl — the undotted per-basis Hessian, its cache, band-path D_k(q), bands, stability checks, cutting
# planes, the committee band plot and committee predictions.
#
# The central checks: the AD per-basis Hessian dotted with θ is the model's own (native, dotted) Hessian, and the
# band-path D_k(q) dotted with θ is the native dynamical matrix — at the fitted θ and at a perturbed θ.

const BP_SRC = "lib/bandpath.jl"

@testset "_band_path" begin
    tested(BP_SRC, :_band_path)
    L = Matrix(fc_native(a_eq(), 2).L)
    @test LIB._band_path(:fcc, L; N_per_seg=3) == ACEWorkflow.fcc_band_path(L; N_per_seg=3)
    @test LIB._band_path(:diamond, L; N_per_seg=3) == ACEWorkflow.fcc_band_path(L; N_per_seg=3)
    @test_throws ErrorException LIB._band_path(:bcc, L; N_per_seg=3)
end

@testset "_site_basis_hessian" begin
    tested(BP_SRC, :_site_basis_hessian)
    V = al12_model(); θ = al12().θ
    sys = rattled_super(a_eq(), 2)
    nlist = LIB.PairList(sys, LIB.cutoff_radius(V))
    _, Rs, Zs, z0 = LIB.get_neighbours(sys, V, nlist, 5)
    Hi = LIB.@no_escape begin LIB._site_basis_hessian(V.model, Rs, Zs, z0, V.ps, V.st) end
    nR = length(Rs)
    @test size(Hi) == (length(θ), 3nR, 3nR)
    @test all(Hi[k, :, :] ≈ Hi[k, :, :]' for k in (1, 17, length(θ)))
    # dotted with θ: the Hessian of the site energy θ·B(R) with respect to the neighbour vectors
    x0 = collect(Float64, reinterpret(Float64, Rs))
    Esite(x) = dot(θ, LIB.evaluate_basis(V.model, [SVector{3}(x[3i-2], x[3i-1], x[3i]) for i in 1:nR], Zs, z0, V.ps, V.st))
    Hs = LIB.@no_escape begin ForwardDiff.hessian(Esite, x0) end
    @test reshape(θ' * reshape(Hi, length(θ), :), 3nR, 3nR) ≈ Hs rtol = 1e-10
end

@testset "undotted_hessian" begin
    tested(BP_SRC, :undotted_hessian)
    sys = rattled_super(a_eq(), 2)                             # off-equilibrium: no symmetry cancellation helps
    Hb = LIB.undotted_hessian(sys, al12_model())
    N3 = 3 * length(sys)
    @test size(Hb) == (N3, N3, length(al12().θ))
    for θ in (al12().θ, θ_perturbed())                         # the AD Hessian dotted with θ is the native one
        Hn = native_hessian(sys, θ)
        @test contract(Hb, θ) ≈ Hn rtol = 1e-10
        @test maximum(abs, contract(Hb, θ) .- Hn) < 1e-10 * maximum(abs, Hn)
    end
    k = 3
    @test Hb[:, :, k] ≈ Hb[:, :, k]'
    @test maximum(abs, sum(reshape(Hb[:, :, k], N3, 3, :); dims=3)) < 1e-10 * maximum(abs, Hb[:, :, k])  # translation invariance
end

@testset "undotted_Hbasis" begin
    tested(BP_SRC, :undotted_Hbasis)
    a = a_eq(); N = 2
    mktempdir() do dir
        r = fake_result(dir)
        f = joinpath(dir, "results", "undotted_Hbasis_$(N)x$(N)x$(N)_a$(round(a; digits=5)).jls")
        f0 = joinpath(dir, "results", "undotted_Hbasis_$(N)x$(N)x$(N).jls")
        mkpath(dirname(f))
        serialize(f0, (H_basis=:stale, a_eq=a + 0.01, N_cell=N))                    # a cache at another a is not used
        Hb = LIB.undotted_Hbasis(r, al12_model(), :Al, a, N)
        @test Hb isa Array{Float64,3}
        @test isfile(f)
        c = deserialize(f)
        @test c.a_eq == a && c.N_cell == N                                            # built at the unrounded a
        _, sup = ACEWorkflow.bulk_prim_super(:Al; a=a, N_cell=N)
        @test contract(Hb, al12().θ) ≈ native_hessian(sup, al12().θ) rtol = 1e-10
        serialize(f, (H_basis=:cached, a_eq=a, N_cell=N))
        @test LIB.undotted_Hbasis(r, al12_model(), :Al, a, N) === :cached              # a cache hit returns the cache
        rm(f); serialize(f0, (H_basis=:legacy, a_eq=a, N_cell=N))
        @test LIB.undotted_Hbasis(r, al12_model(), :Al, a, N) === :legacy              # the pre-a-suffix name still works
        # a tagged request should get its own cache, but it falls back to the untagged legacy file (no caller here
        # passes a tag, so this is latent)
        @test_broken LIB.undotted_Hbasis(r, al12_model(), :Al, a, N; tag="t") !== :legacy
        memo(() -> Hb, :Hb2)
    end
end

@testset "bandpath_Dk" begin
    tested(BP_SRC, :bandpath_Dk)
    a = a_eq()
    mktempdir() do dir
        r = fake_result(dir)
        mkpath(joinpath(dir, "results"))
        serialize(joinpath(dir, "results", "undotted_Hbasis_2x2x2_a$(round(a; digits=5)).jls"),
                  (H_basis=memo(() -> error("undotted_Hbasis must run first"), :Hb2), a_eq=a, N_cell=2))
        bp = LIB.bandpath_Dk(r, al12_model(), :Al, a, 2; N_per_seg=2)
        memo(() -> bp, :bp2)
        q, x, ticks, labels, _ = ACEWorkflow.fcc_band_path(Matrix(fc_native(a, 2).L); N_per_seg=2)
        @test bp.q_list ≈ q && bp.x_vals ≈ x && bp.x_ticks ≈ ticks && bp.labels == labels
        @test bp.qnorm ≈ norm.(q) && bp.Np == 1
        @test length(bp.Bq) == length(q) && size(bp.Bq[1]) == (9, length(al12().θ))
        # D_k(q) dotted with θ is the native dynamical matrix, at the fitted and at a perturbed θ
        for θ in (al12().θ, θ_perturbed())
            m = al12_model(); ACEpotentials.Models.set_linear_parameters!(m, θ)
            prim, sup = ACEWorkflow.bulk_prim_super(:Al; a=a, N_cell=2)
            fc = ACEWorkflow.precompute_force_constants(prim, sup, m)
            Dn = [ACEWorkflow.dynamical_matrix_from_fc(fc, qi) for qi in q]
            # absolute to the band's scale: at Γ both sides are round-off, so a relative test there is meaningless
            @test maximum(norm(reshape(bp.Bq[iq] * θ, 3, 3) - Dn[iq]) for iq in eachindex(q)) < 1e-10 * maximum(norm, Dn)
        end
    end
end

@testset "bands" begin
    tested(BP_SRC, :bands)
    F = LIB.bands([1.0], toy_bp())
    @test size(F) == (3, 3)
    @test F[:, 2] ≈ [-2.0, 1.0, 3.0] .* ACEWorkflow.FREQ_THz          # negative ω² → negative frequency
    @test F[:, 3] ≈ [0.5, 2.0, 4.0] .* ACEWorkflow.FREQ_THz
    @test LIB.bands([4.0], toy_bp()) ≈ 2 .* F                           # linear in θ inside the square root
    bp = memo(() -> error("bandpath_Dk must run first"), :bp2)
    Fr = LIB.bands(al12().θ, bp)
    fc = fc_native(a_eq(), 2)
    λ = [eigvals(Hermitian(ACEWorkflow.dynamical_matrix_from_fc(fc, q))) for q in bp.q_list]   # native ω², ascending
    ω2 = sign.(Fr) .* (Fr ./ ACEWorkflow.FREQ_THz) .^ 2
    @test maximum(maximum(abs, ω2[:, iq] .- λ[iq]) for iq in eachindex(λ)) < 1e-10 * maximum(maximum(abs, l) for l in λ)
    away = findall(bp.qnorm .>= 5e-2)
    @test all(Fr[:, iq] ≈ sign.(λ[iq]) .* sqrt.(abs.(λ[iq])) .* ACEWorkflow.FREQ_THz for iq in away)
    @test maximum(abs, Fr[:, bp.qnorm .< 1e-12]) < 1e-3                  # acoustic at Γ
end

@testset "min_freq_stable" begin
    tested(BP_SRC, :min_freq_stable)
    @test LIB.min_freq_stable([1.0], toy_bp()) ≈ -2 * ACEWorkflow.FREQ_THz            # Γ (−3) is skipped
    @test LIB.min_freq_stable([1.0], toy_bp(); qΓtol=0.0) ≈ -3 * ACEWorkflow.FREQ_THz
    @test LIB.min_freq_stable([1.0], toy_bp(); qΓtol=0.75) ≈ 0.5 * ACEWorkflow.FREQ_THz
    bp = memo(() -> error("bandpath_Dk must run first"), :bp2)
    @test LIB.min_freq_stable(al12().θ, bp) ≈ minimum(LIB.bands(al12().θ, bp)[:, bp.qnorm .>= 5e-2])
end

@testset "soft_modes" begin
    tested(BP_SRC, :soft_modes)
    s = LIB.soft_modes([1.0], toy_bp(), 0.5)
    @test [iq for (iq, _) in s] == [2, 3]                               # −4 at q2 and 0.25 at q3; Γ skipped
    D2 = reshape(toy_bp().Bq[2] * [1.0], 3, 3)
    @test D2 * s[1][2] ≈ -4 .* s[1][2] && norm(s[1][2]) ≈ 1
    @test isempty(LIB.soft_modes([1.0], toy_bp(), -100.0))
    @test length(LIB.soft_modes([1.0], toy_bp(), 100.0)) == 6
end

@testset "cut_row" begin
    tested(BP_SRC, :cut_row)
    rng = MersenneTwister(9)
    herm() = (M = randn(rng, ComplexF64, 3, 3); M + M')
    bp = (; Bq=[hcat([vec(herm()) for _ in 1:4]...) for _ in 1:2], Np=1, qnorm=[0.3, 0.6])
    e = normalize(randn(rng, ComplexF64, 3))
    r = LIB.cut_row(2, e, bp)
    @test r isa Vector{Float64} && length(r) == 4
    for θ in (randn(rng, 4), randn(rng, 4))                             # r·θ = e†D(q; θ)e for every θ
        @test dot(r, θ) ≈ real(e' * reshape(bp.Bq[2] * θ, 3, 3) * e)
    end
    # on a soft mode the cut is its eigenvalue: the cutting plane r·θ ≥ ω²_cut lifts exactly that mode
    (iq, v) = LIB.soft_modes([1.0, 0, 0, 0], bp, 1e6)[1]
    @test dot(LIB.cut_row(iq, v, bp), [1.0, 0, 0, 0]) ≈ eigvals(Hermitian(reshape(bp.Bq[iq][:, 1], 3, 3)))[1]
end

@testset "plot_committee_bands" begin
    tested(BP_SRC, :plot_committee_bands)
    bp = memo(() -> error("bandpath_Dk must run first"), :bp2)
    θ = al12().θ
    members = [θ .* (1 .+ 0.02 .* randn(MersenneTwister(k), length(θ))) for k in 1:3]
    mktempdir() do dir
        fig = LIB.plot_committee_bands(members, θ, bp, "test", joinpath(dir, "bands.png"); ylims=(-1, 12), panel="(a)")
        @test fig isa LIB.Figure
        @test filesize(joinpath(dir, "bands.pdf")) > 1000 && filesize(joinpath(dir, "bands.png")) > 1000   # both, whatever the extension
        ax = fig.content[1]
        @test count(p -> p isa LIB.Lines, ax.scene.plots) == 3 * 4      # 3 branches × (3 members + the mean)
    end
end

@testset "committee_predictions" begin
    tested(BP_SRC, :committee_predictions)
    θ = al12().θ
    mktempdir() do dir
        xyz = small_test_xyz(dir; n=5)
        frames = ExtXYZ.load(xyz)
        same = [copy(θ) for _ in 1:3]
        pr = LIB.committee_predictions(al12_model(), same, xyz; stride=1, point_params=θ)
        @test pr.n == 5 && !pr.per_atom
        @test pr.pE ≈ [energy(s, al12().model) for s in frames]
        @test pr.tE ≈ [ustrip(s[:dft_energy]) for s in frames]
        @test pr.loE ≈ pr.pE && pr.hiE ≈ pr.pE && maximum(abs, pr.dE) < 1e-8      # identical members: no spread
        @test length(pr.pF) == length(pr.tF) == 3 * sum(length, frames) == length(pr.loF)
        # distinct members: lo/hi are the members' extremes, dE pools member − point over members × configs
        mem = [θ .* (1 .+ 0.05 .* randn(MersenneTwister(k), length(θ))) for k in 1:3]
        pr2 = LIB.committee_predictions(al12_model(), mem, xyz; stride=1, point_params=θ)
        Em = [energy(s, (m = al12_model(); ACEpotentials.Models.set_linear_parameters!(m, t); m)) for s in frames, t in mem]
        @test pr2.loE ≈ vec(minimum(Em; dims=2)) && pr2.hiE ≈ vec(maximum(Em; dims=2))
        @test sort(pr2.dE) ≈ sort(vec(Em .- pr2.pE))
        pa = LIB.committee_predictions(al12_model(), mem, xyz; stride=1, point_params=θ, per_atom=true)
        nat = length.(frames)
        @test pa.per_atom && pa.pE ≈ pr2.pE ./ nat && pa.tE ≈ pr2.tE ./ nat && pa.pF ≈ pr2.pF
        @test LIB.committee_predictions(al12_model(), mem, xyz; stride=2, point_params=θ).n == 3
        memo(() -> (; pr=pr2, mem, xyz=nothing), :pr)
        memo(() -> pa, :pr_pa)
    end
end
