# rls_npt/01–04: the multi-volume constrained RLS fit, the 0 K native check, MD analysis, RDF, the figure helpers.

"An ideal FCC crystal of N³ cubic cells (4N³ atoms) at a, as a 3×n position matrix, with optional rattle σ."
function fcc_positions(a, N; σ=0.0, seed=1)
    base = [0 0 0; 0 0.5 0.5; 0.5 0 0.5; 0.5 0.5 0]
    pos = reduce(hcat, [a .* ([i, j, k] .+ base[b, :]) for i in 0:N-1, j in 0:N-1, k in 0:N-1, b in 1:4])
    pos .+ σ .* randn(MersenneTwister(seed), size(pos))
end

@testset "rls_npt/01_constrained_rls.jl" begin
    F = "rls_npt/01_constrained_rls.jl"
    g = al12_globals()

    @testset "fit" begin
        tested(F, :fit)
        m = script_module(F; only=[:fit], globals=(; b_prime=g.b_prime, P=g.P, osqp=OSQP.Model(), Hqp=sparse(g.Hx), qqp=g.qx))
        θk = kkt_θ(g, reshape(g.b_prime, 1, :), [0.0])
        θf = call(m, :fit, zeros(0, g.n), Float64[])
        @test relerr(θf, θk) < 1e-6                               # eps 1e-8: tighter than the al12 QPs
        r = randn(MersenneTwister(19), g.n); lo = dot(r, θk) + 0.05 * abs(dot(r, θk)) + 1e-3
        θr = call(m, :fit, reshape(r, 1, :), [lo])
        @test dot(r, θr) ≈ lo rtol = 1e-6
        @test relerr(θr, kkt_θ(g, vcat(g.b_prime', r'), [0.0, lo])) < 1e-6
    end

    @testset "rmse, minω, ωstr" begin
        tested(F, :rmse, :minω, :ωstr)
        bps = [toy_bp(), merge(toy_bp(), (; qnorm=[0.0, 0.0, 1.0]))]
        m = script_module(F; only=[:rmse, :minω, :ωstr], globals=(; Ap=g.Ap, P=g.P, Yw=g.Yw, bps, vol_scales=[1.0, 1.05]))
        @test call(m, :rmse, g.θ) ≈ sqrt(mean((g.Ap * (g.P * g.θ) .- g.Yw) .^ 2))
        @test call(m, :minω, [1.0]) ≈ [-2.0, 0.5] .* ACEWorkflow.FREQ_THz      # one minimum per volume's path
        @test call(m, :ωstr, [1.0]) == @sprintf("@100%% %+.3f  @105%% %+.3f", -2ACEWorkflow.FREQ_THz, 0.5ACEWorkflow.FREQ_THz)
    end
end

@testset "rls_npt/02_npt.jl" begin
    F = "rls_npt/02_npt.jl"
    g = al12_globals(); bp = bp2()

    @testset "native_minω" begin
        tested(F, :native_minω)
        m = script_module(F; only=[:native_minω], globals=(; element=:Al, N_CELL=2, N_PER_SEG=2, structure=:fcc, qΓtol=5e-2))
        for θ in (g.θ, θ_perturbed())                             # native = the AD band path's minimum, Γ excluded
            @test call(m, :native_minω, al12_model(), θ, g.a) ≈ LIB.min_freq_stable(θ, bp; qΓtol=5e-2) rtol = 1e-6
        end
    end

    @testset "_savepub" begin
        tested(F, :_savepub)
        m = script_module(F; only=[:_savepub])
        mktempdir() do dir
            fig = LIB.Figure(); LIB.lines(fig[1, 1], [1, 2], [1, 2])
            call(m, :_savepub, fig, joinpath(dir, "x"))
            @test filesize(joinpath(dir, "x.pdf")) > 500 && filesize(joinpath(dir, "x.png")) > 500
        end
    end

    @testset "analyze_md" begin
        tested(F, :analyze_md)
        # a stand-in for Molly's System: the four loggers analyze_md reads, and the atoms' symbols
        a, N, nf = 4.05, 2, 6
        frames(σ, collapse) = [begin
                                   p = fcc_positions(a, N; σ, seed=f)
                                   collapse && f > 2 && (p[:, 2] .= p[:, 1] .+ [0.5, 0, 0])   # two atoms 0.5 Å apart
                                   [SVector{3}(p[:, i]) .* u"nm" ./ 10 for i in 1:size(p, 2)]
                               end for f in 1:nf]
        mk(σ, collapse) = (; loggers=(; coords=(; history=frames(σ, collapse)), volume=(; history=fill((N * a)^3 * u"Å^3", nf)),
                                       temp=(; history=fill(200.0u"K", nf)), energy=(; history=fill(-3.0u"eV", nf))))
        m = script_module(F; only=[:analyze_md, :_savepub], globals=(; dt=1.0u"fs", nn_cutoff=3.3, cluster_cutoff=2.2, fcc_coord_tol=1.0))
        Core.eval(m, :(Molly.atomic_symbol(::NamedTuple, i) = :Al))
        mktempdir() do dir
            r = call(m, :analyze_md, mk(0.02, false), joinpath(dir, "ok"), 200.0; log_every=10, equil_frames=2, N_super=N)
            @test r.a_T ≈ a && r.a_T_std < 1e-12 && r.mean_T ≈ 200
            @test r.mean_coord ≈ 12 && r.still_fcc
            @test r.med_nn ≈ a / √2 rtol = 0.02
            @test r.min_pair > 2.2 && r.max_cluster == 1
            for f in ("md_trajectory.extxyz", "md_rdf.pdf", "md_msd.png", "md_convergence.pdf", "md_fcc_survival.png", "md_cluster_analysis.pdf")
                @test isfile(joinpath(dir, "ok", f))
            end
            @test length(ExtXYZ.load(joinpath(dir, "ok", "md_trajectory.extxyz"))) == nf
            bad = call(m, :analyze_md, mk(0.02, true), joinpath(dir, "bad"), 200.0; log_every=10, equil_frames=2, N_super=N)
            @test bad.min_pair ≈ 0.5 rtol = 1e-6                                   # the collapse is caught
            @test bad.max_cluster == 2
        end
    end
end

@testset "rls_npt/03_rdf.jl: rdf_and_coord" begin
    F = "rls_npt/03_rdf.jl"
    tested(F, :rdf_and_coord)
    m = script_module(F)
    a, N = 4.05, 3
    p = fcc_positions(a, N); L = N * a; n = size(p, 2)
    dr, r_max, n_bins = 0.05, L / 2, floor(Int, (L / 2) / 0.05)
    counts, ρacc, coord = call(m, :rdf_and_coord, [p, p], [L, L], n_bins, dr, r_max, 3.3)
    @test all(coord .== 12) && length(coord) == 2n                           # FCC: 12 neighbours within 3.3 Å
    @test ρacc ≈ 2n / L^3
    @test findfirst(>(0), counts) == floor(Int, (a / √2) / dr) + 1           # the first shell at a/√2
    @test sum(counts[1:floor(Int, 3.3 / dr)]) ≈ 2 * 12n                      # each pair counted twice, two frames
end

@testset "rls_npt/04_fig_200K.jl" begin
    F = "rls_npt/04_fig_200K.jl"
    tested(F, :xplot, :shells)
    m = script_module(F; globals=(; xc=[0.0, 2.0, 5.0]), only=[:xplot, :shells])
    @test call(m, :xplot, (; bp=(; x_vals=[0.0, 1.0, 4.0]))) ≈ [0.0, 1.25, 5.0]   # rescaled to the reference path's length
    @test call(m, :shells, 4.0, 6.0) ≈ 4.0 .* sqrt.([0.5, 1.0, 1.5, 2.0])          # ideal FCC shells a√(n/2) below rmax
end
