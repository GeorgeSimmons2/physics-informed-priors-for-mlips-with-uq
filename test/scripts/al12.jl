# al12/01–04: Born constants, the forest, the constrained QPs, the hypercube sampler, bands and their panels.

@testset "al12/01_committee.jl" begin
    F = "al12/01_committee.jl"
    g = al12_globals()

    @testset "born" begin
        tested(F, :born)
        m = script_module(F; globals=(; c11_0=g.c11_0, c12_0=g.c12_0, c44_0=g.c44_0, eV_to_GPa=g.eV_to_GPa), only=[:born])
        C = reshape(reshape(g.H_el, 36, :) * g.θ, 6, 6) .* g.eV_to_GPa
        @test collect(call(m, :born, g.θ)) ≈ [C[1, 1], C[1, 2], C[4, 4]]       # (C11, C12, C44) in GPa
        @test collect(call(m, :born, 2g.θ)) ≈ 2 .* collect(call(m, :born, g.θ))  # linear in θ
    end

    @testset "forest_member" begin
        tested(F, :forest_member)
        Cf = cholesky(Symmetric(g.Hx)); AtX = Cf \ Matrix(g.Ap'); x̃ = Cf \ (g.Ap' * g.Yw)
        leverage = vec(sum(g.Ap' .* AtX; dims=1)); residual = g.Yw .- g.Ap * x̃
        m = script_module(F; globals=(; lin_params=g.P \ x̃, P=g.P, AtX, residual, leverage), only=[:forest_member])
        for i in (1, 57, g.M)
            θi = call(m, :forest_member, i)
            @test dot(g.Ap[i, :], g.P * θi) ≈ g.Yw[i] rtol = 1e-9                   # interpolates its observation
        end
        # the same corrections as src's POPS corrections, with the prior as Γ
        R = ACEWorkflow.corrections(g.Ap, g.Yw, g.P; leverage_percentile=0.0)
        @test call(m, :forest_member, 9) - g.P \ x̃ ≈ R[9, :] rtol = 1e-8
    end

    @testset "constrain_member" begin
        tested(F, :constrain_member)
        m = script_module(F; globals=al12_qp_globals(), only=[:constrain_member])
        i = 11
        θc, st = call(m, :constrain_member, i, zeros(0, g.n), Float64[])
        @test st == :Solved
        @test dot(g.Ap[i, :], g.P * θc) ≈ g.Yw[i] rtol = 1e-4                  # interpolation (OSQP tolerance)
        @test abs(dot(g.b_prime, θc)) < 1e-5 * norm(g.b_prime) * norm(θc)       # a_eq pinned
        @test all(g.born_rows * θc .>= g.born_lower .- 1e-6)
        θk = kkt_θ(g, vcat(g.Ap[i:i, :] * g.P, g.b_prime'), [g.Yw[i], 0.0])              # Born inactive here
        @test all(g.born_rows * θk .> g.born_lower)
        @test relerr(θc, θk) < 1e-4
        # an active cut: r·θ ≥ r·θk + δ binds, and the solution is the KKT point with it as an equality
        r = randn(MersenneTwister(13), g.n); lo = dot(r, θk) + 0.05 * abs(dot(r, θk)) + 1e-3
        θr, _ = call(m, :constrain_member, i, reshape(r, 1, :), [lo])
        @test dot(r, θr) ≈ lo rtol = 1e-4
        @test all(g.born_rows * kkt_θ(g, vcat(g.Ap[i:i, :] * g.P, g.b_prime', r'), [g.Yw[i], 0.0, lo]) .> g.born_lower)
        @test relerr(θr, kkt_θ(g, vcat(g.Ap[i:i, :] * g.P, g.b_prime', r'), [g.Yw[i], 0.0, lo])) < 1e-4
    end

    @testset "mean_fit" begin
        tested(F, :mean_fit)
        m = script_module(F; globals=al12_qp_globals(), only=[:mean_fit])
        θm = call(m, :mean_fit)
        θk = kkt_θ(g, reshape(g.b_prime, 1, :), [0.0])
        @test relerr(θm, θk) < 1e-4 && abs(dot(g.b_prime, θm)) < 1e-5 * norm(g.b_prime) * norm(θm)
        r = randn(MersenneTwister(14), g.n); lo = dot(r, θk) + 0.05 * abs(dot(r, θk)) + 1e-3
        θr = call(m, :mean_fit, reshape(r, 1, :), [lo])
        @test dot(r, θr) ≈ lo rtol = 1e-4
        @test relerr(θr, kkt_θ(g, vcat(g.b_prime', r'), [0.0, lo])) < 1e-4
    end
end

@testset "al12/02_cutting_plane_cloud.jl: solve_one!" begin
    F = "al12/02_cutting_plane_cloud.jl"
    tested(F, :solve_one!)
    g = al12_globals()
    m = script_module(F; only=[:solve_one!], globals=(; Ap=g.Ap, Yw=g.Yw, n_params=g.n, Hqp=sparse(g.Hx), qqp=g.qx,
        b_prime_P=sparse(reshape(g.b_prime' / g.P, 1, g.n)), born_rows_P=sparse(g.born_rows / g.P),
        born_lower=g.born_lower, RHO_INTERVAL=25, Pfac=factorize(g.P)))
    i = 11
    θs, st = call(m, :solve_one!, OSQP.Model(), i, zeros(0, g.n), Float64[])
    @test st == :Solved
    @test relerr(θs, kkt_θ(g, vcat(g.Ap[i:i, :] * g.P, g.b_prime'), [g.Yw[i], 0.0])) < 1e-4
    # cut rows arrive pre-divided by P; same answer as al12/01's constrain_member
    r = randn(MersenneTwister(13), g.n); θk = kkt_θ(g, vcat(g.Ap[i:i, :] * g.P, g.b_prime'), [g.Yw[i], 0.0])
    lo = dot(r, θk) + 0.05 * abs(dot(r, θk)) + 1e-3
    θr, _ = call(m, :solve_one!, OSQP.Model(), i, reshape(r, 1, :) / g.P, [lo])
    m1 = script_module("al12/01_committee.jl"; globals=al12_qp_globals(), only=[:constrain_member])
    @test relerr(θr, call(m1, :constrain_member, i, reshape(r, 1, :), [lo])[1]) < 1e-5
end

@testset "al12/03_hypercube_rejection.jl" begin
    F = "al12/03_hypercube_rejection.jl"
    g = al12_globals(); bp = bp2()
    θm = g.θ
    cut = LIB.min_freq_stable(θm, bp)                           # the mean sits on the margin: proposals are filtered
    G = (; θ_mean=θm, b_double_prime=g.b_double_prime, b_prime=g.b_prime, born_rows=g.born_rows, born_lower=g.born_lower,
         bp, qΓtol=5e-2, cut_margin_THz=cut, n_members=5, max_attempts=20_000, outdir=mktempdir(), ylim=(-1.0, 12.0),
         PLOT_GLOBALS...)
    m = script_module(F; globals=G)
    members = [θm .* (1 .+ 2e-3 .* randn(MersenneTwister(k), g.n)) for k in 1:40]

    @testset "build_and_sample" begin
        tested(F, :build_and_sample)
        Random.seed!(3)
        mem, heig, hb, acc = call(m, :build_and_sample, reduce(hcat, members), "unit")
        @test length(mem) == 5 && 0 < acc < 100                 # some proposals were rejected
        K = dot(θm, g.b_double_prime)
        for θ in mem
            @test all(g.born_lower .<= g.born_rows * θ)
            @test abs(dot(g.b_prime, θ .- θm) / K) <= 0.1
            @test LIB.min_freq_stable(θ, bp; qΓtol=5e-2) >= cut - 1e-6
        end
        V, B = ACEWorkflow.hypercube(Matrix(reduce(hcat, members)' .- θm'))
        @test heig ≈ V && hb ≈ B                                 # the hypercube of the members' deltas
    end

    @testset "band_span" begin
        tested(F, :band_span)
        lo, hi = call(m, :band_span, members[1:3])
        @test lo ≈ [minimum(LIB.bands(θ, bp)) for θ in members[1:3]]
        @test hi ≈ [maximum(LIB.bands(θ, bp)) for θ in members[1:3]]
    end

    @testset "bands_figure" begin
        tested(F, :bands_figure)
        call(m, :bands_figure, members[1:3], "title", 12.345, "unit")
        @test filesize(joinpath(G.outdir, "bands_unit.pdf")) > 1000 && filesize(joinpath(G.outdir, "bands_unit.png")) > 1000
    end

    @testset "cover, rmse" begin
        tested(F, :cover, :rmse)
        @test call(m, :cover, [1.0, 2.0, 3.0], [0.0, 2.5, 2.0], [2.0, 3.0, 4.0]) ≈ 200 / 3
        @test call(m, :rmse, [1.0, 2.0], [1.0, 4.0]) ≈ √2
    end
end

@testset "al12/04_naive_ensemble.jl" begin
    F = "al12/04_naive_ensemble.jl"
    g = al12_globals(); bp = bp2()
    G = (; bp, a_eq=g.a, qΓtol=5e-2, element=:Al, N_cell_fc=2, N_per_seg=2, structure=:fcc, HESS_THREADS=2,
         model=g.model, unstable_tol=-0.05, ylim=(-1.0, 12.0), PLOT_GLOBALS...)
    m = script_module(F; globals=G)

    @testset "undotted_bands, native_bands_one" begin
        tested(F, :undotted_bands, :native_bands_one)
        u = call(m, :undotted_bands, g.θ)
        @test u.F ≈ LIB.bands(g.θ, bp) && u.a == g.a && u.Np == 1
        @test u.min_stable ≈ minimum(u.F[:, bp.qnorm .> 5e-2])
        # the AD (undotted) bands dotted with θ = the native Hessian's bands at the same a
        n = call(m, :native_bands_one, al12_model(), g.θ, g.a)
        @test n.a ≈ g.a rtol = 1e-6
        @test maximum(abs, n.F .- u.F) < 1e-5 * maximum(abs, u.F)
        @test n.x_vals ≈ u.x_vals && n.labels == u.labels
        # an implausible relaxation (outside ±10% of a_ref) falls back to a_ref with a warning
        nb = @test_logs (:warn, r"implausible") match_mode = :any call(m, :native_bands_one, al12_model(), g.θ, 3.0)
        @test nb.a == 3.0
    end

    @testset "native_bands_many" begin
        tested(F, :native_bands_many)
        θs = [g.θ, g.θ .* (1 .+ 1e-3 .* randn(MersenneTwister(15), g.n))]
        many = call(m, :native_bands_many, θs, g.a)
        @test length(many) == 2
        @test all(many[i].F ≈ call(m, :native_bands_one, al12_model(), θs[i], g.a).F for i in 1:2)   # parallel = serial
    end

    @testset "panel!" begin
        tested(F, :panel!)
        u = call(m, :undotted_bands, g.θ)
        soft = merge(u, (; min_stable=-1.0))
        mp = script_module(F; globals=merge(G, (; xticks=bp.x_ticks, labels=bp.labels, Np=1, xref=bp.x_vals)), only=[:panel!])
        fig = LIB.Figure()
        ax = call(mp, :panel!, fig[1, 1], [u, soft], u, "t", "(a)"; showy=false)
        @test ax isa LIB.Axis
        cols = line_colors(ax)
        @test count(==(tocol(PLOT_GLOBALS.RED)), cols) == 3
        @test count(==(tocol(PLOT_GLOBALS.GRY)), cols) == 3
        @test count(==(tocol(PLOT_GLOBALS.BLU)), cols) == 3
        @test !ax.yticklabelsvisible[]                           # showy=false hides the y decorations
    end
end
