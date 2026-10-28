# four_panel/: the Al_12 band cache builder's helpers and the figure's two utilities.

@testset "four_panel/build_bands_cache_Al_12.jl" begin
    F = "four_panel/build_bands_cache_Al_12.jl"
    g = al12_globals(); bp = bp2()
    G = (; bp, a_eq=g.a, qΓtol=5e-2, element=:Al, N_CELL=2, N_PER_SEG=2, structure=:fcc, HESS_THREADS=2,
         model=g.model, RELAX=false)
    m = script_module(F; globals=G)

    @testset "undotted_bands, native_bands_one" begin
        tested(F, :undotted_bands, :native_bands_one)
        u = call(m, :undotted_bands, g.θ)
        @test u.F ≈ LIB.bands(g.θ, bp) && u.minω ≈ minimum(u.F[:, bp.qnorm .> 5e-2]) && u.a == g.a
        # the AD bands dotted with θ = the native Hessian's, at the fitted θ and at a perturbed θ (same a here)
        for θ in (g.θ, θ_perturbed())
            n = call(m, :native_bands_one, al12_model(), θ, g.a)
            @test n.a == g.a
            @test maximum(abs, n.F .- LIB.bands(θ, bp)) < 1e-5 * maximum(abs, n.F)
        end
    end

    @testset "relax_or_reuse" begin
        tested(F, :relax_or_reuse)
        @test call(m, :relax_or_reuse, al12_model(), g.θ, 4.2) == 4.2                       # RELAX off: the saved a
        mr = script_module(F; globals=merge(G, (; RELAX=true)), only=[:relax_or_reuse])
        @test call(mr, :relax_or_reuse, al12_model(), g.θ, 4.2) ≈ g.a rtol = 1e-6           # RELAX on: relaxed
        mi = script_module(F; globals=merge(G, (; RELAX=true, a_eq=3.0)), only=[:relax_or_reuse])
        @test (@test_logs (:warn, r"implausible") match_mode = :any call(mi, :relax_or_reuse, al12_model(), g.θ, 4.2)) == 4.2
    end

    @testset "native_bands_many" begin
        tested(F, :native_bands_many)
        θs = [g.θ, θ_perturbed()]
        many = call(m, :native_bands_many, θs, [g.a, g.a + 0.01])
        @test all(many[i].F ≈ call(m, :native_bands_one, al12_model(), θs[i], [g.a, g.a + 0.01][i]).F for i in 1:2)
        @test many[2].a == g.a + 0.01                                                           # RELAX off: as given
    end
end

@testset "four_panel/plot_four_panel.jl" begin
    F = "four_panel/plot_four_panel.jl"
    tested(F, :centre, :remap)
    m = script_module(F)
    @test call(m, :centre, (; cen=[:a, :b], mean_b=:m), 2) == :b            # per-row centre when the cache has one
    @test call(m, :centre, (; mean_b=:m), 2) == :m                          # else the ensemble's mean model
    from = [0.0, 1.0, 3.0]; to = [0.0, 10.0, 20.0]
    @test call(m, :remap, from, from, to) ≈ to                              # knots map onto knots
    @test call(m, :remap, [0.5, 2.0], from, to) ≈ [5.0, 15.0]               # piecewise linear in between
    @test call(m, :remap, [-1.0, 4.0], from, to) ≈ [-10.0, 25.0]            # extrapolates the end segments
end
