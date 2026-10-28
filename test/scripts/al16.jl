# al16/01–02: the 3-row undotted Hessian and its band path, the rejection predicate, the a-shift estimate,
# reading ensembles, native bands and their panel.  Run on the Al_12 model (the functions are model-generic).

@testset "al16/01_pinned_ensembles.jl" begin
    F = "al16/01_pinned_ensembles.jl"
    g = al12_globals()
    G = (; element=:Al, model=g.model, N_CELL=2, N_PER_SEG=2, BUILD_THREADS=4, lin_params=g.θ,
         b_prime=g.b_prime, b_double_prime=g.b_double_prime, ACCEPT_TOL=-0.05)
    m = script_module(F; globals=G)

    @testset "_site_basis_hessian" begin
        tested(F, :_site_basis_hessian)
        V = al12_model()
        sys = rattled_super(g.a, 2)
        _, Rs, Zs, z0 = LIB.get_neighbours(sys, V, LIB.PairList(sys, LIB.cutoff_radius(V)), 3)
        H16 = LIB.@no_escape begin call(m, :_site_basis_hessian, V.model, Rs, Zs, z0, V.ps, V.st) end
        H = LIB.@no_escape begin LIB._site_basis_hessian(V.model, Rs, Zs, z0, V.ps, V.st) end
        @test H16 ≈ H rtol = 1e-12                                 # the same per-site Hessian as lib's
    end

    @testset "undotted_rows" begin
        tested(F, :undotted_rows)
        sys = rattled_super(g.a, 2)
        r = 5
        Hr = call(m, :undotted_rows, sys, al12_model(), r)
        @test size(Hr) == (3, 3 * length(sys), g.n)
        # the reference atom's three rows of the AD Hessian, dotted with θ, are the native Hessian's rows
        for θ in (g.θ, θ_perturbed())
            Hn = native_hessian(sys, θ)
            @test reshape(reshape(Hr, 9 * length(sys), :) * θ, 3, :) ≈ Hn[3r-2:3r, :] rtol = 1e-10
        end
    end

    @testset "bandpath_rows" begin
        tested(F, :bandpath_rows)
        bpr = call(m, :bandpath_rows, al12_model(), g.a)
        bp = bp2()
        @test bpr.q_list ≈ bp.q_list && bpr.qnorm ≈ bp.qnorm && bpr.a_ref == g.a && bpr.Np == 1
        for θ in (g.θ, θ_perturbed())                              # = native D(q) and = the full undotted path
            mθ = al12_model(); ACEpotentials.Models.set_linear_parameters!(mθ, θ)
            prim, sup = ACEWorkflow.bulk_prim_super(:Al; a=g.a, N_cell=2)
            fc = ACEWorkflow.precompute_force_constants(prim, sup, mθ)
            Dn = [ACEWorkflow.dynamical_matrix_from_fc(fc, q) for q in bpr.q_list]
            @test maximum(norm(reshape(bpr.Bq[iq] * θ, 3, 3) - Dn[iq]) for iq in eachindex(Dn)) < 1e-10 * maximum(norm, Dn)
            @test maximum(norm((bpr.Bq[iq] - bp.Bq[iq]) * θ) for iq in eachindex(Dn)) < 1e-10 * maximum(norm, Dn)
        end
    end

    @testset "is_physical, minw" begin
        tested(F, :is_physical, :minw)
        bp = toy_bp()                                              # ω² = (−9,0,1)θ at Γ, (−4,1,9)θ, (¼,4,16)θ
        counters = (; n_prop=Ref(0), n_rej_K=Ref(0), n_rej_ω=Ref(0))
        mp = script_module(F; only=[:is_physical, :minw],
                           globals=merge(counters, (; bp, keep_q=[3], b_double_prime=[1.0], ACCEPT_TOL=-0.05)))
        @test call(mp, :is_physical, [1.0])                        # q3 only: all ω > 0
        @test !call(mp, :is_physical, [-1.0])                      # b″·θ ≤ 0
        @test counters.n_prop[] == 2 && counters.n_rej_K[] == 1 && counters.n_rej_ω[] == 0
        @test call(mp, :minw, [1.0]) ≈ 0.5 * ACEWorkflow.FREQ_THz
        mq = script_module(F; only=[:is_physical, :minw],
                           globals=merge(counters, (; bp, keep_q=[2, 3], b_double_prime=[1.0], ACCEPT_TOL=-0.05)))
        @test !call(mq, :is_physical, [1.0])                        # q2 has ω = −2 THz·FREQ < ACCEPT_TOL
        @test counters.n_rej_ω[] == 1
        @test call(mq, :minw, [1.0]) ≈ -2 * ACEWorkflow.FREQ_THz
    end

    @testset "da" begin
        tested(F, :da)
        @test call(m, :da, g.θ) == 0
        # the Newton estimate −b′·δ/(b″·θ) of the lattice-constant shift agrees with relaxing the perturbed model
        θp = g.θ .+ 2e-4 .* (g.θ .* randn(MersenneTwister(16), g.n))
        mp = al12_model(); ACEpotentials.Models.set_linear_parameters!(mp, θp)
        Δ = ACEWorkflow.relax_lattice_constant(mp, :Al) - g.a
        @test abs(Δ) > 1e-4
        @test call(m, :da, θp) ≈ Δ rtol = 0.05
    end
end

@testset "al16/02_bands.jl" begin
    F = "al16/02_bands.jl"
    g = al12_globals(); bp = bp2()
    G = (; element=:Al, model=g.model, N_CELL=2, N_PER_SEG=2, structure=:fcc, qΓtol=5e-2, QOI_THREADS=2,
         n_params=g.n, UNSTABLE=-0.05, ylim=(-1.0, 12.0), xt=bp.x_ticks, lbl=bp.labels, xref=bp.x_vals, Np=1,
         PLOT_GLOBALS...)

    @testset "read_ens" begin
        tested(F, :read_ens)
        mktempdir() do dir
            M = rand(MersenneTwister(17), 4, g.n)
            writedlm(joinpath(dir, "ensemble_x.csv"), M, ',')
            writedlm(joinpath(dir, "ensemble_bad.csv"), M[:, 1:5], ',')
            m = script_module(F; globals=merge(G, (; SRC=dir)), only=[:read_ens])
            ens = call(m, :read_ens, "x")
            @test length(ens) == 4 && ens[2] == M[2, :]
            @test_throws ErrorException call(m, :read_ens, "bad")       # wrong width
            @test_throws ErrorException call(m, :read_ens, "none")      # missing file
        end
    end

    m = script_module(F; globals=G)
    @testset "bands_native, bands_many" begin
        tested(F, :bands_native, :bands_many)
        b = call(m, :bands_native, al12_model(), g.θ)              # relaxes, then the native Hessian's bands
        @test b.a ≈ g.a rtol = 1e-6
        @test maximum(abs, b.F .- LIB.bands(g.θ, bp)) < 1e-5 * maximum(abs, b.F)
        @test b.minω ≈ minimum(b.F[:, bp.qnorm .> 5e-2])
        θs = [g.θ, g.θ .* (1 .+ 1e-3 .* randn(MersenneTwister(18), g.n))]
        many = call(m, :bands_many, θs)
        @test all(many[i].F ≈ call(m, :bands_native, al12_model(), θs[i]).F for i in 1:2)
    end

    @testset "panel!" begin
        tested(F, :panel!)
        b = call(m, :bands_native, al12_model(), g.θ)
        mp = script_module(F; globals=merge(G, (; mean_b=b)), only=[:panel!])
        fig = LIB.Figure()
        call(mp, :panel!, fig[1, 1], [b, merge(b, (; minω=-1.0)), merge(b, (; minω=-2.0))], "t")
        ax = fig.content[1]
        @test count(==(tocol(PLOT_GLOBALS.RED)), line_colors(ax)) == 6         # two soft members × 3 branches
        txt = only(filter(p -> p isa LIB.Makie.Text, ax.scene.plots))
        @test occursin("2/3 unstable", string(txt.text[]))
    end
end
