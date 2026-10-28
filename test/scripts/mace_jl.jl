# mace/3_qp.jl and mace/pops/p2_cloud.jl: the readout QPs, their loaders and error measures, on synthetic data.

"A synthetic readout problem: energies X (per atom), y; force Grams; 4 property rows R; θ0 with the shift last."
function readout_problem(; n=60, p=6, nF=300, seed=20)
    rng = MersenneTwister(seed)
    X = randn(rng, n, p); X[:, end] .= 1.0                     # the constant (shift) column
    θ0 = randn(rng, p); y = X * θ0 .+ 0.05 .* randn(rng, n)
    G = randn(rng, nF, p); f = G * θ0 .+ 0.05 .* randn(rng, nF)
    R = randn(rng, 4, p)
    (; X, y, θ0, G, f, GtG=G' * G, Gty=G' * f, yty=dot(f, f), nF=Float64(nF), R, n, p)
end

@testset "mace/3_qp.jl" begin
    F = "mace/3_qp.jl"
    d = readout_problem(); k = 1:d.p-1
    S = vec(sqrt.(mean(d.X[:, k] .^ 2, dims=1)))
    tr = (; X=d.X, y=d.y, GtG=d.GtG, Gty=d.Gty, yty=d.yty, nF=d.nF)

    @testset "env, rd, row, split_data" begin
        tested(F, :env, :rd, :row, :split_data)
        mktempdir() do dir
            D = rand(MersenneTwister(21), 5, 3); E = hcat(rand(5), zeros(5), rand(5), [1.0, 2, 3, 4, 5])
            writedlm(joinpath(dir, "D_s.csv"), D, ','); writedlm(joinpath(dir, "E_s.csv"), E, ',')
            writedlm(joinpath(dir, "GtG_s.csv"), Matrix(1.0I, 3, 3), ','); writedlm(joinpath(dir, "Gty_s.csv"), [1.0, 2, 3], ',')
            writedlm(joinpath(dir, "fstats_s.csv"), [7.0, 9.0], ',')
            m = script_module(F; globals=(; DIR=dir), only=[:env, :rd, :row, :split_data])
            withenv("QP_TEST_X" => "0.25") do
                @test call(m, :env, "QP_TEST_X", 1.0) === 0.25
            end
            @test call(m, :env, "QP_TEST_UNSET", 3) === 3.0
            @test call(m, :rd, "D_s.csv") == D && call(m, :row, "Gty_s.csv") == [1.0, 2, 3]
            s = call(m, :split_data, "s")
            @test s.X ≈ D ./ E[:, 4] && s.y ≈ (E[:, 1] .- E[:, 3]) ./ E[:, 4]     # per atom, frozen part removed
            @test s.GtG == I(3) && s.Gty == [1.0, 2, 3] && s.yty == 7 && s.nF == 9
        end
    end

    @testset "value, erms, frms" begin
        tested(F, :value, :erms, :frms)
        m = script_module(F; globals=(; props=Dict("C11" => ([1.0, 2.0], 10.0))), only=[:value, :erms, :frms])
        @test call(m, :value, "C11", [0.5, 1.0]) ≈ 12.5                           # base + row·θ
        θ = randn(MersenneTwister(22), d.p)
        @test call(m, :erms, tr, θ) ≈ 1e3 * sqrt(mean(abs2, d.y .- d.X * θ))
        @test call(m, :frms, tr, θ) ≈ 1e3 * sqrt(mean(abs2, d.f .- d.G * θ))       # exact from the Grams
    end

    @testset "solve" begin
        tested(F, :solve)
        lo, hi = fill(-Inf, 4), fill(Inf, 4)
        G = (; tr, n=d.n, k, S, θ0=d.θ0, R=d.R, RTOL=1e-12)
        m = script_module(F; globals=merge(G, (; lo, hi)), only=[:solve])
        fw = 0.1
        H = d.X' * d.X / d.n + fw * d.GtG / d.nF; g = d.X' * d.y / d.n + fw * d.Gty / d.nF
        # unconstrained: the minimiser over θ[k] with the shift held at θ0
        θu = copy(d.θ0); θu[k] = H[k, k] \ (g[k] - H[k, end] * d.θ0[end])
        θl, _, rank, _ = call(m, :solve, fw; constrained=false)
        @test θl ≈ θu rtol = 1e-10
        @test θl[end] == d.θ0[end] && rank == d.p - 1
        θc, _, _, _ = call(m, :solve, fw)                                          # inactive windows: same point
        @test θc ≈ θu rtol = 1e-6
        # one window forced onto the first property: the KKT point with it as an equality
        v = dot(d.R[1, :], θu) + 0.5
        mc = script_module(F; globals=merge(G, (; lo=[v; fill(-Inf, 3)], hi=[v + 1e-9; fill(Inf, 3)])), only=[:solve])
        θa, _, _, _ = call(mc, :solve, fw)
        @test dot(d.R[1, :], θa) ≈ v rtol = 1e-7
        Hk = H[k, k]; gk = g[k] - H[k, end] * d.θ0[end]
        @test θa[k] ≈ kkt(Hk, -gk, reshape(d.R[1, k], 1, :), [v - d.R[1, end] * d.θ0[end]]) rtol = 1e-6
        # contradictory windows (one property row twice, disjoint windows) are an error, not a silent answer
        Ri = copy(d.R); Ri[2, :] = Ri[1, :]
        mi = script_module(F; globals=merge(G, (; R=Ri, lo=[0.0, 1.0, -Inf, -Inf], hi=[0.1, 1.1, Inf, Inf])), only=[:solve])
        @test_throws ErrorException call(mi, :solve, fw)
        # RTOL drops degenerate directions
        Xd = copy(d.X); Xd[:, 2] = Xd[:, 1]
        trd = merge(tr, (; X=Xd, GtG=zeros(d.p, d.p), Gty=zeros(d.p)))
        Sd = vec(sqrt.(mean(Xd[:, k] .^ 2, dims=1)))
        md = script_module(F; globals=merge(G, (; tr=trd, S=Sd, lo, hi, RTOL=1e-10)), only=[:solve])
        @test call(md, :solve, fw; constrained=false)[3] == d.p - 2
    end
end

@testset "mace/pops/p2_cloud.jl" begin
    F = "mace/pops/p2_cloud.jl"
    rng = MersenneTwister(23)
    Q = Matrix(qr(randn(rng, 5, 5)).Q); λ = [1e-14, 0.5, 1.0, 2.0, 4.0]
    Hs = Symmetric(Q * Diagonal(λ) * Q')
    keep = λ .> 1e-12 * maximum(λ); Ur, λr = Q[:, keep], λ[keep]

    @testset "env, rd, mrow" begin
        tested(F, :env, :rd, :mrow)
        mktempdir() do dir
            writedlm(joinpath(dir, "v.csv"), [1.0, 2.0], ',')
            m = script_module(F; globals=(; MAIN=dir), only=[:env, :rd, :mrow])
            @test call(m, :mrow, "v.csv") == [1.0, 2.0] && call(m, :rd, dir, "v.csv") == reshape([1.0, 2.0], 2, 1)
            @test call(m, :env, "P2_TEST_UNSET", 0.5) === 0.5
        end
    end

    @testset "Hinv" begin
        tested(F, :Hinv)
        m = script_module(F; globals=(; Ur, λr), only=[:Hinv])
        v = randn(rng, 5)
        x = call(m, :Hinv, v)
        @test Hs * x ≈ Ur * (Ur' * v)                       # the inverse on the kept subspace (a pseudo-inverse)
        @test abs(dot(Q[:, 1], x)) < 1e-12                   # nothing along the dropped direction
        V = randn(rng, 5, 3)
        @test call(m, :Hinv, V) ≈ hcat([call(m, :Hinv, V[:, j]) for j in 1:3]...)
    end

    @testset "osqp_solve" begin
        tested(F, :osqp_solve)
        qr_ = randn(rng, length(λr))
        m = script_module(F; globals=(; Psp=sparse(Diagonal(λr)), qr=qr_, Ur, RHO_INTERVAL=25), only=[:osqp_solve])
        A = randn(rng, 2, length(λr))
        z, st = call(m, :osqp_solve, A, fill(-1e6, 2), fill(1e6, 2))      # inactive: ζ = −q/λ
        @test st == :Solved
        @test z ≈ Ur * (-qr_ ./ λr) rtol = 1e-7
        z2, _ = call(m, :osqp_solve, A, [0.3, -1e6], [0.3, 1e6])           # an equality row
        ζ = kkt(Matrix(Diagonal(λr)), qr_, A[1:1, :], [0.3])
        @test z2 ≈ Ur * ζ rtol = 1e-6
    end
end
