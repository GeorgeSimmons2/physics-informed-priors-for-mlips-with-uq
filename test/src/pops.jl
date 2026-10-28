# src/POPS/POPSRegression.jl — POPS corrections, the hypercube and its samplers.

const POPS_SRC = "src/POPS/POPSRegression.jl"

@testset "corrections" begin
    tested(POPS_SRC, :corrections)
    (; X, Y) = ls_problem()
    n, p = size(X)
    for Γ in (Matrix(1.0I, p, p), Matrix(Diagonal(0.5 .+ rand(MersenneTwister(2), p))))
        λ = 1 / n
        C = Γ' * Γ * λ + X' * X
        c = C \ (X' * Y)                                     # the default coefficients
        h = [dot(X[i, :], C \ X[i, :]) for i in 1:n]          # leverages, computed independently
        R = ACEWorkflow.corrections(X, Y, Γ)
        keep = findall(h .>= quantile(h, 0.5))
        @test size(R) == (length(keep), p)
        @test all(0 .< h .< 1)
        # each pointwise correction (θ-space, δx = Γ r) makes its own observation exact
        @test all(isapprox(dot(X[i, :], c .+ Γ * R[k, :]), Y[i]; rtol=1e-10, atol=1e-10) for (k, i) in enumerate(keep))
        # and the default coefficients are the regularised least-squares solution
        Rc = ACEWorkflow.corrections(X, Y, Γ; coeffs=c)
        @test Rc ≈ R
    end
    # explicit coefficients: corrections are taken around them
    c0 = zeros(p)
    R0 = ACEWorkflow.corrections(X, Y, Matrix(1.0I, p, p); coeffs=c0, leverage_percentile=0.9)
    C = X' * X + I / n
    h = [dot(X[i, :], C \ X[i, :]) for i in 1:n]
    keep = findall(h .>= quantile(h, 0.9))
    @test size(R0, 1) == length(keep)
    @test all(isapprox(dot(X[i, :], R0[k, :]), Y[i]; rtol=1e-10) for (k, i) in enumerate(keep))
end

@testset "hypercube" begin
    tested(POPS_SRC, :hypercube)
    rng = MersenneTwister(3)
    m, p, r = 120, 7, 4
    pc = randn(rng, m, r) * randn(rng, r, p)                  # rank 4 in a 7-dimensional space
    V, B = ACEWorkflow.hypercube(pc)
    @test size(V) == (p, r)                                   # the null directions are cut
    @test V' * V ≈ I(r)
    @test pc * V * V' ≈ pc                                    # nothing of the cloud is lost
    proj = pc * V
    @test B[1, :] ≈ vec(minimum(proj; dims=1))
    @test B[2, :] ≈ vec(maximum(proj; dims=1))
    _, Bc = ACEWorkflow.hypercube(pc; percentile_clipping=10.0)
    @test all(Bc[1, :] .>= B[1, :]) && all(Bc[2, :] .<= B[2, :])
    @test Bc[1, :] ≈ [quantile(proj[:, j], 0.1) for j in 1:r]
end

@testset "sample_hypercube" begin
    tested(POPS_SRC, :sample_hypercube)
    rng = MersenneTwister(4)
    p, d, N = 6, 3, 25
    V = Matrix(qr(randn(rng, p, d)).Q)[:, 1:d]
    B = vcat(-rand(rng, d)', rand(rng, d)')
    c = randn(rng, p)
    Random.seed!(11)
    M, δθ = ACEWorkflow.sample_hypercube(V, B, c; number_of_committee_members=N)
    @test size(M) == (p, N)
    Δ = M .- c
    z = V' * Δ                                                # box coordinates of each member
    @test all(B[1, :] .- 1e-12 .<= z .<= B[2, :] .+ 1e-12)
    @test Δ ≈ V * z                                           # members stay in the span of V
    @test δθ ≈ Δ * Δ' / N
    # member i is row i of rand(N, d): changing N changes every member at a fixed seed
    Random.seed!(11); U = rand(N, d)
    @test z ≈ (B[1, :] .+ (B[2, :] .- B[1, :]) .* U')
    Random.seed!(11); M2, _ = ACEWorkflow.sample_hypercube(V, B, c; number_of_committee_members=N + 1)
    @test !(M2[:, 1] ≈ M[:, 1])
end

@testset "rejection_sample_hypercube" begin
    tested(POPS_SRC, :rejection_sample_hypercube)
    rng = MersenneTwister(5)
    p, d, N = 5, 3, 12
    V = Matrix(qr(randn(rng, p, d)).Q)[:, 1:d]
    B = vcat(-ones(d)', ones(d)')
    c = zeros(p)
    # predicate form: every accepted member satisfies it; draws are rand(d) per proposal
    feasible(θ) = θ[1] > 0
    Random.seed!(21)
    M, δθ = @test_logs (:info, r"accepted 12 / \d+ proposals") ACEWorkflow.rejection_sample_hypercube(
        V, B, c, feasible; number_of_committee_members=N)
    @test size(M) == (p, N)
    @test all(M[1, :] .> 0)
    @test δθ ≈ (M .- c) * (M .- c)' / N
    Random.seed!(21)
    first_ok = nothing
    while first_ok === nothing
        u = rand(d); δ = V * (B[1, :] .+ (B[2, :] .- B[1, :]) .* u)
        feasible(c .+ δ) && (first_ok = δ)
    end
    @test M[:, 1] ≈ c .+ first_ok
    # an infeasible predicate stops at max_attempts
    @test_throws ErrorException ACEWorkflow.rejection_sample_hypercube(V, B, c, θ -> false;
                                                                        number_of_committee_members=2, max_attempts=50)
    # linear-constraint form: l ≤ A c ≤ u for every member; equality rows are refused
    A = randn(rng, 2, p); l = [-0.2, -Inf]; u = [Inf, 0.3]
    Random.seed!(22)
    Mc, _ = ACEWorkflow.rejection_sample_hypercube(V, B, c, A, (l, u); number_of_committee_members=N)
    @test size(Mc) == (p, N)
    @test all(l .<= A * Mc .<= u)
    @test_throws ArgumentError ACEWorkflow.rejection_sample_hypercube(V, B, c, A, ([0.0, -Inf], [0.0, Inf]))
end
