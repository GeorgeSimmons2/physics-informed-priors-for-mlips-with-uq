# p2_cloud.jl — the POPS clouds of the elastic-constrained MACE readout (and of the unconstrained one).
#
# Observations r = every training energy (per atom) and force component, with the mean model's loss weights
#   w_r = 1/n_E (energies),  w_F/n_F (force components),   H = Σ w_r a_r a_rᵀ,  g = Σ w_r y_r a_r
# (the objective of ../3_qp.jl; the shift column held at θ0).  Leverage h_r = w_r a_rᵀ H⁻¹ a_r; the top
# LEV_PERCENTILE of rows by leverage are kept, as in the ACE pipeline (constrained_optimization/al12/02).
#
# CONSTRAINED cloud, one QP per kept row r (the mean model's QP plus exact interpolation of r):
#     min ½ θᵀHθ − gᵀθ   s.t.  a_r·θ = y_r,   C11, C12, C44 ∈ target(1 ± TOL),   |P(a0)| ≤ P_TOL
# solved in z = S(θ − θc) around the mean θc ($OUT/mace/results/theta_constrained_elastic.csv), in the
# eigenbasis of the scaled Hessian exactly as ../3_qp.jl (directions with λ > RTOL·λmax; OSQP sees a diagonal
# P — on the dense Hessian it declares the problem dual infeasible), with a pinned rho schedule so reruns follow
# the same iterate path.  The same eigen-decomposition gives H⁻¹ for the leverage and the closed form below.
# UNCONSTRAINED cloud: the closed-form POPS correction about MACE's trained readout weights θ0,
#     θ_r = θ0 + H⁻¹ a_r (y_r − a_r·θ0) / (a_rᵀ H⁻¹ a_r),   same kept rows
# (the smallest change of θ0, in the loss metric, that interpolates row r).  θ_LS, the unconstrained optimum
# of this loss, is reported for reference only: it is within ‖S(θ_LS − θ0)‖ ≈ 2 of θ0.
# Checks printed: H from the rows reproduces the mean model (the QP without interpolation returns θc);
# every constrained member interpolates its row and sits inside the windows.
# In:  $OUT/mace/pops/rows/{XE,yE,XF,yF}_train.csv (p1, p1b), $OUT/mace/results/*.csv (1_constraints.py, 3_qp.jl)
# Out ($OUT/mace/pops): cloud.jls (Θ_con, Θ_unc as 81 × N, kept rows, leverage, centres), cloud.txt
# Run: julia --project -t 16 mace/pops/p2_cloud.jl      Env: OUT, LEV_PERCENTILE (0.5), FORCES_WEIGHT (0.1), RHO_INTERVAL (25), RTOL (1e-12)

using OSQP, SparseArrays, LinearAlgebra, DelimitedFiles, Printf, Statistics, Serialization

env(k, d) = parse(Float64, get(ENV, k, string(d)))
TOL, P_TOL, FW, LEV, RTOL = 1e-3, 1e-3, env("FORCES_WEIGHT", 0.1), env("LEV_PERCENTILE", 0.5), env("RTOL", 1e-12)
RHO_INTERVAL = parse(Int, get(ENV, "RHO_INTERVAL", "25"))
ROOT = get(ENV, "OUT", joinpath(@__DIR__, "..", "..", "repro", "saved"))   # the route root
MAIN, ROWS = joinpath(ROOT, "mace", "results"), joinpath(ROOT, "mace", "pops", "rows")
OUT = joinpath(ROOT, "mace", "pops"); mkpath(OUT)
GPA = 160.21766208
BLAS.set_num_threads(1)

rd(dir, f) = readdlm(joinpath(dir, f), ',', Float64)
mrow(f) = vec(rd(MAIN, f))
meta = Dict(String(s[1]) => parse(Float64, s[2]) for s in split.(readlines(joinpath(MAIN, "meta.txt"))))
a0, V = meta["a0"], meta["V"]
θ0, fz, θc = mrow("theta0.csv"), mrow("frozen.csv"), mrow("theta_constrained_elastic.csv")
p = length(θ0); k = 1:p-1

# constraint rows in GPa (as ../3_qp.jl) and their windows, absolute
R = [GPA / V .* mrow("d2D_de11.csv")'
     GPA / V .* mrow("d2D_de12.csv")'
     GPA / V .* mrow("d2D_de44.csv")'
     GPA / (3a0^2) .* mrow("dD_da.csv")']
base = [GPA / V * fz[3], GPA / V * fz[4], GPA / V * fz[5], GPA / (3a0^2) * fz[1]]
target = [116.3, 64.8, 30.9]
lo = [target .* (1 - TOL); -P_TOL] .- base
hi = [target .* (1 + TOL);  P_TOL] .- base

# observations
XE, yE, XF, yF = rd(ROWS, "XE_train.csv"), vec(rd(ROWS, "yE_train.csv")), rd(ROWS, "XF_train.csv"), vec(rd(ROWS, "yF_train.csv"))
nE, nF = length(yE), length(yF)
A = vcat(XE, XF); y = vcat(yE, yF); w = vcat(fill(1 / nE, nE), fill(FW / nF, nF))
isE = vcat(trues(nE), falses(nF))
H = A' * (w .* A); g = A' * (w .* y)
S = vec(sqrt.(mean(XE[:, k] .^ 2, dims=1)))
Hs = Symmetric(H[k, k] ./ (S * S'))
λ, U = eigen(Hs)
keepdir = λ .> RTOL * maximum(λ)
Ur, λr = U[:, keepdir], λ[keepdir]
Hinv(v) = Ur * ((Ur' * v) ./ λr)                    # H_s⁻¹ on the kept subspace
@printf("%d energy + %d force rows × %d columns; scaled Hessian condition %.1e, %d of %d directions kept\n",
        nE, nF, p, maximum(λ) / minimum(λ), count(keepdir), p - 1)

# check: the mean model's QP rebuilt from the rows returns θc (z = 0)
Ak = sparse(R[:, k] ./ S')
qc = -(g[k] .- H[k, :] * θc) ./ S
Psp = sparse(Diagonal(λr))
qr = Ur' * qc
AkU = Matrix(Ak) * Ur
function osqp_solve(Aqp, l, u)                       # Aqp in ζ (z = Ur ζ); returns z
    m = OSQP.Model()
    OSQP.setup!(m; P=Psp, q=qr, A=sparse(Aqp), l=l, u=u, eps_abs=1e-9, eps_rel=1e-9,
                max_iter=200_000, polish=true, verbose=false, adaptive_rho_interval=RHO_INTERVAL)
    res = OSQP.solve!(m)
    Ur * res.x, res.info.status
end
z_mean, st = osqp_solve(AkU, lo .- R * θc, hi .- R * θc)
@printf("mean-model check: QP from the rows returns ‖z − 0‖ = %.2e (status %s)\n", norm(z_mean), st)

# leverage → kept rows
B = A[:, k] ./ S'                                   # rows in z-space
lev = w .* vec(sum(B .* Hinv(Matrix(B'))', dims=2))
thresh = quantile(lev, 1 - LEV)
kept = findall(>=(thresh), lev)
N = length(kept)
@printf("leverage ∈ [%.2e, %.2e]; top %.0f%% → %d rows (%d energies, %d force components)\n",
        minimum(lev), maximum(lev), 100LEV, N, count(isE[kept]), count(.!isE[kept]))

# constrained cloud: one QP per kept row
Θ_con = zeros(p, N); status = fill(:unset, N)
t = @elapsed Threads.@threads for j in 1:N
    r = kept[j]
    Aqp = vcat(AkU, reshape(Ur' * B[r, :], 1, :))
    e = y[r] - dot(A[r, :], θc)
    z, st = osqp_solve(Aqp, vcat(lo .- R * θc, e), vcat(hi .- R * θc, e))
    θ = copy(θc); θ[k] .+= z ./ S
    Θ_con[:, j] = θ; status[j] = st
end
ok = status .== :Solved
interp = [abs(dot(A[kept[j], :], Θ_con[:, j]) - y[kept[j]]) for j in 1:N]
inwin = [all(lo .- 1e-6 .<= R * Θ_con[:, j] .<= hi .+ 1e-6) for j in 1:N]
@printf("constrained cloud: %d QPs in %.1f s; %d Solved; max interpolation residual %.1e; %d inside the windows\n",
        N, t, count(ok), maximum(interp[ok]), count(inwin .& ok))

# unconstrained cloud: closed-form corrections about MACE's trained weights θ0
z_ls = Hinv(-(g[k] .- H[k, :] * θ0) ./ S)
θ_LS = copy(θ0); θ_LS[k] .+= z_ls ./ S
Bk = B[kept, :]'                                     # (p−1) × N
HiB = Hinv(Bk)
e0 = y[kept] .- A[kept, :] * θ0
Δz = HiB .* (e0 ./ vec(sum(Bk .* HiB, dims=1)))'
Θ_unc = repeat(θ0, 1, N); Θ_unc[k, :] .+= Δz ./ S
@printf("unconstrained cloud: %d closed-form members about θ0; max interpolation residual %.1e\n",
        N, maximum(abs.(vec(sum(A[kept, :]' .* Θ_unc, dims=1)) .- y[kept])))

serialize(joinpath(OUT, "cloud.jls"), (; Θ_con=Θ_con[:, ok], Θ_unc, kept, kept_con=kept[ok], lev, isE, θc, θ_LS, θ0,
                                        R, lo, hi, base, status))
txt = """
POPS clouds of the elastic-constrained MACE readout (p2_cloud.jl)
rows: $nE energies + $nF force components; leverage percentile $LEV → $N kept ($(count(isE[kept])) E, $(count(.!isE[kept])) F)
constrained: $(count(ok)) / $N Solved, all inside the ±$(100TOL)% windows: $(all(inwin[ok])); max interpolation residual $(maximum(interp[ok]))
unconstrained: $N closed-form members about θ0 (MACE as trained); for reference ‖S(θ_LS − θ0)‖ = $(norm(z_ls))
mean-model check: ‖z‖ = $(norm(z_mean))
"""
write(joinpath(OUT, "cloud.txt"), txt)
print(txt)
