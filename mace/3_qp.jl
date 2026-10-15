# 3_qp.jl — refit the readout weights θ of the trained MACE model on its whole training set (energies
# and forces), subject to the elastic constants and the lattice constant.
#
#   min_θ  ½ mean_frames ((E_DFT − E_frozen − D·θ)/N)²  +  ½ w_F mean_components (F_DFT − F_frozen − G·θ)²
#   s.t.   C11, C12, C44 ∈ target·(1 ± TOL)   and   |P(A0)| ≤ P_TOL           (P = 0 ⇔ A0 is the equilibrium)
#
# The loss has MACE's own form: per-atom energy MSE + w_F × force-component MSE, with w_F = 0.1 — the
# 1000 : 100 energy : force weights of MACE's final (SWA) training stage.  The constant (per-atom shift)
# column is held at its trained value.  C_ij = (160.2/V)(frozen + row·θ), B = E_aa/(9a).
# Solved in z = S(θ − θ0) (S = column RMS of D/N), restricted to eigen-directions of the scaled Hessian
# with eigenvalue > RTOL·max — no penalty is added; with forces in the loss every direction is kept.
# Force RMSEs are exact from the Grams.  Also printed: the unconstrained refit (LS) and a w_F scan.
# In:  $OUT from 1_constraints.py and 2_design.py
# Out: $OUT/theta_constrained_elastic.csv, theta_ls_elastic.csv, qp_elastic.txt
# Run: julia --project=.. 3_qp.jl     Env: OUT, FORCES_WEIGHT (0.1), TOL (1e-3), P_TOL (1e-3 GPa), RTOL (1e-12)

using OSQP, SparseArrays, LinearAlgebra, DelimitedFiles, Printf, Statistics

env(k, d) = parse(Float64, get(ENV, k, string(d)))
TOL, P_TOL, RTOL, FW = env("TOL", 1e-3), env("P_TOL", 1e-3), env("RTOL", 1e-12), env("FORCES_WEIGHT", 0.1)
DIR = joinpath(get(ENV, "OUT", joinpath(@__DIR__, "..", "repro", "saved")), "mace", "results")   # the route root is $OUT
GPA = 160.21766208
C_T = Dict("C11" => 116.3, "C12" => 64.8, "C44" => 30.9)            # lib.C_TARGET
B_T = (C_T["C11"] + 2C_T["C12"]) / 3

rd(f) = readdlm(joinpath(DIR, f), ',', Float64)
row(f) = vec(rd(f))
meta = Dict(String(s[1]) => parse(Float64, s[2]) for s in split.(readlines(joinpath(DIR, "meta.txt"))))
a0, V = meta["a0"], meta["V"]
θ0, fz = row("theta0.csv"), row("frozen.csv")                         # fz: dE_da, d2E_da2, d2E_de11/12/44
p = length(θ0)

props = Dict("C11" => (GPA / V .* row("d2D_de11.csv"), GPA / V * fz[3]),
             "C12" => (GPA / V .* row("d2D_de12.csv"), GPA / V * fz[4]),
             "C44" => (GPA / V .* row("d2D_de44.csv"), GPA / V * fz[5]),
             "B"   => (GPA / (9a0) .* row("d2D_da2.csv"), GPA / (9a0) * fz[2]),
             "P"   => (GPA / (3a0^2) .* row("dD_da.csv"), GPA / (3a0^2) * fz[1]))
value(name, θ) = props[name][2] + dot(props[name][1], θ)
active = ["C11", "C12", "C44", "P"]
target = Dict("C11" => C_T["C11"], "C12" => C_T["C12"], "C44" => C_T["C44"], "P" => 0.0)
R = reduce(vcat, [props[c][1]' for c in active])
base = [props[c][2] for c in active]
lo = [c == "P" ? -P_TOL : target[c] * (1 - TOL) for c in active] .- base
hi = [c == "P" ?  P_TOL : target[c] * (1 + TOL) for c in active] .- base

function split_data(s)
    D, E, fst = rd("D_$s.csv"), rd("E_$s.csv"), rd("fstats_$s.csv")
    (X = D ./ E[:, 4], y = (E[:, 1] .- E[:, 3]) ./ E[:, 4], GtG = rd("GtG_$s.csv"), Gty = row("Gty_$s.csv"),
     yty = fst[1], nF = fst[2])
end
tr, te = split_data("train"), split_data("test")
n = length(tr.y)
erms(d, θ) = 1e3 * sqrt(mean(abs2, d.y .- d.X * θ))
frms(d, θ) = 1e3 * sqrt(max(d.yty - 2dot(θ, d.Gty) + dot(θ, d.GtG * θ), 0) / d.nF)

k = 1:p-1                                                            # constant column held at θ0
S = vec(sqrt.(mean(tr.X[:, k] .^ 2, dims=1)))

function solve(fw; constrained=true)
    H = tr.X' * tr.X / n + fw * tr.GtG / tr.nF
    g = tr.X' * tr.y / n + fw * tr.Gty / tr.nF
    Hs = Symmetric(H[k, k] ./ (S * S'))
    qs = -(g[k] .- H[k, :] * θ0) ./ S
    λ, U = eigen(Hs)
    keep = λ .> RTOL * maximum(λ)
    Ur, λr = U[:, keep], λ[keep]
    qr = Ur' * qs
    if constrained
        m = OSQP.Model()
        OSQP.setup!(m; P=sparse(Diagonal(λr)), q=qr, A=sparse((R[:, k] ./ S') * Ur), l=lo .- R * θ0, u=hi .- R * θ0,
                    eps_abs=1e-9, eps_rel=1e-9, max_iter=200_000, polish=true, verbose=false)
        res = OSQP.solve!(m)
        res.info.status in (:Solved, :Solved_inaccurate) || error("OSQP status $(res.info.status) at w_F = $fw")
        res.info.status == :Solved || @warn "OSQP $(res.info.status) at w_F = $fw: check the constrained values"
        z = Ur * res.x
    else
        z = Ur * (-qr ./ λr)
    end
    θ = copy(θ0); θ[k] .+= z ./ S
    θ, norm(z), count(keep), maximum(λ) / minimum(λ)
end

θls, nls, _, _ = solve(FW; constrained=false)
θc, nc, rank, cond = solve(FW)
lines = [@sprintf("elastic constraints at A0 %.4f Å (model a0 %.4f); w_F %.3g; train %d frames / %d force comps × %d cols; test %d frames",
                  a0, meta["a_model"], FW, n, Int(tr.nF), p, length(te.y)),
         @sprintf("Hessian condition %.1e, kept %d of %d directions (RTOL %.0e), TOL %.0e", cond, rank, p - 1, RTOL, TOL),
         @sprintf("%-12s %8s %8s %8s %8s | %7s %7s %7s %7s %9s", "", "E train", "E test", "F train", "F test", "C11", "C12", "C44", "B", "P(A0)"),
         @sprintf("%-12s %35s | %7.1f %7.1f %7.1f %7.1f", "target", "", C_T["C11"], C_T["C12"], C_T["C44"], B_T)]
for (name, t) in (("MACE", θ0), ("LS", θls), ("constrained", θc))
    push!(lines, @sprintf("%-12s %8.2f %8.2f %8.1f %8.1f | %7.1f %7.1f %7.1f %7.1f %+9.2e", name, erms(tr, t), erms(te, t),
                          frms(tr, t), frms(te, t), value("C11", t), value("C12", t), value("C44", t), value("B", t), value("P", t)))
end
push!(lines, @sprintf("‖S δθ‖: LS %.3e, constrained %.3e   (E meV/atom, F meV/Å, moduli GPa at A0)", nls, nc))
push!(lines, "w_F scan (constrained): w_F → E test, F test, ‖S δθ‖")
for fw in (0.01, 0.1, 1.0, 10.0)
    t, nz, _, _ = solve(fw)
    push!(lines, @sprintf("  %6.2f   %8.2f %8.1f   %.3e", fw, erms(te, t), frms(te, t), nz))
end
txt = join(lines, "\n")
println(txt)
write(joinpath(DIR, "qp_elastic.txt"), txt * "\n")
writedlm(joinpath(DIR, "theta_constrained_elastic.csv"), θc', ',')
writedlm(joinpath(DIR, "theta_ls_elastic.csv"), θls', ',')
