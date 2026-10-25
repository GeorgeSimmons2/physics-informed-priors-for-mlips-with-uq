# verify/mean_qp.jl — judge the Al_12 constrained mean θ_mean on what it predicts, at the method's own precision.

include(joinpath(@__DIR__, "..", "lib", "bandpath.jl"))

function check_mean_qp(label, published, reproduced; cut_margin=0.15, N_per_seg=[20, 20, 20, 20, 60])
    (isfile(published) && isfile(reproduced)) || return Check(label, :converged, :missing, "not produced yet: $reproduced")
    θp, θr = vec(readdlm(published, ',')), vec(readdlm(reproduced, ','))
    result = load_model(:Al, 12, 4, 6, 2; dataset_name=""); model = result.model; n = length(result.lin_params)
    P = result.P; Ap = Diagonal(result.W)*result.A/P; Yw = result.W.*result.Y; λ = 1.0/size(Ap, 1)
    a_eq = ACEWorkflow.relax_lattice_constant(model, :Al)
    lb(a) = ustrip.(u"eV", ACEpotentials.Models.potential_energy_basis(ACEWorkflow.Elasticity.reference_system(:Al; a=a), model))
    b1 = ForwardDiff.derivative(lb, a_eq); b2 = ForwardDiff.derivative(a -> ForwardDiff.derivative(lb, a), a_eq)
    He = reshape(elastic_hessian_basis(model; element=:Al, a=a_eq), 36, n)
    born = vcat(He[22,:]', (He[1,:] .- He[7,:])', (He[1,:] .+ 2 .* He[7,:])', b2'); lower = [0.1, 1.0, 0.1, 1e-9]
    # the exact optimum: equality-constrained KKT (valid while the Born rows are slack, checked below)
    H = Symmetric(Ap'*Ap .+ λ.*(P'*P)); q = -(Ap'*Yw); g = vec(b1'/P); F = cholesky(H)
    x0 = -(F\q); Hg = F\g; θs = P\(x0 .- Hg .* (dot(g, x0)/dot(g, Hg)))
    minimum(born*θs .- lower) > 0 || return Check(label, :converged, :fail, "a Born row is active at the exact optimum; this check assumes they are slack")
    Aw = Diagonal(result.W)*result.A                                      # acts on θ (Ap acts on x = Pθ)
    res = norm(Aw*θp .- Yw); pred(θa, θb) = norm(Aw*(θa .- θb)) / res     # in units of the published fit's residual
    bp = bandpath_Dk(result, model, :Al, a_eq, 4; N_per_seg=N_per_seg)
    Fp = bands(θp, bp); dω(θ) = maximum(abs.(bands(θ, bp) .- Fp))
    d_pred, d_band = pred(θr, θp), dω(θr); r_pred, r_band = pred(θs, θp), dω(θs)
    ok_c = abs(dot(b1, θr)) <= 1e-6 && minimum(born*θr .- lower) > 0 && min_freq_stable(θr, bp) >= cut_margin - 1e-3
    ok = ok_c && d_pred <= r_pred && d_band <= r_band
    Check(label, :converged, ok ? :ok : :fail,
          @sprintf("rerun vs published: predictions %.2e, bands max|Δω| %.3f THz — within the published solution's own distance from the exact optimum (%.2e, %.3f THz); constraints %s; raw ‖Δθ‖/‖θ‖ = %.2e",
                   d_pred, d_band, r_pred, r_band, ok_c ? "hold" : "VIOLATED", norm(θr .- θp)/norm(θp)))
end

# The constrained RLS θ (rls_npt/01): the same objective and flat directions as θ_mean plus a cutting-plane set
# over six volumes.  It is judged at the converged tier's 1e-2 — on its weighted training predictions (well
# conditioned), not on raw θ (not).
# Its per-volume phonons are in the stage's log; its consequence, the 200 K NPT, is judged on its own.
function check_predictions(label, published, reproduced; tol=1e-2)
    (isfile(published) && isfile(reproduced)) || return Check(label, :converged, :missing, "not produced yet: $reproduced")
    θp, θr = vec(readdlm(published, ',')), vec(readdlm(reproduced, ','))
    result = load_model(:Al, 12, 4, 6, 2; dataset_name="")
    Aw = Diagonal(result.W)*result.A; Yw = result.W.*result.Y; res = norm(Aw*θp .- Yw)
    rmse(θ) = sqrt(mean((Aw*θ .- Yw).^2)); d = norm(Aw*(θp .- θr)) / res
    Check(label, :converged, d <= tol ? :ok : :fail,
          @sprintf("training predictions differ by %.2e of the published fit's residual (tol %.0e); weighted RMSE %.5f→%.5f; raw ‖Δθ‖/‖θ‖ = %.2e",
                   d, tol, rmse(θp), rmse(θr), norm(θr .- θp)/norm(θp)))
end
