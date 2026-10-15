# verify/bp_sensitivity.jl — the same cloud members scored on the saved- and refit-route Hessians (a_eq 3e-9 Å apart):
# shows the near-Γ minimum frequency moves by ~0.04 THz, which is why verify reports the refit cloud diagnostics as info.
# Run after both routes: julia --project -t 8 verify/bp_sensitivity.jl   (from the repository root)
include(joinpath(@__DIR__, "..", "lib", "bandpath.jl"))
using Random
s = deserialize("repro/saved/al12/cloud/committee_full_cloud.jls")
bps = Dict()
for (lab, root) in (("saved", "models"), ("refit", "repro/refit/models"))
    ENV["ACE_MODELS"] = root
    result = load_model(:Al, 12, 4, 6, 2; dataset_name=""); model = result.model
    a_eq = ACEWorkflow.relax_lattice_constant(model, :Al)
    bps[lab] = (bandpath_Dk(result, model, :Al, a_eq, 4; N_per_seg=[20, 20, 20, 20, 60]), a_eq)
end
Θ = size(s.Θ, 1) == 91 ? s.Θ : permutedims(s.Θ)
Random.seed!(0); idx = rand(1:size(Θ, 2), 200)
w(θ, k) = min_freq_stable(θ, bps[k][1])
d = [abs(w(Θ[:,i], "saved") - w(Θ[:,i], "refit")) for i in idx]
@printf("a_eq saved %.12f, refit %.12f\n", bps["saved"][2], bps["refit"][2])
@printf("same θ, two band paths: |Δ min ω| median %.4f, 90%% %.4f, max %.4f THz (200 members)\n", median(d), quantile(d, 0.9), maximum(d))
# where along the path is the minimum, and how close to the Γ cut?
bp = bps["saved"][1]; θ = Θ[:, idx[1]]
iq = argmin([bp.qnorm[k] < 5e-2 ? Inf : minimum(eigvals(Hermitian(reshape(bp.Bq[k]*θ, 3, 3)))) for k in eachindex(bp.Bq)])
@printf("member %d: minimum at q-point %d, |q| = %.4f (Γ cut 0.05)\n", idx[1], iq, bp.qnorm[iq])
