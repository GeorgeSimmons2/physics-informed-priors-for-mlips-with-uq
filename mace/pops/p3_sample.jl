# p3_sample.jl — committees from the POPS clouds of p2, with the ACE pipeline's sampler
# (src/POPS/POPSRegression.jl, included unchanged; as constrained_optimization/al12/03).
#
# CONSTRAINED: hypercube(Θ_con − θc) — PCA of the cloud's deltas, bounds at the extreme projections — then
#   rejection_sample_hypercube with the same windows as every member (C11, C12, C44 ± 0.1%, |P(a0)| ≤ 1e-3 GPa).
# UNCONSTRAINED: hypercube(Θ_unc − θ0) about MACE's trained weights and sample_hypercube (no rejection), as the
# ACE naive ensemble.
# Seed 1234 before each sampler.  rejection_sample_hypercube has the prefix property: the first 30 members of
# an N = 100 run are the members of an N = 30 run.
# Out ($OUT/mace/pops): committee_{constrained,unconstrained}.csv (N × 81, one member per row), sample.txt
# Run: julia --project mace/pops/p3_sample.jl      Env: OUT, N_MEMBERS (100), MAX_ATTEMPTS (5000000)

include(joinpath(@__DIR__, "..", "..", "src", "POPS", "POPSRegression.jl"))
using .POPSRegression
using Random, Serialization, DelimitedFiles, LinearAlgebra, Printf, Statistics

N = parse(Int, get(ENV, "N_MEMBERS", "100"))
MAX_ATTEMPTS = parse(Int, get(ENV, "MAX_ATTEMPTS", "5000000"))
OUT = joinpath(get(ENV, "OUT", joinpath(@__DIR__, "..", "..", "repro", "saved")), "mace", "pops")
c = deserialize(joinpath(OUT, "cloud.jls"))

heig, hb = hypercube(Matrix((c.Θ_con .- c.θc)'))
Random.seed!(1234)
t = @elapsed (com_con, _) = rejection_sample_hypercube(heig, hb, c.θc, c.R, (c.lo, c.hi);
                                                       number_of_committee_members=N, max_attempts=MAX_ATTEMPTS)
heig_u, hb_u = hypercube(Matrix((c.Θ_unc .- c.θ0)'))
Random.seed!(1234)
com_unc, _ = sample_hypercube(heig_u, hb_u, c.θ0; number_of_committee_members=N)

viol = [maximum(max.(c.lo .- c.R * com_con[:, i], c.R * com_con[:, i] .- c.hi, 0)) for i in 1:N]
txt = @sprintf("""
committees from the POPS clouds (p3_sample.jl), seed 1234, %d members each
constrained:   %d cloud members → hypercube with %d directions (width mean %.3g, max %.3g); rejection with the windows [%.1f s]
               all members inside the windows: %s (max violation %.1e)
unconstrained: %d cloud members about θ0 → hypercube with %d directions (width mean %.3g, max %.3g); no rejection
""", N, size(c.Θ_con, 2), size(heig, 2), mean(hb[2, :] .- hb[1, :]), maximum(hb[2, :] .- hb[1, :]), t,
     all(viol .<= 0), maximum(viol), size(c.Θ_unc, 2), size(heig_u, 2), mean(hb_u[2, :] .- hb_u[1, :]), maximum(hb_u[2, :] .- hb_u[1, :]))
print(txt)
write(joinpath(OUT, "sample.txt"), txt)
writedlm(joinpath(OUT, "committee_constrained.csv"), com_con', ',')
writedlm(joinpath(OUT, "committee_unconstrained.csv"), com_unc', ',')
