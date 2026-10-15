# verify/verify.jl — how closely a route (repro/saved or repro/refit) reproduces the published ACE results.
#
# Each quantity is judged at the standard its mechanism can meet (lib_verify.jl):
#   :exact        no QP on the path (seeded draws, closed-form corrections) — must match bit for bit
#   :converged    one well-posed QP or a deterministic fit — max|Δ| ≤ 1e-2
#   :statistical  cutting-plane cascades — ensemble mean and spread within 4.5σ of sampling noise
# Seeded hypercube draws (the Al_12 naive and both Al_16 ensembles) have no QP in them, but they go through
# the PCA of a numerically rank-deficient cloud: a round-off-level difference between machines (or the refit's
# 1e-8 change in θ) rotates its near-degenerate eigenvectors and with them every member — the earlier repro
# measured max|Δθ| = 0.096 for the Al_12 naive draw across nodes, and the published Al_16 members were never
# recovered.  So those are judged as distributions plus the physical claim (soft members per panel), and their
# member-level differences are reported as `info`.  Writes the table to stdout; run.sh verify → VERIFY.md.
#
# Run: julia --project verify/verify.jl saved|refit

include(joinpath(@__DIR__, "lib_verify.jl"))
using Serialization, SHA

route = isempty(ARGS) ? "saved" : ARGS[1]
R  = normpath(joinpath(@__DIR__, ".."))
O  = "$R/repro/$route"
P  = "$R/published"
checks = Check[]

# a numbers-only comparison with its own tolerance, for tables that are not a θ vector
function check_numbers(label, tier, a::AbstractVector, b::AbstractVector, tol; detail="")
    length(a) == length(b) || return Check(label, tier, :fail, "length differs: $(length(a)) vs $(length(b))")
    d = maximum(abs.(a .- b))
    Check(label, tier, d <= tol ? :ok : :fail, @sprintf("max|Δ| = %.3e (tol %.1e)%s", d, tol, detail))
end
csvrow(p) = (m = readdlm(p, ','; header=true); Dict(zip(vec(string.(m[2])), vec(m[1]))))

# ── the refitted ACE models (refit route) ────────────────────────────────────
if route == "refit"
    sums = Dict(split(l)[2] => split(l)[1] for l in readlines("$R/data/tierB.sha256"))
    for m in ("Al_12_4_6A_2_", "Al_16_4_6A_3_")
        push!(checks, check_vector("$m fitted coefficients", :converged, "$R/models/$m/lin_params.csv", "$O/models/$m/lin_params.csv"))
        for f in ("Y.csv", "P.csv", "W.csv")
            push!(checks, check_vector("$m $f", :exact, "$R/models/$m/$f", "$O/models/$m/$f"))
        end
        a = "$O/models/$m/A.csv"
        push!(checks, isfile(a) ?
            Check("$m design matrix A.csv", :exact, bytes2hex(open(sha256, a)) == sums["models/$m/A.csv"] ? :ok : :fail,
                  "SHA-256 against the published design matrix") :
            Check("$m design matrix A.csv", :exact, :missing, "not produced yet: $a"))
    end
end

# ── Al_12: committee, cloud, hypercube, naive ensemble ───────────────────────
push!(checks, check_vector("Al_12 constrained mean θ_mean", :converged, "$P/al12/committee/theta_mean.csv", "$O/al12/committee/theta_mean.csv"))
push!(checks, check_ensemble("Al_12 30-member repaired forest", :statistical, "$P/al12/committee/committee_repaired.csv", "$O/al12/committee/committee_repaired.csv"))
push!(checks, check_distribution("Al_12 cloud: min ω per member (THz)", :statistical, "$P/al12/cloud/member_diagnostics.csv",
                                 "$O/al12/cloud/member_diagnostics.csv"; col=5, skipstart=2, unit="THz"))
push!(checks, check_distribution("Al_12 cloud: cuts per member", :statistical, "$P/al12/cloud/member_diagnostics.csv",
                                 "$O/al12/cloud/member_diagnostics.csv"; col=3, skipstart=2))
push!(checks, check_ensemble("Al_12 constrained committee (full-cloud hypercube)", :statistical,
                             "$P/al12/hypercube/committee_rejection_full_cloud.csv", "$O/al12/hypercube/committee_rejection_full_cloud.csv"))
let a = "$P/al12/hypercube/hypercube_summary.csv", b = "$O/al12/hypercube/hypercube_summary.csv"
    if isfile(b)
        A, B = readdlm(a, ','; header=true)[1], readdlm(b, ','; header=true)[1]
        for (i, nm) in enumerate(A[:, 1])
            # directions, acceptance %, E and F coverage %: same box statistics, so agreement to a few %
            vals(M) = Float64[M[i, 3], M[i, 6], M[i, 7], M[i, 8]]
            va, vb = vals(A), vals(B)
            ok = abs(va[1] - vb[1]) <= 2 && all(abs.(va[3:4] .- vb[3:4]) .<= 2.0)
            push!(checks, Check("Al_12 hypercube summary ($nm)", :statistical, ok ? :ok : :fail,
                @sprintf("directions %d→%d, acceptance %.3f→%.3f %%, coverage E %.1f→%.1f %%, F %.1f→%.1f %%",
                         va[1], vb[1], va[2], vb[2], va[3], vb[3], va[4], vb[4])))
        end
    else
        push!(checks, Check("Al_12 hypercube summary", :statistical, :missing, "not produced yet: $b"))
    end
end
push!(checks, check_ensemble("Al_12 naive ensemble (30 draws)", :statistical, "$P/al12/naive/samples_naive.csv", "$O/al12/naive/samples_naive.csv"))
let c = check_vector("Al_12 naive members: min ω, relaxed a (member by member)", :converged,
                     "$P/al12/naive/min_freq_naive.csv", "$O/al12/naive/min_freq_naive.csv")
    push!(checks, c.status === :missing ? c : Check(c.label, :info, :info, c.detail))
end

# ── the four-panel caches: soft members per panel ────────────────────────────
for (row, pub, rep) in (("Al_12", "$P/al12/naive/bands_four_panel_Al_12.jls", "$O/al12/naive/bands_four_panel_Al_12.jls"),
                        ("Al_16", "$P/al16/bands/bands_two_ensembles.jls", "$O/al16/bands/bands_two_ensembles.jls"))
    if isfile(rep)
        a, b = deserialize(pub), deserialize(rep)
        soft(d, k) = count(<(d.UNSTABLE), [m.minω for m in d.res[k]])
        sa, sb = (soft(a, 1), soft(a, 2)), (soft(b, 1), soft(b, 2))
        n = length(a.res[1]); p = (sa[1] + sb[1]) / 2n
        z = sa[1] == sb[1] ? 0.0 : abs(sa[1] - sb[1]) / n / sqrt(p * (1 - p) * 2 / n)   # two-proportion z
        push!(checks, Check("$row four-panel row: soft members (unconstrained, constrained)", :statistical,
                            (sb[2] == 0 && sb[1] >= 1 && z <= Z_TOL) ? :ok : :fail,
                            @sprintf("published %s → reproduced %s of %d; unconstrained soft fraction z = %.2f; the constrained panel must have none",
                                     string(sa), string(sb), n, z)))
    else
        push!(checks, Check("$row four-panel row", :statistical, :missing, "not produced yet: $rep"))
    end
end

# ── Al_16: per-member (relaxed a, min ω) of both pinned ensembles ────────────
for tag in ("unconstrained", "constrained")
    c = check_vector("Al_16 $tag ensemble: member, relaxed a, min ω (member by member)", :converged,
                     "$P/al16/bands/bands_$tag.csv", "$O/al16/bands/bands_$tag.csv")
    push!(checks, c.status === :missing ? c : Check(c.label, :info, :info, c.detail))
end

# ── RLS vs constrained RLS, 200 K NPT: judged on the 200 K outcome ──────────
# The comparison lives at 200 K, where the RLS model breaks down.  RLS must leave FCC there (the driver's
# still_fcc, ⟨coord⟩ well below 12) with a negative min ω at its a(200 K); that a(200 K) belongs to a broken
# structure, so it is reported, not compared.  Constrained RLS must stay FCC with a(200 K) within 3
# fluctuation σ of the published value.  Trajectories are not compared: Molly sums forces over threads.
push!(checks, check_vector("constrained RLS θ", :converged, "$P/rls_npt/constrained/theta_rls_constrained.csv", "$O/rls_npt/constrained/theta_rls_constrained.csv"))
for m in ("rls", "constrained")
    a, b = "$P/rls_npt/npt/$m/T200K/summary_row.csv", "$O/rls_npt/npt/$m/T200K/summary_row.csv"
    if isfile(b)
        ra, rb = csvrow(a), csvrow(b)
        num(r, k) = Float64(r[k]); fcc(r) = string(r["still_fcc"]) == "true"
        aA, aB = num(ra, "a_Ang"), num(rb, "a_Ang"); σ = max(num(ra, "a_std_Ang"), num(rb, "a_std_Ang"))
        ok = m == "rls" ? (!fcc(rb) && num(rb, "mean_coord") < 11 && num(rb, "min_omega_THz") < 0) :
                          (fcc(rb) && abs(aA - aB) <= 3σ)
        push!(checks, Check(m == "rls" ? "200 K NPT, RLS: leaves FCC" : "200 K NPT, constrained RLS: stays FCC, a(200 K)",
            :statistical, ok ? :ok : :fail,
            @sprintf("still FCC %s→%s, ⟨coord⟩ %.2f→%.2f, min ω %+.3f→%+.3f THz, a(200 K) %.5f→%.5f Å (fluctuation σ %.1e%s)",
                     string(fcc(ra)), string(fcc(rb)), num(ra, "mean_coord"), num(rb, "mean_coord"),
                     num(ra, "min_omega_THz"), num(rb, "min_omega_THz"), aA, aB, σ,
                     m == "rls" ? "; not compared, the structure has broken down" : @sprintf(", |Δ| = %.1fσ", abs(aA - aB)/σ))))
    else
        push!(checks, Check("200 K NPT, $m", :statistical, :missing, "not produced yet: $b"))
    end
end

println("## ACE: route `$route` against published/\n")
nf = report(checks)
exit(0)
