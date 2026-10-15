# verify/coverage_noise.jl — how much the test-set coverage of a 30-member committee varies from draw to draw.
#
# The coverage al12/03 reports is the fraction of test values inside ONE 30-member committee's min–max envelope,
# so it fluctuates with the draw.  run.sh reruns al12/03 with 600 members into al12/hypercube_noise/;
# rejection_sample_hypercube has the prefix property, so its members 1–30 are the route's own committee and
# 31–600 are 19 more independent draws from the same box.  This scores each disjoint 30-member committee exactly
# as al12/03 does and writes the mean and standard deviation per hypercube and quantity, which verify.jl uses as
# the yardstick for the coverage in hypercube_summary.csv.
#
#   julia --project verify/coverage_noise.jl <parity_calibration_predictions.jls> <test.xyz> <out.csv>

using Serialization, Statistics, Printf, DelimitedFiles, ExtXYZ

d = deserialize(ARGS[1]); K = 30
frames = ExtXYZ.load(ARGS[2])[1:d.stride:end]; nat = length.(frames)
cover(t, lo, hi) = 100 * (1 - mean((t .< lo) .| (t .> hi)))

function noise(pr, N)
    nc = length(pr.pE); @assert nc == length(frames) "$nc configurations vs $(length(frames)) frames"
    DE = reshape(pr.dE, N, nc)                                   # member × configuration (lib/parity_calibration.jl)
    offs = cumsum(vcat(0, 3 .* nat .* N)); @assert offs[end] == length(pr.dF)
    cE, cF = Float64[], Float64[]
    for k in 1:N ÷ K
        m = (k-1)*K+1:k*K
        push!(cE, cover(pr.tE, pr.pE .+ vec(minimum(DE[m, :]; dims=1)), pr.pE .+ vec(maximum(DE[m, :]; dims=1))))
        lo, hi = Float64[], Float64[]
        for c in 1:nc                                            # forces: 3·n_atoms × member, atom-major
            blk = reshape(pr.dF[offs[c]+1:offs[c+1]], 3nat[c], N)[:, m]
            append!(lo, vec(minimum(blk; dims=2))); append!(hi, vec(maximum(blk; dims=2)))
        end
        push!(cF, cover(pr.tF, pr.pF .+ lo, pr.pF .+ hi))
    end
    return cE, cF
end

open(ARGS[3], "w") do io
    println(io, "cloud,quantity,n_committees,mean_pct,sd_pct,min_pct,max_pct")
    for (name, pr) in (("full_cloud", d.pr_f), ("forest_30", d.pr_3))
        for (q, c) in zip(("E", "F"), noise(pr, d.n_members))
            @printf(io, "%s,%s,%d,%.4f,%.4f,%.4f,%.4f\n", name, q, length(c), mean(c), std(c), extrema(c)...)
            @printf("%-10s %s coverage over %d committees of %d: mean %.2f %%, sd %.2f, range [%.1f, %.1f]\n",
                    name, q, length(c), K, mean(c), std(c), extrema(c)...)
        end
    end
end
