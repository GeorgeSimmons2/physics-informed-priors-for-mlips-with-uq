# al12/03_replot_parity.jl — redraw al12/03's test-set parity and calibration figures from the predictions it
# saved (no sampling, no model evaluation), e.g. after a styling change.  run.sh figures calls it.
#
#   OUT=<route root> julia --project al12/03_replot_parity.jl        Env: OUTDIR ($OUT/al12/hypercube), FIGW (510)

include(joinpath(@__DIR__, "..", "lib", "bandpath.jl"))
include(joinpath(@__DIR__, "..", "lib", "parity_calibration.jl"))

ROUT   = get(ENV, "OUT", normpath(joinpath(@__DIR__, "..", "repro", "saved")))
outdir = get(ENV, "OUTDIR", "$ROUT/al12/hypercube")
FIGW   = parse(Float64, get(ENV, "FIGW", "510"))

d = deserialize("$outdir/parity_calibration_predictions.jls")
for (pr, ttl, tag) in ((d.pr_f, d.label_full, "fullcloud"), (d.pr_3, d.label_30, "forest30"))
    parity_figure(pr, ttl, tag, outdir; FIGW=FIGW)
    calibration_figure(pr, ttl, tag, outdir; FIGW=FIGW)
end
