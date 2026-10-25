# ace_models/fit_models.jl — refit the two ACE models from the DFT training set (the refit route).
#
#   Al_12_4_6A_2_   total degree 12, smoothness 4, cutoff 6 Å, order 2   (91 basis functions)
#   Al_16_4_6A_3_   total degree 16, smoothness 4, cutoff 6 Å, order 3   (684 basis functions)
#
# Both are fitted on all 7,893 frames of data/Al/manual_df_train_Al.extxyz (146,958 energy and
# force rows), weights E 30 / F 1, with ACEpotentials' smoothness prior P and the regularised
# solve θ = P⁻¹ (ÃᵀÃ + PᵀP/N)⁻¹ Ãᵀ ỹ of build_model (src/Models/Models.jl).  The fit is
# deterministic: with the pinned package versions it reproduces the published models' coefficients
# up to floating-point summation order.
#
# Out: $ACE_MODELS/<name>_/{<name>.json, A.csv, Y.csv, P.csv, W.csv, lin_params.csv}
# Run: ACE_MODELS=<dir> julia --project ace_models/fit_models.jl [12] [16]      Env: NPROCS (8)
# The design-matrix assembly runs on NPROCS Distributed workers (ACEfit.assemble uses pmap).

using Distributed
addprocs(parse(Int, get(ENV, "NPROCS", "8")); exeflags="--project=$(Base.active_project())")
@everywhere using ACEWorkflow

haskey(ENV, "ACE_MODELS") || error("set ACE_MODELS: the refit must not overwrite the published models/")
abspath(ENV["ACE_MODELS"]) == abspath(joinpath(@__DIR__, "..", "models")) &&
    error("ACE_MODELS points at the published models/ — choose another directory")
DATA = joinpath(@__DIR__, "..", "data", "Al", "manual_df_train_Al.extxyz")
which = isempty(ARGS) ? ["12", "16"] : ARGS
for w in which
    w == "12" ? build_model(:Al, 12, 4, 6.0, 2; training_xyz=DATA, dataset_name="") :
    w == "16" ? build_model(:Al, 16, 4, 6.0, 3; training_xyz=DATA, dataset_name="") :
                error("unknown model $w (12 or 16)")
end
