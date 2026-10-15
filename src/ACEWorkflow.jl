module ACEWorkflow

# The pieces of the ACE workflow the three results use: the POPS sampler, the strain and lattice
# machinery, phonons from supercell force constants, and model fitting / loading.

include("POPS/POPSRegression.jl")
include("Elasticity/Elasticity.jl")
include("Phonons/Phonons.jl")
include("Models/Models.jl")

using .POPSRegression
using .Elasticity
using .Phonons
using .Models

export corrections, hypercube, sample_hypercube, rejection_sample_hypercube
export elastic_hessian_basis, relax_lattice_constant
export FREQ_THz, precompute_force_constants, dynamical_matrix_from_fc, fcc_band_path, bulk_prim_super
export load_model, build_model, model_name, model_dir

end
