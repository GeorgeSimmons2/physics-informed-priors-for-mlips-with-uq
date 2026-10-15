module Phonons

using LinearAlgebra, StaticArrays
using Unitful
using AtomsCalculatorsUtilities.SitePotentials: hessian
using AtomsBuilder: bulk

include("phonon_bands.jl")

export FREQ_THz, precompute_force_constants, dynamical_matrix_from_fc, fcc_band_path, bulk_prim_super

end
