module Elasticity

using LinearAlgebra, StaticArrays, ForwardDiff
using AtomsBase
using AtomsBuilder
using ACEpotentials
using Unitful

include("get_hessian.jl")

export elastic_hessian_basis, relax_lattice_constant, reference_system, lattice_matrix

end
