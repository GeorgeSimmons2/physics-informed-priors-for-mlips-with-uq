# The strain and lattice machinery the stages use: the reference FCC system, the strained-cell
# design (∂²B/∂ε² → C_ij rows), and the variable-cell relaxation that gives a_eq.
using StaticArrays, LinearAlgebra, AtomsBuilder
using ForwardDiff
using Unitful
using AtomsBase
using GeometryOptimization

function Voigt_strain_to_3x3(strain_vector::SVector{6,T}) where {T}
    s1, s2, s3, s4, s5, s6 = strain_vector
    return @SMatrix [
        s1       0.5*s6   0.5*s5
        0.5*s6   s2       0.5*s4
        0.5*s5   0.5*s4   s3
    ]
end


function lattice_matrix(cell_vectors)
    a, b, c = cell_vectors
    return SMatrix{3,3,eltype(a)}(hcat(a, b, c))
end

lattice_tuple(L) = (L[:,1], L[:,2], L[:,3])

function reference_system(element; a=nothing)
    if isnothing(a)
        return bulk(element)
    else
        a_u = a isa Unitful.Quantity ? a : a * u"Å"
        return bulk(element; a=a_u, T=typeof(ustrip(a_u)))
    end
end

function fractional_positions_unitless(sys)
    L0 = SMatrix{3,3,Float64}(ustrip.(lattice_matrix(sys.cell.cell_vectors)))
    Linv = inv(L0)

    fracs = Vector{SVector{3,Float64}}(undef, length(sys))
    for i in 1:length(sys)
        r = SVector{3,Float64}(ustrip.(sys[i].position))
        fracs[i] = Linv * r
    end
    return fracs
end

function rebuild_periodic_system_unitful(sys; Lnew_unitless, fracs)
    # Attach units to lattice (Å)
    Lnew = Lnew_unitless .* u"Å"

    atoms_new = Vector{Atom}(undef, length(sys))
    for i in 1:length(sys)
        at = sys[i]
        rnew = Lnew * fracs[i]                      # rnew is SVector{3,Quantity}
        atoms_new[i] = Atom(at.species, collect(rnew), missing)
    end

    return periodic_system(atoms_new, lattice_tuple(Lnew))
end

function strained_cell_design(ε::SVector{6,T}; model, element, a=nothing) where {T}
    sys0 = reference_system(element; a=a)

    ϵ = Voigt_strain_to_3x3(ε)
    F = ϵ + one(ϵ)

    # reference lattice, *unitless numeric in Å*
    L0 = SMatrix{3,3,T}(ustrip.(lattice_matrix(sys0.cell.cell_vectors)))

    # strained lattice, still unitless
    L1 = F * L0

    # fractional coords (unitless)
    fracs = fractional_positions_unitless(sys0)

    # rebuild with unitful lattice + unitful positions
    sys1 = rebuild_periodic_system_unitful(sys0; Lnew_unitless=L1, fracs=fracs)

    return ACEpotentials.Models.potential_energy_basis(sys1, model)
end

# Variant that expresses the lattice-constant change as an isotropic pre-scale δ

function elastic_hessian_basis(model; element=:Al, a=nothing, ε0=nothing)
    ε_ref = isnothing(ε0) ? @SVector(zeros(6)) : ε0
    f(ε) = strained_cell_design(ε; model=model, element=element, a=a)
    nparams = length(f(ε_ref))
    H_6x6xd = Array{Float64}(undef, 6, 6, nparams)
    for k in 1:nparams
        g(ε) = ustrip.(f(ε)[k])
        H_6x6xd[:, :, k] = ForwardDiff.hessian(g, ε_ref)
    end
    return H_6x6xd
end

"""
    relax_lattice_constant(model, element) -> Float64

Equilibrium lattice constant (Å) of `element`'s ground-state structure under `model`, from a
variable-cell energy minimisation (GeometryOptimization.minimize_energy!) of the primitive cell.
"""
function relax_lattice_constant(model, element::Symbol)
    sys = reference_system(element)
    res = minimize_energy!(sys, model; variablecell=true)
    optsys = res.system
    return ustrip(u"Å", optsys.cell.cell_vectors[1][2] * 2)  # plain Float64 in Å
end
