# test/fixtures.jl — shared inputs: the published Al_12 model and its a_eq, small cells, seeded synthetic problems.
# Anything expensive is built once, on first use (`memo`).

using LinearAlgebra, Statistics, Random, DelimitedFiles, Printf, Serialization
using StaticArrays, Unitful, ForwardDiff
using AtomsBase, AtomsBuilder, ExtXYZ
using ACEpotentials, ACEWorkflow
using ACEpotentials: potential_energy
import AtomsCalculatorsUtilities.SitePotentials: hessian

const AL12_DIR = joinpath(REPO, "models", "Al_12_4_6A_2_")
const TEST_XYZ = joinpath(REPO, "data", "Al", "manual_df_test_Al.xyz")
const TRAIN_XYZ = joinpath(REPO, "data", "Al", "subset_20_percent.extxyz")

const _MEMO = Dict{Symbol,Any}()
memo(f, key) = get!(f, _MEMO, key)

"The published Al_12 model with its fitted θ, from the json and lin_params.csv (load_model would read the 259 MB A.csv)."
function al12()
    memo(:al12) do
        model, _ = ACEpotentials.load_model(joinpath(AL12_DIR, "Al_12_4_6A_2.json"))
        θ = vec(readdlm(joinpath(AL12_DIR, "lin_params.csv"), ','))
        ACEpotentials.Models.set_linear_parameters!(model, θ)
        (; model, θ)
    end
end
"A fresh copy of the Al_12 model at its fitted θ (for tests that set other parameters)."
al12_model() = (m = deepcopy(al12().model); ACEpotentials.Models.set_linear_parameters!(m, al12().θ); m)
a_eq() = memo(:a_eq) do; ACEWorkflow.relax_lattice_constant(al12_model(), :Al) end

"A stand-in for load_model's result whose `dir` is a temp directory, so Hessian caches never touch models/."
fake_result(dir; θ=al12().θ) = (; lin_params=θ, name="Al_12_4_6A_2", dir)

energy(sys, model) = ustrip(u"eV", potential_energy(sys, model))
basis_energy(sys, model) = ustrip.(u"eV", ACEpotentials.Models.potential_energy_basis(sys, model))

"FCC Al with the deformation gradient F = I + ε applied to cell and positions, built without src/'s helpers."
function strained_fcc(ε3::AbstractMatrix, a)
    s = bulk(:Al; a=a * u"Å")
    F = I + ε3
    cv = [F * ustrip.(u"Å", v) for v in cell_vectors(s)]
    atoms = [Atom(:Al, F * ustrip.(u"Å", position(s, i)) * u"Å") for i in 1:length(s)]
    periodic_system(atoms, Tuple(v * u"Å" for v in cv))
end
voigt3(e) = [e[1] e[6]/2 e[5]/2; e[6]/2 e[2] e[4]/2; e[5]/2 e[4]/2 e[3]]

"Seeded least-squares problem: X (n×p, full rank), Y = Xβ + noise."
function ls_problem(; n=200, p=8, seed=1)
    rng = MersenneTwister(seed)
    X = randn(rng, n, p) .* (1 .+ 3 .* rand(rng, n))      # rows of different leverage
    β = randn(rng, p)
    (; X, Y=X * β .+ 0.1 .* randn(rng, n), β)
end

"A small frame list from the test set, written to a temp file (committee prediction tests)."
function small_test_xyz(dir; n=6)
    frames = ExtXYZ.load(TEST_XYZ)[1:n]
    f = joinpath(dir, "small_test.xyz"); ExtXYZ.save(f, frames); f
end

"Native (dotted) force-constant data for an N_cell³ cubic supercell of Al_12 at a."
fc_native(a, N_cell) = memo(Symbol("fc_", a, "_", N_cell)) do
    prim, sup = ACEWorkflow.bulk_prim_super(:Al; a=a, N_cell=N_cell)
    ACEWorkflow.precompute_force_constants(prim, sup, al12_model())
end

# lib/ is loaded whole (it has top-level constants) into its own module; parity_calibration.jl expects bandpath.jl
const LIB = Core.eval(Main, :(module LibUnderTest
    include($(joinpath(REPO, "lib", "bandpath.jl")))
    include($(joinpath(REPO, "lib", "parity_calibration.jl")))
end))

"Al_12's band path on a 2×2×2 cell, 2 q-points per segment (the identities hold for any cell)."
bp2() = memo(:bp2) do
    mktempdir() do dir
        LIB.bandpath_Dk(fake_result(dir), al12_model(), :Al, a_eq(), 2; N_per_seg=2)
    end
end

"min ½xᵀHx + qᵀx  s.t.  E x = d, by a dense KKT solve (the reference for every QP helper)."
function kkt(H, q, E, d)
    n, m = size(H, 1), size(E, 1)
    (Matrix([H E'; E zeros(m, m)]) \ [-q; d])[1:n]
end

"An N_cell³ cubic Al supercell at a with every atom displaced by a seeded random vector of rms σ (Å)."
function rattled_super(a, N_cell; σ=0.03, seed=7)
    _, sup = ACEWorkflow.bulk_prim_super(:Al; a=a, N_cell=N_cell)
    rng = MersenneTwister(seed)
    atoms = [Atom(:Al, (ustrip.(u"Å", position(sup, i)) .+ σ .* randn(rng, 3)) * u"Å") for i in 1:length(sup)]
    periodic_system(atoms, Tuple(cell_vectors(sup)))
end
θ_perturbed() = al12().θ .* (1 .+ 0.1 .* randn(MersenneTwister(8), length(al12().θ)))
contract(H, θ) = (n = size(H, 1); reshape(reshape(H, n * n, :) * θ, n, n))
native_hessian(sys, θ) = (m = al12_model(); ACEpotentials.Models.set_linear_parameters!(m, θ); ustrip.(hessian(sys, m)))

# A hand-built band path: one parameter, three q-points (the first at Γ), diagonal D so the answers are known.
toy_bp() = (; Bq=[reshape(ComplexF64.(vec(Diagonal(d))), 9, 1) for d in ([-9.0, 0.0, 1.0], [-4.0, 1.0, 9.0], [0.25, 4.0, 16.0])],
            Np=1, qnorm=[0.0, 0.5, 1.0])
