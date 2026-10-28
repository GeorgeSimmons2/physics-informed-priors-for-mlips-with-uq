# src/Elasticity/get_hessian.jl — strain and lattice machinery, the strained-cell design and its strain Hessian.

const EL_SRC = "src/Elasticity/get_hessian.jl"
const EL = ACEWorkflow.Elasticity

@testset "Voigt_strain_to_3x3" begin
    tested(EL_SRC, :Voigt_strain_to_3x3)
    E = EL.Voigt_strain_to_3x3(SVector(1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
    @test E isa SMatrix{3,3}
    @test E == [1 3 2.5; 3 2 2; 2.5 2 3]                      # engineering shears: ε_ij = γ/2
    @test E == E'
    @test EL.Voigt_strain_to_3x3(SVector{6}(zeros(6))) == zeros(3, 3)
end

@testset "lattice_matrix, lattice_tuple" begin
    tested(EL_SRC, :lattice_matrix, :lattice_tuple)
    cv = (SVector(1.0, 2, 3), SVector(4.0, 5, 6), SVector(7.0, 8, 10))
    L = EL.lattice_matrix(cv)
    @test L isa SMatrix{3,3}
    @test L == hcat(cv...)                                    # columns are the cell vectors
    @test EL.lattice_tuple(L) == cv
    s = bulk(:Al)
    @test EL.lattice_tuple(EL.lattice_matrix(cell_vectors(s))) == Tuple(cell_vectors(s))
end

@testset "reference_system" begin
    tested(EL_SRC, :reference_system)
    @test cell_vectors(EL.reference_system(:Al)) == cell_vectors(bulk(:Al))
    for a in (4.0, 4.0u"Å")
        s = EL.reference_system(:Al; a=a)
        @test length(s) == 1
        @test all(norm(ustrip.(u"Å", v)) ≈ 4.0 / √2 for v in cell_vectors(s))   # primitive fcc: |a_i| = a/√2
    end
    # the element type follows a, so the system can carry ForwardDiff duals (lattice_basis differentiates through it)
    d = ForwardDiff.Dual(4.0, 1.0)
    @test eltype(ustrip.(cell_vectors(EL.reference_system(:Al; a=d))[1])) <: ForwardDiff.Dual
end

@testset "fractional_positions_unitless, rebuild_periodic_system_unitful" begin
    tested(EL_SRC, :fractional_positions_unitless, :rebuild_periodic_system_unitful)
    s = bulk(:Al; a=4.0u"Å", cubic=true)
    f = EL.fractional_positions_unitless(s)
    @test sort(f; by=Tuple) ≈ sort([SVector(0.0, 0, 0), SVector(0.0, 0.5, 0.5), SVector(0.5, 0, 0.5), SVector(0.5, 0.5, 0)]; by=Tuple)
    L = ustrip.(EL.lattice_matrix(cell_vectors(s)))
    s2 = EL.rebuild_periodic_system_unitful(s; Lnew_unitless=L, fracs=f)
    @test all(ustrip.(position(s2, i)) ≈ ustrip.(position(s, i)) for i in 1:length(s))
    @test all(ustrip.(cell_vectors(s2)[k]) ≈ ustrip.(cell_vectors(s)[k]) for k in 1:3)
    F = [1.01 0.02 0; 0 0.99 0; 0 0 1.03]
    s3 = EL.rebuild_periodic_system_unitful(s; Lnew_unitless=F * L, fracs=f)
    @test all(ustrip.(position(s3, i)) ≈ F * ustrip.(position(s, i)) for i in 1:length(s))
    @test unit(position(s3, 1)[1]) == u"Å"
    @test EL.fractional_positions_unitless(s3) ≈ f
end

@testset "strained_cell_design" begin
    tested(EL_SRC, :strained_cell_design)
    (; model, θ) = al12()
    a = a_eq()
    b0 = ustrip.(u"eV", EL.strained_cell_design(SVector{6}(zeros(6)); model, element=:Al, a))
    @test length(b0) == length(θ)
    @test dot(b0, θ) ≈ energy(bulk(:Al; a=a * u"Å"), model) rtol = 1e-12
    e = SVector(0.01, -0.005, 0.002, 0.004, -0.003, 0.006)
    b = ustrip.(u"eV", EL.strained_cell_design(e; model, element=:Al, a))
    @test dot(b, θ) ≈ energy(strained_fcc(voigt3(e), a), model) rtol = 1e-10
end

@testset "elastic_hessian_basis" begin
    tested(EL_SRC, :elastic_hessian_basis)
    (; model, θ) = al12()
    a = a_eq()
    H = ACEWorkflow.elastic_hessian_basis(model; element=:Al, a)
    @test size(H) == (6, 6, length(θ))
    C = reshape(reshape(H, 36, :) * θ, 6, 6)                 # θ-contracted: ∂²E/∂ε_I∂ε_J (eV)
    @test C ≈ C'
    # against a central finite difference of the energy of independently strained cells
    E(e) = energy(strained_fcc(voigt3(e), a), model)
    h = 1e-3
    unit6(i) = (v = zeros(6); v[i] = 1; v)
    fd(i, j) = (E(h * (unit6(i) + unit6(j))) - E(h * (unit6(i) - unit6(j))) -
                E(h * (-unit6(i) + unit6(j))) + E(-h * (unit6(i) + unit6(j)))) / (4h^2)
    for (i, j) in ((1, 1), (1, 2), (4, 4), (1, 4))
        @test C[i, j] ≈ fd(i, j) atol = 2e-3 * maximum(abs, C)
    end
    # cubic symmetry
    @test C[1, 1] ≈ C[2, 2] ≈ C[3, 3]
    @test C[1, 2] ≈ C[1, 3] ≈ C[2, 3]
    @test C[4, 4] ≈ C[5, 5] ≈ C[6, 6]
    @test maximum(abs, C[1:3, 4:6]) < 1e-8 * C[1, 1]
    # in GPa: Born-stable and of a metal's magnitude (this model: 141/60/14 GPa; its C44 is half aluminium's 31)
    V = abs(det(ustrip.(EL.lattice_matrix(cell_vectors(bulk(:Al; a=a * u"Å"))))))
    C11, C12, C44 = (C[1, 1], C[1, 2], C[4, 4]) .* (160.21766 / V)
    @test C11 - C12 > 0 && C11 + 2C12 > 0 && C44 > 0
    @test 10 < C11 < 300 && 1 < C44 < 100
    # ε0 shifts the expansion point
    H1 = ACEWorkflow.elastic_hessian_basis(model; element=:Al, a, ε0=SVector(0.01, 0, 0, 0, 0, 0))
    @test !(H1 ≈ H)
end

@testset "relax_lattice_constant" begin
    tested(EL_SRC, :relax_lattice_constant)
    (; model) = al12()
    a = a_eq()
    @test a isa Float64
    @test 3.95 < a < 4.15
    E(x) = energy(bulk(:Al; a=x * u"Å"), model)
    @test E(a) < E(a - 0.01) && E(a) < E(a + 0.01)            # an energy minimum
    @test abs((E(a + 1e-4) - E(a - 1e-4)) / 2e-4) < 1e-3      # zero pressure (eV/Å per atom)
end
