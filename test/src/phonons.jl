# src/Phonons/phonon_bands.jl — force constants from one supercell Hessian, the Bloch sum, the FCC path.

const PH_SRC = "src/Phonons/phonon_bands.jl"
const PH = ACEWorkflow.Phonons

@testset "FREQ_THz" begin
    # sqrt(eV / (Å² amu)) / 2π in THz, from Unitful's constants rather than the hand-entered ones
    @test ACEWorkflow.FREQ_THz ≈ ustrip(u"THz", sqrt(1u"eV" / (1u"Å"^2 * 1u"u")) / 2π) rtol = 1e-9
    @test ACEWorkflow.FREQ_THz ≈ 15.633 atol = 1e-3
end

@testset "_lattice_mat" begin
    tested(PH_SRC, :_lattice_mat)
    cv = cell_vectors(bulk(:Al; a=4.0u"Å"))
    L = PH._lattice_mat(cv)
    @test L isa SMatrix{3,3}
    @test all(L[:, k] == cv[k] for k in 1:3)
end

@testset "bulk_prim_super" begin
    tested(PH_SRC, :bulk_prim_super)
    prim, sup = ACEWorkflow.bulk_prim_super(:Al; a=4.0, N_cell=2)
    @test length(prim) == 1 && length(sup) == 32
    @test all(norm(ustrip.(u"Å", v)) ≈ 4.0 / √2 for v in cell_vectors(prim))
    @test ustrip.(u"Å", hcat(cell_vectors(sup)...)) ≈ 8.0 * I(3)       # a cubic 2a × 2a × 2a box
    p0, s0 = ACEWorkflow.bulk_prim_super(:Al)
    @test cell_vectors(p0) == cell_vectors(bulk(:Al)) && length(s0) == 108
end

@testset "precompute_force_constants" begin
    tested(PH_SRC, :precompute_force_constants)
    a = a_eq()
    fc = fc_native(a, 2)
    @test fc.Np == 1 && fc.Ns == 32
    @test all(fc.s2p_map .== 0)                                         # one atom class
    @test norm(ustrip.(position(ACEWorkflow.bulk_prim_super(:Al; a=a, N_cell=2)[2], fc.p2s_map[1] + 1))) < 1e-10
    @test fc.masses ≈ [26.9815] rtol = 1e-4
    @test size(fc.H) == (96, 96)
    @test fc.H ≈ fc.H'
    @test maximum(abs, sum(reshape(fc.H, 96, 3, 32); dims=3)) < 1e-8 * maximum(abs, fc.H)   # acoustic sum rule
    # the Hessian is the model's own (dotted) one
    _, sup = ACEWorkflow.bulk_prim_super(:Al; a=a, N_cell=2)
    @test fc.H ≈ ustrip.(hessian(sup, al12_model()))
end

@testset "dynamical_matrix_from_fc" begin
    tested(PH_SRC, :dynamical_matrix_from_fc)
    a = a_eq()
    fc2, fc3 = fc_native(a, 2), fc_native(a, 3)
    G = 2π / a
    q = G .* [0.21, 0.37, 0.05]
    D = ACEWorkflow.dynamical_matrix_from_fc(fc3, q)
    @test size(D) == (3, 3) && D ≈ D'                                   # Hermitian
    @test ACEWorkflow.dynamical_matrix_from_fc(fc3, -q) ≈ conj(D)       # time reversal
    Gvec = 2π * inv(transpose(Matrix(fc3.L)))[:, 2]                     # a reciprocal lattice vector of the primitive cell
    @test ACEWorkflow.dynamical_matrix_from_fc(fc3, q .+ Gvec) ≈ D      # periodic in the reciprocal lattice
    @test maximum(abs, eigvals(Hermitian(ACEWorkflow.dynamical_matrix_from_fc(fc3, zeros(3))))) < 1e-8 * norm(D)  # acoustic at Γ
    # at X, commensurate with both the 2×2×2 and 3×3×3 cells, the result cannot depend on the cell
    X = G .* [1.0, 0, 0]
    @test ACEWorkflow.dynamical_matrix_from_fc(fc2, X) ≈ ACEWorkflow.dynamical_matrix_from_fc(fc3, X) rtol = 1e-8
    # at an incommensurate q only the minimum-image wrap makes D converge with the cell.  ACE's force constants reach
    # 2 r_cut (two atoms sharing a neighbour), so no cell here isolates them: D changes ≈7% from 3×3×3 to 4×4×4 and
    # ≈2% from 4×4×4 to 5×5×5; without the wrap it does not converge (≈45%)
    D4 = ACEWorkflow.dynamical_matrix_from_fc(fc_native(a, 4), q)
    D5 = ACEWorkflow.dynamical_matrix_from_fc(fc_native(a, 5), q)
    @test norm(D4 - D5) < norm(D - D4)
    @test norm(D4 - D5) < 0.05 * norm(D5)
    # and the frequencies there are aluminium's (X: TA ≈ 5.8, LA ≈ 9.7 THz experimentally), two degenerate TA
    ω = sqrt.(max.(eigvals(Hermitian(ACEWorkflow.dynamical_matrix_from_fc(fc3, X))), 0)) .* ACEWorkflow.FREQ_THz
    @test ω[1] ≈ ω[2] rtol = 1e-6
    @test 4 < ω[1] < 7.5 && 8 < ω[3] < 11.5
end

@testset "fcc_band_path" begin
    tested(PH_SRC, :fcc_band_path)
    a = 4.0
    L = Matrix(ustrip.(u"Å", hcat(cell_vectors(bulk(:Al; a=a * u"Å"))...)))
    q, x, ticks, labels, starts = ACEWorkflow.fcc_band_path(L; N_per_seg=4)
    G = 2π / a
    @test length(q) == 5 * 4 + 1
    @test labels == ["Γ", "X", "U", "L", "Γ", "K"]
    # the high-symmetry points at their Cartesian positions (units of 2π/a)
    at(k) = q[4k + 1] ./ G
    @test at(0) ≈ [0, 0, 0] atol = 1e-12
    @test at(1) ≈ [1, 0, 0] || at(1) ≈ [0, 1, 0] || at(1) ≈ [0, 0, 1]
    @test norm(at(2)) ≈ √(1 + 1 / 8)                                    # U = (1, ¼, ¼)
    @test norm(at(3)) ≈ √3 / 2                                          # L = (½, ½, ½)
    @test norm(at(4)) ≈ 0 atol = 1e-12
    @test norm(q[end] ./ G) ≈ 3√2 / 4                                   # K = (¾, ¾, 0)
    seglen = G .* [1, √2 / 4, √6 / 4, √3 / 2, 3√2 / 4]
    @test ticks ≈ vcat(0, cumsum(seglen))
    @test issorted(x) && x[end] ≈ ticks[end]
    @test starts == [true; falses(length(q) - 1)]
    q2, _, _, _, _ = ACEWorkflow.fcc_band_path(L; N_per_seg=[1, 2, 3, 4, 5])
    @test length(q2) == 1 + 2 + 3 + 4 + 5 + 1
    @test_throws ArgumentError ACEWorkflow.fcc_band_path(L; N_per_seg=[1, 2, 3])
end
