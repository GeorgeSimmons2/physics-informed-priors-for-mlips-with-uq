# test/scripts/common.jl — globals shared by the Al script tests, built the way the scripts build them, with a
# small synthetic design in place of the 146,958-row one.

using OSQP, SparseArrays

"The Al_12 scripts' globals: the real model, θ, a_eq, Born and a_eq rows; a 400-row design consistent with θ."
al12_globals() = memo(:al12g) do
    (; θ) = al12(); model = al12_model(); a = a_eq(); n = length(θ)
    sys0 = ACEWorkflow.Elasticity.reference_system(:Al; a=a)
    L0 = ustrip.(ACEWorkflow.Elasticity.lattice_matrix(sys0.cell.cell_vectors))
    eV_to_GPa = 160.2176621 / abs(det(L0))
    H_el = ACEWorkflow.elastic_hessian_basis(model; element=:Al, a=a)
    c11_0 = reshape(H_el, 36, n)[1, :]; c12_0 = reshape(H_el, 36, n)[7, :]; c44_0 = reshape(H_el, 36, n)[22, :]
    lb(x) = basis_energy(ACEWorkflow.Elasticity.reference_system(:Al; a=x), model)
    b_prime = ForwardDiff.derivative(lb, a)
    b_double_prime = ForwardDiff.derivative(x -> ForwardDiff.derivative(lb, x), a)
    born_rows = vcat(c44_0', (c11_0 .- c12_0)', (c11_0 .+ 2 .* c12_0)', b_double_prime')
    born_lower = [0.1, 1.0, 0.1, 1e-9]
    P = readdlm(joinpath(AL12_DIR, "P.csv"), ',')
    rng = MersenneTwister(12); M = 400
    Ap = randn(rng, M, n); Yw = Ap * (P * θ) .+ 1e-3 .* randn(rng, M); λ = 1.0 / M
    # the QP's unconstrained minimum is the fitted model (x = Pθ), as in the real problem: with 400 random rows
    # instead of 146,958 the prior would otherwise shrink θ far from any physical model
    Hx = Ap' * Ap .+ λ .* (P' * P); qx = -(Hx * (P * θ))
    (; model, θ, a, n, eV_to_GPa, H_el, c11_0, c12_0, c44_0, b_prime, b_double_prime, born_rows, born_lower,
       P, Ap, Yw, λ, M, Hx, qx)
end

"The QP globals al12/01's constrain_member and mean_fit read."
function al12_qp_globals()
    g = al12_globals()
    (; born_rows=g.born_rows, born_lower=g.born_lower, Ap=g.Ap, Yw=g.Yw, b_prime=g.b_prime, P=g.P,
       Hqp=sparse(g.Hx), qqp=g.qx, osqp=OSQP.Model(), RHO_INTERVAL=25, n_params=g.n)
end

"θ from the dense KKT solve of the Al_12 QP with equality rows E_θ (θ-space) = d."
kkt_θ(g, Eθ, d) = g.P \ kkt(g.Hx, g.qx, Eθ / g.P, d)          # θ-space rows (an x-space row a is aᵀP)
relerr(a, b) = norm(a - b) / norm(b)

const PLOT_GLOBALS = (; FIGW=300.0, TITLE=13, LAB=12, TICK=11, RED=LIB.RGBf(0.8, 0.15, 0.15),
                        GRY=LIB.RGBAf(0.45, 0.45, 0.45, 0.3), BLU=LIB.RGBf(0.0, 0.447, 0.698))
"Colours of the Lines plots in an axis, as RGBA (Makie stores them converted)."
tocol(c) = LIB.CairoMakie.Makie.to_color(c)
line_colors(ax) = [tocol(p.color[]) for p in ax.scene.plots if p isa LIB.Lines]

# lattice_basis is defined in seven scripts; each copy must give the energy and its a-derivatives
const LATTICE_BASIS_SCRIPTS = ["al12/01_committee.jl", "al12/02_cutting_plane_cloud.jl", "al12/03_hypercube_rejection.jl",
                               "al12/04_naive_ensemble.jl", "al16/01_pinned_ensembles.jl",
                               "four_panel/build_bands_cache_Al_12.jl", "rls_npt/01_constrained_rls.jl"]
@testset "lattice_basis (7 scripts)" begin
    g = al12_globals()
    E(x) = energy(bulk(:Al; a=x * u"Å"), g.model)
    for f in LATTICE_BASIS_SCRIPTS
        tested(f, :lattice_basis)
        m = script_module(f; globals=(; element=:Al, model=g.model), only=[:lattice_basis])
        lbf = getfield(m, :lattice_basis)
        @test dot(Base.invokelatest(lbf, 4.1), g.θ) ≈ E(4.1) rtol = 1e-12
        x = g.a + 0.05                                   # off equilibrium, so the derivative is not ~0
        bp1 = Base.invokelatest(ForwardDiff.derivative, lbf, x)
        @test dot(bp1, g.θ) ≈ (E(x + 1e-5) - E(x - 1e-5)) / 2e-5 rtol = 1e-5
        @test abs(dot(Base.invokelatest(ForwardDiff.derivative, lbf, g.a), g.θ)) < 1e-3   # θ's own equilibrium (relax tolerance)
    end
end
