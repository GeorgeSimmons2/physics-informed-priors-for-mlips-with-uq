# lib/parity_calibration.jl — coverage/RMSE helpers, the parity and calibration figures, fast committee predictions.

const PC_SRC = "lib/parity_calibration.jl"
pr_fixture() = memo(() -> error("lib/bandpath.jl's committee_predictions test must run first"), :pr).pr

@testset "pc_per_atom" begin
    tested(PC_SRC, :pc_per_atom)
    @test LIB.pc_per_atom((; per_atom=true)) && !LIB.pc_per_atom((; per_atom=false))
    @test !LIB.pc_per_atom((; pE=[1.0]))                          # older cached results had no field: total energies
end

@testset "pc_cover, pc_rmse" begin
    tested(PC_SRC, :pc_cover, :pc_rmse)
    @test LIB.pc_cover([1.0, 2.0, 3.0], [0.0, 2.5, 2.0], [2.0, 3.0, 4.0]) ≈ 200 / 3     # 2 of 3 inside
    @test LIB.pc_cover([1.0], [1.0], [1.0]) == 100                                     # bounds are inclusive
    @test LIB.pc_rmse([1.0, 2.0], [1.0, 4.0]) ≈ √2
    @test LIB.pc_rmse(zeros(3), zeros(3)) == 0
end

@testset "pc_rmse_label" begin
    tested(PC_SRC, :pc_rmse_label)
    pr = (; pE=[1.0, 2.0], tE=[1.0, 2.0 + 2e-3 * √2], pF=[0.0, 0.0], tF=[0.1, 0.1])
    @test LIB.pc_rmse_label(pr, :F) == "0.1 eV/Å"
    @test LIB.pc_rmse_label(pr, :E) == "0.002 eV"                                      # total energies: eV
    @test LIB.pc_rmse_label(merge(pr, (; per_atom=true)), :E) == "2.0 meV/atom"        # per atom: meV/atom
end

@testset "pc_parity!" begin
    tested(PC_SRC, :pc_parity!)
    fig = LIB.Figure(); ax = LIB.Axis(fig[1, 1])
    t = [1.0, 2.0, 3.0]; p = [1.1, 1.9, 3.2]
    LIB.pc_parity!(ax, t, p, p .- 0.1, p .+ 0.2, LIB.PC_BLU)
    kinds = typeof.(ax.scene.plots)
    @test any(k -> k <: LIB.Lines, kinds) && any(k -> k <: LIB.Errorbars, kinds) && any(k -> k <: LIB.Scatter, kinds)
    l = only(filter(x -> x isa LIB.Lines, ax.scene.plots))
    @test [q[1] for q in l[1][]] ≈ [1.0, 3.2] && [q[2] for q in l[1][]] ≈ [1.0, 3.2]          # the y = x line over the data
end

@testset "pc_calib!" begin
    tested(PC_SRC, :pc_calib!)
    fig = LIB.Figure(); ax = LIB.Axis(fig[1, 1]; yscale=log10)
    rng = MersenneTwister(10)
    t = randn(rng, 2000); p = zeros(2000); spread = 2 .* randn(rng, 2000)
    l1, l2 = LIB.pc_calib!(ax, t, p, spread)
    for l in (l1, l2)
        pts = l[1][]
        x = [q[1] for q in pts]; y = [q[2] for q in pts]
        dx = x[2] - x[1]
        @test minimum(y) >= 1e-3 - 1e-9                           # empty bins floored for the log axis
        @test sum(y[y .> 1e-3]) * dx ≈ 1 atol = 0.02              # a normalised density (the floors add < 2%)
    end
    # both histograms share the bin edges, symmetric over ±max|value|/MAE
    xs = [q[1] for q in l1[1][]]
    mae = mean(abs.(t))
    @test xs[1] ≈ -maximum(abs.(vcat(t, spread))) / mae rtol = 1e-6
end

@testset "parity_figure, calibration_figure" begin
    tested(PC_SRC, :parity_figure, :calibration_figure)
    pr = pr_fixture()
    mktempdir() do dir
        f = LIB.parity_figure(pr, "title", "unit", dir; FIGW=300.0)
        @test f isa LIB.Figure
        @test filesize(joinpath(dir, "parity_unit.pdf")) > 1000 && filesize(joinpath(dir, "parity_unit.png")) > 1000
        @test f.scene.viewport[].widths ≈ [300, 180]               # built at its display width
        g = LIB.calibration_figure(pr, "title", "unit", dir)
        @test filesize(joinpath(dir, "calibration_unit.pdf")) > 1000 && isfile(joinpath(dir, "calibration_unit.png"))
        @test g.scene.viewport[].widths ≈ [510, round(0.58 * 510)]          # Makie rounds to whole points
        # panel titles carry the coverage and RMSE of the data
        axs = filter(x -> x isa LIB.Axis, f.content)
        @test occursin("cov $(round(LIB.pc_cover(pr.tE, pr.loE, pr.hiE), digits=1))%", axs[1].title[])
        @test occursin(LIB.pc_rmse_label(pr, :F), axs[2].title[])
    end
end

@testset "committee_predictions_fast" begin
    tested(PC_SRC, :committee_predictions_fast)
    θ = al12().θ
    mem = memo(() -> error("committee_predictions test must run first"), :pr).mem
    mktempdir() do dir
        xyz = small_test_xyz(dir; n=5)
        for pa in (false, true)
            slow = LIB.committee_predictions(al12_model(), mem, xyz; stride=1, point_params=θ, per_atom=pa)
            fast = LIB.committee_predictions_fast(al12_model(), mem, xyz; stride=1, point_params=θ, per_atom=pa)
            for k in (:pE, :tE, :loE, :hiE, :pF, :tF, :loF, :hiF, :dE)
                @test getfield(fast, k) ≈ getfield(slow, k) rtol = 1e-10
            end
            # dF holds the same deviations in another order (member-major, the slow path atom-major); the
            # calibration histogram does not see the order
            @test sort(fast.dF) ≈ sort(slow.dF) rtol = 1e-10
            @test fast.n == slow.n && fast.per_atom == pa
        end
        one = LIB.committee_predictions_fast(al12_model(), mem, xyz; stride=1, point_params=θ, nthreads=1)
        many = LIB.committee_predictions_fast(al12_model(), mem, xyz; stride=1, point_params=θ, nthreads=4)
        @test all(getfield(one, k) == getfield(many, k) for k in (:pE, :loE, :hiE, :pF, :dE, :dF))   # identical, not just close
    end
end
