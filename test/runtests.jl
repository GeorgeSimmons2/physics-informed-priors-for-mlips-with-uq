# Unit tests for src/, lib/ and the functions the figure scripts define.
#
#   JULIA_DEPOT_PATH=.julia julia --project=. -t 8 test/runtests.jl        (bash run.sh test runs this)
#   TEST_GROUPS=src,lib,scripts selects groups (default all); with all three, a coverage check fails if any top-level
#   function in src/, lib/ or a script has no test.

include("harness.jl")
const GROUPS = split(get(ENV, "TEST_GROUPS", "src,lib,scripts"), ",")
const FILES = Dict(
    "src"     => ["src/pops.jl", "src/elasticity.jl", "src/phonons.jl", "src/models.jl"],
    "lib"     => ["lib/bandpath.jl", "lib/parity_calibration.jl"],
    "scripts" => ["scripts/common.jl", "scripts/al12.jl", "scripts/al16.jl", "scripts/four_panel.jl", "scripts/rls_npt.jl",
                  "scripts/mace_jl.jl"])

t0 = time()
include("fixtures.jl")
@printf("fixtures loaded in %.0f s\n", time() - t0)

@testset "constrained-pops-repro" begin
    for g in ("src", "lib", "scripts")
        g in GROUPS || continue
        @testset "$g" begin
            for f in FILES[g]
                t = time()
                include(f)
                @printf("  %-26s %6.0f s\n", f, time() - t)
            end
        end
    end
    if all(g -> g in GROUPS, ("src", "lib", "scripts"))
        @testset "every function has a test" begin
            missing_tests = setdiff(inventory(), COVERED)
            isempty(missing_tests) || println("untested: ", join(["$f: $n" for (f, n) in missing_tests], ", "))
            @test isempty(missing_tests)
        end
    end
end
