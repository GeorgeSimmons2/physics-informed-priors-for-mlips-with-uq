# src/Models/Models.jl — model names and directories, build and load.

const MO_SRC = "src/Models/Models.jl"

"The first n frames of an extxyz file, copied as text (keeps every key as written)."
function head_frames(src, dst, n)
    lines = readlines(src); out = String[]; i = 1
    for _ in 1:n
        nat = parse(Int, strip(lines[i])); append!(out, lines[i:i+nat+1]); i += nat + 2
    end
    write(dst, join(out, "\n") * "\n"); dst
end

@testset "model_name" begin
    tested(MO_SRC, :model_name)
    @test ACEWorkflow.model_name(:Al, 12, 4, 6, 2) == "Al_12_4_6A_2"
    @test ACEWorkflow.model_name([:Ti, :Al], 20, 5, 6.0, 3) == "AlTi_20_5_6A_3"            # sorted, order-independent
    @test ACEWorkflow.model_name([:Al, :Ti], 20, 5, 6.0, 3) == ACEWorkflow.model_name([:Ti, :Al], 20, 5, 6.0, 3)
    @test_throws InexactError ACEWorkflow.model_name(:Al, 12, 4, 6.5, 2)                   # rcut must be a whole Å
end

@testset "model_dir" begin
    tested(MO_SRC, :model_dir)
    withenv("ACE_MODELS" => nothing) do
        @test ACEWorkflow.model_dir("X") == joinpath(REPO, "models", "X")
        @test ACEWorkflow.model_dir(:Al, 12, 4, 6, 2) == AL12_DIR[1:end-1]            # the published model's directory
    end
    mktempdir() do d
        withenv("ACE_MODELS" => d) do
            @test ACEWorkflow.model_dir([:Al], 12, 4, 6, 2) == joinpath(d, "Al_12_4_6A_2")
        end
    end
end

@testset "build_model, load_model" begin
    tested(MO_SRC, :build_model, :load_model)
    mktempdir() do dir
        withenv("ACE_MODELS" => dir) do
            xyz = head_frames(TRAIN_XYZ, joinpath(dir, "train.xyz"), 12)
            @test_throws ArgumentError ACEWorkflow.load_model(:Al, 6, 4, 5.0, 2)           # missing, nothing to build from
            b = ACEWorkflow.build_model(:Al, 6, 4, 5.0, 2; training_xyz=xyz)
            for f in ("Al_6_4_5A_2.json", "A.csv", "Y.csv", "P.csv", "W.csv", "lin_params.csv")
                @test isfile(joinpath(dir, "Al_6_4_5A_2", f))
            end
            @test b.name == "Al_6_4_5A_2" && b.dir == joinpath(dir, "Al_6_4_5A_2")
            # θ solves the regularised normal equations in x = Pθ: (ÃᵀÃ + PᵀP/n) x = Ãᵀ(W y)
            Y = vec(readdlm(joinpath(b.dir, "Y.csv"), ','))
            Ap = Diagonal(b.W) * b.A / b.P
            x = b.P * b.lin_params
            @test (Ap' * Ap + b.P' * b.P / size(Ap, 1)) * x ≈ Ap' * (b.W .* Y) rtol = 1e-8
            l = ACEWorkflow.load_model(:Al, 6, 4, 5.0, 2)
            @test l.lin_params == b.lin_params
            @test l.A ≈ b.A && l.P ≈ b.P && l.W ≈ b.W
            s = ExtXYZ.load(xyz)[3]                                                       # the loaded model carries θ
            @test energy(s, l.model) ≈ energy(s, b.model) rtol = 1e-12
            # build_model's docstring promises load_model's tuple, but it returns the weighted W.*Y as Y
            @test_broken b.Y ≈ l.Y
            @test b.Y ≈ b.W .* l.Y
            # stride and dataset_name: every second frame, into <name>_<dataset>
            h = ACEWorkflow.build_model(:Al, 6, 4, 5.0, 2; training_xyz=xyz, stride=2, dataset_name="half")
            @test h.dir == joinpath(dir, "Al_6_4_5A_2_half")
            @test size(h.A, 1) < size(b.A, 1)
            @test ACEWorkflow.load_model(:Al, 6, 4, 5.0, 2; dataset_name="half").lin_params == h.lin_params
            # a missing model with training data is built on the way in
            n = ACEWorkflow.load_model(:Al, 5, 4, 5.0, 2; training_xyz=xyz)
            @test isfile(joinpath(dir, "Al_5_4_5A_2", "Al_5_4_5A_2.json")) && n.name == "Al_5_4_5A_2"
        end
    end
end
