# test/harness.jl — load the functions a script defines without running the script, and keep track of which
# functions the tests cover.
#
# The scripts are top-level pipelines (hours of work) whose helpers read script globals.  `script_module(path;
# globals)` parses the file, keeps its `using`/`import` lines, its `include`s (lib/*.jl) and every top-level function
# definition, drops everything else, and evaluates them in a fresh module after binding `globals`.  A test then calls
# the script's own function on small, controlled inputs.  Scripts and src/ are never edited.

using Test

const REPO = normpath(joinpath(@__DIR__, ".."))

# ── recognising definitions ──────────────────────────────────────────────────────────────────────────────────────
undoc(ex) = ex isa Expr && ex.head === :macrocall && length(ex.args) >= 3 && endswith(string(ex.args[1]), "@doc\"") ?
            ex.args[end] : ex          # a docstring parses as Core.@doc (a GlobalRef or a dotted name)
function callsig(lhs)
    while lhs isa Expr && lhs.head in (:where, :(::))
        lhs = lhs.args[1]
    end
    lhs isa Expr && lhs.head === :call ? lhs : nothing
end
isfundef(ex) = (ex = undoc(ex); ex isa Expr && (ex.head === :function ? ex.args[1] isa Expr :
                                                 ex.head === :(=) && callsig(ex.args[1]) !== nothing))
function funname(ex)
    ex = undoc(ex)
    s = callsig(ex.args[1])
    s === nothing && (s = ex.args[1])                         # `function f end`
    n = s isa Expr ? s.args[1] : s
    n isa Expr && n.head === :. ? n.args[end].value : n       # Base.f → f
end
isinclude(ex) = ex isa Expr && ex.head === :call && ex.args[1] === :include
isusing(ex) = ex isa Expr && ex.head in (:using, :import)

"Top-level expressions of a file with their line numbers, descending into `module … end` blocks."
function toplevel(path)
    out = Tuple{LineNumberNode,Any}[]
    walk(args, ln) = for e in args
        if e isa LineNumberNode
            ln = e
        elseif e isa Expr && e.head === :module
            walk(e.args[3].args, ln)
        else
            push!(out, (ln, e))
        end
    end
    walk(Meta.parseall(read(path, String); filename=path).args, LineNumberNode(1, Symbol(path)))
    out
end

"Names of the top-level functions a file defines (each name once)."
defined_functions(path) = unique([funname(e) for (_, e) in toplevel(path) if isfundef(e)])

# ── loading a script's functions ─────────────────────────────────────────────────────────────────────────────────
"""
    script_module(relpath; globals=(;), only=nothing) -> Module

A fresh module holding `relpath`'s `using`/`import` lines, `include`s and top-level function definitions (all, or
those named in `only`), with `globals` bound first.  Nothing else in the script runs.
"""
function script_module(relpath; globals=(;), only=nothing)
    path = joinpath(REPO, relpath)
    m = Core.eval(Main, :(module $(gensym(Symbol(basename(relpath)))) end))
    for (k, v) in pairs(globals)
        Core.eval(m, Expr(:global, Expr(:(=), k, QuoteNode(v))))        # quoted: a Symbol value stays a Symbol
    end
    for (ln, e) in toplevel(path)
        keep = isusing(e) || isinclude(e) || (isfundef(e) && (only === nothing || funname(e) in only))
        keep && Core.eval(m, Expr(:toplevel, ln, e))
    end
    m
end

"Call a function of a script module (its methods are newer than the caller's world)."
call(m::Module, f::Symbol, args...; kw...) = Base.invokelatest(getfield(m, f), args...; kw...)

# ── coverage bookkeeping ─────────────────────────────────────────────────────────────────────────────────────────
const COVERED = Set{Tuple{String,Symbol}}()
"Mark `file`'s function `name` as tested (called inside the testset that tests it)."
tested(file, names::Symbol...) = foreach(n -> push!(COVERED, (file, n)), names)

"Every function defined at top level in src/, lib/ and the scripts, as (file, name)."
function inventory()
    files = String[]
    for (root, _, fs) in walkdir(joinpath(REPO, "src")), f in fs
        endswith(f, ".jl") && push!(files, relpath(joinpath(root, f), REPO))
    end
    for d in ("lib", "al12", "al16", "four_panel", "rls_npt", "mace", "mace/pops", "ace_models"),
        f in readdir(joinpath(REPO, d))
        endswith(f, ".jl") && push!(files, joinpath(d, f))
    end
    [(f, n) for f in sort(files) for n in defined_functions(joinpath(REPO, f))]
end
