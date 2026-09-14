# V31 -- source-level checks on the scripts the offline suite cannot load.
#
# `exp_square_HIT/tools/run_d6.jl` needs CUDA and IncompressibleNavierStokes, which this suite
# deliberately does not depend on (see `runtests.jl`), so nothing here ever macroexpanded it. That
# gap cost a Snellius run: `@printf` requires its format argument to be a *single string literal*,
# a `"..." * "..."` concatenation parses fine and fails only at macroexpansion, and for a
# documented function macroexpansion happens when the **docstring** is processed. So the error
# pointed at the docstring of `run_ic` (line 231) while the defect was 140 lines further down, and
# it appeared only on the cluster, after the copy and the queue.
#
# The check is therefore on the parsed source, not on a loaded module: it needs no packages and so
# covers every script in the tree, including the drivers this suite cannot import.
@testitem "V31 every @printf format argument is a string literal" default_imports = false begin
    using Test

    root = normpath(joinpath(@__DIR__, "..", ".."))   # code_base/lib

    """Walk an AST and collect `@printf`/`@sprintf` calls whose format argument is not a literal."""
    function bad_formats(ex, file, acc = String[])
        ex isa Expr || return acc
        if ex.head === :macrocall && ex.args[1] isa Symbol &&
           String(ex.args[1]) in ("@printf", "@sprintf")
            rest = ex.args[3:end]
            if !isempty(rest)
                # the io argument is optional, so the format is the first or the second argument
                fmt = rest[1] isa String ? rest[1] : (length(rest) > 1 ? rest[2] : nothing)
                if !(fmt isa String)
                    ln = ex.args[2]
                    push!(acc, string(relpath(file, root), ":",
                                      ln isa LineNumberNode ? ln.line : "?", "  ",
                                      String(ex.args[1]), " fmt = ", sprint(show, fmt)))
                end
            end
        end
        for a in ex.args
            bad_formats(a, file, acc)
        end
        return acc
    end

    """Parse every `.jl` file under `root` and return the offenders plus the file count."""
    function scan(root)
        bad = String[]
        nfiles = 0
        for (dir, _, files) in walkdir(root), f in files
            endswith(f, ".jl") || continue
            p = joinpath(dir, f)
            ast = try
                Meta.parseall(read(p, String); filename = p)
            catch
                continue      # a file this Julia cannot parse is not this test's business
            end
            nfiles += 1
            bad_formats(ast, p, bad)
        end
        return bad, nfiles
    end

    bad, nfiles = scan(root)

    @test nfiles > 50                     # the walk found the tree, not an empty directory
    @test isempty(bad) || (println("\n  ", join(bad, "\n  ")); false)

    # Positive control: a scanner that has never seen an offender is not a test. The first three
    # are the exact shape that broke run_d6.jl on Snellius; the last three are what is fine.
    parse1(code) = bad_formats(Meta.parseall(code; filename = "control.jl"), "control.jl")
    @test length(parse1(raw"""@printf("a %d b " * "c %d\n", 1, 2)""")) == 1
    @test length(parse1(raw"""@printf(io, "a %d " * "b\n", 1)""")) == 1
    @test length(parse1(raw"""@sprintf("%d" * "%d", 1, 2)""")) == 1
    @test isempty(parse1(raw"""@printf("a %d b c %d\n", 1, 2)"""))
    @test isempty(parse1(raw"""@printf(io, "a %d\n", 1)"""))
    @test isempty(parse1(raw"""@printf(stdout, "plain\n")"""))
end

# V35 -- every `using` in a runnable script resolves from `[deps]`, `[weakdeps]` or `@stdlib`.
#
# Gotcha #53 was `2_HF_ref.jl` carrying `using CairoMakie`, which is not in `lib/RikFlow`'s
# `[deps]`: a 20-hour job that would have died at load with "Package CairoMakie not found in current
# path", after the queue wait, with nothing done. Nothing in the file plotted. The rule written down
# then -- "before submitting a long job, check every `using` in the driver against the project's
# `[deps]`" -- was a habit, and on 2026-09-14 the same defect was found in **two more drivers on the
# critical path**: `4_setup_search.jl` (`using DataFrames`) and `5_train_LinReg.jl`
# (`using CairoMakie`). A habit that fails three times is a test.
#
# 🔑 This is the class V31 cannot reach: a parse check sees the `using` but not whether it resolves.
# Like V31 it works on source text, so it needs no packages and covers the drivers this suite cannot
# import. Stdlibs are exempt -- they resolve from `@stdlib` whatever the project says, which is why
# `using Printf` works undeclared and hides the fact that the check is needed at all.
@testitem "V35 every `using` in a script resolves from [deps] or @stdlib" default_imports = false begin
    using Test

    const RF = normpath(joinpath(@__DIR__, ".."))

    """The union of `[deps]` and `[weakdeps]` names in a Project.toml, by line scan (no TOML dep)."""
    function project_names(path)
        names, inside = Set{String}(), false
        for line in eachline(path)
            s = strip(line)
            if startswith(s, "[")
                inside = s in ("[deps]", "[weakdeps]")
            elseif inside && occursin("=", s) && !startswith(s, "#")
                push!(names, strip(split(s, "=")[1]))
            end
        end
        return names
    end

    stdlib_names() = Set(readdir(joinpath(Sys.BINDIR, "..", "share", "julia", "stdlib",
                                          "v$(VERSION.major).$(VERSION.minor)")))

    "Top-level `using X` / `import X` package names in a file, first name only."
    function imported(path)
        out = String[]
        for line in eachline(path)
            m = match(r"^\s*(?:using|import)\s+([A-Za-z][A-Za-z0-9_]*)", line)
            m === nothing && continue
            push!(out, m.captures[1])
        end
        return out
    end

    deps = project_names(joinpath(RF, "Project.toml"))
    stdlibs = stdlib_names()
    allowed = union(deps, stdlibs, Set(["RikFlow"]))

    # The scripts that are actually submitted. Analysis drivers run under `analysis/Project.toml`
    # and are checked against that one instead; plotting and notebook helpers are not job scripts.
    dirs = [(joinpath(RF, "exp_square_HIT"), allowed),
            (joinpath(RF, "exp_square_HIT", "tools"), allowed),
            (joinpath(RF, "analysis"),
             union(project_names(joinpath(RF, "analysis", "Project.toml")), stdlibs,
                   Set(["RikFlow"])))]

    # 🔑 In a function, not a bare loop: assigning to `nfiles` inside a top-level `for` when a
    # global of that name exists makes it a new local and the read throws. Gotcha #47's last bullet,
    # which bit three times in one session and twice more on 2026-09-14.
    function scan_imports(dirs, root)
        bad, nfiles = String[], 0
        for (dir, ok) in dirs
            isdir(dir) || continue
            for f in sort(filter(x -> endswith(x, ".jl"), readdir(dir)))
                path = joinpath(dir, f)
                nfiles += 1
                for pkg in imported(path)
                    pkg in ok || push!(bad, "$(relpath(path, root)): using $pkg")
                end
            end
        end
        return bad, nfiles
    end

    bad, nfiles = scan_imports(dirs, RF)

    # 🔴 Known, pre-existing, and NOT on the fit/online/D6 path. Listed with a reason each rather
    # than waved through, because the point of this test is that a new one cannot appear quietly.
    #
    #   1_spinnup.jl / figs_paper.jl / plot_spinnup_output.jl -- CairoMakie. The last two really do
    #     plot, so the fix is a dependency decision (add CairoMakie, or move them under a project
    #     that has it), not a deletion. 1_spinnup.jl is 2_HF_ref.jl's defect again and IS a job
    #     script, so it is the one of the three that would actually bite; it is not rerun in the
    #     current rebaseline only because the archived spin-up field is reused deliberately.
    #   compute_ks*.jl -- DataFrames, genuinely used. P2r/R2 scores through the analysis metric
    #     layer instead (one scoring path, gotcha #36), so these are off the critical path;
    #     compute_ks.jl separately loads a filename nothing writes (#55).
    #   analysis/ou_replay.jl -- IncompressibleNavierStokes, genuinely needed for its CPU mini-solve.
    #     `analysis/Project.toml` excludes INS on purpose, so this one script must run under
    #     `lib/RikFlow`'s project. Deliberate, and recorded here so it is not "fixed".
    sep = Base.Filesystem.path_separator
    known = Set(replace.([
        "exp_square_HIT/1_spinnup.jl: using CairoMakie",
        "exp_square_HIT/compute_ks.jl: using DataFrames",
        "exp_square_HIT/compute_ks_DDN_smag_LF.jl: using DataFrames",
        "exp_square_HIT/figs_paper.jl: using CairoMakie",
        "exp_square_HIT/plot_spinnup_output.jl: using CairoMakie",
        "analysis/ou_replay.jl: using IncompressibleNavierStokes",
    ], "/" => sep))
    novel = setdiff(Set(bad), known)

    @test nfiles > 15                      # the walk found the tree, not an empty directory

    # The gate: no NEW unresolvable import. This is exactly what would have caught
    # 4_setup_search.jl and 5_train_LinReg.jl before they reached the queue.
    @test isempty(novel) ||
          (println("\n  NEW unresolvable imports:\n  ", join(sort(collect(novel)), "\n  ")); false)

    # The debt, visible rather than hidden: broken until the six above are decided.
    @test_broken isempty(bad)

    # Positive control: a scanner that has never seen an offender is not a test.
    let ok = Set(["JLD2", "Printf"])
        probe(line) = [p for p in [match(r"^\s*(?:using|import)\s+([A-Za-z][A-Za-z0-9_]*)", line)]
                       if p !== nothing && !(p.captures[1] in ok)]
        @test length(probe("using CairoMakie")) == 1
        @test length(probe("import DataFrames")) == 1
        @test isempty(probe("using JLD2"))
        @test isempty(probe("using Printf"))
        @test isempty(probe("# using CairoMakie   -- a comment is not an import"))
    end
end

# V36 -- no top-level `for`/`while` assigns a name that is already bound at top level.
#
# Julia's soft scope: inside a top-level loop, assigning to a name that already exists as a global
# creates a NEW LOCAL, and reading it throws `UndefVarError`. The REPL special-cases this and
# assigns to the global with only a warning, so a script written interactively runs there and dies
# when run as a script.
#
# 🔴 `4_setup_search.jl` had exactly this and had therefore **never run as a script** -- its
# `i = 0; for ...; i += 1; end` counter threw on the first real invocation, 2026-09-14. Gotcha #47
# already records three instances in one session, and two more happened while writing V35. Six is
# enough.
#
# 🔑 This is the class #47 says a parse check cannot reach and an `include` would not catch either:
# it is a runtime error in a code path that only executes when the script is run. A *static* check
# for the shape does reach it, which is what this is. The fix is always the same -- put the loop in
# a function -- and never `global`.
@testitem "V36 no top-level loop assigns a top-level binding (soft scope)" default_imports = false begin
    using Test

    const RF36 = normpath(joinpath(@__DIR__, ".."))

    "Names bound by a plain top-level assignment or `const`."
    function toplevel_bindings(ast)
        names = Set{Symbol}()
        for ex in ast.args
            ex isa Expr || continue
            if ex.head === :(=) && ex.args[1] isa Symbol
                push!(names, ex.args[1])
            elseif ex.head === :const && ex.args[1] isa Expr && ex.args[1].args[1] isa Symbol
                push!(names, ex.args[1].args[1])
            end
        end
        return names
    end

    """Names assigned anywhere inside `ex`, NOT descending into function bodies -- those introduce
    a hard scope and are exactly the recommended fix, so they must not be flagged."""
    function assigned_in(ex, acc = Set{Symbol}())
        ex isa Expr || return acc
        ex.head in (:function, :(->)) && return acc
        if ex.head === :(=) || (let h = string(ex.head); endswith(h, "=") && length(h) > 1 end)
            ex.args[1] isa Symbol && push!(acc, ex.args[1])
        end
        for a in ex.args
            assigned_in(a, acc)
        end
        return acc
    end

    function offenders(ast)
        globals = toplevel_bindings(ast)
        out = Symbol[]
        for ex in ast.args
            ex isa Expr && ex.head in (:for, :while) || continue
            append!(out, intersect(assigned_in(ex), globals))
        end
        return out
    end

    function scan36(dirs, root)
        bad, nfiles = String[], 0
        for d in dirs
            dir = joinpath(root, d)
            isdir(dir) || continue
            for f in sort(filter(x -> endswith(x, ".jl"), readdir(dir)))
                p = joinpath(dir, f)
                nfiles += 1
                ast = try
                    Meta.parseall(read(p, String); filename = p)
                catch
                    continue
                end
                for n in offenders(ast)
                    push!(bad, "$(relpath(p, root)): top-level loop assigns global `$n`")
                end
            end
        end
        return bad, nfiles
    end

    bad, nfiles = scan36(["exp_square_HIT", joinpath("exp_square_HIT", "tools"), "analysis"], RF36)

    @test nfiles > 15
    @test isempty(bad) || (println("\n  soft-scope traps:\n  ", join(sort(bad), "\n  ")); false)

    # Positive control: a scanner that has never seen an offender is not a test. The first is the
    # exact shape that broke 4_setup_search.jl.
    parse36(code) = offenders(Meta.parseall(code; filename = "control.jl"))
    @test parse36("i = 0\nfor x in 1:3\n    i += 1\nend\n") == [:i]
    @test parse36("acc = []\nfor x in 1:3\n    acc = vcat(acc, x)\nend\n") == [:acc]
    @test parse36("n = 0\nwhile n < 3\n    n += 1\nend\n") == [:n]
    # And the three shapes that are fine.
    @test isempty(parse36("f() = (i = 0; for x in 1:3; i += 1; end; i)\n"))   # inside a function
    @test isempty(parse36("acc = []\nfor x in 1:3\n    push!(acc, x)\nend\n"))  # mutation, not assignment
    @test isempty(parse36("for x in 1:3\n    y = x\nend\n"))                    # no such global
end
