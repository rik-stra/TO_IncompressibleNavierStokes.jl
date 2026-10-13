# Gate-off reruns against their gated originals (handoff_gate_off_2026-10-13, Steps 3-4).
#
#   julia --project=analysis analysis/gate_pairs.jl check      # report only (default)
#   julia --project=analysis analysis/gate_pairs.jl assemble   # + complete out/D6_LinReg1_nogate
#
# Long runs: for TO_LRS/<c>_nogate/ against TO_LRS/<c>/, per replica, the first dQ column where the
# two differ must be the gated run's first firing (a zero column at step >= 101); anything earlier
# means the rerun does not reproduce the original (software or hardware drift), so the pair is not
# common random numbers. Also checks `gate_threshold == 0` in every gate-off file.
#
# Hindcast: the 35 rerun members in D6_LinReg1_nogate carry `pair_first_diff`/`pair_expected`/
# `pair_ok` (written by run_d6.jl's D6_PAIR_WITH). `assemble` then copies every ORIGINAL member
# that never fired (gate census 0, so it is its own gate-off run) into D6_LinReg1_nogate, never
# overwriting, and refuses if any rerun is missing or any pair is not OK.

using JLD2, Printf

const OUTD = joinpath(@__DIR__, "..", "exp_square_HIT", "output")
const LRS_CLOSURES = ("LinReg1", "LinReg6")
const D6_GATED = joinpath(OUTD, "D6_LinReg1")
const D6_NOGATE = joinpath(OUTD, "D6_LinReg1_nogate")
const SKIP = 100
const PAT = r"^d6_online_ic(\d+)_m(\d+)\.jld2$"

firstdiff(a, b) = something(findfirst(c -> !isequal(view(a, :, c), view(b, :, c)),
                                      1:min(size(a, 2), size(b, 2))), 0)
"First identically-zero column of `dQ` at or after `from` (a firing), or -1."
function firstzero(dQ, from)
    for c in from:size(dQ, 2)
        all(iszero, view(dQ, :, c)) && return c
    end
    return -1
end

function check_lrs()
    allok = true
    for c in LRS_CLOSURES
        dg, dn = joinpath(OUTD, "TO_LRS", c), joinpath(OUTD, "TO_LRS", c * "_nogate")
        isdir(dn) || (println("TO_LRS/$(c)_nogate: absent"); allok = false; continue)
        for r in 1:5
            f = "data_online_tsim100.0_replica$r.jld2"
            isfile(joinpath(dn, f)) || (println("  $c r$r: no gate-off file"); allok = false; continue)
            g = jldopen(io -> io["data_online"].dQ, joinpath(dg, f))
            n, thr = jldopen(io -> (io["data_online"].dQ, get(io, "gate_threshold", missing)), joinpath(dn, f))
            fz = firstzero(g, SKIP + 1)            # the gated run's first firing (column), -1 none
            fd = firstdiff(n, g)
            ok = thr === 0.0 && (fz < 0 ? fd == 0 : fd == fz)
            allok &= ok
            @printf("  %-8s r%d: first diff col %6d (t = %7.3f), gated first firing col %6d -- %s%s\n",
                    c, r, fd, fd * 2.5e-3, fz, ok ? "OK" : "🔴 MISMATCH",
                    thr === 0.0 ? "" : "  (gate_threshold = $thr)")
        end
    end
    return allok
end

function check_d6(; assemble = false)
    isdir(D6_NOGATE) || (println("D6_LinReg1_nogate: absent"); return false)
    fired, quiet = String[], String[]
    for f in readdir(D6_GATED)
        occursin(PAT, f) || continue
        dQ, nw = load(joinpath(D6_GATED, f), "dQ", "nwarm")
        (firstzero(dQ, nw + 1) > 0 ? fired : quiet) |> v -> push!(v, f)
    end
    @printf("D6_LinReg1: %d members fired, %d never did\n", length(fired), length(quiet))
    ok = true
    for f in fired
        p = joinpath(D6_NOGATE, f)
        if !isfile(p)
            println("  missing rerun $f"); ok = false; continue
        end
        d = load(p)
        good = get(d, "pair_ok", false) === true && d["gate_threshold"] == 0.0
        ok &= good
        @printf("  %-26s first diff %4d, expected %4d, gate %g, %s%s\n", f, d["pair_first_diff"],
                d["pair_expected"], d["gate_threshold"], good ? "OK" : "🔴 NOT PAIRED",
                get(d, "diverged", false) === true ? "  (diverged at step $(d["nstep"]))" : "")
    end
    if assemble
        ok || error("not assembling: a rerun is missing or not paired (see above)")
        n = 0
        for f in quiet
            dst = joinpath(D6_NOGATE, f)
            isfile(dst) && continue
            cp(joinpath(D6_GATED, f), dst)
            n += 1
        end
        nfiles = count(f -> occursin(PAT, f), readdir(D6_NOGATE))
        @printf("assembled: copied %d never-fired originals; D6_LinReg1_nogate now holds %d members\n",
                n, nfiles)
        nfiles == length(fired) + length(quiet) || error("expected $(length(fired) + length(quiet)) members")
    end
    return ok
end

mode = isempty(ARGS) ? "check" : ARGS[1]
mode in ("check", "assemble") || error("usage: gate_pairs.jl [check|assemble]")
println("== long runs"); lok = check_lrs()
println("== hindcast"); dok = check_d6(; assemble = mode == "assemble")
@printf("\nlong runs paired: %s; hindcast paired: %s\n", lok, dok)
