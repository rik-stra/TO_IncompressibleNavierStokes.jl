# Stage G of the Snellius grid (plan step 1+, 2026-09-28): score everything pulled back, in one pass.
#
#     julia --startup-file=no --project=analysis analysis/p4grid_collect.jl
#
# 1. Stage B again (offline tables), from the fits under P4GRID_FITROOT.
# 2. Stage D: the plan's PRIMARY score, `score_d6.jl`'s `paired_primary` (paired fair CRPS, leads
#    <= 0.5 TU, 6 bands, 90% IC-block bootstrap; negative = the fit better), for every pair of
#    d6_pairs.csv (fit -> matched M0, written by `p4grid_select.jl --after-smoke`), D6 dirs under
#    P4GRID_D6ROOT/D6mini_p4grid/<tag>. Stage-3 rule (plan §7, declared): a config PASSES if the CI
#    excludes zero in its favour (hi < 0) in >= 2 of its 3 fit seeds. Writes d6_paired.csv and
#    long_list.txt (the passing configs' best-D6 seed + their matched M0s) for stage E.
# 3. Stage F: every P4GRID_D6ROOT/D6mini_LinReg<n>_<suffix> against D6mini_LinReg<n> (same family,
#    CRN), same scorer.
# 4. Stage E: 100 TU replicas (data_online_tsim100.0_replica<i>.jld2) of the long_list fits. Pass/fail
#    gate, not a score (plan §7 stage 4): every replica complete (40001 columns) and finite, and the
#    whole-run level offset max_k |mean_k - mu_k| / sd_k (median over replicas) no larger than the
#    matched M0's. The per-20 TU chunk drift/basin view is `m4_screen_long.jl` (command printed).
#    Reference: the tracked record at t <= 74 TU only.
#
# Environment: P4GRID_FITROOT (default exp_square_HIT/output/TO_LSTM/p4grid), P4GRID_D6ROOT (default
# analysis/output/snellius_p4grid -- where the RUNBOOK's rsync puts the pulled D6 dirs, kept apart
# from the desktop's own D6mini_* dirs), P4GRID_SELOUT (default analysis/output/p4grid: the tables,
# and where d6_pairs.csv is read from). Plus score_d6.jl's own (D6_EXCLUDE_ICS, ...).

using JLD2, Statistics, Printf

module Sel
include(joinpath(@__DIR__, "p4grid_select.jl"))
end
module Sc
include(joinpath(@__DIR__, "score_d6.jl"))
end

const TO = Sel.TO
const FITROOT = get(ENV, "P4GRID_FITROOT", joinpath(TO, "p4grid"))
const D6ROOT = get(ENV, "P4GRID_D6ROOT", joinpath(@__DIR__, "output", "snellius_p4grid"))
const SELOUT = get(ENV, "P4GRID_SELOUT", joinpath(@__DIR__, "output", "p4grid"))
g(x) = @sprintf("%g", x)
nmembers(d) = isdir(d) ? count(f -> occursin(r"^d6_online_ic\d+_m\d+\.jld2$", f), readdir(d)) : 0
cfgkey(tag) = replace(tag, r"_s\d+$" => "")

"`paired_primary(a, b)`, or the reason it cannot be scored."
function pair(a, b, truth; io)
    na, nb = nmembers(a), nmembers(b)
    (na == 0 || nb == 0) && return (; ok = false, why = "no members (A $na, B $nb)")
    try
        r = Sc.paired_primary(a, b; truth, io)
        return (; ok = true, why = "", r)
    catch err
        return (; ok = false, why = sprint(showerror, err))
    end
end

function stage_d(truth; io)
    pf = joinpath(SELOUT, "d6_pairs.csv")
    isfile(pf) || (println(io, "\n(stage D: no $pf -- run p4grid_select.jl --after-smoke first)"); return [])
    pairs = [split(l, ",") for l in readlines(pf)[2:end] if !isempty(strip(l))]
    println(io, "\n==== Stage D: paired mini-D6, fit - matched M0 (negative = fit better)")
    rows = []
    for (fit, m0) in pairs
        t, t0 = basename(fit), basename(m0)
        p = pair(joinpath(D6ROOT, "D6mini_p4grid", t), joinpath(D6ROOT, "D6mini_p4grid", t0), truth; io)
        push!(rows, (; tag = t, cfg = cfgkey(t), m0 = t0, fit, m0dir = m0, p))
        p.ok || @printf(io, "  %-38s vs %-20s: not scored -- %s\n", t, t0, p.why)
    end
    println(io, "\n  summary")
    @printf(io, "  %-38s %-20s %9s %21s %4s  %s\n", "fit", "matched M0", "A - B", "90% CI", "K", "")
    for x in rows
        x.p.ok || continue
        r = x.p.r
        @printf(io, "  %-38s %-20s %+9.5f [%+9.5f, %+9.5f] %4d  %s\n", x.tag, x.m0, r.diff, r.lo, r.hi, r.K,
                r.hi < 0 ? "fit better" : r.lo > 0 ? "M0 better" : "unresolved")
    end
    mkpath(SELOUT)
    open(joinpath(SELOUT, "d6_paired.csv"), "w") do f
        println(f, "fit,matched_m0,diff,lo,hi,K,score_fit,score_m0,status")
        for x in rows
            if x.p.ok
                r = x.p.r
                println(f, join((x.fit, x.m0dir, g(r.diff), g(r.lo), g(r.hi), r.K, g(r.score_a), g(r.score_b), "ok"), ","))
            else
                println(f, join((x.fit, x.m0dir, "", "", "", "", "", "", "\"" * replace(x.p.why, "\"" => "'") * "\""), ","))
            end
        end
    end
    println(io, "\n  stage-3 rule: CI excludes 0 in the fit's favour in >= 2 of 3 seeds")
    long = String[]
    for c in unique(getfield.(rows, :cfg))
        xs = filter(x -> x.cfg == c && x.p.ok, rows)
        nwin = count(x -> x.p.r.hi < 0, xs)
        ok = nwin >= 2
        @printf(io, "  %-34s %d/%d seeds resolved in favour -> %s\n", c, nwin, length(xs), ok ? "PASS (to stage E)" : "fail")
        if ok
            b = xs[argmin([x.p.r.diff for x in xs])]
            push!(long, b.fit); push!(long, b.m0dir)
        end
    end
    long = unique(long)
    open(f -> foreach(l -> println(f, l), long), joinpath(SELOUT, "long_list.txt"), "w")
    @printf(io, "  wrote %s/{d6_paired.csv,long_list.txt}: %d dir(s) for stage E\n", SELOUT, length(long))
    isempty(long) || @printf(io, "  stage E: P4GRID_ONLINE_LIST=batch_scripts/p4grid/long_list.txt P4GRID_TSIM=100 sbatch --array=1-%d batch_scripts/run_p4grid_online.sh\n",
                             3 * length(long))
    return rows
end

function stage_f(truth; io)
    isdir(D6ROOT) || return []
    ds = filter(d -> occursin(r"^D6mini_LinReg\d+_\w+$", d), readdir(D6ROOT))
    isempty(ds) && return []
    println(io, "\n==== Stage F: ridge + colour, D6mini_LinReg<n>_<variant> - D6mini_LinReg<n> (negative = variant better)")
    out = []
    for d in sort(ds)
        base = match(r"^(D6mini_LinReg\d+)_", d)[1]
        p = pair(joinpath(D6ROOT, d), joinpath(D6ROOT, base), truth; io)
        p.ok || @printf(io, "  %s vs %s: not scored -- %s\n", d, base, p.why)
        push!(out, (; d, base, p))
    end
    return out
end

"Whole-run census of the 100 TU replicas in `dir`."
function long_census(dir, ref)
    fs = sort(filter(x -> occursin(r"^data_online_tsim100\.0_replica\d+\.jld2$", x), readdir(dir)))
    reps = map(fs) do x
        o = jldopen(fh -> fh["data_online"], joinpath(dir, x), "r")
        n = size(o.q, 2)
        dq = o.dQ[:, 101:end]
        (; n, ok = n >= 40001 && all(isfinite, o.q),
         clamp = count(j -> all(iszero, view(dq, :, j)), axes(dq, 2)),
         off = maximum(abs.((vec(mean(o.q; dims = 2)) .- ref.mu) ./ ref.sd)))
    end
    return (; nrep = length(reps), allok = !isempty(reps) && all(r -> r.ok, reps),
            clamp = sum((r.clamp for r in reps); init = 0),
            off = isempty(reps) ? NaN : median([r.off for r in reps]), ns = [r.n for r in reps])
end

function stage_e(; io)
    lf = joinpath(SELOUT, "long_list.txt")
    isfile(lf) || return nothing
    dirs = filter(!isempty, strip.(readlines(lf)))
    isempty(dirs) && return nothing
    ref = Sel.reference()
    println(io, "\n==== Stage E: 100 TU free runs (gate; reference t <= 74 TU)")
    cen = Dict(d => (p = startswith(d, "/") ? d : joinpath(TO, d); isdir(p) ? long_census(p, ref) : nothing) for d in dirs)
    for d in dirs
        c = cen[d]
        c === nothing && (@printf(io, "  %-40s missing\n", d); continue)
        @printf(io, "  %-40s reps %d  complete+finite %-5s  clamp %5d  level offset %.2f sd  (columns %s)\n", d, c.nrep,
                string(c.allok), c.clamp, c.off, join(c.ns, "/"))
    end
    println(io, "  per-20 TU drift / basins: REF_TU_MAX=74 julia --project=analysis analysis/m4_screen_long.jl ",
            join(dirs, " "))
    return cen
end

function main(; io = stdout)
    println(io, "fits: $FITROOT\nD6 dirs: $D6ROOT\ntables: $SELOUT")
    if isdir(FITROOT)
        res = Sel.offline([FITROOT]; io)
        Sel.write_offline(res, SELOUT; io)
    else
        println(io, "(no fit root $FITROOT)")
    end
    truth = Sc.load_truth()
    d = stage_d(truth; io)
    f = stage_f(truth; io)
    e = stage_e(; io)
    println(io, "\nonline smoke views: TSCREEN=20 REF_TU_MAX=74 M4_SCREEN_SUBDIR=p4grid julia --project=analysis analysis/m4_screen.jl")
    return (; d, f, e)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
