# Aggregate D6 score files into section 4c's summary table, and diff the two exclusion policies.
#
# Usage:
#   julia --startup-file=no --project=analysis analysis/d6_delta.jl
#
# 🔴 **Why this script exists at all.** Section 4c's summary rows -- mean skill over the 30
# (band, lead) cells, the short- and long-lead means, the count inside S7's band, the median
# spread-skill ratio -- were never computed by any script. They were read off `score_d6.jl`'s
# printed tables by hand. So the numbers in the report could not be regenerated, and a change of
# policy could not be *diffed*, only re-read. Everything here is derived from the saved
# `d6_scores_*.jld2` and nothing is retyped.
#
# ---------------------------------------------------------------------------------------------
# The two policies, and what is and is not comparable between them
# ---------------------------------------------------------------------------------------------
#
# **IC-level** (`d6_scores_D6_<model>.jld2`): the three ICs where a LinReg1 member diverged are
# dropped from every closure. K = 87, M = 10. The kept set is not a random subset for LinReg1 --
# it is "the ICs where the model did not break" -- so its skill column is conditioned in its own
# favour.
#
# **Member-level** (`..._thin.jld2`): those ICs are kept and the diverged *member* is dropped from
# every closure, each IC then thinned to a common M. K = 90, M = 9. The IC-level conditioning is
# gone; member-level survivorship remains, because a diverged member has no value past its blow-up
# under any policy.
#
# 🔴 **`skill / climatology` is NOT directly comparable across the two, and the reason is arithmetic
# rather than physical.** `climatological_skill` is `sigma * sqrt(1 + 1/M)` (`ts_score.jl:953`), so
# the denominator itself moves with `M`: 1.04881 sigma at M = 10 against 1.05409 sigma at M = 9, a
# **+0.50%** shift that would appear as a uniform improvement in every cell of every closure. It
# cancels exactly in any *between-closure* comparison within one policy, which is what the Pareto
# front is read from -- and it does not cancel between policies. So the between-policy delta is
# reported twice: once as published (each file's own denominator) and once on a common M = 10
# denominator, which is the one to read.

using Statistics
using Printf
using JLD2

const HERE = @__DIR__
const OUT = joinpath(HERE, "output")

"The three closures D6 ran, in the order section 4c's tables list them."
const MODELS = ["LinReg1", "LinReg7", "DDN"]

"S7's spread-skill acceptance band. `metrics.md` #18; the width is doing real work -- see section 3."
const S7_BAND = (0.8, 1.25)

"""
    SHORT_TU, LONG_TU

The lead windows, in TU, that the summary rows pool: **short** is at or below the level's 1/e time
(0.290-0.355 TU), **long** is past its `rho = 0.1` crossing (0.430-0.600 TU).

🔑 **Stated in physical time, not as grid positions.** Since 2026-09-17 every band shares one grid
(`score_d6.jl`'s `LEADS`), so a lead is the same forecast horizon for all six and these windows mean
something on their own. Under the old per-QoI grid the same two rows pooled positions, and position
2 was 50 steps for `Z[0,6]` and 109 for `Z[16,32]` -- a factor 2.2 in horizon inside one column.

Leads between the two windows are reported but pooled into neither: that is the crossover, where
section 1 measures the reference's own residual autocorrelation as marginal.
"""
const SHORT_TU = (0.0, 0.3)
const LONG_TU = (0.6, Inf)

"""
    cells(sc; m_ref = nothing)

The 30 `(band, lead)` cells of one score file as `(; skillratio, ratio)`, each `6 x 5`.

`skillratio` is the RMSE of the ensemble mean as a fraction of the climatological level, which is
what section 4c calls skill: **1.0 is "no better than climatology"** and lower is better. `ratio` is
the finite-`M`-corrected spread-skill ratio, already corrected in the score file.

`m_ref` rescales the climatological denominator to a reference `M`, undoing the `sqrt(1 + 1/M)`
factor the file was written with. Pass it whenever two files with different `M` are compared; leave
it `nothing` to reproduce the published numbers exactly.
"""
function cells(sc; m_ref = nothing)
    lev = sc["level"]
    dt = sc["dt"]
    nq = length(lev.skill)
    nl = length(first(lev.skill))
    all(==(nl), length.(lev.skill)) ||
        error("ragged lead grid: $(length.(lev.skill)) leads per QoI, expected $nl everywhere")
    # 🔴 The file's `clim_level` is `sigma * sqrt(1 + 1/M)` at that file's own `M`, so putting it on
    # a reference `M` means multiplying by `sqrt(1 + 1/m_ref) / sqrt(1 + 1/M_file)` -- NOT its
    # reciprocal. Inverted, the factor is applied twice instead of cancelling, and at M 10 -> 9 that
    # is a spurious 1.0% improvement in every cell: enough to reverse the sign of the `M` effect and
    # make dropping a member look like it *helps* the ensemble mean, which it cannot.
    scale = m_ref === nothing ? 1.0 :
            sqrt(1 + 1 / m_ref) / sqrt(1 + 1 / lev.M)
    sr = [lev.skill[i][t] / (lev.clim_level[i] * scale) for i in 1:nq, t in 1:nl]
    rt = [lev.ratio[i][t] for i in 1:nq, t in 1:nl]
    # One grid for every band since 2026-09-17, so a lead column is one horizon in TU. Asserted
    # rather than assumed: pooling by physical time is only meaningful while that holds.
    all(==(lev.leads[1]), lev.leads) ||
        error("the bands no longer share a lead grid; SHORT_TU/LONG_TU pool by physical time and " *
              "would be averaging different horizons into one row")
    tu = lev.leads[1] .* dt
    return (; skillratio = sr, ratio = rt, K = lev.K, M = lev.M, nq, nl, tu)
end

"""
    summarise(sc; m_ref = nothing)

Section 4c's summary row for one score file.

⚠️ `in_band` counts cells, not QoIs. 30 cells over 6 bands are **not** 30 independent tests: the
bands move together (section 1 measures the level's six 1/e times as spanning only 1.22x), so a
count of 23 against 2 is a large difference in kind and not a 21-fold difference in evidence.
"""
function summarise(sc; m_ref = nothing)
    c = cells(sc; m_ref)
    inb = count(r -> S7_BAND[1] <= r <= S7_BAND[2], c.ratio)
    short = findall(t -> SHORT_TU[1] <= t <= SHORT_TU[2], c.tu)
    long = findall(t -> LONG_TU[1] <= t <= LONG_TU[2], c.tu)
    return (; K = c.K, M = c.M,
            mean_skill = mean(c.skillratio),
            short = mean(c.skillratio[:, short]),
            long = mean(c.skillratio[:, long]),
            in_band = inb, ncells = length(c.ratio),
            median_ratio = median(c.ratio))
end

"""
    POLICIES

The three scored sets, and what each pair of them isolates.

    A  87 ICs, M = 10   the published policy: whole ICs dropped
    C  87 ICs, M =  9   the isolation control: same ICs, one fewer member
    B  90 ICs, M =  9   the member-level policy: ICs kept, the member dropped

🔴 **A -> B confounds two changes**, `K` and `M`, and a spread-skill ratio is `M`-free only for a
perfectly reliable ensemble -- which is precisely what is in question. So the delta is decomposed:
**A -> C is the `M` effect** (no initial condition changes) and **C -> B is the initial conditions**
(no `M` change). Only the second is the survivorship this narrowing was asked to remove.
"""
const POLICIES = [("A", "", "IC-level: every IC with a divergence dropped"),
                  ("C", "_thin_k87", "the same ICs at the member-level M (control)"),
                  ("B", "_thin", "member-level: ICs kept, the member dropped")]

"Load one policy's score file for `model`, or `nothing` when it has not been written."
function load_scores(model::AbstractString, suffix::AbstractString = "")
    p = joinpath(OUT, "d6_scores_D6_$(model)$(suffix).jld2")
    isfile(p) || return nothing
    return load(p)
end
load_scores(model::AbstractString; thin::Bool) = load_scores(model, thin ? "_thin" : "")

fmt(x) = @sprintf("%.4f", x)
signed4(x) = @sprintf("%+.4f", x)
pct(a, b) = b == 0 ? "n/a" : @sprintf("%+.1f%%", 100 * (a - b) / b)

"""
    report(; io = stdout)

Both policies, per closure, with the delta -- and the front table under each.

🔑 The front is the claim (S2', `claude_memory.md` #17), so the last block is the one that decides
whether this narrowing changes anything: **it matters only if it moves the LinReg1-vs-LinReg7
ordering on an axis.** The accuracy axis is a 2.6% gap on the IC-level policy, which is inside the
range three added ICs can move.
"""
function report(; io = stdout)
    have = Dict{String,Any}()
    for m in MODELS, (_, suffix, _) in POLICIES
        s = load_scores(m, suffix)
        s === nothing || (have[m * suffix] = s)
    end
    for (tag, suffix, what) in POLICIES
        gone = [m for m in MODELS if !haskey(have, m * suffix)]
        isempty(gone) && continue
        println(io, "🔴 policy $tag ($what) missing for ", join(gone, ", "), ".")
        println(io, "   See the header of this file for the three runs.")
        return nothing
    end

    # 🔴 PAIRING CHECK, and it has already caught one. D6's whole power is that every closure
    # forecasts from the same ICs, and policy A holds that by naming the diverged ICs in
    # `D6_EXCLUDE_ICS` for *every* closure -- the one closure that diverged loses them anyway, the
    # other two do not unless told. Re-scoring without it leaves LinReg1 on 87 and the others on 90:
    # a comparison of different experiments, with nothing in the tables to show it, because the
    # column headings here used to be hardcoded strings rather than read off the files.
    for (tag, suffix, what) in POLICIES
        ks = [(m, summarise(have[m * suffix]).K, summarise(have[m * suffix]).M) for m in MODELS]
        allequal([k for (_, k, _) in ks]) && allequal([mm for (_, _, mm) in ks]) ||
            error("policy $tag ($what) is NOT paired: " *
                  join(("$m K=$k M=$mm" for (m, k, mm) in ks), ", ") *
                  ". Every closure must be scored over the same initial conditions and the same " *
                  "member count; re-run that policy with the same environment for all three.")
    end

    # Every skill number is put on the A denominator, so the M = 10 -> 9 shift in
    # `climatological_skill` cannot masquerade as a change in the forecasts.
    mref = summarise(have[MODELS[1]]).M
    S = Dict((m, suffix) => summarise(have[m * suffix]; m_ref = mref)
             for m in MODELS, (_, suffix, _) in POLICIES)

    println(io, "\nD6 exclusion policy -- IC-level (A) against member-level (B), with the")
    println(io, "M-isolation control (C) between them. All skill on a common M = $mref denominator.")
    println(io, "="^104)
    for m in MODELS
        a, c, b = S[(m, "")], S[(m, "_thin_k87")], S[(m, "_thin")]
        @printf(io, "\n%s\n", m)
        @printf(io, "  %-26s %10s %10s %10s %12s %12s\n",
                "", (@sprintf("A %d@%d", a.K, a.M)), (@sprintf("C %d@%d", c.K, c.M)),
                (@sprintf("B %d@%d", b.K, b.M)), "A->C  (M)", "C->B  (ICs)")
        for (name, fa, fc, fb) in (("mean skill, all cells", a.mean_skill, c.mean_skill, b.mean_skill),
                                   ("  short leads (<= 0.3 TU)", a.short, c.short, b.short),
                                   ("  long leads (>= 0.6 TU)", a.long, c.long, b.long))
            @printf(io, "  %-26s %10s %10s %10s %12s %12s\n", name, fmt(fa), fmt(fc), fmt(fb),
                    signed4(fc - fa), signed4(fb - fc))
        end
        @printf(io, "  %-26s %10d %10d %10d %12s %12s\n", "in S7's band / $(a.ncells)",
                a.in_band, c.in_band, b.in_band,
                @sprintf("%+d", c.in_band - a.in_band), @sprintf("%+d", b.in_band - c.in_band))
        @printf(io, "  %-26s %10s %10s %10s %12s %12s\n", "median spread-skill",
                fmt(a.median_ratio), fmt(c.median_ratio), fmt(b.median_ratio),
                signed4(c.median_ratio - a.median_ratio), signed4(b.median_ratio - c.median_ratio))
    end

    println(io, "\n", "="^104)
    println(io, "The front, under each policy. Stability is a property of the RUN and does not move:")
    println(io, "member 0.9967 / 1.0000 / 1.0000, IC 0.9667 / 1.0000 / 1.0000 under all three.")
    for (tag, suffix, what) in POLICIES
        k0 = summarise(have[MODELS[1] * suffix])
        println(io, "\n  policy $tag -- $what  (K = $(k0.K), M = $(k0.M))")
        @printf(io, "    %-10s %14s %14s %18s\n", "", "mean skill", "in band", "more skilful?")
        # Each policy is internally consistent, so the front is read on its own file's denominator.
        ss = Dict(m => summarise(have[m * suffix]) for m in MODELS)
        best = argmin(m -> ss[m].mean_skill, MODELS)
        for m in MODELS
            @printf(io, "    %-10s %14s %14s %18s\n", m, fmt(ss[m].mean_skill),
                    "$(ss[m].in_band)/$(ss[m].ncells)", m == best ? "<-- best" : "")
        end
        d = ss["LinReg7"].mean_skill - ss["LinReg1"].mean_skill
        @printf(io, "    LinReg7 - LinReg1 = %s (%s of LinReg1) -- %s\n", signed4(d),
                pct(ss["LinReg7"].mean_skill, ss["LinReg1"].mean_skill),
                d > 0 ? "LinReg1 more skilful" : "LinReg7 more skilful")
    end

    println(io, "\n⚠️ Read the accuracy axis against the stability axis, never alone. Under either " *
                "policy\n   LinReg1 and LinReg7 are two points on S2''s front; what the policy can " *
                "change is\n   which of them holds the accuracy end of it.")
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    report()
end
