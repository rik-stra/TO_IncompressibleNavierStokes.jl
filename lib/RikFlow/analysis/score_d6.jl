# Score D6: metric #17 lead-resolved spread-skill, and RH-3 lead-resolved rank histograms.
#
# The last two baseline metrics, and the only two D6 exists for. Both are expectations over
# **initial conditions** at a fixed lead, and every archived online run is one trajectory from one
# initial condition (`6_online_TO_LRS.jl:56-59`), so with the archive alone each lead has exactly one
# verification instance and an RMSE from one sample is not an RMSE.
#
# Usage:
#   julia --startup-file=no --project=analysis analysis/score_d6.jl            # score what is there
#   julia --startup-file=no --project=analysis analysis/score_d6.jl --preview  # design only, no runs
#
# Writes `analysis/output/d6_scores.jld2` and prints the tables that go into
# `analysis/results.md`. ⚠️ The report belongs in `results.md`, beside the script that made it --
# not in `meta_files/`, and not left in `analysis/output/`, which `.gitignore:12` excludes.
#
# ---------------------------------------------------------------------------------------------
# Three decisions that determine whether the numbers mean anything
# ---------------------------------------------------------------------------------------------
#
# **Scored on the QoI level `q`, with the correction `dQ` reported beside it.** "Score `dQ`, never
# `q`" is regime-B-scoped and applying it here would be an error (`claude_memory.md` gotcha #27).
# D6 is regime C -- free-running, no replayed predictor stream -- and free-running the level is what
# the claims are about: the long-term QoI distribution, the QoI decorrelation time, the QoI spread.
# The choice changes verdicts: LinReg73 reads 0.997 on the correction and **0.809** on the level.
# The correction is reported as secondary because the level's own temporal statistic is null
# (gotcha #28), so a level-only temporal claim would make every configuration look equally good.
#
# **Leads are per QoI and in physical time.** `T_int` spans 0.2489-0.5430 TU across the six bands, so
# one grid in units of `t_int` cannot serve them all. The grid is `{0.25, 0.5, 1, 2, 5, 10} x
# T_int(i)`; the largest entry, 2172 steps, is what sets the 2172-step forecast, and short leads are
# free within a run. 🔴 These are the LEVEL's timescales since 2026-09-16 -- see `T_INT`.
#
# **Truth is the high-fidelity reference, not the tracked record.** The ICs are cut from the tracked
# record's velocity fields, but the tracked run is an LF simulation nudged onto the reference, and
# what a forecast should be scored against is the reference itself. The two agree to 3e-5-2.3e-3 of
# a standard deviation (`results.md` section 1), so this is a small choice -- but it is a choice, and
# `TRUTH_SOURCE` names it.

using LinearAlgebra
using Statistics
using Random
using Printf
using JLD2
using Dates

const HERE = @__DIR__
const SRC = normpath(joinpath(HERE, "..", "src"))
include(joinpath(SRC, "ts_score.jl"))
include(joinpath(HERE, "extract_qois.jl"))
include(joinpath(HERE, "build_d6_ics.jl"))
# `load_ensemble`, for the ordinal-0 validation comparison against the archived LinReg1 runs.
include(joinpath(HERE, "extract_archive.jl"))
# `load_rebaseline`, for the ordinal-0 validation comparison against R2's own LinReg1 runs.
include(joinpath(HERE, "extract_rebaseline.jl"))

const OUT = joinpath(HERE, "output")
const DT = 2.5e-3                      # HIT LES time step, TU
const SEED = 20260909

"QoI band labels. HIT shells are [0,6] [7,15] [16,32]; rows alternate enstrophy / energy."
const LABELS = ["Z[0,6]", "E[0,6]", "Z[7,15]", "E[7,15]", "Z[16,32]", "E[16,32]"]

"""
Index of `Z[16,32]`, named because it is excluded from the validation verdict.

Commit `09954be1` (2025-06-04, *"exclude derivative of nyqist freq"*) changed how that QoI is
computed and **every archived record predates it**, so it is a different quantity today
(`claude_memory.md` gotcha #45). Derived with `findfirst` rather than hard-coded, so a change to the
QoI set cannot leave a stale `5` behind pointing at the wrong band.
"""
const IZ1632 = something(findfirst(==("Z[16,32]"), LABELS))

"""
Integral timescale per QoI, in TU, measured on the reference `dQ` (`analysis/results.md` section 1).

🔴 Not one number. These span a factor 36.8, and `T_exp` disagrees with `T_int` by up to 4x within a
single QoI. Every document in the project said 0.04 TU; neither estimator gives that
(`claude_memory.md` gotcha #30).

🔴 **These are the LEVEL's decorrelation times on R1, changed 2026-09-16 (Rik).** They used to be
the **correction's**, on the **archive**: `[0.1118, 0.0082, 0.0923, 0.0669, 0.2926, 0.3017]`.

D6 scores the forecast of the QoI *level*, so the level's timescale is what has to saturate, and
the old grid's longest lead (1207 steps) was only 5.6x the slowest level timescale -- a grid that
could have stopped before the slowest band saturated. The old docstring said as much: *"the level
decorrelates far more slowly, so the level's saturation lead may lie beyond 10 x T_int"*. It now
does not.

🔑 `10 x max(T_INT) / dt = 10 x 0.5430 / 2.5e-3 = 2172` steps exactly, which is `N_LEAD`. The two
constants are derived from the same number and must move together; `build_d6_ics.jl`'s `T_INT_MAX`
is the other half.

⚠️ Still a decay constant, not `1 + 2*sum(rho)`. See `build_d6_ics.jl`'s `T_INT_MAX`.
"""
const T_INT = [0.2489, 0.4732, 0.4893, 0.4742, 0.5430, 0.5395]

"Which record supplies the verification truth. See the header."
const TRUTH_SOURCE = get(ENV, "D6_TRUTH", "hf_reference")

const D6_DIR = get(ENV, "D6_OUT",
                   normpath(joinpath(HERE, "..", "exp_square_HIT", "output", "D6")))

# ---------------------------------------------------------------------------------------------
# index alignment
# ---------------------------------------------------------------------------------------------
#
# 🔑 Asserted, not assumed, and tested against planted step indices in `test/test_d6_score.jl`.
# Run step `s` is tracked step `n_k + s`; the forecast's `t = 0` is at run step `nwarm`, because the
# first `nwarm` steps replay the record's correction rather than predicting it.

"""
    forecast_column(lead; nwarm = N_WARM)

Column of a run's own `q` holding lead `lead`. Run step `nwarm + lead`, and `q` carries the initial
state, so column `nwarm + lead + 1` (`RikFlow.jl:294`).
"""
forecast_column(lead::Integer; nwarm::Integer = N_WARM) = nwarm + lead + 1

"""
    truth_column(n_k, lead; nwarm = N_WARM)

Column of the reference `q` verifying lead `lead` of a run launched from step `n_k`: tracked step
`n_k + nwarm + lead`, plus the same initial-state offset.
"""
truth_column(n_k::Integer, lead::Integer; nwarm::Integer = N_WARM) = n_k + nwarm + lead + 1

"""
    forecast_column_dq(lead; nwarm = N_WARM)
    truth_column_dq(n_k, lead; nwarm = N_WARM)

The same two, for `dQ`. `dQ` has **no** initial-state column -- it is `nstep` long against `q`'s
`nstep + 1` -- so both offsets drop by one. Keeping the two pairs separate is deliberate: a single
"column" helper shared between `q` and `dQ` is precisely the off-by-one that would misalign the
secondary metric while the primary one looked right.
"""
forecast_column_dq(lead::Integer; nwarm::Integer = N_WARM) = nwarm + lead
truth_column_dq(n_k::Integer, lead::Integer; nwarm::Integer = N_WARM) = n_k + nwarm + lead

# ---------------------------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------------------------

"""
    load_members(dir = D6_DIR)

Every `d6_online_ic<k>_m<member>.jld2` in `dir`, grouped by IC and sorted by member.

Refuses a ragged ensemble: `spread_skill`'s finite-`M` correction is a function of `M`, so an IC
with fewer members than the rest would be silently down-weighted and its correction wrong.
"""
function load_members(dir = D6_DIR)
    isdir(dir) || return nothing
    pat = r"^d6_online_ic(\d+)_m(\d+)\.jld2$"
    byic = Dict{Int,Vector{Tuple{Int,String}}}()
    for f in readdir(dir)
        m = match(pat, f)
        m === nothing && continue
        push!(get!(byic, parse(Int, m[1]), Tuple{Int,String}[]),
              (parse(Int, m[2]), joinpath(dir, f)))
    end
    isempty(byic) && return nothing
    ks = sort(collect(keys(byic)))
    Ms = [length(byic[k]) for k in ks]
    allequal(Ms) || error("ragged ensemble: members per IC are $(sort(unique(Ms))). " *
                          "The finite-M correction is a function of M; fix the runs, do not average.")
    for k in ks
        sort!(byic[k], by = first)
        first.(byic[k]) == collect(1:Ms[1]) || error("IC $k has member ids $(first.(byic[k]))")
    end
    return (; ks, M = Ms[1], files = byic)
end

"Relative deviation, in units of each QoI's own sd, allowed across the replayed warm-up window."
const VALID_GATE_TOL = 1e-2

"Relative deviation at which two trajectories are called separated, for reporting the split column."
const VALID_DIVERGE_TOL = 1e-1

"""
    validation_verdict(qr, qa, dqr, dqa, nwarm; gate_tol, diverge_tol, iexcl)

Decide whether one validation member reproduces its archived replica, and say where it stops.

🔴 **Why this is not an rms over the whole run, which is what it used to be.** The previous criterion
was `max` per-QoI relative rms over all 1309 columns `< 1e-2`. Measured 2026-09-11, that criterion
is **unreachable by construction** and was reporting correct behaviour as failure:

  * **Two archived replicas of LinReg1** -- same code, same inputs, different seed -- are
    **1.08 sd apart** on that statistic. The run scored 0.98-2.5. So the aggregate saturates at the
    seed-to-seed scale and has no resolution left past roughly column 200; asking it for `1e-2`
    demands near-bit-identity from a chaotic, stochastic, Float32 system.
  * The docstring's premise -- *every input is the archive's, so the output must be too* -- is
    **false** after commit `09954be1` (2025-06-04, *"exclude derivative of nyqist freq"*). That
    commit zeroes the Nyquist wavenumber before `∂` is built, and `∂` feeds `get_vi_functions`
    (`RikFlow.jl:141`), so the Z-QoI direction vectors and hence `tau` changed with it. The run is
    **not the same dynamical system** as the archive, whatever inputs it is given
    (`claude_memory.md` gotchas #45, #46).

So the check is moved to where it has power. Two windows, and they test different things:

 1. **The warm-up, columns `1:nwarm`.** Here the sampler emits `spinnup_data` -- the archive's own
    `dQ[:, 1:nwarm]` -- verbatim, so `dqr` must be **bit-identical** to `dqa` there. That is exact,
    and it is the sharpest single check in the whole D6 path: it proves the IC package's warm-up
    slice, the history layout and the deployment wiring at once. It was never asserted before.
    With `dQ` pinned, any drift in `q` over the same window is the solver and the forcing alone --
    IC alignment, `ou_advance` at its identity point, `tau`. Measured 3.9e-3 sd at column 100
    against `gate_tol = 1e-2`; a wrong slice, a misphased chain or a seed mismatch is `O(1)` here.
 2. **Past the warm-up** the sampler runs, the trajectories separate, and separation is *expected*.
    Reported, never gated: the first column at which the deviation crosses `diverge_tol`, and the
    full-window rms **beside the archive's own replica-to-replica rms**, which is the scale at which
    that statistic saturates.

⚠️ The model-testing window is therefore narrow, and saying so is part of the result: columns
`1:nwarm` are replayed and test nothing about the sampler, and by ~50 columns past `nwarm` the pair
has decorrelated. Reproducing the archive's *trajectory* is not something this design can ask for.

🔴 **`iexcl` defaults to `nothing` since 2026-09-16: NOTHING is excluded.** It existed to drop
`Z[16,32]`, a different quantity today than when the archive
was written (gotcha #45), off by ~1.06e-3 relative on an identical velocity field, so including it
would report a known convention change as a defect. It is still measured and printed.

Returns `(; ok, dq_identical, gate, diverge_col, rel_full, rel_warm, nwarm, n)`.
"""
function validation_verdict(qr::AbstractMatrix, qa::AbstractMatrix,
                            dqr::AbstractMatrix, dqa::AbstractMatrix, nwarm::Integer;
                            gate_tol::Real = VALID_GATE_TOL,
                            diverge_tol::Real = VALID_DIVERGE_TOL,
                            iexcl::Union{Integer,Nothing} = nothing)
    n = min(size(qr, 2), size(qa, 2))
    nw = min(Int(nwarm), n)
    sd = vec(std(view(qa, :, 1:n); dims = 2))
    dev(c) = abs.(view(qr, :, c) .- view(qa, :, c)) ./ sd
    keep = iexcl === nothing ? collect(axes(qr, 1)) : [i for i in axes(qr, 1) if i != iexcl]

    # The exact half: over the replayed window the sampler is emitting stored numbers, so anything
    # but bit-identity means the warm-up slice or the history layout is wrong.
    nd = min(size(dqr, 2), size(dqa, 2), nw)
    dq_identical = nd > 0 && view(dqr, :, 1:nd) == view(dqa, :, 1:nd)

    rel_warm = vec(sqrt.(mean(abs2, view(qr, :, 1:nw) .- view(qa, :, 1:nw); dims = 2))) ./ sd
    rel_full = vec(sqrt.(mean(abs2, view(qr, :, 1:n) .- view(qa, :, 1:n); dims = 2))) ./ sd
    gate = maximum(maximum(dev(c)[keep]) for c in 1:nw)
    dcol = findfirst(c -> maximum(dev(c)[keep]) > diverge_tol, 1:n)

    return (; ok = dq_identical && gate <= gate_tol, dq_identical, gate,
            diverge_col = dcol, rel_full, rel_warm, nwarm = nw, n)
end

"""
    replica_spread(ens, n)

Relative rms between distinct archived replicas of one configuration over their first `n` columns,
per QoI, as `(; median, lo, hi, npairs)`.

🔑 **This is the number that makes the validation report readable.** The replicas differ only in
their model seed, so this is what "as different as two correct runs of the same thing" measures on
the same statistic -- 1.08 sd on LinReg1. Printing the run-vs-archive rms without it invites reading
a saturated statistic as a defect, which is exactly what happened on 2026-09-11.
"""
function replica_spread(ens, n::Integer)
    R = length(ens.q)
    vals = Float64[]
    npairs = 0
    for i in 1:R, j in (i + 1):R
        a, b = Float64.(ens.q[i]), Float64.(ens.q[j])
        m = min(n, size(a, 2), size(b, 2))
        sd = vec(std(view(a, :, 1:m); dims = 2))
        append!(vals, vec(sqrt.(mean(abs2, view(a, :, 1:m) .- view(b, :, 1:m); dims = 2))) ./ sd)
        npairs += 1
    end
    isempty(vals) && return (; median = NaN, lo = NaN, hi = NaN, npairs = 0)
    s = sort(vals)
    med = isodd(length(s)) ? s[(length(s) + 1) ÷ 2] :
          (s[length(s) ÷ 2] + s[length(s) ÷ 2 + 1]) / 2
    return (; median = med, lo = first(s), hi = last(s), npairs)
end

"""
    compare_validation(; dir = D6_DIR, io = stdout)

Check the validation run (ordinal 0) against the archived online ensemble it launched from.

The validation IC is `fields[1]` of the 10 TU tracked record -- the initial condition every archived
online run started from (`paper_runs/online_sgs.jl:50`), so `n_k = 0`, `ou_advance = 0` is the
identity point of the replay, and `run_d6.jl` gives its members the archive's own model seeds
`Xoshiro(236 + member)`. What that buys, and what it does not, is `validation_verdict`'s docstring;
the short version is that the **replayed warm-up window** is the part with a right answer, and the
free-running remainder is reported rather than gated.

`Z[16,32]` is excluded from the verdict and flagged wherever it appears (gotcha #45).
"""
function compare_validation(; dir = D6_DIR, io = stdout)
    isdir(dir) || (println(io, "no run directory at $dir"); return nothing)
    pat = r"^d6_valid_ic(\d+)_m(\d+)\.jld2$"
    files = sort([(parse(Int, m[2]), joinpath(dir, f))
                  for f in readdir(dir) for m in (match(pat, f),) if m !== nothing])
    if isempty(files)
        println(io, "no validation runs in $dir (run `tools/run_d6.jl 0`)")
        return nothing
    end
    arch = try
        load_rebaseline("LinReg1")
    catch err
        println(io, "no rebaselined LinReg1 ensemble to compare against: ", err)
        return nothing
    end

    println(io, "\nValidation: ordinal 0 against R2's own LinReg1 ensemble (not the archive)")
    @printf(io, "  oracle: %s, %d replicas\n", arch.label, length(arch.q))
    rows = NamedTuple[]
    for (member, path) in files
        d = load(path)
        get(d, "validation", false) ||
            error("$path is not marked as a validation run; the glob picked up a scored file")
        member <= length(arch.q) ||
            (println(io, "  member $member has no archived replica"); continue)
        v = validation_verdict(Float64.(d["q"]), Float64.(arch.q[member]),
                               Float64.(d["dQ"]), Float64.(arch.dQ[member]), d["nwarm"])
        push!(rows, (; member, seed = d["seed"], v...))
    end
    isempty(rows) && return rows

    nw, n = rows[1].nwarm, rows[1].n
    println(io, "\n  GATE -- the replayed warm-up, columns 1:$nw. `dQ` is the record's own slice")
    println(io, "  emitted verbatim, so it must be bit-identical; `q` then moves under the solver,")
    println(io, "  the OU forcing and `tau` alone.")
    @printf(io, "  %8s %10s %14s %14s   %s\n",
            "member", "dQ ident", "max deviation", "verdict", "per-QoI rel rms over warm-up")
    for r in rows
        @printf(io, "  %8d %10s %14.3e %14s   %s\n", r.member, r.dq_identical, r.gate,
                r.ok ? "pass" : "FAIL",
                join((@sprintf("%8.1e", x) for x in r.rel_warm), " "))
    end

    println(io, "\n  REPORTED, not gated -- past the warm-up the sampler runs and the pair separates.")
    sp = replica_spread(arch, n)
    @printf(io, "  %8s %14s %14s   %s\n",
            "member", "diverges at", "rel rms 1:$n", "per-QoI rel rms over the full window")
    for r in rows
        @printf(io, "  %8d %14s %14.3e   %s%s\n", r.member,
                r.diverge_col === nothing ? "never" : string(r.diverge_col),
                maximum(r.rel_full),
                join((@sprintf("%8.1e", x) for x in r.rel_full), " "), "")
    end
    @printf(io, "  scale: two R2 replicas of this configuration are %.2f apart on the same\n",
            sp.median)
    @printf(io, "         statistic (range %.2f-%.2f over %d pairs), so a full-window rms near that\n",
            sp.lo, sp.hi, sp.npairs)
    println(io, "         value is saturation, not a defect. Only the gate above is a verdict.")

    npass = count(r -> r.ok, rows)
    @printf(io, "\n  => %d of %d members pass the warm-up gate (tol %.0e)\n",
            npass, length(rows), VALID_GATE_TOL)
    println(io, npass == length(rows) ?
            "  ✅ IC packaging, warm-up slice, history layout, ou_advance and the solver path all\n" *
            "     reproduce the archive over the window that has a right answer." :
            "  🔴 the gate failed. Check the warm-up slice, ou_advance and the seed before " *
            "trusting\n     anything scored -- see `validation_verdict`'s docstring.")
    return rows
end

"""
    load_truth(source = TRUTH_SOURCE)

The verification truth as `(; q, dQ)`. `q` is `N_Q x 40001`; `dQ` is `N_Q x 40000` and comes from
the tracked record in either case, because the high-fidelity reference has no correction of its own
-- `dQ` is a property of the tracking run.
"""
function load_truth(source = TRUTH_SOURCE)
    dd = joinpath(HERE, "data")
    # 🔴 R1's record and the REGENERATED reference, not the archive's (2026-09-16).
    #
    # D6's ICs are cut from R1 (`build_d6_ics.jl`). Scoring forecasts launched from R1 against paper
    # 2's archived truth would compare a run of one dynamical system against another (memory #45,
    # #46) -- and nothing about the output would have looked wrong. This was the last place the
    # rebaselined D6 path still reached into the archive.
    trk = joinpath(dd, "data_track_dns512_les64_Re2000.0_tsim100.0_f64_lmwray3_qois.jld2")
    isfile(trk) || error("no extracted R1 tracked record at $trk; run analysis/extract_qois.jl " *
                         "on exp_square_HIT/output/data_track_..._f64_lmwray3.jld2")
    t = load(trk)
    if source == "hf_reference"
        hf = joinpath(dd, "hf_reference_new_tsim100.0_f64_lmwray3_qois.jld2")
        isfile(hf) || error("no extracted regenerated HF reference at $hf; run " *
                            "analysis/extract_archive.jl new-reference")
        return (; q = load(hf, "q_ref"), dQ = t["dQ"], source)
    elseif source == "tracked"
        return (; q = t["q"], dQ = t["dQ"], source)
    else
        error("D6_TRUTH must be \"hf_reference\" or \"tracked\", got $(repr(source))")
    end
end

"""
    assemble(ens, truth, grid; level = true)

Build `(fc, truth)` as `K x N_Q x M x L` and `K x N_Q x L` on the sorted `grid` of step leads.

`level = true` takes the QoI level `q` from both sides, `false` the correction `dQ`; the two use
different column offsets and that is the point of them being separate functions above.

Every column index is bounds-checked against the record. A lead that runs past the reference throws
rather than being clipped, because a clipped lead reads as a saturated one.
"""
function assemble(ens, truth, grid::AbstractVector{<:Integer}; level::Bool = true)
    K = length(ens.ks)
    M = ens.M
    L = length(grid)
    ref = level ? truth.q : truth.dQ
    nq = size(ref, 1)
    fc = Array{Float64}(undef, K, nq, M, L)
    tr = Array{Float64}(undef, K, nq, L)
    fcol = level ? forecast_column : forecast_column_dq
    tcol = level ? truth_column : truth_column_dq

    for (ik, k) in pairs(ens.ks)
        n_k = nothing
        warm = nothing
        for (im, (mid, path)) in pairs(ens.files[k])
            d = load(path)
            n_k === nothing && (n_k = d["n_k"])
            warm === nothing && (warm = d["nwarm"])
            d["n_k"] == n_k || error("members of IC $k disagree on n_k")
            # V28: the burn-in offset must be identical across all members of one IC. If it were
            # not, lead 0 would mean a different forecast time for different members and the
            # ensemble at a "lead" would not be an ensemble at a lead at all.
            d["nwarm"] == warm || error("members of IC $k disagree on nwarm ($warm vs $(d["nwarm"]))")
            d["k"] == k || error("$path says k = $(d["k"])")
            x = level ? d["q"] : d["dQ"]
            nwarm = d["nwarm"]
            for (j, ℓ) in pairs(grid)
                c = fcol(ℓ; nwarm)
                c <= size(x, 2) || error("lead $ℓ needs column $c of a $(size(x, 2))-column run")
                fc[ik, :, im, j] = @view x[:, c]
            end
            if im == 1
                for (j, ℓ) in pairs(grid)
                    c = tcol(n_k, ℓ; nwarm)
                    c <= size(ref, 2) ||
                        error("lead $ℓ from IC k = $k needs reference column $c of " *
                              "$(size(ref, 2)); the IC pool cap was violated")
                    tr[ik, :, j] = @view ref[:, c]
                end
            end
        end
    end
    return fc, tr
end

# ---------------------------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------------------------

fmt_lead(ℓ) = @sprintf("%d (%.4f TU)", ℓ, ℓ * DT)

"""
    tie_count(fc, tr, i, j)

How often the truth exactly equals one of the members, for QoI `i` at grid column `j`.

Trap 7 of the handoff: `metrics.md` #4 says the inherited stabiliser "produces exact duplicates by
construction", but the clamp has been measured never to fire on HIT -- min `|q*|` is 2.19e-2 against
a 1e-2 threshold -- so ties should be **absent**. The seeded tie-breaking stays as insurance and the
count is reported, because a non-zero count here would mean the census is wrong.
"""
function tie_count(fc::AbstractArray{<:Real,4}, tr::AbstractArray{<:Real,3}, i::Integer, j::Integer)
    K, _, M, _ = size(fc)
    n = 0
    for k in 1:K
        any(m -> fc[k, i, m, j] == tr[k, i, j], 1:M) && (n += 1)
    end
    return n
end

"""
    clamp_report(ens)

How often the inherited stabiliser fired during the forecast, counted **exactly**.

🔑 The clamp is `any(abs.(q_star) .< 1e-2) && (dQ .= 0)` in the `LinReg` path
(`time_series_methods.jl:162,165,190,193`), so a step on which it fired has a `dQ` column that is
**identically zero**. That is a direct, exact indicator and needs no `q_star`.

⚠️ It needs one, because `online_sgs` does not return `q_star`: `allocate_arrays_outputs` stores it
only in `:TRACK_REF` mode (`RikFlow.jl:110-119`), and `to_sgs_term` is out of scope to change.
`clamp_census` -- which wants `q_star` -- can still be run on a reconstruction
`q_star ~ q[:, 2:end] - dQ`, accurate to the O(||sgs||^2) gap of 5.9e-6-5.7e-4 relative (phase-0
check 0.4), which against a 1e-2 threshold and a measured minimum `|q*|` of 2.19e-2 cannot flip the
census. The exact zero-`dQ` count below is preferred anyway, and reported per IC.

The warm-up columns are excluded: there `dQ` is replayed from the record, not predicted.
"""
function clamp_report(ens)
    fired = Int[]
    total = 0
    for k in ens.ks, (_, path) in ens.files[k]
        dQ = load(path, "dQ")
        nwarm = load(path, "nwarm")
        n = 0
        for c in (nwarm + 1):size(dQ, 2)
            all(iszero, @view dQ[:, c]) && (n += 1)
        end
        push!(fired, n)
        total += size(dQ, 2) - nwarm
    end
    return (; nfired = sum(fired), nsteps = total, per_run = fired,
            rate = total == 0 ? NaN : sum(fired) / total)
end

"""
    report_grids(leads; io = stdout)

The lead grids themselves, which are a result: they say what the runs can and cannot resolve.
"""
function report_grids(leads; io = stdout)
    println(io, "\nLead grids -- {0.25, 0.5, 1, 2, 5, 10} x T_int(i), per QoI, in physical time")
    @printf(io, "  %-10s %8s   %s\n", "QoI", "T_int", "leads [steps]")
    for i in eachindex(leads)
        @printf(io, "  %-10s %8.4f   %s\n", LABELS[i], T_INT[i], join(leads[i], ", "))
    end
    @printf(io, "  union: %d distinct leads, longest %s of %d available\n",
            length(union_grid(leads)), fmt_lead(maximum(union_grid(leads))), N_LEAD)
end

"""
    report_scores(ss, rh, sat; label, io = stdout)

Metric #17 and RH-3 side by side, per QoI and lead.

The two belong in one table because neither is readable alone: a flat histogram is reliability, not
skill, and a ratio near 1 says nothing about whether the shape is right.
"""
function report_scores(ss, rh, sat; label, io = stdout)
    println(io, "\n", label, "  (K = ", ss.K, ", M = ", ss.M,
            ", finite-M correction ", @sprintf("%.4f", ss.correction), ")")
    for i in eachindex(ss.leads)
        s = sat[i]
        @printf(io, "\n  %-10s  saturation lead: %s\n", LABELS[i],
                s === nothing ? "NOT REACHED within the grid -- reported, not extrapolated" :
                fmt_lead(s))
        @printf(io, "    %10s %10s %10s %8s %8s %18s %18s\n",
                "lead", "spread", "skill", "ratio", "chi2_eff", "slope [95% CI]",
                "convexity [95% CI]")
        for (t, ℓ) in pairs(ss.leads[i])
            h = rh.hist[i][t]
            @printf(io, "    %10d %10.4g %10.4g %8.3f %8.1f  %6.2f [%6.2f,%6.2f]  %6.2f [%6.2f,%6.2f]\n",
                    ℓ, ss.spread[i][t], ss.skill[i][t], ss.ratio[i][t], h.chi2_eff,
                    h.slope, h.slope_ci[1], h.slope_ci[2],
                    h.convexity, h.convexity_ci[1], h.convexity_ci[2])
        end
    end
end

"""
    saturation(ss, truth_q, M)

The saturation lead per QoI, and the climatological level it is measured against.
"""
function saturation(ss, ref::AbstractMatrix, M::Integer)
    lev = [climatological_skill(collect(float.(view(ref, i, :))), M) for i in 1:size(ref, 1)]
    sat = [saturation_lead(ss.leads[i], ss.skill[i]; sat_level = lev[i]) for i in eachindex(lev)]
    return sat, lev
end

# ---------------------------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------------------------

"""
    preview(; io = stdout)

What the scorer will do, without any runs: the lead grids, and the index alignment written out for
one IC so the arithmetic can be read rather than trusted.
"""
function preview(; io = stdout)
    leads = lead_grid(T_INT; dt = DT, nlead = N_LEAD)
    report_grids(leads; io)
    sel = select_ics(; K = 180)
    k, n_k = sel.k[1], sel.n[1]
    println(io, "\nIndex alignment, worked for the first IC (ordinal 1, k = $k, n_k = $n_k)")
    @printf(io, "  %8s %14s %16s %14s %16s\n",
            "lead", "run q col", "reference q col", "run dQ col", "reference dQ col")
    for ℓ in (0, 1, leads[2][1], leads[6][end])
        @printf(io, "  %8d %14d %16d %14d %16d\n", ℓ,
                forecast_column(ℓ), truth_column(n_k, ℓ),
                forecast_column_dq(ℓ), truth_column_dq(n_k, ℓ))
    end
    @printf(io, "  last reference column used: %d of %d\n",
            truth_column(last(sel.n), maximum(union_grid(leads))), N_REF + 1)
    println(io, "\nNo scored runs found. Submit exp_square_HIT/batch_scripts/run_d6.sh, pull the " *
                "output back into\n$(D6_DIR), then re-run this script.")
    # The validation run is independent of the scored set and worth reporting on its own, because
    # it is the check that has to pass before the scored numbers mean anything.
    compare_validation(; io)
    return leads
end

"""
    main(; dir = D6_DIR, preview_only = false, outdir = OUT, io = stdout)

Score whatever runs are in `dir`, print the tables to `io`, and write `d6_scores.jld2` under
`outdir`.

`outdir` and `io` are parameters so `test/test_d6_score.jl` can drive this whole path on synthetic
members. That matters more than it looks: without it, the first execution of `main` would be on
pilot data, which is to say after GPU time had already been spent.
"""
function main(; dir = D6_DIR, preview_only::Bool = false, outdir = OUT, io = stdout)
    leads = lead_grid(T_INT; dt = DT, nlead = N_LEAD)
    ens = preview_only ? nothing : load_members(dir)
    if ens === nothing
        preview(; io)
        return nothing
    end

    grid = union_grid(leads)
    truth = load_truth()
    report_grids(leads; io)
    @printf(io, "\nD6: %d initial conditions x %d members from %s\n", length(ens.ks), ens.M, dir)
    @printf(io, "truth: %s\n", truth.source)

    # The validation run first: if the D6 path does not reproduce the archived trajectory from the
    # archived inputs, nothing below is worth reading.
    compare_validation(; dir, io)

    cl = clamp_report(ens)
    @printf(io, "clamp: fired on %d of %d forecast steps (%.3g%%)%s\n",
            cl.nfired, cl.nsteps, 100 * cl.rate,
            cl.nfired == 0 ? " -- as measured everywhere else on HIT, it never fires" :
            " ⚠️ every number below is then partly the stabiliser's, not the model's")

    rng = Xoshiro(SEED)
    pos = lead_positions(grid, leads)
    out = Dict{Symbol,Any}()
    for (label, level) in (("LEVEL q  -- primary (gotcha #27)", true),
                           ("CORRECTION dQ -- secondary (gotcha #28)", false))
        fc, tr = assemble(ens, truth, grid; level)
        ss = spread_skill_by_lead(fc, tr; grid, leads)
        rh = rank_histogram_by_lead(fc, tr; grid, leads, rng)
        ref = level ? truth.q : truth.dQ
        sat, lev = saturation(ss, ref, ens.M)
        report_scores(ss, rh, sat; label, io)
        tag = level ? :level : :correction
        out[tag] = (; leads, grid, ss.ratio, ss.spread, ss.skill, ss.pooled_ratio,
                    ss.K, ss.M, ss.correction,
                    saturation = [s === nothing ? -1 : s for s in sat],
                    clim_level = lev,
                    counts = [[h.counts for h in hs] for hs in rh.hist],
                    slope = [[h.slope for h in hs] for hs in rh.hist],
                    slope_ci = [[collect(h.slope_ci) for h in hs] for hs in rh.hist],
                    convexity = [[h.convexity for h in hs] for hs in rh.hist],
                    convexity_ci = [[collect(h.convexity_ci) for h in hs] for hs in rh.hist],
                    chi2_eff = [[h.chi2_eff for h in hs] for hs in rh.hist],
                    n_eff = [[h.n_eff for h in hs] for hs in rh.hist],
                    blocklen = [[h.blocklen for h in hs] for hs in rh.hist],
                    ties = [[tie_count(fc, tr, i, j) for j in pos[i]] for i in eachindex(leads)])
        nties = sum(sum.(out[tag].ties))
        @printf(io, "\n  ties (truth exactly equal to a member): %d of %d instances.%s\n",
                nties, ss.K * sum(length, leads),
                nties == 0 ? " As expected -- the clamp never fires on HIT." :
                " ⚠️ Unexpected; metrics.md #4's duplicate-by-construction case was ruled out.")
        nsat = count(s -> s !== nothing, sat)
        @printf(io, "\n  %d of %d QoIs saturate inside the grid.%s\n", nsat, length(sat),
                nsat == length(sat) ? "" :
                " ⚠️ The rest are reported as not reached; do not extrapolate.")
    end

    mkpath(outdir)
    p = joinpath(outdir, "d6_scores.jld2")
    # Named explicitly rather than splatted: `jldsave`'s keywords must be symbols, and a `Dict`
    # splat is the kind of thing that works until the dictionary's key type changes.
    jldsave(p; level = out[:level], correction = out[:correction],
            labels = LABELS, T_int = T_INT, dt = DT, truth = truth.source,
            ics = ens.ks, clamp_nfired = cl.nfired, clamp_nsteps = cl.nsteps,
            written = string(now()))
    @printf(io, "\nwrote %s (%.1f kB)\n", p, filesize(p) / 1024)
    println(io, "⚠️  The report goes into `analysis/results.md`, not into meta_files/ and not left " *
            "here:\n    `.gitignore:12` is `*output/`.")
    return out
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(; preview_only = ("--preview" in ARGS))
end
