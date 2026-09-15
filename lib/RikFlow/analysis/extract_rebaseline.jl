# Extract the QoI time series from the REBASELINED online runs (P2r's R2, 2026-09-15).
#
# Third sibling of `extract_qois.jl` (tracking records) and `extract_archive.jl` (paper 2's
# archive). This one handles the runs produced by the rebaselined pipeline on the merged solver:
# the four closures of `results.md` §4b, launched from the regenerated HF reference and the new
# 100 TU tracking record.
#
# 🔴 These are NOT the archive. `extract_archive.jl`'s `load_ensemble` reads paper 2's frozen runs
# on the pre-`09954be1` Nyquist convention; the files here are post-merge, Float64 throughout, and
# score against `load_new_reference()`. The two must never be pooled -- they measure different
# dynamical systems (memory #45, #46). Separate cache names keep that mistake from being one
# `load_ensemble(name)` away.
#
# Same reason for caching as the siblings: each online file is ~289 MB of which the QoI arrays are
# ~2 MB. JLD2 stores `data_online` as one compound dataset, so `fields` (41 snapshots of a 64^3 x 3
# velocity field) cannot be skipped on read. Pay it once.
#
# Usage:
#   julia --startup-file=no --project=analysis analysis/extract_rebaseline.jl           # all four
#   julia --startup-file=no --project=analysis analysis/extract_rebaseline.jl LinReg1   # just one

using JLD2, Printf, Dates

# `extract_qois.jl` and `extract_archive.jl` define this to the same value, and a driver includes
# more than one of the three, so an unconditional `const` would be a redefinition.
isdefined(@__MODULE__, :DATA_DIR) || const DATA_DIR = joinpath(@__DIR__, "data")

"""
    REBASE_ROOT

Where the rebaselined runs live: `exp_square_HIT/output/`, inside this repository rather than in
the gitignored archive. Overridable with `RIKFLOW_REBASE_ROOT` so a copy on another machine (or a
staged pull off the cluster) can be scored without editing the script.
"""
const REBASE_ROOT = get(ENV, "RIKFLOW_REBASE_ROOT",
                        normpath(joinpath(@__DIR__, "..", "exp_square_HIT", "output")))

"""
    EXPECTED_STEPS

QoI columns a completed 100 TU run must have: `tsim / dt + 1 = 100 / 2.5e-3 + 1`.

🔴 **A replica that stops early still writes a perfectly valid file, and two of its arrays lie.**
`q` comes from `stack(outputs.qoihist)` and is genuinely short, but `dQ` and `tau` are
*preallocated* by `TO_Setup(; nstep)` and keep their full 40 000 columns, with everything past the
last completed step left at zero. Measured on `LinReg5` replica 5 (2026-09-15): it diverged at
t = 37.81 TU with `Z[0,6]` reaching 3.0e7, `q` is (6, 15125), and `dQ` columns 15074-40000 are
identically zero -- which a clamp census reads as **24 883 firings (62% of steps)** that never
happened. `compute_ks.jl:44` has always guarded this with `length(q_rep[i][1,:]) < 40000`; this
extractor did not, and every statistic downstream would have been wrong with nothing looking odd.
"""
const EXPECTED_STEPS = 40001

"""
    lrs_cell(key, label)

One TO-LRS entry. 🔴 **`dir` is DERIVED from `key`, never written out.**
`6_online_TO_LRS.jl:94` writes to `output/TO_LRS/<name>/`, so the directory and the configuration
name are the same string by construction. Repeating it by hand is how `LinReg5` and `LinReg6` came
to point at `TO_LRS/LinReg1` -- a directory that already holds the lambda = 0 runs, so the lambda
probe would have scored LinReg1's trajectories three times and labelled two of them 1e-5 and 1e-4.
That reads as "lambda makes no difference" and nothing in the output would have looked wrong.
"""
lrs_cell(key, label) =
    (; key, label, dir = joinpath("TO_LRS", key),
     pattern = r"^data_online_tsim100\.0_replica(\d+)\.jld2$",
     nominal = 5, stochastic = true, clampable = true)

"""
    REBASE_MODELS

The closures R2 scores, in the order `results.md` reports them: the ordering of the result
(best summed KS first), not alphabetical. `LinReg5`/`LinReg6` are the lambda probe and are skipped
with a warning until their online ensembles land.

`dir`/`pattern` locate the replica files; `nominal` is how many replicas the run was launched with,
so a missing file reads as an unstable replica rather than as a smaller ensemble -- that is how
`compute_ks.jl` has always decided stability and the count has to survive the extraction to be
reportable. `stochastic` records whether the closure carries a `dQ` stream at all: Smagorinsky and
no-model are deterministic LF runs with no TO correction, so their files have `q` and nothing else,
and every `dQ`-derived diagnostic is undefined for them rather than zero.

🔴 `clampable` is a **separate** flag and it is not redundant with `stochastic`. The stabilizer
`any(abs.(q_star) .< 1e-2) && (dQ .= 0)` lives only in the `LinReg` path
(`time_series_methods.jl:162,165,190,193`); `to_sgs_term` computes `q_star` at all only for
`ANN`/`LinReg`, and `MVG_sampler`'s `get_next_item_timeseries` never receives it. So the DDN carries
a `dQ` stream whose columns are never identically zero **because nothing can zero them**, and
reporting that as "clamped 0 times" beside a closure that clamps on 1.8% of steps states the
asymmetry backwards (memory #59).
"""
const REBASE_MODELS = (
    lrs_cell("LinReg1", "LinReg1 (h=5, lambda=0, :normal)"),
    # The precision-regularization probe (2026-09-15). Fitted and on disk; the online ensembles are
    # submitted through `batch_scripts/submit_lrs.sh` and are skipped with a warning until they land.
    lrs_cell("LinReg5", "LinReg5 (h=5, lambda=1e-5, :normal)"),
    lrs_cell("LinReg6", "LinReg6 (h=5, lambda=1e-4, :normal)"),
    (key = "DDN", label = "DDN",
     dir = "TO_DDN", pattern = r"^DDN_data_online_tsim100\.0_replica(\d+)\.jld2$",
     nominal = 5, stochastic = true, clampable = false),
    (key = "nomodel", label = "no model",
     dir = "no_model", pattern = r"^data_no_sgs_tsim100\.0\.jld2$",
     nominal = 1, stochastic = false, clampable = false),
    (key = "smag", label = "Smagorinsky c_s = 0.07",
     dir = "smag", pattern = r"^data_smag_0\.07_tsim100\.0\.jld2$",
     nominal = 1, stochastic = false, clampable = false),
)
"The spec for `key`, or an error naming what is available."
function rebase_model(key::AbstractString)
    i = findfirst(m -> m.key == key, REBASE_MODELS)
    i === nothing &&
        error("no rebaselined model $key; have " * join((m.key for m in REBASE_MODELS), ", "))
    return REBASE_MODELS[i]
end

"""
    rebase_replica_files(spec)

The replica files for `spec`, as `(index, path)` sorted by index.

The single-run baselines match a pattern with no capture group, so their index is 1 by position.
Anything matching but unparseable is an error rather than a skip: a silently dropped replica is
exactly the failure `replica_groups` exists to prevent in the archive.
"""
function rebase_replica_files(spec)
    dir = joinpath(REBASE_ROOT, spec.dir)
    isdir(dir) || return Tuple{Int,String}[]
    out = Tuple{Int,String}[]
    for f in sort(readdir(dir))
        m = match(spec.pattern, f)
        m === nothing && continue
        idx = isempty(m.captures) ? length(out) + 1 : parse(Int, m.captures[1])
        push!(out, (idx, joinpath(dir, f)))
    end
    return sort(out; by = first)
end

"""
    extract_rebaseline(key; force = false)

Cache one rebaselined closure's online QoIs as `analysis/data/online_new_<key>_qois.jld2`.

Stores `q` (one `N_Q x nstep+1` matrix per replica) and, for the stochastic closures, `dQ` and
`tau`. `q_star` is reconstructed the same way `extract_ensemble` does it --
`q_star[:, n] = q[:, n+1] - dQ[:, n]`, exact because `time_series_methods.jl:163-166` enforces
`dQ^n = q^n - q^{n*}` -- because `online_sgs` still does not return it (memory #35) and the clamp
census needs it.

`missing_replicas` records the launched-minus-present count, which is the stability fraction's
numerator and is not recoverable from the cache otherwise.
"""
function extract_rebaseline(key::AbstractString; force = false)
    spec = rebase_model(key)
    out = joinpath(DATA_DIR, "online_new_$(key)_qois.jld2")
    if isfile(out) && !force
        @printf("  exists, skipping: %s (%.2f MB)\n", basename(out), filesize(out) / 2^20)
        return out
    end
    entries = rebase_replica_files(spec)
    isempty(entries) && (@warn "no rebaselined runs for $key under $(joinpath(REBASE_ROOT, spec.dir))";
                         return nothing)
    mkpath(DATA_DIR)

    @printf("  %-10s %d of %d replica(s) present in %s\n", key, length(entries), spec.nominal,
            spec.dir)
    flush(stdout)

    qs, dQs, taus, srcs, bytes = Matrix{Float64}[], Matrix{Float64}[], Matrix{Float64}[],
                                 String[], Int[]
    kept = Int[]
    incomplete = NamedTuple[]
    t0 = time()
    for (i, p) in entries
        @printf("    replica %d  %-52s %.0f MB ... ", i, basename(p), filesize(p) / 2^20)
        flush(stdout)
        d = load(p, "data_online")
        q = Array{Float64}(d.q)
        if size(q, 2) < EXPECTED_STEPS
            # Stopped early. Report it loudly, keep the evidence, and keep it OUT of the ensemble:
            # `dQ` is preallocated, so this replica's tail is zeros that are not clamp firings, and
            # a short `q` is not a sample of the same length as the others.
            peak = maximum(abs, view(q, :, size(q, 2)))
            push!(incomplete, (; replica = i, nsteps = size(q, 2),
                               t_end = (size(q, 2) - 1) * 2.5e-3, peak_abs_q = peak,
                               source = abspath(p)))
            @printf("q = %s  INCOMPLETE (%.2f of 100 TU, |q|max %.3g) -- EXCLUDED\n",
                    size(q), (size(q, 2) - 1) * 2.5e-3, peak)
            flush(stdout)
            d = nothing
            GC.gc()
            continue
        end
        push!(qs, q)
        push!(kept, i)
        if spec.stochastic
            hasproperty(d, :dQ) || error("$p has no dQ but $key is declared stochastic")
            push!(dQs, Array{Float64}(d.dQ))
            push!(taus, hasproperty(d, :tau) ? Array{Float64}(d.tau) : zeros(Float64, 0, 0))
        end
        push!(srcs, abspath(p))
        push!(bytes, filesize(p))
        @printf("q = %s\n", size(q))
        flush(stdout)
        d = nothing
        GC.gc()                          # 285 MB of velocity snapshots per file
    end
    q_star = spec.stochastic ? [q[:, 2:end] .- dQ for (q, dQ) in zip(qs, dQs)] : Matrix{Float64}[]

    jldsave(out; q = qs, dQ = dQs, tau = taus, q_star,
            replica_index = kept, nominal_replicas = spec.nominal,
            missing_replicas = spec.nominal - length(entries),
            unstable_replicas = length(incomplete), incomplete,
            stochastic = spec.stochastic, clampable = spec.clampable,
            key = spec.key, label = spec.label,
            sources = srcs, source_bytes = bytes, root = abspath(REBASE_ROOT),
            extracted = string(now()))
    isempty(incomplete) ||
        @printf("    %d of %d replica(s) stopped early and were EXCLUDED: %s\n",
                length(incomplete), length(entries),
                join((@sprintf("r%d at %.1f TU", c.replica, c.t_end) for c in incomplete), ", "))
    @printf("    %d complete replica(s) in %.1f s; wrote %s (%.2f MB)\n", length(qs), time() - t0,
            basename(out), filesize(out) / 2^20)
    flush(stdout)
    return out
end

"""
    load_rebaseline(key)

Load a cached rebaselined ensemble, extracting it first if needed.

Returns `(; q, q_star, dQ, tau, replica_index, nominal_replicas, missing_replicas, stochastic,
clampable, key, label, sources)`, one matrix per replica.
"""
function load_rebaseline(key::AbstractString)
    p = joinpath(DATA_DIR, "online_new_$(key)_qois.jld2")
    isfile(p) || extract_rebaseline(key)
    isfile(p) || error("extraction produced no cache at $p")
    d = load(p)
    return (; q = d["q"], q_star = d["q_star"], dQ = d["dQ"], tau = d["tau"],
            replica_index = d["replica_index"], nominal_replicas = d["nominal_replicas"],
            missing_replicas = d["missing_replicas"],
            unstable_replicas = get(d, "unstable_replicas", 0),
            incomplete = get(d, "incomplete", NamedTuple[]), stochastic = d["stochastic"],
            clampable = d["clampable"], key = d["key"], label = d["label"],
            sources = d["sources"], path = p)
end

if abspath(PROGRAM_FILE) == @__FILE__
    targets = isempty(ARGS) ? [m.key for m in REBASE_MODELS] : ARGS
    println("extracting rebaselined online QoIs to ", DATA_DIR)
    println("  root: ", REBASE_ROOT)
    flush(stdout)
    for t in targets
        println()
        println("== ", t)
        flush(stdout)
        extract_rebaseline(t)
    end
end
