# Where M4's drivers get their QoIs. `include`d by `11_train_StochLSTM.jl`, `tools/m4_lr_scan.jl`
# and `tools/m4_stride_scan.jl`.
#
# 🔑 **One implementation, because there were three.** Each driver had its own `load_qois` with its
# own fallback, which is three chances to point at the wrong record -- and pointing at the wrong
# record is not a slow run, it is a different dynamical system (below). `ladder_setup.jl` exists
# for exactly this reason on the linear side.
#
# Resolution order:
#   1. `RIKFLOW_QOI_CACHE`, if set -- an explicit path always wins and is never second-guessed.
#   2. the extracted cache under `analysis/data/`, discovered by pattern.
#   3. `RIKFLOW_TRACK_FILE` or the default tracking record -- correct, but 2.7 GB to read.
#
# ⚠️ **The cache is ~7 MB against the record's 2.7 GB** and holds exactly what these drivers read.
# Generate it once, on a login node, before submitting anything:
#
#     julia --project=analysis analysis/extract_qois.jl \
#         exp_square_HIT/output/data_track_dns512_les64_Re2000.0_tsim100.0_f64_lmwray3.jld2
#
# 🔴 **Not generated on demand here.** `extract_qois.jl` writes one file at a fixed path and the
# §6.3 grid submits fifteen jobs at once: all fifteen would read 2.7 GB and race to write the same
# output. A warning that says what to run is the right behaviour; a helpful auto-write is not.

using JLD2
using Printf

"""
    m4_phase(msg)

Print a timestamped, **flushed** progress line.

🔴 **This exists because a 20-minute Snellius job produced no output at all and was cancelled
blind** (2026-09-18). Julia's streams are block-buffered when redirected to a file, which a SLURM
job always is, so a long phase looks identical to a hang: the log is empty either way and there is
nothing to tell you whether to wait or to kill it. Every phase that can take minutes prints one of
these before it starts and one after, with the elapsed time, so the log localises a stall to a
phase even if the job is killed.

🔑 It flushes BOTH streams. `@info` goes to stderr and the driver's own `println`s go to stdout;
flushing one and not the other reorders the log in exactly the way that makes it untrustworthy.
"""
function m4_phase(msg)
    println("[", Libc.strftime("%H:%M:%S", time()), "] ", msg)
    flush(stdout)
    flush(stderr)
end

"""
    m4_probe_step(spec, X, Y, steps; cfg, device, stride, batch, label)

Time a handful of optimiser steps on `device` and print the per-update cost, **before** committing
to a full scan.

🔴 **This is the answer to a 20-minute job that printed nothing and was cancelled blind.** The
scans pass `verbose = false`, so a single point — 3000 updates — produces no output at all from
start to finish; on the CPU that is ~10 minutes of silence and on an untried device it is
unbounded. A probe costs a few updates and turns "is it hung or is it slow?" into a number in the
log before the first point starts.

🔑 **The first call is compilation, not cost** — gotcha #44, and doubly so on a GPU, where the
first launch also pays PTX compilation. So one epoch is run and discarded before the timed ones,
and both numbers are printed: a large gap between them IS the compilation, and a large *steady*
number is the thing worth stopping for.
"""
function m4_probe_step(spec, X, Y, steps; cfg, device, stride, batch, epochs_planned = nothing,
                       label = "probe")
    m4_phase("$label: compiling one update on the selected device ...")
    t0 = time()
    RikFlow.train_stochlstm(spec, X, Y, steps; cfg.L, cfg.burn, stride, epochs = 1, batch,
                            cfg.lr, cfg.beta, cfg.val_frac, seed = 1, verbose = false, device)
    tc = time() - t0
    m4_phase(@sprintf("%s: first epoch (compilation included) %.2f s", label, tc))

    n = 3
    t0 = time()
    RikFlow.train_stochlstm(spec, X, Y, steps; cfg.L, cfg.burn, stride, epochs = n, batch,
                            cfg.lr, cfg.beta, cfg.val_frac, seed = 1, verbose = false, device)
    ts = (time() - t0) / n
    # 🔴 Project from the epochs the CALLER actually plans, not from the update budget. An epoch is
    # `updates_per_epoch` optimiser steps and that factor is 2-5 here, so projecting a 3000-UPDATE
    # budget as 3000 epochs overstates the cost by exactly that factor -- which this message did on
    # its first outing.
    if epochs_planned === nothing
        m4_phase(@sprintf("%s: steady %.3f s/epoch", label, ts))
    else
        m4_phase(@sprintf("%s: steady %.3f s/epoch -> %d epochs is about %.1f min",
                          label, ts, epochs_planned, ts * epochs_planned / 60))
    end
    return (; compile = tc, steady = ts)
end

"""
    m4_geometry(steps, ntrain; L, burn, stride, batch)

Training segments, optimiser steps per epoch, and scored rows per epoch for one `(stride, batch)`.

🔑 **Shared so the scan and the walltime estimate cannot disagree.** `updates/epoch` is not
`ceil(nseg/batch)`: segments are grouped by LENGTH first and each group is batched separately, so
the trailing short segment always costs an update of its own. Getting that wrong understates the
step count, which is exactly the quantity a walltime is set from.
"""
function m4_geometry(steps, ntrain; L, burn, stride, batch)
    segs = RikFlow.segment_indices(view(steps, 1:ntrain); L, burn, stride)
    lens = Dict{Int,Int}()
    for sg in segs
        lens[length(sg.rows)] = get(lens, length(sg.rows), 0) + 1
    end
    upd = sum(cld(n, batch) for (_, n) in lens)
    return (; nseg = length(segs), upd, scored = sum(length(sg.score) for sg in segs))
end

"""
    m4_save_progress(path; complete, payload...)

Write a scan's results so far, **atomically**, after every point.

🔴 **Because a scan that is killed must not lose everything.** `jldsave` used to run once, after
the last point, so a walltime kill — or a cancelled job, which is how the first Snellius run
ended — threw away every completed point. A scan point costs minutes; there is no reason to make
the last one a single point of failure for the first four.

🔑 **Atomic: written to a `.tmp` and renamed.** `jldsave` is not atomic, so a kill *during* the
write leaves a truncated file that is worse than no file — it looks like a result and fails, or
worse loads, later. `mv` within one filesystem is atomic, so a reader sees either the previous
complete file or the new one, never a half-written one.

`complete` is stored so a consumer can tell a finished scan from a partial one without counting
rows. ⚠️ **A partial file is not a result**: its `thresholds` are computed from the points that
happen to have finished, so a ranking read off one is a ranking over a subset.
"""
function m4_save_progress(path::AbstractString; complete::Bool, payload...)
    tmp = path * ".tmp"
    jldsave(tmp; complete, payload...)
    mv(tmp, path; force = true)
    return path
end

"""
    QOI_PATTERN

Which extracted caches count as *this* project's tracking record.

🔴 **`_f64_lmwray3` is load-bearing and that is why the pattern carries it.** `analysis/data/` also
holds caches of paper 2's ARCHIVED records, which sit on the pre-`09954be1` Nyquist convention --
a **different dynamical system**, not a less accurate measurement of this one (`claude_memory.md`
#45, #46). A loose `data_track_*` match would eventually pick one, and nothing about the resulting
run would look wrong. `11_train_StochLSTM.jl`'s `track_file` default carries the same suffix for
the same reason.
"""
const QOI_PATTERN = r"^data_track_dns512_.*_f64_lmwray3_qois\.jld2$"

"""
    find_qoi_cache(data_dir)

The one matching cache under `data_dir`, or `nothing`.

⚠️ **Several matches is an error, not a choice.** `analysis/postrun_lstm.jl` already refuses an
ambiguous cache rather than guessing; guessing here would silently decide which record a whole
sweep was fitted on.
"""
function find_qoi_cache(data_dir::AbstractString)
    isdir(data_dir) || return nothing
    hits = filter(f -> occursin(QOI_PATTERN, f), readdir(data_dir))
    isempty(hits) && return nothing
    length(hits) == 1 || error(
        "find_qoi_cache: $(length(hits)) caches match under $data_dir -- " *
        "$(sort(hits)). Set RIKFLOW_QOI_CACHE explicitly rather than let the driver pick.")
    return joinpath(data_dir, only(hits))
end

"""
    load_m4_qois(; track_file, data_dir) -> (; q, q_star, source)

The `(q, q_star)` pair every M4 driver fits on, plus where it came from.

`source` is returned and logged so a run records which record it used -- the field that caught
D6's `load_truth` still pointing at the archive (`claude_memory.md` #66).
"""
function load_m4_qois(; track_file::AbstractString,
                      data_dir::AbstractString = normpath(joinpath(@__DIR__, "..", "..",
                                                                   "analysis", "data")))
    explicit = get(ENV, "RIKFLOW_QOI_CACHE", "")
    if !isempty(explicit)
        isfile(explicit) || error("RIKFLOW_QOI_CACHE=$explicit does not exist")
        m4_phase("reading QoIs from RIKFLOW_QOI_CACHE: $explicit")
        t0 = time()
        d = load(explicit)
        m4_phase(@sprintf("QoIs read in %.1f s", time() - t0))
        return (; q = d["q"], q_star = d["q_star"], source = explicit)
    end

    cache = find_qoi_cache(data_dir)
    if cache !== nothing
        m4_phase("reading QoIs from the extracted cache: $cache")
        t0 = time()
        d = load(cache)
        m4_phase(@sprintf("QoIs read in %.1f s", time() - t0))
        return (; q = d["q"], q_star = d["q_star"], source = cache)
    end

    isfile(track_file) || error(
        "no QoI source found.\n" *
        "  RIKFLOW_QOI_CACHE : unset\n" *
        "  cache under       : $data_dir (no file matching $(QOI_PATTERN.pattern))\n" *
        "  tracking record   : $track_file (missing)\n" *
        "Set RIKFLOW_QOI_CACHE, or extract the cache once:\n" *
        "  julia --project=analysis analysis/extract_qois.jl <tracking.jld2>")
    @warn """no extracted QoI cache under $data_dir, falling back to the 2.7 GB tracking record.
             Slower to start, and heavy on memory when several jobs overlap. Extract it once:
               julia --project=analysis analysis/extract_qois.jl $track_file""" track_file
    d = load(track_file, "data_track")
    return (; d.q, d.q_star, source = track_file)
end
