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
        d = load(explicit)
        @info "QoIs from RIKFLOW_QOI_CACHE" path = explicit
        return (; q = d["q"], q_star = d["q_star"], source = explicit)
    end

    cache = find_qoi_cache(data_dir)
    if cache !== nothing
        d = load(cache)
        @info "QoIs from the extracted cache" path = cache
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
