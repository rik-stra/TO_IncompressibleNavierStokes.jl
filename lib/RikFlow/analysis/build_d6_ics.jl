# Build D6's initial-condition packages: one small file per initial condition, sliced once from the
# 100 TU tracked record.
#
# D6 is the multi-IC ensemble the baseline metric set is missing -- `K` initial conditions x `M`
# members, each a genuine forecast -- and it exists because every archived online run is one
# trajectory from one initial condition (`6_online_TO_LRS.jl:56-59` hard-codes
# `ustart = data_track.fields[1].u` and varies only the model seed). Spread and skill are both
# expectations over initial conditions, so with one IC every lead has exactly one verification
# instance and an RMSE from one sample is not an RMSE.
#
# Why one file per IC rather than one big file: JLD2 stores `data_track` as a single compound
# dataset, so `fields` cannot be partially read from the source and the slicing has to happen once,
# locally. Paying it once turns the 5-IC pilot into a ~17 MB copy instead of a 1.29 GB one, and lets
# a Snellius array task read exactly the one file it needs.
#
# Usage:
#   julia --startup-file=no --project=analysis analysis/build_d6_ics.jl          # K = 180
#   julia --startup-file=no --project=analysis analysis/build_d6_ics.jl 5        # a 5-IC pilot set
#
# Writes `analysis/output/d6_ics/d6_ic_<k>.jld2` plus a manifest. That directory is gitignored
# (`.gitignore:12` is `*output/`), which is what we want for ~600 MB of velocity snapshots.

using JLD2, Printf, Dates

# --------------------------------------------------------------------------------------------
# the record's own geometry
# --------------------------------------------------------------------------------------------
#
# These are properties of `data_track2_dns512_les64_Re2000.0_tsim100.0.jld2` and every one of them
# is asserted against the file in `build_d6_ics`, never assumed. They are named here because
# `select_ics` has to be callable -- and testable -- without opening a 1.29 GB file.

"Step spacing between saved fields: `params_track.savefreq`."
const FIELD_STRIDE = 100

"Time spacing between saved fields, in TU. `FIELD_STRIDE * Δt` with `Δt = 2.5e-3`."
const FIELD_DT = 0.25

"Columns in the reference `q`/`dQ`, i.e. the length of the truth the forecasts are scored against."
const N_REF = 40000

"Saved fields in the record. `fields[k].n = FIELD_STRIDE * (k-1)`, `fields[k].t = FIELD_DT * (k-1)`."
const N_FIELDS = 401

"""
End of M0's fit window in TU. `LinReg1` was fitted on steps 400-4000, i.e. `t in [1, 10]`.

An initial condition inside that window measures short-lead spread on data the conditional mean has
already seen, so the pool starts strictly after it.
"""
const FIT_END_TU = 10.0

"""
Warm-up steps replayed from the record before the forecast starts (the driver's `spinnup_data`).

🔴 **220 steps = 0.55 TU = 1.01x the slowest LEVEL decorrelation time** (Rik, 2026-09-15). The
previous 100 steps is 0.25 TU: a full decorrelation time for `Z[0,6]` (T = 0.2489) but only 0.46x
for `E[16,32]` (T = 0.5395), so the slowest bands entered every forecast still carrying the
record's state rather than the model's. Costs about one field of IC pool.
"""
const N_WARM = 220

"""
    N_WARM_DRIVER

The warm-up the **online drivers** use: `dQ_data = data_track.dQ[:, 1:100]`
(`6_online_TO_LRS.jl`, and `paper_runs/online_sgs.jl:62` for the archive).

🔴 **Fixed by what it reproduces, so it does NOT follow `N_WARM`.** The validation IC exists to
reproduce a driver run column for column, and that driver replayed 100 steps. Building it with the
scored set's 220 would mean validating against inputs nobody ever ran -- which is how the check
would have passed while testing nothing. The scored ICs are free to use a longer warm-up because
they are a different experiment; the validation is not.
"""
const N_WARM_DRIVER = 100

"""
Forecast length in steps. 2172 steps = 5.43 TU = 10x the slowest **LEVEL** decorrelation time.

🔴 **Set from `T(q)`, not `T(dQ)`** (Rik, 2026-09-15). D6 scores the forecast of the QoI *level*, so
the level's timescale is what has to saturate; the correction's says nothing about when skill dies.
The previous value, 1208 steps, was 10x the slowest **dQ** timescale on the **archive**
(`T_int(dQ) = 0.3017`); that is only 5.6x the level's on R1 -- a forecast that could have ended
before the slowest band saturated, discoverable only by re-running the whole ensemble.

Measured on R1's tracked record, decay constant per band, in TU:

    level q : 0.2489  0.4732  0.4893  0.4742  0.5430  0.5395   (max 0.5430 = 217 steps)
    dQ      : 0.1162  0.0081  0.0589  0.0666  0.3800  0.3745   (max 0.3800 = 152 steps)

Cost: the IC pool falls from 346 to 337 fields, and K = 90, M = 10 goes from 6.7 to 12.0 GPU-h.
Cheap against re-running the ensemble.
"""
const N_LEAD = 2172

"""
The slowest **level** decorrelation time in TU on R1 -- the yardstick for the forecast length and
for the achieved IC spacing.

⚠️ This is the DECAY CONSTANT `T` in `rho(tau) = exp(-tau/T)`, which is what `N_eff = K*tanh(L/(2TK))`
is derived for. `plot_hf_new_vs_archive.jl`'s `integrated_time` returns `1 + 2*sum(rho)`, i.e. **2T**,
and memory #58's "0.94-1.08 TU" is in that convention. Measured ratio between the two estimators on
all six bands: 1.985-1.997. Feeding 2T into the N_eff formula halves the ceiling.
"""
const T_INT_MAX = 0.5430

# 🔴 R1's record, not the archive's. The packages on disk before 2026-09-15 were cut from
# `data_track2_...` -- paper 2's archived tracked record, on the pre-`09954be1` Nyquist convention
# and therefore a different dynamical system (memory #45, #46, #60). Nothing in a filename or a
# directory listing showed it; only `provenance.source` did.
const DEFAULT_TRACK_FILE = get(ENV, "RIKFLOW_TRACK_FILE",
    normpath(joinpath(@__DIR__, "..", "exp_square_HIT", "output",
        "data_track_dns512_les64_Re2000.0_tsim100.0_f64_lmwray3.jld2")))

"""
    VALIDATION_TRACK_FILE

The record ordinal 0's validation IC is cut from: **R1's own**, the same one the scored ICs come
from.

🔴 **Switched from paper 2's archived 10 TU record on 2026-09-16 (Rik).** The oracle is now **R2's
`LinReg1` replica 1**, not the archive's. The reason is that the archive is a *different dynamical
system* -- pre-`09954be1`, Float32, its own reference (memory #45, #46) -- so a failed reproduction
against it was ambiguous: driver bug, or system difference? The `Z[16,32]` carve-out that
`validation_verdict` used to need was the symptom of exactly that, and it is gone with this change.

R2's replica 1 is the right oracle because every confound disappears: same record D6's ICs come
from, same Float64 precision, same Nyquist convention, same solver, same reference. A failed
reproduction now means the D6 driver is wrong, full stop.

⚠️ **What is given up, and why it does not matter.** The archive comparison was the only thing in D6
tying back to paper 2's published numbers. G1 already does that job and does it better -- it
reproduces the archived coefficients to 1.6e-4 and the published KS table to the digit -- so this
was the weaker of two overlapping checks.

⚠️ `fields[1]` of R1 is `n_k = 0`, `t = 0`, inside the fit window and the identity point of the OU
replay (`ou_advance = 0`). It is the field R2's online runs launched from, and it stays excluded
from `select_ics` and from everything scored.
"""
const VALIDATION_TRACK_FILE = get(ENV, "RIKFLOW_VALIDATION_TRACK", DEFAULT_TRACK_FILE)

"""
    DRIVER_SEED_BASE

The online drivers' model-seed base: `Xoshiro(seeds.to + i + 2)` with `seeds.to = 234`, so replica
`i` uses `Xoshiro(236 + i)`.

🔑 **Both drivers use the identical expression** -- `6_online_TO_LRS.jl:37-41,83` and the archive's
`paper_runs/online_sgs.jl:84` -- so the value did not change when ordinal 0's oracle moved from the
archive to R2 (`VALIDATION_TRACK_FILE`). Only what it is reproducing did.

The validation run reuses this stream, which is what makes the comparison column-by-column rather
than merely distributional. Every *scored* member goes through `member_seed`, i.e. `hash((:d6, k,
member))`, and therefore shares a stream with nothing else.
"""
const DRIVER_SEED_BASE = 236

const DEFAULT_IC_DIR = joinpath(@__DIR__, "output", "d6_ics")

"""
The keys of `params_track` an IC package carries forward.

Deliberately a subset. `params_track` also holds `ArrayType`, `backend`, `filters` and `ref_reader`,
and those are useless or harmful here: the first two are set by whichever machine runs the forecast,
`ref_reader` carries the whole tracked QoI series and would add ~2 MB to *every* IC file, and all
four come back from JLD2 as reconstructed placeholder types on a machine without CUDA. What is kept
is plain numbers, strings and tuples, and it is exactly the set `online_sgs` reads.
"""
const PARAM_KEYS = (:D, :Re, :lims, :qois, :nles, :Δt, :ou_bodyforce)

# --------------------------------------------------------------------------------------------
# IC selection
# --------------------------------------------------------------------------------------------

"""
    select_ics(; K = 180, nwarm = N_WARM, nlead = N_LEAD, nref = N_REF, kmin = ..., kmax = ..., ...)

Choose `K` field indices as evenly spaced over the usable pool as the pool allows, and assert both
constraints that define the pool for every one of them. Returns
`(; k, n, t, ordinal, spacing_tu, spacing_fields, pool, kmin, kmax)`.

**Two exclusions, both of which cost K.**

 1. **The training window.** `t_k > FIT_END_TU`, so `k >= 42`.
 2. 🔴 **The reference length.** Truth is the 40 001-column reference. A run from `n_k` replays
    `nwarm` warm-up steps and then forecasts `nlead`, so it needs `n_k + nwarm + nlead <= nref`,
    i.e. `n_k <= 37608` and `k <= 377` at the current `nwarm = 220, nlead = 2172`.

So the pool is `k in [42, 377]`, **336 fields**, and at exactly 0.5 TU spacing (every second field)
that yields **K = 168**, not 180. Reaching `K = 180` needs 0.47 TU spacing. None of these numbers is
hard-coded: the bounds are re-derived here from `nwarm`, `nlead`, `nref` and the record's grid, any
supplied `kmin`/`kmax` are checked against them, and the achieved spacing is reported rather than
assumed. (Before 2026-09-15 the pool was `[42, 387]`, 346 fields, at `nwarm = 100, nlead = 1208`.)

🔴 **At K = 180 the spacing is BELOW the decorrelation time, not above it.** 0.4679 TU against the
slowest level timescale `T_INT_MAX = 0.5430` is a ratio of **0.86**; on the old dQ-based yardstick
it read 1.60. So adjacent ICs at K = 180 are genuinely correlated. Every interval over these `K`
instances needs a block bootstrap over initialisation time with a **QoI-dependent** block length,
and nothing may claim `K` independent instances.

🔑 **K = 90 is the intended production setting** -- build all 180 packages (cheap, no GPU) and submit
`--array=1-179:2`. That doubles the spacing to 0.94 TU, i.e. 1.7x the slowest timescale, and
`select_ics` is strictly monotone so the halved set is exactly every second IC of the full one.
"""
# 🔴 `kmin`/`kmax` are DERIVED from the other constants, not written down. They used to be literal
# 42 and 387, correct for `nwarm = 100, nlead = 1208` and silently wrong for anything else -- and
# changing `N_LEAD` is exactly what this file is for. The checks below still re-derive and compare,
# so a caller passing explicit bounds is validated rather than trusted.
default_kmin(; dt_field = FIELD_DT, tfit_end = FIT_END_TU) = floor(Int, tfit_end / dt_field) + 2
default_kmax(; nwarm = N_WARM, nlead = N_LEAD, nref = N_REF, nfields = N_FIELDS,
             nstride = FIELD_STRIDE) = min(nfields, div(nref - nwarm - nlead, nstride) + 1)

function select_ics(; K::Integer = 180,
                    nwarm::Integer = N_WARM, nlead::Integer = N_LEAD, nref::Integer = N_REF,
                    nfields::Integer = N_FIELDS, nstride::Integer = FIELD_STRIDE,
                    dt_field::Real = FIELD_DT, tfit_end::Real = FIT_END_TU,
                    kmin::Integer = default_kmin(; dt_field, tfit_end),
                    kmax::Integer = default_kmax(; nwarm, nlead, nref, nfields, nstride))
    K >= 1 || error("select_ics: K must be at least 1, got $K")

    step_of(k) = nstride * (k - 1)
    time_of(k) = dt_field * (k - 1)

    # Re-derive the pool rather than trusting the arguments, so that a change to nlead, to the
    # reference length or to the fit window shows up as a failure here instead of as a quietly
    # truncated forecast several hundred GPU-hours later.
    kmin_req = floor(Int, tfit_end / dt_field) + 2          # first k with t_k strictly > tfit_end
    kmax_req = min(nfields, div(nref - nwarm - nlead, nstride) + 1)

    kmin >= kmin_req || error("select_ics: kmin = $kmin is inside the fit window; " *
                              "t_$kmin = $(time_of(kmin)) is not > $tfit_end (need kmin >= $kmin_req)")
    kmax <= kmax_req || error("select_ics: kmax = $kmax overruns the reference; " *
                              "n_$kmax + $nwarm + $nlead = $(step_of(kmax) + nwarm + nlead) > $nref " *
                              "(need kmax <= $kmax_req)")

    pool = kmax - kmin + 1
    K <= pool || error("select_ics: K = $K exceeds the usable pool of $pool fields " *
                       "(k in [$kmin, $kmax] after excluding the fit window and the reference cap). " *
                       "Reduce K, or shorten nlead.")

    # Evenly as the pool allows, endpoints included, deterministic. `round` on a `range` cannot
    # collide while the step exceeds one field, and K <= pool guarantees that; `allunique` says so
    # rather than leaving it to the argument above.
    k = K == 1 ? [kmin] : round.(Int, range(kmin, kmax; length = K))
    allunique(k) || error("select_ics: K = $K produced repeated field indices")
    issorted(k) || error("select_ics: index selection is not sorted")

    n = step_of.(k)
    t = time_of.(k)

    for (i, ki) in pairs(k)
        t[i] > tfit_end ||
            error("select_ics: IC $i (k = $ki) is inside the fit window: t = $(t[i]) <= $tfit_end")
        n[i] + nwarm + nlead <= nref ||
            error("select_ics: IC $i (k = $ki) overruns the reference: " *
                  "$(n[i]) + $nwarm + $nlead > $nref")
    end

    dfield = K == 1 ? Float64[] : Float64.(diff(k))
    spacing_fields = K == 1 ? NaN : (kmax - kmin) / (K - 1)
    spacing_tu = spacing_fields * dt_field

    return (; k, n, t, ordinal = collect(1:K), spacing_tu, spacing_fields,
            spacing_min_tu = isempty(dfield) ? NaN : minimum(dfield) * dt_field,
            spacing_max_tu = isempty(dfield) ? NaN : maximum(dfield) * dt_field,
            pool, kmin, kmax, nwarm, nlead, nref, K)
end

"""
    report_ics(sel; io = stdout)

Print what `select_ics` achieved, including the two facts that must not be hidden: the requested
`K = 180` comes out at 0.48 TU spacing rather than the nominal 0.5, and that spacing is only
1.6x the slowest QoI's integral timescale.
"""
function report_ics(sel; io = stdout)
    @printf(io, "  pool           k in [%d, %d]  (%d fields, after the fit window and the reference cap)\n",
            sel.kmin, sel.kmax, sel.pool)
    @printf(io, "  selected       K = %d  ordinals 1..%d -> k = %d .. %d\n",
            sel.K, sel.K, first(sel.k), last(sel.k))
    @printf(io, "  steps          n = %d .. %d       last forecast column %d of %d\n",
            first(sel.n), last(sel.n), last(sel.n) + sel.nwarm + sel.nlead, sel.nref)
    @printf(io, "  times          t = %.2f .. %.2f TU  (fit window ends at %.1f TU)\n",
            first(sel.t), last(sel.t), FIT_END_TU)
    if sel.K > 1
        @printf(io, "  spacing        %.4f TU mean  (%.2f fields; %.2f .. %.2f TU realised)\n",
                sel.spacing_tu, sel.spacing_fields, sel.spacing_min_tu, sel.spacing_max_tu)
        @printf(io, "  ⚠️  that is %.2f x the slowest QoI's T_int = %.4f TU, so the ICs are only\n",
                sel.spacing_tu / T_INT_MAX, T_INT_MAX)
        @printf(io, "      weakly independent: a block bootstrap over initialisation time, with a\n")
        @printf(io, "      QoI-dependent block length, is mandatory. Never claim K independent instances.\n")
    end
    return nothing
end

# --------------------------------------------------------------------------------------------
# packaging
# --------------------------------------------------------------------------------------------

"Path of the package for field index `k`."
ic_path(k::Integer, dir = DEFAULT_IC_DIR) = joinpath(dir, "d6_ic_$(k).jld2")

"Path of the manifest describing a whole IC set."
manifest_path(dir = DEFAULT_IC_DIR) = joinpath(dir, "d6_ic_manifest.jld2")

"""
    warmup_range(n_k, nwarm = N_WARM)

The warm-up slice of the reference `dQ` for a run launched from the field at step `n_k`.

🔑 This is the off-by-one to get right. `dQ[:, m]` is the correction **at** step `m`, and a run
launched from the field at step `n_k` takes its first solver step at `n_k + 1`, so the slice is
`n_k .+ (1:nwarm)`. Sanity: `k = 1` gives `n = 0` and `dQ[:, 1:100]`, which is exactly the archived
driver's hard-coded slice (`6_online_TO_LRS.jl:62`). A generalisation that does not reduce to that
at `k = 1` is wrong.
"""
warmup_range(n_k::Integer, nwarm::Integer = N_WARM) = n_k .+ (1:nwarm)

"""
    ic_q_column(n_k)

The reference `q` column holding the QoIs of the field at step `n_k`.

`q` has `nstep + 1` columns because `qoisaver` fires on the initial state as well
(`RikFlow.jl:294`), so step `m` is column `m + 1`. The packages carry this column so a forecast can
assert on the compute node that its own first `q` column -- the QoIs of `ustart` -- is the one the
record says it is, before it spends nineteen seconds forecasting from the wrong field.
"""
ic_q_column(n_k::Integer) = n_k + 1

"""
    q_window_range(n_k, ncol; half = 2)

Reference `q` columns to ship with an IC package, and the offset of each from the IC's own column.

🔑 **Why a window and not just the one column.** The driver has to check on the compute node that
the field it was handed really is at step `n_k`, before it spends ten members on it. Comparing the
run's recomputed `q[:, 1]` against a single stored value cannot do that: a one-step misalignment
moves these QoIs by 2e-4 to 3e-3 relative, which is the same size as other, benign discrepancies —
in particular `Z[16,32]`, where the current code and the archived code disagree by ~1.06e-3 on the
*same* field (`claude_memory.md` gotcha #45). An absolute threshold therefore cannot separate
"wrong step" from "known code difference", and the 1e-3 one that shipped first could not.

With a window the test becomes scale-free: the claimed column must be the **best** match among its
neighbours. Any discrepancy that is constant across columns — which a code difference in one QoI is
— cancels out of that comparison entirely. Measured margin at the right column: 1.2e-7 median
relative against ~9e-4 one column away, about 7000x.

Returns `(cols, offsets)`; `offsets[j] == 0` marks the IC's own column. Clamped at the ends of the
record, so `n_k = 0` yields offsets `0:half`.
"""
function q_window_range(n_k::Integer, ncol::Integer; half::Integer = 2)
    c = ic_q_column(n_k)
    lo = max(1, c - half)
    hi = min(ncol, c + half)
    return lo:hi, (lo - c):(hi - c)
end

"""
    build_d6_ics(; K = 180, track_file, outdir, force = false)

Slice the tracked record once and write one `d6_ic_<k>.jld2` per selected initial condition.

Each package holds the velocity field `u`, its step `n_k` and time `t_k`, the `nwarm`-column
warm-up slice of the reference `dQ`, the reference QoI column at the IC, the `PARAM_KEYS` subset of
`params_track`, and provenance. Returns the selection.
"""
function build_d6_ics(; K::Integer = 180, track_file = DEFAULT_TRACK_FILE, outdir = DEFAULT_IC_DIR,
                      force::Bool = false)
    isfile(track_file) || error("no such tracking file: $track_file")
    mkpath(outdir)

    @printf("reading %s (%.2f GB) ...\n", basename(track_file), filesize(track_file) / 2^30)
    t0 = time()
    d, params_track = jldopen(track_file, "r") do io
        haskey(io, "data_track") || error("no data_track in $track_file; keys = $(keys(io))")
        io["data_track"], io["params_track"]
    end
    @printf("  read in %.1f s\n", time() - t0)

    # The record's geometry, asserted rather than assumed. `select_ics` derives the pool from these
    # numbers, so if the record ever changes shape the failure belongs here and not downstream.
    fields = d.fields
    nfields = length(fields)
    nref = size(d.dQ, 2)
    nfields == N_FIELDS || error("record has $nfields fields, expected $N_FIELDS")
    nref == N_REF || error("record has $nref dQ columns, expected $N_REF")
    size(d.q, 2) == nref + 1 ||
        error("q has $(size(d.q, 2)) columns, expected nref + 1 = $(nref + 1); " *
              "the initial-state offset that `ic_q_column` and the scorer's truth alignment rely on")
    all(fields[k].n == FIELD_STRIDE * (k - 1) for k in 1:nfields) ||
        error("field step spacing is not $FIELD_STRIDE; the n_k = $FIELD_STRIDE (k-1) map is wrong")
    all(isapprox(fields[k].t, FIELD_DT * (k - 1); atol = 1e-5) for k in 1:nfields) ||
        error("field time spacing is not $FIELD_DT TU")
    params_track.savefreq == FIELD_STRIDE ||
        error("params_track.savefreq = $(params_track.savefreq), expected $FIELD_STRIDE")

    sel = select_ics(; K, nfields, nref)
    println("\nIC selection")
    report_ics(sel)

    params = NamedTuple{PARAM_KEYS}(map(k -> getproperty(params_track, k), PARAM_KEYS))
    provenance = (; source = abspath(track_file), source_bytes = filesize(track_file),
                  built = string(now()), nwarm = sel.nwarm, nlead = sel.nlead, nref = sel.nref,
                  K = sel.K, spacing_tu = sel.spacing_tu)

    println("\nwriting $(sel.K) packages to $outdir")
    written = 0
    bytes = 0
    for (i, k) in pairs(sel.k)
        out = ic_path(k, outdir)
        n_k = sel.n[i]
        t_k = sel.t[i]

        if isfile(out) && !force
            bytes += filesize(out)
            continue
        end

        rows = warmup_range(n_k, sel.nwarm)
        last(rows) <= nref || error("warm-up slice for k = $k runs past the record")
        dQ_warm = Array(d.dQ[:, rows])
        size(dQ_warm, 2) == sel.nwarm ||
            error("warm-up slice for k = $k has $(size(dQ_warm, 2)) columns, expected $(sel.nwarm)")
        any(isnan, dQ_warm) && error("warm-up slice for k = $k contains NaN")

        u = Array(fields[k].u)
        any(isnan, u) && error("field k = $k contains NaN")
        fields[k].n == n_k || error("field k = $k is at step $(fields[k].n), expected $n_k")

        wcols, woff = q_window_range(n_k, size(d.q, 2))
        jldsave(out; u, n_k, t_k, k, ordinal = i, dQ_warm,
                q_at_ic = Array(d.q[:, ic_q_column(n_k)]),
                q_window = Array(d.q[:, wcols]), q_window_offsets = collect(woff),
                params, provenance)
        written += 1
        bytes += filesize(out)
    end

    manifest = manifest_path(outdir)
    jldsave(manifest; k = sel.k, n = sel.n, t = sel.t, ordinal = sel.ordinal, K = sel.K,
            kmin = sel.kmin, kmax = sel.kmax, spacing_tu = sel.spacing_tu, provenance, params)

    @printf("  %d written, %d already present; %.1f MB total, %.2f MB each\n",
            written, sel.K - written, bytes / 2^20, bytes / 2^20 / sel.K)
    @printf("  manifest: %s\n", basename(manifest))
    println("\nto copy the first 5 (the pilot): " *
            join(basename.(ic_path.(sel.k[1:min(5, sel.K)], outdir)), ", "))
    return sel
end

# --------------------------------------------------------------------------------------------
# the validation IC
# --------------------------------------------------------------------------------------------

"Path of the validation package. A distinct name, for the reason in `build_validation_ic`."
validation_path(dir = DEFAULT_IC_DIR) = joinpath(dir, "d6_ic_validation.jld2")

"""
    build_validation_ic(; track_file = VALIDATION_TRACK_FILE, outdir, nwarm, nlead, force)

Build the one IC that is **not** for scoring: `fields[1]` of the 10 TU tracked record, i.e. the
initial condition R2's online runs launched from -- `fields[1]` of R1's own tracked record.

🔑 **Why it exists.** A D6 run from here has `n_k = 0`, so `ou_advance = 0` and the OU chain starts
at zero, which is exactly what the online driver does and is the identity point of the whole replay
mechanism. Give it the driver's model seeds (`DRIVER_SEED_BASE`) and its `q` must reproduce **R2's
`LinReg1` replica 1** over the replayed window. That is a correctness check on the entire D6 path --
IC packaging, warm-up slicing, `ou_advance`, the driver, the output format -- against a trajectory
produced by a separate run of the same system.

🔴 **The oracle moved from paper 2's archive to R2 on 2026-09-16** (Rik); see
`VALIDATION_TRACK_FILE` for why. The archive is a different dynamical system, so a failure against
it could not distinguish a driver bug from the system difference.

🔴 **It is deliberately kept out of `select_ics` and out of everything scored**, for two
independent reasons, and merging it in would break both:

 1. `t_1 = 0` is **inside** M0's fit window, so its short-lead spread would be measured on data the
    conditional mean has already seen. `select_ics` asserts `t_k > 10` precisely to exclude it.
 2. V28 requires D6's IC set to be **disjoint** from the online runs' IC, which is this one.
    `test_d6_ics.jl` asserts `!(1 in select_ics(; K).k)`; that test is only meaningful while this
    package stays outside the selection.

Hence the separate filename, and hence `run_d6.jl` writing its members as `d6_valid_ic1_m*.jld2`
where the scorer's own glob cannot see them.
"""
function build_validation_ic(; track_file = VALIDATION_TRACK_FILE, outdir = DEFAULT_IC_DIR,
                             nwarm::Integer = N_WARM_DRIVER, nlead::Integer = N_LEAD,
                             force::Bool = false)
    isfile(track_file) || error("no such tracking file: $track_file")
    out = validation_path(outdir)
    if isfile(out) && !force
        @printf("validation IC exists, skipping: %s (%.2f MB)\n", basename(out),
                filesize(out) / 2^20)
        return out
    end
    mkpath(outdir)

    @printf("reading %s (%.2f GB) ...\n", basename(track_file), filesize(track_file) / 2^30)
    d, params_track = jldopen(track_file, "r") do io
        haskey(io, "data_track") || error("no data_track in $track_file; keys = $(keys(io))")
        io["data_track"], io["params_track"]
    end

    # The geometry checks that apply to *this* record. It is 10 TU, not 100, so `N_FIELDS` and
    # `N_REF` do not; what must hold is the field grid and that the warm-up slice fits.
    fields = d.fields
    fields[1].n == 0 || error("fields[1] is at step $(fields[1].n), expected 0")
    isapprox(fields[1].t, 0.0; atol = 1e-6) || error("fields[1].t = $(fields[1].t), expected 0")
    params_track.savefreq == FIELD_STRIDE ||
        error("params_track.savefreq = $(params_track.savefreq), expected $FIELD_STRIDE")
    size(d.q, 2) == size(d.dQ, 2) + 1 || error("q is not one column longer than dQ")
    size(d.dQ, 2) >= nwarm || error("record has $(size(d.dQ, 2)) dQ columns, need >= $nwarm")

    rows = warmup_range(0, nwarm)
    first(rows) == 1 && last(rows) == nwarm ||
        error("the warm-up slice at n = 0 is $(rows), expected 1:$nwarm")
    dQ_warm = Array(d.dQ[:, rows])
    any(isnan, dQ_warm) && error("warm-up slice contains NaN")
    u = Array(fields[1].u)
    any(isnan, u) && error("field contains NaN")

    params = NamedTuple{PARAM_KEYS}(map(k -> getproperty(params_track, k), PARAM_KEYS))
    provenance = (; source = abspath(track_file), source_bytes = filesize(track_file),
                  built = string(now()), nwarm, nlead, K = 0, spacing_tu = NaN,
                  nref = size(d.dQ, 2))
    wcols, woff = q_window_range(0, size(d.q, 2))
    jldsave(out; u, n_k = 0, t_k = 0.0, k = 1, ordinal = 0, dQ_warm,
            q_at_ic = Array(d.q[:, ic_q_column(0)]),
            q_window = Array(d.q[:, wcols]), q_window_offsets = collect(woff),
            params, provenance, validation = true, driver_seed_base = DRIVER_SEED_BASE)

    @printf("wrote %s (%.2f MB)\n", basename(out), filesize(out) / 2^20)
    println("  ⚠️  validation only: t_1 = 0 is inside M0's fit window and this is the archived " *
            "runs' own IC,\n      so it is excluded from `select_ics` and from everything scored. " *
            "Run it as ordinal 0.")
    return out
end

"""
    DDN_TRAIN_RANGE

The window the DDN's multivariate Gaussian is fitted on: `7_online_DDN.jl:30`, `400:4000`.

🔴 The same window the LRS is fitted on (`5_train_LinReg.jl`, `train_range = (400, 4000)`). Fitting
the two closures on different data would make every D6 comparison between them a comparison of
training sets as much as of models.
"""
const DDN_TRAIN_RANGE = 400:4000

ddn_path(outdir = DEFAULT_IC_DIR) = joinpath(outdir, "d6_ddn_traindata.jld2")

"""
    build_ddn_traindata(; track_file, outdir, force)

Write the `dQ` slice the DDN is fitted on, once, beside the IC packages.

🔑 **Why a file and not a field in every package.** `MVG_sampler` fits its Gaussian in its own
constructor, so a D6 task needs the training `dQ` -- 6 x 3601 Float64, 173 kB. Re-reading the
2.6 GB tracked record in each of K x M array tasks is out of the question, and copying the same
slice into all 180 IC packages would waste 31 MB to say one thing 180 times. One small file, read
by every task.

⚠️ **The slice is stored, not the fitted distribution.** Storing the distribution would create a
second path into the DDN that could drift from `MVG_sampler`'s constructor; re-fitting a 6-component
Gaussian from 3601 samples costs microseconds, so there is no reason to have two.
"""
function build_ddn_traindata(; track_file = DEFAULT_TRACK_FILE, outdir = DEFAULT_IC_DIR,
                             force::Bool = false)
    out = ddn_path(outdir)
    if isfile(out) && !force
        @printf("DDN training data exists, skipping: %s (%.2f MB)
", basename(out),
                filesize(out) / 2^20)
        return out
    end
    isfile(track_file) || error("no such tracking file: $track_file")
    mkpath(outdir)
    @printf("reading %s (%.2f GB) for the DDN training slice ...
", basename(track_file),
            filesize(track_file) / 2^30)
    flush(stdout)
    dQ_train, params = jldopen(track_file, "r") do io
        d = io["data_track"]
        p = io["params_train" in keys(io) ? "params_train" : "params_track"]
        (Array(d.dQ[:, DDN_TRAIN_RANGE]), NamedTuple(k => getproperty(p, k) for k in PARAM_KEYS))
    end
    size(dQ_train, 2) == length(DDN_TRAIN_RANGE) ||
        error("DDN slice has $(size(dQ_train, 2)) columns, expected $(length(DDN_TRAIN_RANGE))")
    any(isnan, dQ_train) && error("DDN training slice contains NaN")
    jldsave(out; dQ_train, train_range = DDN_TRAIN_RANGE, params,
            provenance = (; source = abspath(track_file), source_bytes = filesize(track_file),
                          built = string(now())))
    @printf("wrote %s (%.2f MB), dQ_train = %s
", basename(out), filesize(out) / 2^20,
            size(dQ_train))
    return out
end

if abspath(PROGRAM_FILE) == @__FILE__
    # `--force` rewrites packages that already exist, which is what a change to the package format
    # needs (the `q_window` addition of 2026-09-11, for instance).
    force = "--force" in ARGS
    args = filter(!=("--force"), ARGS)
    if !isempty(args) && args[1] == "validation"
        build_validation_ic(; force)
    else
        K = isempty(args) ? 180 : parse(Int, args[1])
        build_d6_ics(; K, force)
        println()
        build_validation_ic(; force)
        println()
        build_ddn_traindata(; force)
    end
end
