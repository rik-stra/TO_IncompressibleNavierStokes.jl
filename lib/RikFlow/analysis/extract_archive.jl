# Extract the QoI time series from paper 2's archive into small, self-contained files.
#
# Sibling of `extract_qois.jl`, which does the same for *tracking* runs. This one handles the two
# other archive objects the metric layer needs:
#
#   D3  the 100 TU high-fidelity reference, `data_train_...tsim100.0.jld2` (1.39 GB), whose QoIs
#       live at `data_train.data[1].qoi_hist` as a vector of per-step vectors.
#   D5  the online ensembles, `TO_LRS/<name>/data_online_tsim100.0_replica<i>.jld2` (144 MB each),
#       which carry `q`, `dQ`, `tau` and 41 velocity snapshots.
#
# The same reason applies as for tracking files: JLD2 stores each of these as one compound dataset,
# so `fields` cannot be skipped on read even though it is ~90% of the bytes and nothing in the
# time-series work looks at it. Pay the read once, cache the QoIs.
#
# The archive is gitignored (`*output/`) and lives outside the repository. Paths come from the
# environment, defaulting to this machine's layout:
#
#   RIKFLOW_ARCHIVE      paper 2's frozen archive  (LinReg1, 63, 64 + the HF reference)
#   RIKFLOW_DEV_ARCHIVE  the working repository    (LinReg1, 63, 64, 73, 74, `_rand_initial_dQ`)
#
# Usage:
#   julia --project=<env> analysis/extract_archive.jl            # everything available
#   julia --project=<env> analysis/extract_archive.jl reference  # just D3
#   julia --project=<env> analysis/extract_archive.jl LinReg1    # one configuration's ensemble

using JLD2, Printf, Dates

# `extract_qois.jl` defines the same constant to the same value, and a driver includes both, so
# defining it again unconditionally would be a const redefinition.
isdefined(@__MODULE__, :DATA_DIR) || const DATA_DIR = joinpath(@__DIR__, "data")

const FROZEN = get(ENV, "RIKFLOW_ARCHIVE",
                   raw"C:\Users\rik\Documents\julia_code\INS_paper_summer2025\lib\RikFlow\exp_square_HIT\output\paper_data_HIT")
const DEV = get(ENV, "RIKFLOW_DEV_ARCHIVE",
                raw"C:\Users\rik\Documents\julia_code\IncompressibleNavierStokes.jl\lib\RikFlow\exp_square_HIT\paper_runs")

const REFERENCE_FILE = "data_train_dns512_les64_Re2000.0_freeze_10_tsim100.0.jld2"

"""
    model_dir(name)

Where a named configuration's model and online runs live, and which root it came from. The frozen
archive keeps them under `TO_LRS/<name>/` and the working repository under `output/online/<name>/`;
the frozen one wins when both are present, because its trajectories are the ones paper 2 reports.

✅ **The two roots are the same runs under two filenames.** Measured 2026-09-10 on `q` over all
40 001 columns: frozen `data_online_tsim100.0_replica<i>.jld2` is **bit-identical** to the working
repository's `..._replica<i>_rand_initial_dQ.jld2` for LinReg1 (replicas 1-3) and LinReg63/64
(replicas 1-2). The `_rand_initial_dQ` suffix is a **stale label, not a treatment**: the script that
wrote those files, `paper_runs/online_sgs.jl:62,84`, seeds `spinnup_data` from
`data_track.dQ[:, 1:100]` exactly as `6_online_TO_LRS.jl:62,83` does, with the same
`Xoshiro(seeds.to + i + 2)` and the same `seeds.to = 234`. Nothing randomises the initial `dQ`.

⚠️ An earlier version of this comment said they were *"separate samples"* that *"must never be
pooled"*. Wrong on the reason, right on the conclusion -- see `replica_groups`.

The frozen root holds LinReg1, 63 and 64; LinReg73 and 74 exist only in the working repository, so
their runs come through the `_rand_initial_dQ` filename and are the same convention as the rest.
"""
function model_dir(name::AbstractString)
    f = joinpath(FROZEN, "TO_LRS", name)
    isfile(joinpath(f, "LinReg.jld2")) && return (f, :frozen)
    d = joinpath(DEV, "output", "online", name)
    isfile(joinpath(d, "LinReg.jld2")) && return (d, :dev)
    return (nothing, nothing)
end

"""
    replica_groups(dir)

The online replica files in `dir`, grouped by **run family** and sorted by replica index.

🔴 Grouping is not tidiness, but the reason is **duplication, not treatment mixing**. A
configuration directory can hold the same runs under two filenames:

  - `data_online_tsim100.0_replica<i>.jld2` -- the runs paper 2 reports.
  - `data_online_dns512_les64_Re2000.0_tsim100.0_replica<i>_rand_initial_dQ.jld2` -- **the same
    runs**. The suffix is a stale label; nothing randomises the initial `dQ`. See the header.

The frozen archive's `LinReg1` holds five of the first and **one** of the second, and that one is
bit-identical to `replica1`. Globbing `data_online*` returns six files, and treating them as a
six-member ensemble would present **five distinct runs with replica 1 counted twice** -- inflating
its weight to 2/6 of every marginal statistic and understating the ensemble spread. That is what
this function prevents.

⚠️ An earlier version of this docstring called them *"separate experiments"* and warned about mixed
warm starts. The guard was right; the reason was not.

Returns a `Dict` from family label to a vector of `(index, filename)`.
"""
function replica_groups(dir)
    pat = r"^data_online_?(.*)_replica(\d+)(.*)\.jld2$"
    groups = Dict{String,Vector{Tuple{Int,String}}}()
    for f in readdir(dir)
        m = match(pat, f)
        m === nothing && continue
        label = isempty(m.captures[3]) ? m.captures[1] : m.captures[1] * m.captures[3]
        push!(get!(groups, label, Tuple{Int,String}[]), (parse(Int, m.captures[2]), f))
    end
    for v in values(groups)
        sort!(v; by = first)
    end
    return groups
end

"The family with the most replicas; ties broken by label for determinism."
function primary_family(groups)
    isempty(groups) && return nothing
    ks = sort(collect(keys(groups)))
    return argmax(k -> (length(groups[k]), -findfirst(==(k), ks)), ks)
end

# ---------------------------------------------------------------------------------------------
# D3 -- the high-fidelity reference
# ---------------------------------------------------------------------------------------------

"""
    extract_reference(; force = false)

Cache the 100 TU HF reference QoI trajectory as `q_ref`, `N_Q × nstep+1`.

`qoi_hist` is stored as a vector of per-step vectors, so it is `stack`ed into a matrix here and
never again. 40 001 points over 100 TU at Δt = 2.5e-3.
"""
function extract_reference(; force = false)
    src = joinpath(FROZEN, REFERENCE_FILE)
    out = joinpath(DATA_DIR, "hf_reference_tsim100.0_qois.jld2")
    isfile(out) && !force && (@printf("  exists, skipping: %s (%.2f MB)\n", basename(out),
                                      filesize(out) / 2^20); return out)
    isfile(src) || (@warn "no HF reference at $src"; return nothing)
    mkpath(DATA_DIR)

    @printf("  reading %s (%.2f GB) ...\n", basename(src), filesize(src) / 2^30)
    flush(stdout)                        # Julia buffers stdout when redirected
    t0 = time()
    q_ref, params = jldopen(src, "r") do io
        d = io["data_train"]
        hist = d.data[1].qoi_hist
        (stack(hist), "params_train" in keys(io) ? string(io["params_train"]) : "")
    end
    jldsave(out; q_ref, source = abspath(src), source_bytes = filesize(src),
            params = params, extracted = string(now()))
    @printf("  read in %.1f s; wrote %s (%.2f MB), q_ref = %s\n", time() - t0, basename(out),
            filesize(out) / 2^20, size(q_ref))
    flush(stdout)
    return out
end

# ---------------------------------------------------------------------------------------------
# D3' -- the REGENERATED high-fidelity reference (2026-09-14)
#
# Same object as D3, from `2_HF_ref.jl` on the merged solver: Float64, LMWray3, corrected Nyquist
# convention, the archive's own spin-up field as its initial condition. It is a *different
# realisation* from D3, not a refinement of it (the OU chain draws into a Float64 buffer, which
# consumes the random stream differently, and the stepper changed) -- see gotchas #45, #46, #52.
# Cached here for the same reason as D3: the QoIs are ~2 MB of a 2.58 GB file.
# ---------------------------------------------------------------------------------------------

const NEW_REFERENCE_FILE = get(
    ENV, "RIKFLOW_HF_NEW",
    normpath(joinpath(@__DIR__, "..", "exp_square_HIT", "output",
        "data_train_dns512_les64_Re2000.0_freeze_10_tsim100.0_f64_lmwray3.jld2")),
)

const NEW_REFERENCE_CACHE = "hf_reference_new_tsim100.0_f64_lmwray3_qois.jld2"

"""
    extract_new_reference(; force = false)

Cache the regenerated 100 TU HF reference QoI trajectory as `q_ref`, `N_Q x nstep+1`.

Stores `dt_sample` alongside it, computed from the run's own `savefreq * Δt` rather than assumed.
D3's spacing is a constant in the consumers because the archive cannot be re-read cheaply; there is
no reason to repeat that here, and a run with different sampling must not land silently on D3's
axis.

Also stores `comptime` (the solver's own wall time, which is the only record of what the run cost)
and `nfields` (the stored filtered LES fields -- 401 at the production `plotfreq = 1000`).
"""
function extract_new_reference(; force = false)
    src = NEW_REFERENCE_FILE
    out = joinpath(DATA_DIR, NEW_REFERENCE_CACHE)
    isfile(out) && !force && (@printf("  exists, skipping: %s (%.2f MB)\n", basename(out),
                                      filesize(out) / 2^20); return out)
    isfile(src) || (@warn "no regenerated HF reference at $src"; return nothing)
    mkpath(DATA_DIR)

    @printf("  reading %s (%.2f GB) ...\n", basename(src), filesize(src) / 2^30)
    flush(stdout)
    t0 = time()
    q_ref, dt_sample, comptime, nfields, params = jldopen(src, "r") do io
        d = io["data_train"]
        e = d.data[1]
        p = io["params_train"]
        # `params_train` comes back as a JLD2 reconstruction (it carries `ArrayType`, `backend` and
        # `filters`, none of which exist in this environment), so its fields are read through
        # `getproperty` and never splatted or reconstructed.
        (stack(e.qoi_hist), Float64(p.savefreq) * Float64(p.Δt), Float64(d.comptime),
         length(e.u), string(p))
    end
    jldsave(out; q_ref, dt_sample, comptime, nfields, source = abspath(src),
            source_bytes = filesize(src), params = params, extracted = string(now()))
    @printf("  read in %.1f s; wrote %s (%.2f MB), q_ref = %s at Δt_sample = %.4g\n",
            time() - t0, basename(out), filesize(out) / 2^20, size(q_ref), dt_sample)
    @printf("  run cost %.1f s = %.2f h; %d stored LES fields\n", comptime, comptime / 3600,
            nfields)
    flush(stdout)
    return out
end

# ---------------------------------------------------------------------------------------------
# D5 -- the online ensembles
# ---------------------------------------------------------------------------------------------

"""
    extract_ensemble(name; force = false)

Cache one configuration's online replica QoIs as `q` (a vector of `N_Q × nstep+1` matrices) and
`dQ` (`N_Q × nstep`).

`q_star` is not stored by the online driver, but it is recoverable exactly: `dQ^n = q^n - q^{n*}`
is enforced at `time_series_methods.jl:163-166`, and with `q[:, n+1]` the corrected QoI at step `n`
that gives `q_star[:, n] = q[:, n+1] - dQ[:, n]`. It is reconstructed and stored, because every
regime-B and regime-A score needs the predictor stream and nothing else in the archive carries it.

**One cache per configuration**, `online_<name>_<root>_qois.jld2`, holding the `primary_family` --
the family with the most replicas. The other families in a directory are the same runs under a stale
filename (see `replica_groups`), so caching them produced a duplicate that nothing ever loaded: for
`LinReg1` a 3.85 MB copy of replica 1 alone. They are now reported and skipped instead, and the
family that was taken is recorded inside the cache under `"family"`.
"""
function extract_ensemble(name::AbstractString; force = false)
    dir, root = model_dir(name)
    dir === nothing && (@warn "no archived model for $name"; return nothing)
    groups = replica_groups(dir)
    isempty(groups) && (@warn "no online replicas in $dir"; return nothing)
    mkpath(DATA_DIR)

    label = primary_family(groups)
    for other in sort(collect(keys(groups)))
        other == label && continue
        @printf("  skipping duplicate family %-45s %d replica(s) -- same runs, stale filename\n",
                other, length(groups[other]))
    end

    outs = String[]
    let entries = groups[label]
        out = joinpath(DATA_DIR, "online_$(name)_$(root)_qois.jld2")
        push!(outs, out)
        if isfile(out) && !force
            @printf("  exists, skipping: %s (%.2f MB)\n", basename(out), filesize(out) / 2^20)
            return outs
        end
        @printf("  family %-45s %d replica(s): %s\n", label, length(entries),
                string(first.(entries)))
        flush(stdout)

        qs, dQs, taus = Matrix{Float32}[], Matrix{Float32}[], Matrix{Float32}[]
        t0 = time()
        for (_, f) in entries
            p = joinpath(dir, f)
            @printf("    %-72s %.0f MB ... ", f, filesize(p) / 2^20)
            flush(stdout)
            d = load(p, "data_online")
            push!(qs, Array(d.q))
            push!(dQs, Array(d.dQ))
            push!(taus, hasproperty(d, :tau) ? Array(d.tau) : zeros(Float32, 0, 0))
            @printf("q = %s\n", size(d.q))
            flush(stdout)
            d = nothing
            GC.gc()                      # 144 MB of velocity snapshots per file
        end
        q_star = [q[:, 2:end] .- dQ for (q, dQ) in zip(qs, dQs)]
        jldsave(out; q = qs, dQ = dQs, tau = taus, q_star,
                replicas = last.(entries), replica_index = first.(entries),
                family = label, name = name, root = string(root),
                source_dir = abspath(dir), extracted = string(now()))
        @printf("    %d replicas in %.1f s; wrote %s (%.2f MB)\n", length(entries), time() - t0,
                basename(out), filesize(out) / 2^20)
        flush(stdout)
    end
    return outs
end

"""
    load_ensemble(name; root = nothing)

Load a cached online ensemble, extracting it first if needed. Returns
`(; q, q_star, dQ, tau, replicas, name, root)` with one matrix per replica.
"""
function load_ensemble(name::AbstractString; root = nothing, family = nothing)
    dir, r = model_dir(name)
    dir === nothing && error("no archived model for $name")
    root = something(root, r)
    # `family` is accepted only so a caller can assert which one it got; the cache name no longer
    # carries it, because the families are the same runs and only the primary one is cached.
    p = joinpath(DATA_DIR, "online_$(name)_$(root)_qois.jld2")
    isfile(p) || extract_ensemble(name)
    isfile(p) || error("extraction produced no cache at $p")
    d = load(p)
    return (; q = d["q"], q_star = d["q_star"], dQ = d["dQ"], tau = d["tau"],
            replicas = d["replicas"], replica_index = d["replica_index"],
            family = d["family"], name = d["name"], root = Symbol(d["root"]))
end

"""
    load_reference()

Load the cached HF reference QoI trajectory, extracting it first if needed.
"""
function load_reference()
    p = joinpath(DATA_DIR, "hf_reference_tsim100.0_qois.jld2")
    isfile(p) || extract_reference()
    return load(p, "q_ref")
end

"""
    load_new_reference()

Load the cached **regenerated** HF reference, extracting it first if needed.

Returns a NamedTuple, not a bare matrix: `dt_sample` is what keeps a consumer off D3's hard-coded
axis, and `comptime`/`nfields` are the only surviving record of what the 20.7 h run produced.
"""
function load_new_reference()
    p = joinpath(DATA_DIR, NEW_REFERENCE_CACHE)
    isfile(p) || extract_new_reference()
    isfile(p) || error("extraction produced no cache at $p")
    d = load(p)
    return (; q_ref = d["q_ref"], dt_sample = d["dt_sample"], comptime = d["comptime"],
            nfields = d["nfields"], source = d["source"], path = p)
end

const ARCHIVED_CONFIGS = ("LinReg1", "LinReg63", "LinReg64", "LinReg73", "LinReg74")

if abspath(PROGRAM_FILE) == @__FILE__
    targets = isempty(ARGS) ? ["reference", "new-reference", ARCHIVED_CONFIGS...] : ARGS
    println("extracting archive QoIs to ", DATA_DIR)
    println("  frozen root: ", FROZEN)
    println("  dev root:    ", DEV)
    flush(stdout)
    for t in targets
        println()
        println("== ", t)
        flush(stdout)
        if t == "reference"
            extract_reference()
        elseif t == "new-reference"
            extract_new_reference()
        else
            extract_ensemble(t)
        end
    end
end
