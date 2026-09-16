# D6: one initial condition, M members, on Snellius.
#
# One array task = one IC = M members run sequentially. `ARGS[1]` is the **ordinal** (1..K) of the
# initial condition, not the field index `k`; the map from one to the other is `select_ics`'s and is
# deterministic, so scaling `--array=1-5` to `--array=1-180` renumbers nothing and the pilot's five
# ICs are the first five of the same set the full run uses.
#
# Usage:
#   julia --project tools/run_d6.jl <ordinal>
#
# Environment overrides, all optional:
#   D6_IC_DIR    where the `d6_ic_<k>.jld2` packages live
#   D6_MODEL     the fitted M0 to deploy (default: LinReg1 under this experiment's output tree)
#   D6_OUT       where to write `d6_online_ic<k>_m<member>.jld2`
#   D6_MEMBERS   M, default 10
#
# 🔴 Three things in here are load-bearing and none of them is obvious.
#
#  1. **`ou_advance = n_k`.** The initial condition is the reference's field at step `n_k`, and the
#     OU forcing that produced it is `n_k` steps into its own chain. `Setup` starts every chain at
#     zero, so without the replay every member's forcing is out of phase with its own initial
#     condition -- which inflates skill without inflating spread and biases the spread-skill ratio
#     downward, toward a false "over-confident" verdict. Verified by measurement in
#     `analysis/ou_replay.jl`; see `claude_memory.md` gotcha #33.
#  2. **`savefreq > nt`, so at most the single `t = 0` field is held.** `params_track` carries
#     `savefreq = 100`; at 1308 steps that is ~13 snapshots x 3.3 MB per run, i.e. **80 GB** over
#     1800 runs against **160 MB** of QoIs.
#
#     🔴 **`savefreq > nt` does NOT give zero fields, and an earlier version of this comment said
#     it did.** `fieldsaver` appends when `state.n % nupdate == 0` (`processors.jl:306-316`), and
#     `qoisaver`'s initializer does `state[] = state[]` -- *"invokes all processors on initial
#     state!"*, `RikFlow.jl:332` -- which is what gives `q` its `nstep+1` columns and therefore the
#     offset the whole index alignment rests on. `fieldsaver` is registered **before** `qoisaver`
#     (`LFsims.jl:61,159`, deliberately), so it is already listening at that notification and
#     `0 % anything == 0`. Measured 2026-09-10: exactly **1** field, always.
#
#     That is 3.3 MB of transient memory per run and **nothing on disk**: `jldsave` below writes
#     `q`, `dQ` and `tau` and no `fields` key at all, which is what actually prevents the 80 GB.
#     The assertion therefore allows the one initial field and refuses any more.
#  3. **The IC and the forcing are shared across the M members; only the model seed varies.** That
#     is what makes the rank histogram measure the surrogate's own dispersion and nothing else.
#
# ⚠️ Julia buffers stdout when redirected, so an empty `.out` file says nothing about progress --
# two healthy jobs have already been killed on that misreading. Every progress line here is
# followed by `flush(stdout)`.

if false                                                        #src
    include("../../src/RikFlow.jl")                             #src
    include("../../../../src/IncompressibleNavierStokes.jl")    #src
end                                                             #src

using Random
using JLD2
using Printf
using Dates
using RikFlow
using IncompressibleNavierStokes
using CUDA

# `select_ics`, `ic_path`, `manifest_path`, `N_WARM`, `N_LEAD` -- the one definition of D6's IC set.
include(normpath(joinpath(@__DIR__, "..", "..", "analysis", "build_d6_ics.jl")))

const EXP_DIR = normpath(joinpath(@__DIR__, ".."))

"""
    ic_dir()

Where the IC packages are. `D6_IC_DIR` wins; otherwise this experiment's own output tree (where a
Snellius copy lands) and then the local build directory, so the same script runs on both machines
without editing.
"""
function ic_dir()
    haskey(ENV, "D6_IC_DIR") && return ENV["D6_IC_DIR"]
    for d in (joinpath(EXP_DIR, "output", "d6_ics"), DEFAULT_IC_DIR)
        isfile(manifest_path(d)) && return d
    end
    error("no IC packages found: set D6_IC_DIR, or run analysis/build_d6_ics.jl first")
end

model_file() = get(ENV, "D6_MODEL", joinpath(EXP_DIR, "output", "TO_LRS", "LinReg1", "LinReg.jld2"))

"""
    closure()

Which closure to forecast with: `:lrs` (default) or `:ddn`, from `D6_CLOSURE`.

🔴 **The DDN needs this to enter D6 at all**, and until 2026-09-16 it could not: `MVG_sampler` took
no `spinnup_data`, so a DDN forecast would have started sampling immediately while the LRS was being
advanced through `nwarm` recorded steps. The two closures would then forecast from **different
physical states** and every lead-resolved difference would carry that offset (memory #55).
"""
function closure()
    c = Symbol(lowercase(get(ENV, "D6_CLOSURE", "lrs")))
    c in (:lrs, :ddn) || error("D6_CLOSURE must be lrs or ddn; got $c")
    return c
end

"The DDN's training slice, written once beside the IC packages by `analysis/build_d6_ics.jl`."
ddn_file() = get(ENV, "D6_DDN_DATA", joinpath(ic_dir(), "d6_ddn_traindata.jld2"))
out_dir() = get(ENV, "D6_OUT", joinpath(EXP_DIR, "output", "D6"))
n_members() = parse(Int, get(ENV, "D6_MEMBERS", "10"))

"""
    member_seed(k, member)

The model seed for one member. Shared IC, shared forcing, distinct seed -- so the ensemble's spread
is the surrogate's own and nothing else's.

The value is written into the output file, which is what makes a run reproducible: `hash` is not
promised to be stable across Julia versions, so the recorded seed, not this expression, is the
record of what was drawn.
"""
member_seed(k::Integer, member::Integer) = hash((:d6, Int(k), Int(member)))

"""
    validation_seed(member)

The **archived** driver's seed for replica `member`: `Xoshiro(seeds.to + i + 2)` with
`seeds.to = 234` (`6_online_TO_LRS.jl:37-41,83`), i.e. `Xoshiro(236 + member)`.

Used only by ordinal 0. Reusing the archive's stream is what makes the validation run comparable
to the archived trajectory column by column instead of merely in distribution; every *scored*
member goes through `member_seed` and therefore shares a stream with nothing archived.
"""
validation_seed(member::Integer) = UInt64(DRIVER_SEED_BASE + Int(member))

"""
    check_ic_alignment(q0, q_window, offsets; qois, median_tol = 1e-4, margin = 10.0)

Verify on the compute node that the field the driver was handed really is at the step its package
claims, by asking which reference column it best matches.

🔴 **Why this is a comparison and not a threshold.** The first version of this check asserted
`maximum(abs.(q0 - q_at_ic) ./ abs.(q_at_ic)) < 1e-3` against a single stored column. That cannot
work, and it failed the first validation run at 1.06e-3. Two discrepancies live in that range and
they mean opposite things:

  * **a one-step misalignment** moves these QoIs by 2.1e-4 to 3.2e-3 relative — the thing to catch;
  * **`Z[16,32]`** disagrees by **1.06e-3** between the current code and the archived code on the
    *same* velocity field, reproducibly and on CPU as well as GPU (`claude_memory.md` gotcha #45).
    The other five QoIs recompute to 6e-8 to 2.2e-7, i.e. Float32 round-off.

So an absolute bound on the max over QoIs cannot separate "wrong step" from "known code
difference". Two changes fix that:

 1. **Compare across columns, not against a tolerance.** The claimed column must be the *best*
    match in its window. Any discrepancy that is constant across neighbouring columns — which a
    per-QoI code difference is — cancels out of that comparison entirely.
 2. **Aggregate with the median over QoIs, not the maximum.** The median ignores one bad band by
    construction. Measured: **1.2e-7** at the right column against **~9e-4** one column away, a
    margin of roughly 7000x, where the max-based statistic had no margin at all.

Returns `(; ok, message, report, best_offset, median_rel, per_qoi)`. `report` is printed by the
driver for member 1 so the per-QoI numbers are in the log whether or not the check passes — that is
what turns the next failure into one line instead of an investigation.
"""
function check_ic_alignment(q0::AbstractVector, q_window::AbstractMatrix,
                            offsets::AbstractVector; qois = nothing,
                            median_tol::Real = 1e-4, margin::Real = 10.0)
    labels = qois === nothing ? ["q$i" for i in eachindex(q0)] :
             ["$(q[1])[$(q[2]),$(q[3])]" for q in qois]
    relof(c) = abs.(Float64.(q0) .- Float64.(view(q_window, :, c))) ./
               max.(abs.(Float64.(view(q_window, :, c))), floatmin(Float64))
    med(v) = (s = sort(v); n = length(s);
              isodd(n) ? s[(n + 1) ÷ 2] : (s[n ÷ 2] + s[n ÷ 2 + 1]) / 2)

    meds = [med(relof(c)) for c in axes(q_window, 2)]
    icentre = findfirst(==(0), offsets)
    icentre === nothing && return (; ok = false,
        message = "IC package's q_window_offsets $(collect(offsets)) contains no 0, so the IC's " *
                  "own column is not in the window; rebuild with analysis/build_d6_ics.jl",
        report = "", best_offset = nothing, median_rel = NaN, per_qoi = Float64[])

    ibest = argmin(meds)
    per = relof(icentre)

    io = IOBuffer()
    println(io, "  IC alignment (recomputed q[:,1] against the record's own column):")
    @printf(io, "    %-10s %15s %15s %12s\n", "QoI", "run q[:,1]", "record", "rel")
    for i in eachindex(q0)
        flag = per[i] > 1e-3 ? "  <- see gotcha #45" : ""
        @printf(io, "    %-10s %15.7e %15.7e %12.3e%s\n", labels[i], Float64(q0[i]),
                Float64(q_window[i, icentre]), per[i], flag)
    end
    @printf(io, "    median rel at offset 0: %.3e   (tolerance %.0e)\n", meds[icentre], median_tol)
    @printf(io, "    neighbours:")
    for (j, off) in pairs(offsets)
        j == icentre && continue
        @printf(io, "  %+d: %.2e", off, meds[j])
    end
    println(io)
    report = String(take!(io))

    if offsets[ibest] != 0
        return (; ok = false, report,
            message = "IC MISALIGNED: the run's q[:,1] matches the record at offset " *
                      "$(offsets[ibest]) (median rel $(meds[ibest])), not at the package's own " *
                      "column (offset 0, median rel $(meds[icentre])). The field is not at step " *
                      "the package claims.\n" * report,
            best_offset = offsets[ibest], median_rel = meds[icentre], per_qoi = per)
    end
    runner = minimum(meds[j] for j in eachindex(meds) if j != icentre)
    if !(meds[icentre] <= median_tol && runner >= margin * meds[icentre])
        return (; ok = false, report,
            message = "IC alignment is not convincing: median rel at offset 0 is " *
                      "$(meds[icentre]) (tolerance $median_tol) and the nearest neighbour is " *
                      "$runner, a margin of $(runner / meds[icentre])x against the required " *
                      "$(margin)x.\n" * report,
            best_offset = 0, median_rel = meds[icentre], per_qoi = per)
    end
    return (; ok = true, message = "", report, best_offset = 0, median_rel = meds[icentre],
            per_qoi = per)
end

"""
    load_ic(ordinal; dir = ic_dir())

The IC package for array-task `ordinal`, with the manifest cross-checked against `select_ics`.

The manifest is `select_ics`'s recorded output, so comparing the two is not circular bookkeeping: it
catches an IC set built with a different `K`, `nlead` or record from the one this driver assumes,
which would otherwise show up only as a silently misaligned truth at scoring time.
"""
function load_ic(ordinal::Integer; dir = ic_dir())
    # 🔑 Ordinal 0 is the validation IC -- `fields[1]` of R1, the initial condition R2's own
    # initial condition. It is not in the manifest and must not be: it sits inside M0's fit window
    # and V28 needs D6's scored set disjoint from it. See `build_validation_ic`.
    if ordinal == 0
        p = validation_path(dir)
        isfile(p) || error("no validation IC at $p; run " *
                           "`julia --project=analysis analysis/build_d6_ics.jl validation`")
        pkg = load(p)
        get(pkg, "validation", false) || error("$p is not marked as a validation package")
        pkg["n_k"] == 0 || error("validation IC is at step $(pkg["n_k"]), expected 0")

        # 🔴 **Ordinal 0 used to check NOTHING but those two lines, and it is the one run whose whole
        # job is catching mistakes.** Scored ordinals cross-check the manifest against `select_ics`
        # below and fail loudly on a stale IC set; the validation package is not in the manifest, so
        # it had no equivalent. Measured 2026-09-16: `ic_dir()` prefers
        # `exp_square_HIT/output/d6_ics`, which still held the **2026-09-10 archive-built** packages
        # (`data_track_trackingnoise_..._tsim10.0`, Float32, `nwarm = 100, nlead = 1208`). A
        # validation run picked them up silently, returned Float32 everywhere, and could not have
        # passed its own `dQ` bit-identity gate -- with nothing in the output saying why.
        prov = pkg["provenance"]
        prov.nlead == N_LEAD || error(
            "validation IC at $p was built with nlead = $(prov.nlead), but this driver is at " *
            "N_LEAD = $N_LEAD. The IC set is stale -- rebuild with analysis/build_d6_ics.jl, or " *
            "point D6_IC_DIR at the current one.")
        prov.nwarm == N_WARM_DRIVER || error(
            "validation IC at $p was built with nwarm = $(prov.nwarm), expected " *
            "N_WARM_DRIVER = $N_WARM_DRIVER (what the online drivers replay).")
        eltype(pkg["u"]) === eltype(pkg["dQ_warm"]) || error(
            "validation IC mixes precisions: u is $(eltype(pkg["u"])), dQ_warm is " *
            "$(eltype(pkg["dQ_warm"]))")
        return pkg
    end
    man = load(manifest_path(dir))
    K = man["K"]
    1 <= ordinal <= K || error("ordinal $ordinal is outside 1..$K")
    sel = select_ics(; K)
    man["k"] == sel.k || error("manifest disagrees with select_ics(K = $K); the IC packages were " *
                               "built from a different configuration than this driver assumes")
    k = man["k"][ordinal]
    pkg = load(ic_path(k, dir))
    pkg["ordinal"] == ordinal || error("package $(ic_path(k, dir)) says ordinal $(pkg["ordinal"])")
    pkg["n_k"] == man["n"][ordinal] || error("package and manifest disagree on n_k for k = $k")
    return pkg
end

"""
    run_ic(ordinal; M = n_members(), force = false, nlead = nothing, od = out_dir())

Run all `M` members for one initial condition and write one file per member.

`nlead` shortens the forecast and `od` redirects the output; both exist for `smoke_d6.jl` and must
be left at their defaults for anything whose numbers are reported. A short run is a pipeline check,
not a measurement -- the lead grid's longest entry is 1086 steps.
"""
function run_ic(ordinal::Integer; M::Integer = n_members(), force::Bool = false,
                nlead = nothing, od = out_dir())
    # 🔴 **Float64, changed 2026-09-16 (Rik).** It was `Float32`, and that was wrong in three ways at
    # once on the rebaselined pipeline:
    #
    #  1. **It did not set the solver's precision.** `online_sgs` takes `T = typeof(Re)`, and the IC
    #     package carries R1's `Re = 2000.0` as a **Float64**. So the solver already ran Float64
    #     while this `T` cast `Δt`, `tsim` and the warm-up down to Float32 around it.
    #  2. **It broke D6's own acceptance gate.** The gate is `dQ` bit-identity over the replayed
    #     window (memory #48). `ArrayType{Float32}(dQ_warm)` on a Float64 package is not
    #     bit-identical to the record -- measured, `Array{Float32}(dQ_warm) == dQ_warm` is `false` --
    #     so the validation could never have passed.
    #  3. **It made the two closures replay different warm-ups.** The LRS went through
    #     `ArrayType{T}(...)` and the DDN through `Array(...)`, so at `T = Float32` they entered the
    #     forecast from different states -- defeating the whole point of giving `MVG_sampler` a
    #     warm-up (memory #55, #66).
    #
    # Everything the package carries is already Float64: `Re`, `Δt`, `u`, `ou_bodyforce`. This makes
    # the driver agree with them instead of casting around them.
    T = Float64
    gpu = CUDA.functional()
    ArrayType = gpu ? CuArray : Array
    backend = gpu ? CUDABackend() : IncompressibleNavierStokes.CPU()

    pkg = load_ic(ordinal)
    validation = get(pkg, "validation", false)
    k, n_k, t_k = pkg["k"], pkg["n_k"], pkg["t_k"]
    params_ic = pkg["params"]
    dQ_warm = pkg["dQ_warm"]
    q_at_ic = pkg["q_at_ic"]
    haskey(pkg, "q_window") || error("IC package has no q_window: it predates the alignment " *
        "check (2026-09-11). Rebuild with analysis/build_d6_ics.jl and re-copy.")
    q_window = pkg["q_window"]
    q_window_offsets = pkg["q_window_offsets"]
    nwarm = pkg["provenance"].nwarm
    nlead = nlead === nothing ? pkg["provenance"].nlead : Int(nlead)
    nlead == pkg["provenance"].nlead ||
        @warn "forecast shortened to $nlead steps from $(pkg["provenance"].nlead); this is a " *
              "pipeline check, not a measurement -- the lead grid reaches 1086 steps"

    Δt = T(params_ic.Δt)
    nt = nwarm + nlead
    tsim = T(Δt * nt)
    # `online_sgs` asserts this too, before it replays anything; asserting it here as well means the
    # job fails in the first second rather than after Julia has compiled the solver.
    round(Int, tsim / Δt) == nt || error("tsim/Δt is not $nt")
    T(tsim) / nt === Δt || error("tsim/nt = $(T(tsim)/nt) is not Δt = $Δt; the OU replay and the " *
                                 "reference would step the chain differently")

    cl = closure()
    if cl === :lrs
        mdl = model_file()
        isfile(mdl) || error("no model at $mdl (set D6_MODEL)")
        hist_len, hist_var = load(mdl, "hist_len", "hist_var")
        dQ_train = nothing
    else
        mdl = ddn_file()
        isfile(mdl) || error("no DDN training data at $mdl; run analysis/build_d6_ics.jl " *
                             "(or set D6_DDN_DATA)")
        dQ_train = load(mdl, "dQ_train")
        # The DDN has no history at all -- that is what makes its warm-up "pseudo". Reported as 0
        # rather than left undefined, so the output file says which closure produced it.
        hist_len, hist_var = 0, :none
    end
    nq = size(params_ic.qois, 1)

    mkpath(od)

    @printf("D6 ordinal %d -> field k = %d, n_k = %d, t_k = %.2f TU\n", ordinal, k, n_k, t_k)
    @printf("  %s, %d members, %d steps (%d warm-up + %d forecast) = %.4f TU, Δt = %g\n",
            gpu ? "GPU (CuArray)" : "CPU (Array)", M, nt, nwarm, nlead, tsim, Δt)
    @printf("  closure %s, model %s: hist_len = %d, hist_var = %s\n", cl,
            basename(mdl), hist_len, hist_var)
    @printf("  ou_advance = %d, savefreq = %d (> nt, so no fields)\n", n_k, nt + 1)
    if validation
        println("  🔑 VALIDATION run, not a scored one. This is R2's own online initial")
        println("     condition, so ou_advance = 0 -- the identity point of the replay -- and the")
        @printf("     model seeds are the driver's, Xoshiro(%d + member). The oracle is R2's\n",
                DRIVER_SEED_BASE)
        println("     LinReg1 replica 1, not paper 2's archive. Output goes to")
        println("     d6_valid_ic1_m*.jld2, which the scorer's glob cannot see. Compare it with")
        println("     `compare_validation` in analysis/score_d6.jl.")
    end
    flush(stdout)

    ustart = ArrayType(pkg["u"])
    params = (; params_ic..., tsim, Δt, ArrayType, backend, savefreq = nt + 1)

    t_all = time()
    walls = Float64[]                  # per-member wall time; member 1 carries the compilation
    for member in 1:M
        out = joinpath(od, validation ? "d6_valid_ic$(k)_m$(member).jld2" :
                                        "d6_online_ic$(k)_m$(member).jld2")
        if isfile(out) && !force
            @printf("  member %2d/%d: exists, skipping\n", member, M)
            flush(stdout)
            continue
        end

        seed = validation ? validation_seed(member) : member_seed(k, member)
        sampler = if cl === :lrs
            q_hist = ArrayType{T}(zeros(T, nq, hist_len))
            hist_var == :q_star_q && (q_hist = cat(q_hist, q_hist, dims = 1))
            RikFlow.LinReg(mdl, Xoshiro(seed), ArrayType;
                           q_hist, spinnup_data = ArrayType{T}(dQ_warm))
        else
            # 🔑 Same `dQ_warm`, same member seed, same IC. The DDN replays the identical warm-up
            # the LRS does, so both enter the forecast from the same physical state -- and the
            # replay does not consume the rng, so this member's first *sampled* dQ is what
            # `Xoshiro(seed)` would have given with no warm-up at all.
            # `Array`, not `ArrayType{T}`: `MVG_sampler` is CPU-side (it samples from a
            # Distributions object and `to_sgs_term` calls `Array(dQ)` anyway). At `T = Float64`
            # this is the same numbers the LRS replays, which is what the comment above requires.
            RikFlow.MVG_sampler(dQ_train, Xoshiro(seed); spinnup_data = Array{T}(dQ_warm))
        end

        t0 = time()
        data = online_sgs(; params..., ustart, time_series_method = sampler, ou_advance = n_k)
        wall = time() - t0

        # 🔴 The 80 GB guard. If this ever fires, stop -- do not "clean up afterwards".
        #
        # One field is expected and unavoidable: the `t = 0` snapshot that `qoisaver`'s
        # `state[] = state[]` forces through the already-registered `fieldsaver` (see item 2 in the
        # header). More than one means `savefreq` is not doing its job. None of them reaches disk --
        # `jldsave` below writes no `fields` key -- so this bounds transient memory, and
        # `smoke_d6.jl` separately asserts the written file carries no fields.
        nfields = length(data.fields)
        nfields <= 1 ||
            error("member $member wrote $nfields velocity fields; savefreq = $(nt + 1) should " *
                  "have written at most the one t = 0 snapshot. 1800 runs of ~13 is 80 GB.")
        all(f -> f.n == 0, data.fields) ||
            error("member $member saved a field at step(s) $([f.n for f in data.fields]); only " *
                  "the t = 0 snapshot is expected at savefreq = $(nt + 1)")
        size(data.q, 2) == nt + 1 ||
            error("q has $(size(data.q, 2)) columns, expected nt + 1 = $(nt + 1)")
        # The run's first `q` column is the QoIs of `ustart`, so it must be the reference column the
        # package was cut from. This is the non-circular half of "the package and the record agree
        # on what step the IC is at" -- the package carries the record's value, this compares it
        # against QoIs recomputed here from the field itself.
        q0 = Array(data.q)[:, 1]
        chk = check_ic_alignment(q0, q_window, q_window_offsets; qois = params_ic.qois)
        chk.ok || error(chk.message)
        member == 1 && print(chk.report)

        jldsave(out; q = Array(data.q), dQ = Array(data.dQ), tau = Array(data.tau),
                k, n_k, t_k, ordinal, member, seed, ou_advance = n_k, validation,
                nwarm, nlead, M, model = abspath(mdl), closure = string(cl), hist_len,
                hist_var,
                tsim, Δt, wall_seconds = wall,
                julia = string(VERSION), device = gpu ? "cuda" : "cpu", written = string(now()))

        push!(walls, wall)
        @printf("  member %2d/%d: %.1f s, q0 median rel %.1e, %.0f kB\n",
                member, M, wall, chk.median_rel, filesize(out) / 1024)
        flush(stdout)
    end

    total = time() - t_all
    @printf("done: ordinal %d (k = %d), %d members in %.1f s (%.1f s/member)\n",
            ordinal, k, M, total, total / M)

    # 🔴 Cost reporting, and why it is not `total / (M * tsim)`.
    #
    # Member 1 pays the whole solver + TO compilation, once per Julia process. Measured on Snellius
    # 2026-09-10 (gpu_a100, 400 steps, M = 1): wall 62.6 s of which only ~3.6 s was stepping --
    # `itertime` fell 0.013 -> 0.29 -> 0.057 -> 0.009 -> **0.0089** s/step as it warmed up. So the
    # naive `wall / tsim` read **62.55 s/TU** against a true steady rate of **3.56 s/TU**: a factor
    # 18 on the one number the handoff asks you to record, and worst exactly on the short runs
    # people do first. plan P2's own 5.8 s/TU smoke figure has the same defect.
    #
    # So: quote the **median over members 2..M** as the rate, because those pay no compilation, and
    # report member 1 separately as the per-task constant. With M = 1 there is no such member and
    # the script says so rather than printing a number that would be wrong.
    if length(walls) > 1
        steady = sort(walls[2:end])[cld(length(walls) - 1, 2)]
        @printf("  compile+first member %.1f s (per array task, once) · steady %.1f s/member = %.3f s/TU\n",
               walls[1], steady, steady / tsim)
        @printf("  ⚠️  write the STEADY %.3f s/TU into handoff_p2c_d6.md section 2, not the\n",
                steady / tsim)
        @printf("      first-member figure. plan's two SBU figures differ by 10x; the SBU rate for\n")
        @printf("      this partition has to come from Snellius accounting, not from either of them.\n")
    elseif !isempty(walls)
        @printf("  ⚠️  M = 1, so this run is all compilation: %.1f s wall is NOT a rate. Re-run\n",
                walls[1])
        @printf("      with M > 1, or read the steady `itertime` off the solver's own log.\n")
    end
    flush(stdout)
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    isempty(ARGS) && error("usage: run_d6.jl <ordinal>   (1..K the array task id; " *
                           "0 = the validation IC, fields[1] of the 10 TU record)")
    run_ic(parse(Int, ARGS[1]))
end
