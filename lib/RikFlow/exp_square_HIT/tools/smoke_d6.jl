# D6 pre-flight. Run this before requesting any allocation.
#
# plan P2's pre-flight was HIT, 1 TU, 400 steps, 1 replica, about 0.01 SBU. This is that, plus the
# two checks D6 adds:
#
#   * 🔴 the written output carries **no** velocity fields. That is the difference between 160 MB
#     and 80 GB over the full 1800 runs, and it is silent if wrong. ⚠️ Note what is and is not
#     claimed: `savefreq > nt` still leaves **one** `t = 0` field in memory, because `qoisaver`'s
#     `state[] = state[]` (`RikFlow.jl:332`) notifies the already-registered `fieldsaver`. What
#     keeps the disk cost at zero is that `run_d6.jl`'s `jldsave` has no `fields` key, and that is
#     what is asserted here.
#   * 🔴 `ou_advance` is actually wired through and actually changes the trajectory. The unit tests
#     prove the replay is correct arithmetic and `analysis/ou_replay.jl` proves the advance count is
#     right, but neither touches `online_sgs`. If the keyword were dropped on the floor between the
#     driver and `Setup`, every member's forcing would be out of phase with its own initial
#     condition, the spread-skill ratio would be biased downward, and nothing would say so.
#
# Usage (from exp_square_HIT/):
#   julia --project tools/smoke_d6.jl
#
# Writes into output/D6_smoke/, which is throwaway. ⚠️ 400 steps is a pipeline check, not a
# measurement: the lead grid reaches 2172 steps and nothing here may be reported as a result.

using Random
using JLD2
using Printf
using RikFlow
using IncompressibleNavierStokes
using CUDA

include(joinpath(@__DIR__, "run_d6.jl"))

const SMOKE_DIR = joinpath(EXP_DIR, "output", "D6_smoke")

"""
Forecast steps for the pre-flight. Default 300, so with the 100 warm-up steps the run is 400 steps
= 1 TU, which is plan P2's pre-flight and its ~0.01 SBU figure. Leave it alone on Snellius.

⚠️ `D6_SMOKE_LEAD` shortens it, and the reason is the CPU. On a GPU this whole script is ~10 s of
compute; on a CPU the same 640 solver steps (400 here plus 2 x 120 for the `ou_advance`
comparison) take 5-20 minutes, because every step carries several 64^3 FFTs for `compute_QoI` plus
`to_sgs_term`. All three things this script actually decides -- the output keys and shapes, that
**no velocity fields were written**, and that **`ou_advance` changes the trajectory** -- are settled
in the first handful of steps, so `D6_SMOKE_LEAD=20` is a complete check in about two minutes when
you only want to know whether the pipeline runs at all.
"""
const SMOKE_LEAD = parse(Int, get(ENV, "D6_SMOKE_LEAD", "300"))

"""
    trajectory(pkg; ou_advance, nlead = 20, seed = 1)

One short forecast from an IC package, returning its `q`. Used only to compare `ou_advance = 0`
against `ou_advance = n_k` with everything else -- the field, the model, the seed -- held fixed.
"""
function trajectory(pkg; ou_advance::Int, nlead::Int = 20, seed = 1)
    # 🔴 Float64, and it must match `run_ic`'s (2026-09-16). This helper builds its own `params`
    # instead of going through `run_ic`, so it carried its own `T` -- and at Float32, with the IC
    # package's `Re` being Float64, `online_sgs`'s own assertion fires:
    # `got Δt = 0.0025, tsim/nt = 0.002499999850988388`. The OU replay and the reference would then
    # step the chain differently, which is exactly what this function exists to test.
    T = Float64
    gpu = CUDA.functional()
    ArrayType = gpu ? CuArray : Array
    backend = gpu ? CUDABackend() : IncompressibleNavierStokes.CPU()
    p = pkg["params"]
    nwarm = pkg["provenance"].nwarm
    nt = nwarm + nlead
    Δt = T(p.Δt)
    mdl = model_file()
    hist_len, hist_var = load(mdl, "hist_len", "hist_var")
    nq = size(p.qois, 1)
    q_hist = ArrayType{T}(zeros(T, nq, hist_len))
    hist_var == :q_star_q && (q_hist = cat(q_hist, q_hist, dims = 1))
    sampler = RikFlow.LinReg(mdl, Xoshiro(seed), ArrayType;
                             q_hist, spinnup_data = ArrayType{T}(pkg["dQ_warm"]))
    d = online_sgs(; p..., tsim = T(Δt * nt), Δt, ArrayType, backend, savefreq = nt + 1,
                   ustart = ArrayType(pkg["u"]), time_series_method = sampler, ou_advance)
    return Array(d.q)
end

function main()
    @printf("D6 pre-flight, %s, %d forecast steps%s\n", CUDA.functional() ? "GPU" : "CPU",
            SMOKE_LEAD, SMOKE_LEAD == 300 ? " (plan P2's 1 TU pre-flight)" : " (SHORTENED)")
    SMOKE_LEAD == 300 || @warn "D6_SMOKE_LEAD = $SMOKE_LEAD, not the 300 that makes this plan P2's " *
                               "1 TU pre-flight. Fine as a pipeline check; do not quote a timing " *
                               "or an SBU figure from it."
    flush(stdout)

    # --- 1. the pre-flight run -----------------------------------------------------------------
    run_ic(1; M = 1, nlead = SMOKE_LEAD, od = SMOKE_DIR, force = true)

    files = filter(f -> startswith(f, "d6_online_"), readdir(SMOKE_DIR))
    @assert length(files) == 1 "expected one member file, got $files"
    d = load(joinpath(SMOKE_DIR, files[1]))
    for key in ("q", "dQ", "tau", "k", "n_k", "t_k", "seed", "ou_advance", "nwarm", "nlead",
                "model", "hist_len", "hist_var", "wall_seconds")
        @assert haskey(d, key) "output is missing key $key"
    end
    nt = d["nwarm"] + d["nlead"]
    @assert size(d["q"], 2) == nt + 1 "q has $(size(d["q"], 2)) columns, expected $(nt + 1)"
    @assert size(d["dQ"], 2) == nt "dQ has $(size(d["dQ"], 2)) columns, expected $nt"
    @assert !any(isnan, d["q"]) "q contains NaN"
    @assert !any(isnan, d["dQ"]) "dQ contains NaN"
    @assert d["ou_advance"] == d["n_k"] "ou_advance is $(d["ou_advance"]), expected n_k = $(d["n_k"])"
    @printf("  keys, shapes, finiteness: ok  (%.0f kB, %.2f s/TU)\n",
            filesize(joinpath(SMOKE_DIR, files[1])) / 1024,
            d["wall_seconds"] / (nt * 2.5e-3))

    # 🔴 The 80 GB check, and the one that actually decides whether the full array is affordable:
    # not "were fields computed" but "were fields *written*". `run_ic` bounds the in-memory count
    # at the single t = 0 snapshot; this asserts none of it reaches the file.
    @assert !haskey(d, "fields") "output carries velocity fields; savefreq did not suppress them"
    @printf("  no velocity fields in the output file: ok  (%.0f kB total)\n",
            filesize(joinpath(SMOKE_DIR, files[1])) / 1024)

    # --- 2. the clamp, counted exactly ---------------------------------------------------------
    # A step on which the stabiliser fired has an identically zero `dQ` column. Measured everywhere
    # else on HIT to never fire; if it does here, every later number is partly the clamp's.
    nfired = count(c -> all(iszero, @view d["dQ"][:, c]), (d["nwarm"] + 1):nt)
    @printf("  clamp fired on %d of %d forecast steps%s\n", nfired, nt - d["nwarm"],
            nfired == 0 ? " (as expected)" : "  ⚠️ UNEXPECTED on HIT")
    flush(stdout)

    # --- 3. ou_advance is wired through and does something --------------------------------------
    pkg = load_ic(1)
    q_on = trajectory(pkg; ou_advance = pkg["n_k"])
    q_off = trajectory(pkg; ou_advance = 0)
    @assert size(q_on) == size(q_off)
    @assert q_on[:, 1] ≈ q_off[:, 1] "the two runs did not start from the same field"
    same = q_on == q_off
    dev = maximum(abs.(q_on .- q_off) ./ max.(abs.(q_off), eps(Float32)))
    @assert !same "ou_advance = $(pkg["n_k"]) produced a trajectory identical to ou_advance = 0; " *
                  "the keyword is not reaching the OU chain and every member's forcing would be " *
                  "$(pkg["n_k"]) steps out of phase with its own initial condition"
    @printf("  ou_advance changes the trajectory: max relative deviation %.3g over %d steps\n",
            dev, size(q_on, 2) - 1)

    println("\npre-flight passed. ⚠️ 400 steps is a pipeline check, not a measurement — the lead " *
            "grid\nreaches 2172 steps. Submit batch_scripts/run_d6.sh with --array=1-5 next, and " *
            "write the\nmeasured s/TU into meta_files/handoff_p2c_d6.md section 2.")
    flush(stdout)
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
