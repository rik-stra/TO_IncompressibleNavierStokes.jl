# The GATE 3 check of a long-run smoke (WORKFLOW.md Stage 3; 2026-10-23, step 3a), run by
# `batch_scripts/smoke_online_lstm.sh` after its replicas:
#
#     julia --project tools/smoke_online_check.jl <tsim> <fit dir> [<fit dir> ...]   # from exp_square_HIT/
#
# `<fit dir>` is relative to output/TO_LSTM, as `run_d6_train_lstm.sh` takes it. For each, the replica
# `data_online_tsim<tsim>_replica1_smoke.jld2` that `12_online_StochLSTM.jl` wrote into the fit dir is
# first MOVED to output/smoke_online/<name>/, so no smoke file sits beside the production replicas,
# then checked:
#   - the run is complete: tsim / Δt steps;
#   - the warm-up `dQ[:, 1:nwarm]` is bit-identical to the record's (#48), the one exact check;
#   - `q` and `dQ` are finite;
#   - the gate fired nowhere after the warm-up (`gate_census`, as `run_d6.jl` counts it; on HIT any
#     firing is an alarm, #65);
#   - the file records the fit dir's own StochLSTM_seed1.jld2.
# Exits 1 if any check fails. ⚠️ Not a measurement: 100 forecast steps, one replica (#44).

using JLD2
using Printf

include(joinpath(@__DIR__, "d6_lib.jl"))    # gate_census

length(ARGS) >= 2 || error("usage: julia tools/smoke_online_check.jl <tsim> <fit dir> [<fit dir> ...]")
tsim = parse(Float64, ARGS[1])
fits = ARGS[2:end]

const EXP = dirname(@__DIR__)
track_file = get(ENV, "RIKFLOW_TRACK_FILE",
                 joinpath(EXP, "output", "data_track_dns512_les64_Re2000.0_tsim100.0_f64_lmwray3.jld2"))
dQ_rec = load(track_file, "data_track").dQ

failures = String[]
for fit in fits
    name = basename(rstrip(fit, '/'))
    fitdir = joinpath(EXP, "output", "TO_LSTM", fit)
    src = joinpath(fitdir, "data_online_tsim$(tsim)_replica1_smoke.jld2")
    dst_dir = joinpath(EXP, "output", "smoke_online", name)
    dst = joinpath(dst_dir, basename(src))
    mkpath(dst_dir)
    isfile(src) && mv(src, dst; force = true)
    if !isfile(dst)
        push!(failures, "$name: no replica file")
        println("🔴 $name: no $(basename(src)) (see $(joinpath(dst_dir, "run.log")))")
        continue
    end
    d, params, nwarm, model_file = load(dst, "data_online", "params", "nwarm", "model_file")
    nexp = round(Int, tsim / params.Δt)
    nstep = size(d.q, 2) - 1
    dQ = d.dQ[:, 1:min(nstep, size(d.dQ, 2))]
    gc = gate_census(dQ, nwarm)
    checks = [
        ("complete ($nstep of $nexp steps)", nstep == nexp),
        ("warm-up dQ[:, 1:$nwarm] bit-identical to the record",
         size(dQ, 2) >= nwarm && view(dQ, :, 1:nwarm) == view(dQ_rec, :, 1:nwarm)),
        ("q and dQ finite", all(isfinite, d.q) && all(isfinite, dQ)),
        ("gate fired $(gc.nfired) / $(gc.nsteps) after the warm-up", gc.nfired == 0),
        ("model_file is the fit dir's", isfile(model_file) &&
         realpath(model_file) == realpath(joinpath(fitdir, "StochLSTM_seed1.jld2"))),
    ]
    for (what, ok) in checks
        println(ok ? "  ok   " : "  🔴   ", rpad(name, 22), what)
        ok || push!(failures, "$name: $what")
    end
end

if isempty(failures)
    println("PASS: $(length(fits)) closure(s)")
else
    println("FAIL:\n  ", join(failures, "\n  "))
    exit(1)
end
