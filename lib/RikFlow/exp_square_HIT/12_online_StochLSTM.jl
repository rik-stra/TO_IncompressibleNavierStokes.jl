# M4 deployed in the solver -- the D-online half of the experiment.
#
#     julia --startup-file=no --project=lib/RikFlow \
#           lib/RikFlow/exp_square_HIT/12_online_StochLSTM.jl <model_index> [replica]
#
# ⚠️ Runs under `--project=lib/RikFlow`, **not** the training environment. The deployed closure is
# stdlib plus the package; Lux is not loaded and must not be. If this script ever needs
# `using Lux`, something has been put on the wrong side of the split.
#
# 🔴 **Sørensen et al. do not do this.** They apply the model as a post-processing step on a
# finished low-fidelity trajectory, outside the solver, and are explicit that this is why theirs is
# *"long-term stable by design"* -- *"there is no interaction with the numerical solver"*. Running
# it as a closure puts the model's own output back into its next input, which is a feedback loop
# they never face and which is where this project has already measured models failing
# (`results.md` §3). `analysis/postrun_lstm.jl` is the faithful post-processing comparison, and
# 🔴 **M4 and M0 must be compared in the SAME mode** -- an M4 scored as a post-processor against an
# M0 scored in-solver is the confound `plan.md` §22 item 9 exists to name.
#
# Structure, the OU inheritance and the replica semantics are `6_online_TO_LRS.jl`'s; read that
# file's comments for why each one is the way it is. Only the closure differs.

using Random
using JLD2
using RikFlow
using IncompressibleNavierStokes
using CUDA

const RF = RikFlow

length(ARGS) >= 1 || error("usage: julia 12_online_StochLSTM.jl <model_index> [replica]")
model_index = parse(Int, ARGS[1])
replica_arg = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : nothing

TO_folder = @__DIR__() * "/output/TO_LSTM"
track_file = get(ENV, "RIKFLOW_TRACK_FILE",
                 @__DIR__() * "/output/data_track_dns512_les64_Re2000.0_tsim100.0_f64_lmwray3.jld2")

# simulation parameters
T = Float64
Re = T(2_000)
Δt = T(2.5e-3)
tsim = T(100)

ArrayType = CuArray
backend = CUDABackend()

seeds = (; dns = 123, ou = 333, to = 234)

inputs = load(TO_folder * "/inputs_lstm.jld2", "inputs")
1 <= model_index <= length(inputs) ||
    error("model_index $model_index is out of range 1:$(length(inputs))")
cfg = inputs[model_index]
out_dir = TO_folder * "/$(cfg.name)/"

if replica_arg !== nothing
    1 <= replica_arg <= cfg.n_replicas ||
        error("replica $replica_arg is out of range: $(cfg.name) declares n_replicas = $(cfg.n_replicas)")
end
replicas = replica_arg === nothing ? (1:cfg.n_replicas) : (replica_arg:replica_arg)

# 🔑 S6 asks for the MEDIAN-seed fit to go online, not seed 1. `11_train_StochLSTM.jl` writes the
# summary that names it; without the summary there is only one seed and it is seed 1.
summary_file = out_dir * "seed_summary.jld2"
deploy_seed = isfile(summary_file) ? load(summary_file, "median_seed") : 1
model_file = out_dir * "StochLSTM_seed$(deploy_seed).jld2"
isfile(model_file) || error("no fitted model at $model_file -- run 11_train_StochLSTM.jl first")
@info "Deploying $(cfg.name)" arch=cfg.arch deploy_seed replicas=collect(replicas)

fit = RF.load_stochlstm(model_file)

# --- the reference run this one inherits from --------------------------------------------------
params_track = load(track_file, "params_track")
data_track = load(track_file, "data_track")

ustart = if data_track.fields[1].u isa Tuple
    stack(ArrayType{T}.(data_track.fields[1].u))
else
    ArrayType{T}(data_track.fields[1].u)
end

# The warm-up window. ⚠️ M4's requirement is its own: the replay has to charge the recurrence, not
# just fill the lag window, so `nwarm` must cover the LSTM's memory. 100 is inherited from the
# linear cells, where it was sized from the measured ACF; `tools/m4_warmup_probe.jl` measures it
# for M4 and this number should be set from that, not assumed.
nwarm = parse(Int, get(ENV, "RIKFLOW_M4_NWARM", "100"))
dQ_data = data_track.dQ[:, 1:nwarm]

params = (;
    params_track...,
    Re = T(2_000),          # 🔴 override: the splat carries the archive's Float32 parameters
    tsim,
    Δt,
    ArrayType,
    backend,
    savefreq = 1000,
)

haskey(params, :ou_bodyforce) ||
    error("params_track carries no ou_bodyforce; this run would be unforced")
params.ou_bodyforce.freeze == 1 || error(
    "inherited ou_bodyforce.freeze = $(params.ou_bodyforce.freeze), expected 1. At the LF step " *
    "that would advance the OU chain on a different schedule from the HF reference.",
)

for i in replicas
    # 🔑 The replica index selects the SEED, not a position in a sequence -- task 3 of an array
    # writes exactly the `..._replica3.jld2` the serial loop would have written. The closure is
    # rebuilt per replica so its recurrent state and history buffer start clean.
    sampler = RF.StochLSTM(fit.spec, fit.weights, fit.scaling;
                           spinnup_data = dQ_data,
                           rng = Xoshiro(seeds.to + i + 2))

    @info "Running sim $i out of $(cfg.n_replicas)"
    data_online = online_sgs(; params..., ustart = ustart, time_series_method = sampler)
    jldsave(out_dir * "data_online_tsim$(tsim)_replica$(i).jld2";
            data_online, params, model_index, deploy_seed, nwarm)
end
