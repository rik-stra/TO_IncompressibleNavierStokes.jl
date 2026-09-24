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
# 🔑 `RIKFLOW_ONLINE_TSIM` shortens the run for a smoke (e.g. 1 TU) -- the first time a model goes
# into the solver, a short run catches a bad path or an immediate blow-up before 100 TU does. The
# output name carries `tsim`, so a smoke never overwrites a full run.
tsim = T(parse(Float64, get(ENV, "RIKFLOW_ONLINE_TSIM", "100")))

# 🔑 `RIKFLOW_ONLINE_DEVICE=cpu` runs the LES on the host -- for short local screening runs, not for
# production. Its output files carry `_cpu`, so a local run can never be confused with, or
# overwritten by, a cluster copy. Different backend = different round-off = a different
# realisation after ~0.5 TU; statistics, not trajectories, are comparable with the GPU runs.
const ONCPU = lowercase(strip(get(ENV, "RIKFLOW_ONLINE_DEVICE", "cuda"))) == "cpu"
ArrayType = ONCPU ? Array : CuArray
backend = ONCPU ? IncompressibleNavierStokes.CPU() : CUDABackend()
devtag = ONCPU ? "_cpu" : ""

seeds = (; dns = 123, ou = 333, to = 234)

inputs = load(TO_folder * "/inputs_lstm.jld2", "inputs")
1 <= model_index <= length(inputs) ||
    error("model_index $model_index is out of range 1:$(length(inputs))")
cfg = inputs[model_index]
# 🔑 `RIKFLOW_M4_MODEL_DIR` deploys a fit from somewhere other than the table's own directory --
# in practice one stride-scan point exported by `tools/m4_export_point.jl`, which cannot be written
# into `<cfg.name>/` without overwriting the fit that lives there. The online replicas are written
# beside the model, so the two directories never mix. `model_index` still supplies `n_replicas`,
# and the fit must be of that cell (checked below).
model_dir_env = strip(get(ENV, "RIKFLOW_M4_MODEL_DIR", ""))
out_dir = isempty(model_dir_env) ? TO_folder * "/$(cfg.name)/" : rstrip(model_dir_env, '/') * "/"
isempty(model_dir_env) || isdir(out_dir) ||
    error("RIKFLOW_M4_MODEL_DIR = $out_dir is not a directory")

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
if !isfile(model_file)
    # 🔑 Most often this is an exported stride-scan point deployed WITHOUT `RIKFLOW_M4_MODEL_DIR`
    # reaching the job (set on its own line instead of on the `sbatch` line). Name the candidates.
    exported = filter(d -> startswith(d, cfg.name * "_") && isdir(joinpath(TO_folder, d)),
                      readdir(TO_folder))
    error("no fitted model at $model_file. RIKFLOW_M4_MODEL_DIR = " *
          (isempty(model_dir_env) ? "<not set>" : "'$model_dir_env'") * ". " *
          (isempty(exported) ? "Run 11_train_StochLSTM.jl first." :
           "Exported fits for $(cfg.name) under $TO_folder: " * join(exported, ", ") *
           " -- deploy one with RIKFLOW_M4_MODEL_DIR=<that dir> on the SAME line as sbatch."))
end
@info "Deploying $(cfg.name)" arch=cfg.arch deploy_seed replicas=collect(replicas)

fit = RF.load_stochlstm(model_file)
# 🔴 An exported fit must belong to the cell the index names: `n_replicas` and the seeds come from
# `cfg`, and deploying cell 2's weights under cell 5's row would label the replicas wrongly.
fit_cell = hasproperty(fit.extras, :cfg) ? fit.extras.cfg.name : cfg.name
fit_cell == cfg.name || error("the model in $out_dir was fitted for $fit_cell, but model_index " *
                              "$model_index is $(cfg.name) -- pass the matching index")
isempty(model_dir_env) || @info "deploying an exported fit" out_dir source_scan =
    get(fit.extras, :source_scan, "?") stride = get(fit.extras, :stride, "?") batch =
    get(fit.extras, :batch, "?")

# --- the reference run this one inherits from --------------------------------------------------
# 🔑 `RIKFLOW_ONLINE_IC` reads the ~6 MB extract `tools/m4_extract_ic.jl` writes instead of the
# 2.7 GB record -- the same three objects, copied, so the run is the same.
ic_file = strip(get(ENV, "RIKFLOW_ONLINE_IC", ""))
if isempty(ic_file)
    params_track = load(track_file, "params_track")
    data_track = load(track_file, "data_track")
    u0 = data_track.fields[1].u
    dQ_rec = data_track.dQ
else
    params_track, u0, dQ_rec = load(ic_file, "params_track", "u0", "dQ")
end

ustart = u0 isa Tuple ? stack(ArrayType{T}.(u0)) : ArrayType{T}(u0)

# The warm-up window. ⚠️ M4's requirement is its own: the replay has to charge the recurrence, not
# just fill the lag window, so `nwarm` must cover the LSTM's memory. 100 is inherited from the
# linear cells, where it was sized from the measured ACF; `tools/m4_warmup_probe.jl` measures it
# for M4 and this number should be set from that, not assumed.
nwarm = parse(Int, get(ENV, "RIKFLOW_M4_NWARM", "100"))
dQ_data = dQ_rec[:, 1:nwarm]

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
    jldsave(out_dir * "data_online_tsim$(tsim)_replica$(i)$(devtag).jld2";
            data_online, params, model_index, deploy_seed, nwarm, model_file)
end
