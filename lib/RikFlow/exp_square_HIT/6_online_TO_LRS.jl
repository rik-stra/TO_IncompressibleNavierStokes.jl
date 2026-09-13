if false                                               #src
    include("../src/RikFlow.jl")                  #src
    include("../../../src/IncompressibleNavierStokes.jl") #src
end

using Random
using JLD2
using RikFlow
using IncompressibleNavierStokes
using CUDA

# parse input ARGS
model_index = parse(Int, ARGS[1])
# or set model_index manually
model_index = 2

inputs_file_name = "/inputs_example.jld2"
TO_folder = @__DIR__()*"/output/TO_LRS"
track_file = @__DIR__()*"/output/data_track_tsim10.0.jld2"

# simulation parameters
T = Float64
Re = T(2_000);
Δt = T(2.5e-3);
tsim = T(100);
# forcing
T_L = 0.01  # correlation time of the forcing
e_star = 0.1 # energy injection rate
k_f = sqrt(2) # forcing wavenumber  
# ⚠️ The forcing here is INHERITED, not set. This script splats `params_track...`, which
# carries `ou_bodyforce` -- freeze, rng_seed and all -- from the tracking run, and a local
# `freeze` would be dead code that reads as if it did something. The value is checked after
# the splat instead, because `freeze = 1` at the LF step is what keeps this run's forcing
# aligned with the HF reference's `freeze = 10` at a ten-times-smaller step (gotcha #33).

# For running on a CUDA compatible GPU
ArrayType = CuArray
backend = CUDABackend()


seeds = (;
    dns = 123, # DNS initial condition
    ou = 333, # OU process
    to = 234, # TO method online sampling
)


## Load data
inputs = load(TO_folder*inputs_file_name, "inputs")
(; name, hist_len, n_replicas, hist_var,tracking_noise) = inputs[model_index]

out_dir = TO_folder*"/$(name)/"

# load reference data

params_track = load(track_file, "params_track");
data_track = load(track_file, "data_track");

# get initial condition
if data_track.fields[1].u isa Tuple
    ustart = stack(ArrayType{T}.(data_track.fields[1].u));
elseif data_track.fields[1].u isa Array{<:Number,4}
    ustart = ArrayType{T}(data_track.fields[1].u);
end
# get ref trajectories to initialize history for the model
dQ_data = data_track.dQ[:,1:100]; # first 100 time steps are not predicted but taken from training data.

params = (;
    params_track...,
    # 🔴 Override the archived Re. The splat above carries the archive's Float32 parameters, and a
    # later key wins — without this the setup is built at Float32 while the script declares
    # Float64, and `typeof(setup.Re)` silently drives every QoI buffer back to single precision.
    # `rf_setup` now refuses that mismatch outright, so this is what keeps the script runnable.
    Re = T(2_000),
    tsim,
    Δt,
    ArrayType,
    backend,
    savefreq = 1000);

# 🔴 The inherited forcing must be the one this run assumes. `freeze = 1` at the LF step of
# 2.5e-3 advances the OU chain on exactly the same schedule, and with exactly the same per-advance
# Delta t, as the HF reference's `freeze = 10` at 2.5e-4 -- 40001 advances of 2.5e-3 either way, each
# forcing field covering the same physical interval (verified 2026-09-13, and bit-identical only in
# Float64: in Float32 the two products are 1 ulp apart). A `params_track` carrying `freeze != 1`
# would silently break that correspondence, so it is checked rather than assumed.
haskey(params, :ou_bodyforce) ||
    error("params_track carries no ou_bodyforce; this run would be unforced")
params.ou_bodyforce.freeze == 1 || error(
    "inherited ou_bodyforce.freeze = $(params.ou_bodyforce.freeze), expected 1. At the LF step " *
    "that would advance the OU chain on a different schedule from the HF reference.",
)

# Run replicas
for i in 1:n_replicas
    LinReg_file_name = out_dir*"LinReg.jld2"
    if hist_len == 0
        q_hist = nothing
    else
        q_hist = ArrayType{T}(zeros(T,size(params.qois,1),hist_len)) 
        if hist_var == :q_star_q
            q_hist = cat(q_hist, q_hist, dims=1)
        end
    end
    time_series_sampler = RikFlow.LinReg(LinReg_file_name, Xoshiro(seeds.to+i+2), ArrayType, q_hist = q_hist, spinnup_data = ArrayType{T}(dQ_data));
    
# run the sim
    @info "Running sim $i out of $n_replicas"
    data_online = online_sgs(; params..., ustart=ustart, time_series_method=time_series_sampler);
# Save tracking data
    jldsave(out_dir*"data_online_tsim$(tsim)_replica$(i).jld2"; data_online, params);
end
