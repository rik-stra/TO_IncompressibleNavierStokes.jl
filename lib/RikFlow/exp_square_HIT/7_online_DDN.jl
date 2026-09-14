if false                                               #src
    include("../src/RikFlow.jl")                  #src
    include("../../../src/IncompressibleNavierStokes.jl") #src
end

using Random
using JLD2
using RikFlow
using IncompressibleNavierStokes
using CUDA

DDN_folder = @__DIR__()*"/output/TO_DDN"
# The tracking record produced by `3_track_ref.jl` on the REGENERATED HF reference.
#
# 🔴 The `_f64_lmwray3` suffix is deliberate and must not be dropped: a Float64/LMWray3 record must
# never be confusable with the archived Float32/RK44 one, which is otherwise identically named. The
# archive sits on the pre-`09954be1` Nyquist convention, which changed `∂` and therefore `tau` and
# `dQ`, so it is a *different dynamical system* rather than a less accurate measurement of this one
# (claude_memory.md #45, #46).
#
# 🔑 100 TU, not 10 (Rik, 2026-09-14). One tracking run carries the 1–10 TU fit window *and* the 401
# fields at 0.25 TU that D6 draws its initial conditions from. `train_range = (400, 4000)` selects
# t ∈ [1, 10] out of whatever record it is given, so fitting "to 10 TU" needs no change here.
track_file = get(ENV, "RIKFLOW_TRACK_FILE",
    @__DIR__()*"/output/data_track_dns512_les64_Re2000.0_tsim100.0_f64_lmwray3.jld2")
# Parameters and the initial field come from here as well as the training `dQ`.
ispath(DDN_folder) || mkpath(DDN_folder)
## DDN inputs
n_replicas = 5
traindata_range = 400:4000

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


# load reference data
data_track = load(track_file, "data_track");
params_track = load(track_file, "params_track");
# get initial condition
if data_track.fields[1].u isa Tuple
    ustart = stack(ArrayType{T}.(data_track.fields[1].u));
elseif data_track.fields[1].u isa Array{<:Number,4}
    ustart = ArrayType{T}(data_track.fields[1].u);
end
# get ref trajectories
dQ_data = data_track.dQ[:,traindata_range];

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

# Run 10 replicas
for i in 1:n_replicas
    #time_series_sampler = RikFlow.Resampler(dQ_data, Xoshiro(seeds.to+i));
    time_series_sampler = RikFlow.MVG_sampler(dQ_data, Xoshiro(seeds.to+i));

# run the sim
    @info "Running sim $i out of $n_replicas"
    data_online = online_sgs(; params..., ustart=ustart, time_series_method=time_series_sampler);
# Save tracking data
    jldsave(DDN_folder*"/DDN_data_online_tsim$(tsim)_replica$(i).jld2"; data_online, params);
end