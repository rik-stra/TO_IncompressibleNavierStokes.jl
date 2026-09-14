if false                                               #src
    include("../src/RikFlow.jl")                  #src
    include("../../../src/IncompressibleNavierStokes.jl") #src
end

using Random
using JLD2
using RikFlow
using IncompressibleNavierStokes
using CUDA

smag_folder = @__DIR__()*"/output/smag"
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
# Parameters and the initial field come from here.
ispath(smag_folder) || mkpath(smag_folder)

# 🔴 `c_s = 0.07` by Rik's instruction, 2026-09-14 — NOT paper 2's 0.071.
#
# Paper 2 tuned `C_s` for best agreement with the reference distributions and reports
# `C_s = 0.071`, summed KS-distance 0.705. Two reasons that number is not simply inherited:
# the tuning was against the *archived* reference, which is on the old Nyquist convention, and the
# merge replaced our Smagorinsky kernels with upstream's (map §9, Q2) — a different implementation
# of the same model. ⚠️ So this baseline is neither paper 2's constant nor paper 2's code, and its
# summed KS must not be compared with 0.705 without saying so.
smag_vals = [0.07]
# simulation parameters
T = Float64
Re = T(2_000);
Δt = T(2.5e-3);
tsim = T(100);
# forcing
T_L = 0.01  # correlation time of the forcing
e_star = 0.1 # energy injection rate
k_f = sqrt(2) # forcing wavenumber  
freeze = 1 # number of time steps to freeze the forcing

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
    ou_bodyforce = (;T_L, e_star, k_f, freeze, rng_seed = seeds.ou),
    savefreq = 1000);


for c_s in smag_vals
    # Build setup and assemble operators
    setup = rf_setup(;
        x = ntuple(α -> LinRange(params.lims[α]..., params.nles[1][α] + 1), params.D),
        Re=params.Re,
        ArrayType,
        backend,
    );

    # The closure moved from `setup.closure_model` + `θ` into the right-hand side and its cache.
    # 🔴 These are upstream's Smagorinsky kernels, not the `smagorinsky_closure_natural` this
    # script used before the merge (map section 9, Q2) - a different implementation of the same
    # model, so this baseline is not numerically the one paper 2 reports.
    force_cache = rf_smag_force_cache(setup; c_s = T(c_s), params.ou_bodyforce);

    # Number of time steps to save
    nt = round(Int, params.tsim / params.Δt)

    to_setup_les = RikFlow.TO_Setup(; 
            params.qois, 
            to_mode = :CREATE_REF,
            params.ArrayType, 
            setup,
            nstep=nt);

    psolver = psolver_spectral(setup);

    # Solve
    @info "Solving LF sim (SMAG)"
    (; u, t), outputs = solve_unsteady(; 
            # LMWray3 by Rik's decision of 2026-09-11: stated, never inherited from the library
            # default. Reproducing an archived run means passing RKMethods.RK44 explicitly.
            method = LMWray3(; T = eltype(ustart)),
            setup,
            start = (; u = ustart),
            force! = rf_smag_navierstokes!,
            force_cache,
            params = rf_params(setup),
            tlims = (T(0), params.tsim),
            params.Δt,
            processors = (;
                log = timelogger(; nupdate = 100),
                fields = fieldsaver(; setup, nupdate = params.savefreq),  # by calling this BEFORE qoisaver, we also save the field at t=0!
                qoihist = RikFlow.qoisaver(; setup, to_setup=to_setup_les, nupdate = 1),
            ),
            psolver,
    );

    q = stack(outputs.qoihist);
    data_online = (;q, fields = outputs.fields);
    # Save tracking data
    jldsave(smag_folder*"/data_smag_$(c_s)_tsim$(tsim).jld2"; data_online, params);
end