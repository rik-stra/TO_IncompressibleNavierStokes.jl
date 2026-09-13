# High fidelity reference simulation of homogeneous isotropic turbulence (HIT).
# Collects qoi reference trajectories.

if false                                               #src
    include("../src/RikFlow.jl")                  #src
    include("../../../src/IncompressibleNavierStokes.jl") #src
    using .IncompressibleNavierStokes                  #src
end     


println("Loading modules...")
t0 = time()
using LoggingExtras
using Random
# 🔴 `using CairoMakie` was here and would have killed the job at load: CairoMakie is not in
# `lib/RikFlow`'s `[deps]` (only `Makie`, and only as a weak dependency behind `RikFlowMakieExt`),
# so the run dies with "Package CairoMakie not found in current path" before the first step —
# after the queue wait, with nothing done. Nothing in this file or in `create_ref_data` plots:
# `plotfreq` reaches `lesdatagen` as `n_plot`, where it decides when to *store* a filtered field,
# and `create_ref_data`'s only `plot = energy_spectrum_plot` line is commented out.
# Gotcha #47's class again — an import that a parse check cannot flag.
using JLD2
using RikFlow
using IncompressibleNavierStokes
using CUDA
t1 = time()

# Write output to file, as the default SLURM file is not updated often enough
# jobid = ENV["SLURM_JOB_ID"]
# logfile = joinpath(@__DIR__, "log_$(jobid).out")
# filelogger = MinLevelLogger(FileLogger(logfile), Logging.Info)
# logger = TeeLogger(global_logger(), filelogger)
# global_logger(logger)

# ---------------------------------------------------------------------------------------------
# Parameters. ONE block, defined once.
#
# ⚠️ This file used to declare the production parameters and then silently overwrite them with a
# "small test parameters" block six lines later, so as committed it ran n_dns = 128, tsim = 0.5
# while appearing to run the production case. Set HF_REF_SMOKE=1 for the small variant instead;
# nothing is shadowed.
#
# 🔴 Float64 and LMWray3 for the regeneration (Rik, 2026-09-11).
#   - Float64 because Float32 leaves the coefficient-level diagnostics unresolved
#     (claude_memory.md gotcha #26: cond*eps(Float32) ~ 0.2), and because it makes the DNS/LES
#     forcing equivalence exact: Float32(2.5e-4)*10 != Float32(2.5e-3), but in Float64 it is.
#   - LMWray3 because it needs one stage vector instead of four (7.1 GiB against 16.2 GiB of ODE
#     cache at 512^3 Float64) and three right-hand-side evaluations per step instead of four.
#
# ⚠️ Consequence, and it is not a defect: the OU chain draws `randn!` into a Float64 buffer, which
# consumes the stream differently from Float32, so this run is a *different realisation* from the
# archive - not a refinement of it. Together with the scheme change, the archived 401 fields are
# reproducible only at field 1, the filtered initial condition, which needs no time stepping.
# `tools/check_ref_401.jl ic` is that check and it passes.
# ---------------------------------------------------------------------------------------------

const SMOKE = get(ENV, "HF_REF_SMOKE", "0") == "1"

T = Float64
ArrayType = CuArray
backend = CUDABackend()

n_dns = SMOKE ? Int(128) : Int(512)
n_les = Int(64)
Re = T(2_000)
Δt = T(2.5e-4)
tsim = SMOKE ? T(0.5) : T(100)
tburn = T(4)

# forcing
T_L = 0.01  # correlation time of the forcing
e_star = 0.1 # energy injection rate
k_f = sqrt(2) # forcing wavenumber
freeze = 10 # number of time steps to freeze the forcing

# What gets written, and how often.
#
# ⚠️ `plotfreq` is only ever tested on steps where `n % savefreq == 0` (`filtersaver` gates its
# inner observable on `savefreq` first), so a `plotfreq` that is not a multiple of `savefreq`
# stores fields at `lcm(savefreq, plotfreq)` instead — quietly, and far fewer of them.
# `ref_data_storage` reports the interval it really gets; read that, not this line.
savefreq = 10      # DNS steps between QoI samples
plotfreq = 1000    # DNS steps between stored (filtered) LES fields -> 401 fields at tsim = 100
# 🔴 Checkpoints are write-only. Nothing in RikFlow reads one back: `create_ref_data` has no resume
# path, so a checkpoint buys a post-mortem restart that somebody has to write by hand, not an
# automatic one. Raising this does not make the run recoverable on its own — it only makes a
# hand-written restart possible at a finer granularity, at 4.3 GiB and a couple of minutes each.
n_checkpoints = 1

seeds = (;
    dns = 123, # DNS initial condition
    ou = 333, # OU process
    to = 234, # TO method online sampling
)

outdir = @__DIR__() *"/output"
indir = @__DIR__() *"/output"
checkpoints_dir = @__DIR__() *"/output/checkpoints"
ispath(outdir) || mkpath(outdir)
ispath(checkpoints_dir) || mkpath(checkpoints_dir)

# ---------------------------------------------------------------------------------------------
# Disk preflight. Before the initial condition is loaded, let alone before 400,000 steps.
#
# 🔴 `create_ref_data` holds every stored field in host memory for the whole run and writes them
# all at the end, and each checkpoint carries the full DNS field *plus* everything accumulated so
# far. At 512^3 Float64 that is 2.58 GiB of output and a 4.33 GiB checkpoint. Running out of space
# is discovered at the end — after the compute is spent and with nothing written — so it is checked
# here instead. `hard = true`: refuse to start rather than fail late.
#
# ⚠️ Free space is not reserved. Another job can take it while this one runs; the 15% margin covers
# the ordinary case, not a busy filesystem.
# ---------------------------------------------------------------------------------------------
_storage = ref_data_storage(;
    ndns = n_dns, nles = n_les, tsim, Δt, savefreq, plotfreq, n_checkpoints, T)
report_ref_data_storage(_storage; label = "$(n_dns)^3 -> $(n_les)^3, $T, tsim = $tsim")
check_output_space([outdir, checkpoints_dir], _storage.peak; hard = true)

# Device and precision are set in the parameter block above.

# 🔑 The archived spin-up is reused rather than re-run, and it is stored in Float32.
#
# Reusing it is deliberate. The spin-up is a plain forced DNS - no TO, no QoIs - so `∂` never
# touches it and the Nyquist change of `09954be1` cannot have tainted it; and it seeds its OU chain
# from `ou_spin = 123` while this run seeds from `ou = 333`, so no forcing state carries over. It
# is only an initial velocity field, and it is a valid one.
#
# Re-running it would produce a different field and leave the new reference sharing nothing with
# the archive - not even field 1. Keeping it preserves the one full-scale anchor that survives.
#
# Float32 -> Float64 promotion is exact (every Float32 is representable). The field is only
# Float32-*accurate*, so its divergence is ~1e-7 rather than ~1e-16; the first pressure projection
# removes that, and 1e-7 on a turbulent field is nothing.
ustart = load(indir*"/u_start_spinnup_$(n_dns)_Re$(Re)_freeze_$(freeze)_tsim$(tburn).jld2", "u_start");
if ustart isa Tuple # old INS data format
    ustart = stack(ArrayType{T}.(ustart));
elseif ustart isa Array{<:Number,4} # new INS data format
    ustart = ArrayType{T}(ustart);
end
@info "initial condition" eltype(ustart) size(ustart)

# Parameters
get_params(nlesscalar) = (;
    D = 3,
    Re,
    lims = ( (T(0) , T(1)) , (T(0) , T(1)), (T(0),T(1)) ),
    qois = [["Z",0,6],["E", 0, 6],["Z",7,15],["E", 7, 15],["Z",16,32],["E", 16, 32]],
    tsim,
    Δt,
    nles = map(n -> (n, n, n), nlesscalar), # LES resolutions
    ndns = (n -> (n, n, n))(n_dns), # DNS resolution
    filters = (FaceAverage(),),
    ArrayType,
    backend,
    ou_bodyforce = (;T_L, e_star, k_f, freeze, rng_seed = seeds.ou ),
)

params_train = (; get_params([n_les])..., savefreq, plotfreq);
t3 = time()
data_train = create_ref_data(; params_train..., ustart, method = LMWray3(; T),
    n_checkpoints, checkpoint_name = checkpoints_dir);
t4 = time()
println("HF simulation done. Time: $(t4-t3) s")
# Save filtered DNS data
# Tagged with precision and stepper: a Float64/LMWray3 reference must not be confusable
# with the archived Float32/RK44 one, which has otherwise the same name.
filename = "$outdir/data_train_dns$(n_dns)_les$(n_les)_Re$(Re)_freeze_$(freeze)_tsim$(params_train.tsim)_f64_lmwray3.jld2"
jldsave(filename; data_train, params_train)