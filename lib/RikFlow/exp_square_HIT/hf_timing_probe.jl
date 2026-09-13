# Wall-clock probe for the 512^3 HF reference run, at the production configuration.
#
# Purpose: measure the per-step cost of `2_HF_ref.jl` on a few thousand steps so the 400,000-step
# production run can be costed before it is launched, and report what it will write to disk.
#
# ⚠️ This is not a short *reference* run: ~0.5 TU, and the OU chain is a different realisation from
# the archive's. It runs the production right-hand side, stepper, precision, forcing and processor
# mix and times them.
#
# It does keep everything it computes — QoI trajectories, the filtered LES fields, and one real
# checkpoint written by `filtersaver`'s own branch and then loaded back — so that every piece of
# machinery the 30-hour run depends on has executed at full scale before that run starts. The
# checkpoint is deleted afterwards (`HFT_KEEP_CHECKPOINT=1` keeps it); the rest goes in the output
# file.
#
# Everything that sets the per-step cost is identical to production: n_dns = 512, Float64, LMWray3,
# fixed Δt = 2.5e-4, freeze = 10, OU forcing, QoIs every 10 steps through the full 512^3 -> 64^3
# filter, and a stored LES field every 1000 steps. Only the number of steps differs.
#
# How it works. Two solves:
#
#   1. a warm-up solve, which pays for CUDA compilation, the FFT plans and the first allocations,
#      and whose only output is a first estimate of the per-step cost;
#   2. a measured solve, continuing from the warmed field, sized from that estimate so it fits the
#      time budget and is a whole number of 1000-step blocks — so its mix of plain steps, QoI steps
#      and field-store steps is exactly production's.
#
# Splitting them is the point: a single solve folds a minute of compilation into the first steps,
# and at 400,000 steps a 10% error in the per-step cost is four hours.
#
# How to read the output. Two independent projections are printed — one from the mean over the
# measured block, one from the per-class medians (plain / QoI / field-store) weighted by how often
# production hits each. They should agree to a few percent. If they do not, the timing is noisy and
# the run was too short.
#
# ⚠️ What the projection does not cover: a shared GPU node, filesystem contention from other jobs,
# and any slowdown that only appears after hours of running. Treat it as a floor and add margin.
#
# 🔑 There is a prior to check the answer against. The archived Float32/RK44 reference recorded
# `comptime = 69_329 s` — 19.3 h — for these same 400,000 steps. Float64 roughly doubles the memory
# traffic that this solver is bound by, and LMWray3 drops one of RK44's four stages, so on
# comparable hardware the expectation is very roughly 25–35 h, less on an H100 if the archive was
# made on an A100. A projection far outside that range is a reason to distrust the probe or the
# configuration before believing the number.

if false                                               #src
    include("../src/RikFlow.jl")                       #src
    include("../../../src/IncompressibleNavierStokes.jl") #src
    using .IncompressibleNavierStokes                  #src
end

println("Loading modules...")
t0 = time()
using JLD2
using Observables      # `on`, for the step timer
using Printf
using Random
using Statistics
using RikFlow
using IncompressibleNavierStokes
using CUDA
t1 = time()
println("Modules loaded. Time: $(t1-t0) s")

# ---------------------------------------------------------------------------------------------
# Parameters. One block, environment-overridable, nothing shadowed.
# ---------------------------------------------------------------------------------------------

T = Float64

# `HFT_DEVICE=cpu` is for smoke-testing the script end to end before it reaches Snellius; gotcha
# #47 records undefined names inside function bodies shipping to the cluster three times because a
# parse check cannot see them. A CPU run measures nothing about GPU cost.
const ONCPU = get(ENV, "HFT_DEVICE", "gpu") == "cpu"
ArrayType = ONCPU ? Array : CuArray
backend = ONCPU ? IncompressibleNavierStokes.CPU() : CUDABackend()

n_dns = parse(Int, get(ENV, "HFT_N_DNS", "512"))
n_les = parse(Int, get(ENV, "HFT_N_LES", "64"))
Re = T(2_000)

# Production values. These are what make the measurement transferable; none of them is a probe
# parameter.
Δt = T(2.5e-4)
# ⚠️ These two must match `2_HF_ref.jl`'s `params_train`. They are overridable so the probe can
# cost a *proposed* saving policy -- that is half of what it is for -- but a projection made with
# one policy does not apply to a run with another: the QoI and field-store steps cost several times
# a plain one.
savefreq = parse(Int, get(ENV, "HFT_SAVEFREQ", "10"))    # DNS steps between QoI samples
plotfreq = parse(Int, get(ENV, "HFT_PLOTFREQ", "1000"))  # DNS steps between stored LES fields
freeze = 10        # DNS steps the OU body force is held fixed
T_L = 0.01
e_star = 0.1
k_f = sqrt(2)

# What the measurement is extrapolated to.
target_tsim = parse(T, get(ENV, "HFT_TARGET_TSIM", "100"))
# The SLURM wall limit the projection is compared against.
wall_hours = parse(Float64, get(ENV, "HFT_WALL_HOURS", "120"))
# Checkpoints the production run will write, for the storage table.
n_checkpoints = parse(Int, get(ENV, "HFT_N_CHECKPOINTS", "1"))

# Probe sizing.
warmup_steps = parse(Int, get(ENV, "HFT_WARMUP", "200"))
# The measured block is sized from the warm-up estimate to fill this budget, then rounded to whole
# `plotfreq` blocks and clamped. Set HFT_MEASURE to override the sizing and fix it outright.
budget_s = parse(Float64, get(ENV, "HFT_BUDGET_S", "900"))
measure_override = parse(Int, get(ENV, "HFT_MEASURE", "0"))
measure_min = parse(Int, get(ENV, "HFT_MEASURE_MIN", "1000"))
measure_max = parse(Int, get(ENV, "HFT_MEASURE_MAX", "20000"))

seeds = (; dns = 123, ou = 333, to = 234)

qois = [["Z", 0, 6], ["E", 0, 6], ["Z", 7, 15], ["E", 7, 15], ["Z", 16, 32], ["E", 16, 32]]
lims = ((T(0), T(1)), (T(0), T(1)), (T(0), T(1)))

outdir = @__DIR__() * "/output"
indir = @__DIR__() * "/output"
checkpoints_dir = outdir * "/checkpoints"
ispath(outdir) || mkpath(outdir)

# ---------------------------------------------------------------------------------------------
# Storage accounting for the production run this is costing.
#
# Printed before anything expensive happens, because it is the other half of the question and it
# costs nothing to answer. The free-space check warns here and errors in `2_HF_ref.jl`: this probe
# writes almost nothing itself.
# ---------------------------------------------------------------------------------------------

store = ref_data_storage(;
    ndns = n_dns,
    nles = n_les,
    tsim = target_tsim,
    Δt,
    savefreq,
    plotfreq,
    n_checkpoints,
    T,
)
report_ref_data_storage(store; label = "production, $(n_dns)^3, $T, n_checkpoints = $n_checkpoints")
check_output_space([outdir, checkpoints_dir], store.peak; hard = false)

# ---------------------------------------------------------------------------------------------
# Initial condition: the archived spin-up, promoted to Float64. Same file and same reasoning as
# `2_HF_ref.jl` and `cfl_probe.jl`.
#
# It matters for timing, not just for realism: the cost of a step is dominated by the grid, but a
# zero or synthetic field is not what production runs, and `maximum(abs, u)` in the logger and the
# QoI path both touch the real values.
# ---------------------------------------------------------------------------------------------

icfile = indir * "/u_start_spinnup_$(n_dns)_Re$(Re)_freeze_10_tsim4.0.jld2"
const SYNTHETIC = get(ENV, "HFT_SYNTHETIC", "0") == "1"

if isfile(icfile)
    println("Loading initial condition: $icfile")
    ustart = load(icfile, "u_start")
    ustart isa Tuple && (ustart = stack(ustart))
    println("  stored as $(eltype(ustart)) $(size(ustart)); promoting to $T")
    ustart = ArrayType{T}(ustart)
elseif SYNTHETIC
    @warn "SYNTHETIC initial condition (HFT_SYNTHETIC=1). Acceptable for a cost measurement — the " *
          "per-step cost is set by the grid — but this is not the production field."
    ustart = nothing   # built below, once `dns` exists
else
    error(
        "No initial condition at:\n    $icfile\n\n" *
        "Copy the archived spin-up there, or set HFT_SYNTHETIC=1 to time the machinery on a " *
        "synthetic field (acceptable here: the per-step cost is set by the grid, not the values).",
    )
end

# ---------------------------------------------------------------------------------------------
# Setups and operators
# ---------------------------------------------------------------------------------------------

gpumem() = ONCPU ? (0, 0) : (CUDA.total_memory() - CUDA.available_memory(), CUDA.total_memory())

dns = rf_setup(; x = ntuple(a -> LinRange(lims[a]..., n_dns + 1), 3), Re, ArrayType, backend)
les = rf_setup(; x = ntuple(a -> LinRange(lims[a]..., n_les + 1), 3), Re, ArrayType, backend)
compression = n_dns ÷ n_les
@info "Grid" n_dns n_les compression

force_cache = ou_force_cache(dns; T_L, e_star, k_f, rng_seed = seeds.ou, freeze)
psolver = psolver_spectral(dns)

if isnothing(ustart)
    ustart = velocityfield(
        dns,
        (dim, x, y, z) ->
            dim == 1 ? T(1) * sinpi(2 * y) * cospi(2 * z) :
            dim == 2 ? T(1) * sinpi(2 * z) * cospi(2 * x) :
            T(1) * sinpi(2 * x) * cospi(2 * y);
        psolver,
    )
end

# `nstep = 1` where production passes `nstep = nt`, and that is not a difference: `TO_Setup` only
# reaches `allocate_arrays_outputs` for `:TRACK_REF` and `:ONLINE`, so under `:CREATE_REF` nothing
# is sized by `nstep` at all. Worth knowing for the production run too — the HF reference holds no
# per-step TO array, however many steps it takes.
to_setup_les = RikFlow.TO_Setup(; qois, to_mode = :CREATE_REF, ArrayType, setup = les, nstep = 1)

# ---------------------------------------------------------------------------------------------
# Processors
# ---------------------------------------------------------------------------------------------

"""
Per-step wall time.

🔑 The `CUDA.synchronize()` is what makes the numbers mean anything. GPU work is queued
asynchronously, so without it a per-step time is the cost of *submitting* the step, and the real
cost only shows up wherever something later forces a sync. The queue drains eventually either way,
so the block total is right regardless — but the per-class breakdown would be nonsense.

It does cost something: syncing every step removes what little overlap there is between steps, so
this is a slight over-estimate, which is the safe direction for a projection. The QoI path already
forces a sync every `savefreq` steps through `Array(q)`.
"""
steptimer() =
    processor() do state
        times = Tuple{Int,Float64}[]
        on(state) do (; u, t, n)
            ONCPU || CUDA.synchronize()
            push!(times, (n, time()))
        end
        # 🔴 No `state[] = state[]` here, unlike every other processor in this repo.
        # Poking the observable fires **every handler registered before this one**, so a second
        # poking processor makes `filtersaver` record its n = 0 sample twice -- which is why
        # `cfl_probe.jl` reported `qoi_hist (6, 17)` for a run with 16 sample points. Harmless
        # there, wrong here: it would put two n = 0 entries in the timing record.
        # The cost is that step 1 is never timed, which at a thousand steps is nothing.
        times
    end

"Build the production processor set. `nplot` is `plotfreq`, or huge to store nothing."
makeprocs(; nplot, nlog, checkpoints = nothing, checkpoint_name = nothing) = (;
    f = RikFlow.filtersaver(
        dns,
        [les],
        (FaceAverage(),),
        [compression],
        [to_setup_les];
        nupdate = savefreq,
        n_plot = nplot,
        checkpoints,
        checkpoint_name,
    ),
    steps = steptimer(),
    log = timelogger(; nupdate = nlog),
)

"""
Per-step costs from a `steptimer` record.

The record starts at step 1 (see [`steptimer`](@ref)), so differencing gives the duration of steps
2..n and step 1 is not reported -- it would carry the processor initialisation anyway.
"""
function steptimes(rec)
    length(rec) < 2 && return (Int[], Float64[])
    ns = [rec[i][1] for i = 2:length(rec)]
    dts = [rec[i][2] - rec[i-1][2] for i = 2:length(rec)]
    ns, dts
end

# ---------------------------------------------------------------------------------------------
# 1. Warm-up solve. Everything expensive and one-off happens here.
# ---------------------------------------------------------------------------------------------

@info "Warm-up solve" steps = warmup_steps stepper = "LMWray3" T Δt freeze
u = ustart
tw0 = time()
(; u), outw = solve_unsteady(;
    setup = dns,
    start = (; u),
    force! = ou_navierstokes!,
    force_cache,
    params = rf_params(dns),
    method = LMWray3(; T),
    docopy = false,
    tlims = (T(0), warmup_steps * Δt),
    Δt,
    processors = makeprocs(; nplot = plotfreq, nlog = max(50, warmup_steps ÷ 4)),
    psolver,
)
tw1 = time()

nsw, dtsw = steptimes(outw.steps)
# The tail of the warm-up is already warm; the head is compilation. Use the last half.
warm_tail = isempty(dtsw) ? NaN : median(dtsw[max(1, end ÷ 2):end])
@printf("warm-up: %d steps in %.1f s; settled per-step %.4f s\n",
    warmup_steps, tw1 - tw0, warm_tail)

used, tot = gpumem()
ONCPU ||
    @printf("GPU memory in use after warm-up: %s of %s\n", humanbytes(used), humanbytes(tot))

# ---------------------------------------------------------------------------------------------
# 2. Measured solve, sized from the warm-up.
# ---------------------------------------------------------------------------------------------

measure_steps = if measure_override > 0
    measure_override
elseif isfinite(warm_tail) && warm_tail > 0
    # Whole `plotfreq` blocks, so the step mix matches production exactly.
    blocks = max(1, floor(Int, budget_s / (warm_tail * plotfreq)))
    clamp(blocks * plotfreq, measure_min, measure_max)
else
    measure_min
end
@printf("measured block: %d steps (~%.0f s at the warm-up rate)\n",
    measure_steps, measure_steps * warm_tail)

# 🔑 A real checkpoint, written by `filtersaver`'s own `n in checkpoints` branch, not imitated
# here with a `jldsave`. That branch is the one the production run uses and it had never executed
# at 512^3 post-merge — it does `Array(u)` on a 3 GiB device field and serialises it together with
# `results`, which is exactly the kind of thing that works on a workstation and fails on a cluster.
#
# It lands on `measure_steps ÷ 2 + 1`, deliberately neither a QoI step nor a field-store step, so
# the write shows up alone in one step's time and that step is excluded from the medians.
# It goes in its own directory so it can never be mistaken for a production checkpoint.
ckpt_dir = outdir * "/checkpoints_probe"
ispath(ckpt_dir) || mkpath(ckpt_dir)
ckpt_n = measure_steps ÷ 2 + 1
ckpt_file = "$ckpt_dir/checkpoint_n$(ckpt_n).jld2"
isfile(ckpt_file) && rm(ckpt_file; force = true)   # never time an overwrite of a stale file

@info "Measured solve" steps = measure_steps checkpoint_at = ckpt_n
tm0 = time()
(; u), outm = solve_unsteady(;
    setup = dns,
    start = (; u),
    force! = ou_navierstokes!,
    force_cache,
    params = rf_params(dns),
    method = LMWray3(; T),
    docopy = false,
    tlims = (T(0), measure_steps * Δt),
    Δt,
    processors = makeprocs(;
        nplot = plotfreq,
        nlog = max(100, measure_steps ÷ 10),
        checkpoints = [ckpt_n],
        checkpoint_name = ckpt_dir,
    ),
    psolver,
)
tm1 = time()
block_wall = tm1 - tm0

ns, dts = steptimes(outm.steps)

# The QoIs are computed anyway — they are part of the cost being measured — so they are kept
# rather than discarded. Two uses, neither of them physics:
#   * a NaN or a wild magnitude here says the 512^3 -> 64^3 filter, the masks, `curl` or
#     `compute_QoI` went wrong on the GPU, and that is worth catching in a 20-minute job rather
#     than 30 hours into the reference run;
#   * it is the second full-scale exercise of the TO path after `cfl_probe.jl`.
# ⚠️ Not reference data. ~0.5 TU total, and the OU chain here is a different realisation from the
# archive's (Float64 draws consume the stream differently — see `2_HF_ref.jl`).
qoi_warm = stack(outw.f.data[1].qoi_hist)
qoi_hist = stack(outm.f.data[1].qoi_hist)
@printf("qoi_hist %s (warm-up %s), every %d steps; finite: %s\n",
    string(size(qoi_hist)), string(size(qoi_warm)), savefreq, all(isfinite, qoi_hist))
for (i, q) in enumerate(eachrow(qoi_hist))
    @printf("  qoi %d  %-12s  min %.4e  max %.4e\n",
        i, string(qois[i]), minimum(q), maximum(q))
end

# The filtered LES fields, kept for the same reason as the QoIs: `filtersaver` has already built
# them and they are the only full-scale filtered output this configuration produces outside the
# reference run itself. Left as a `Vector` of host arrays — the same shape as production's
# `data_train.data[1].u` — so anything that reads a reference file reads these too.
#
# ⚠️ Unlike the QoIs these are not free to store: one is 6.58 MiB at 64^3. A default-sized block
# holds a handful; `HFT_MEASURE_MAX=20000` would hold 21. `HFT_SAVE_FIELDS=0` drops them.
const SAVE_FIELDS = get(ENV, "HFT_SAVE_FIELDS", "1") == "1"
fields = SAVE_FIELDS ? outm.f.data[1].u : typeof(outm.f.data[1].u)()
fields_warm = SAVE_FIELDS ? outw.f.data[1].u : typeof(outw.f.data[1].u)()
fieldbytes = sum(sizeof, fields; init = 0) + sum(sizeof, fields_warm; init = 0)
@printf("stored fields: %d measured + %d warm-up, %s%s\n",
    length(fields), length(fields_warm), humanbytes(fieldbytes),
    SAVE_FIELDS ? "" : "   (HFT_SAVE_FIELDS=0: dropped)")
# One short: `steptimer` does not poke the observable, so step 1 has no predecessor to difference
# against.
@assert length(ns) == measure_steps - 1 "step record is $(length(ns)), expected $(measure_steps - 1)"

# Classify each step by what production does on it. The checkpoint step is in none of the classes:
# its time is a plain step plus the write, and it is separated out below.
#
# 🔴 The OU class is `(n - 1) % freeze == 0`, not `n % freeze == 0`, and getting it wrong hid the
# single biggest per-step cost at 512^3. `solve_unsteady` advances the chain at the **top** of
# iteration `it`, gated on `mod(stepper.n, freeze) == 0` where `stepper.n` is still `it - 1`; the
# step then ends at `n = it`. So the cost lands on n = 1, 11, 21, ... Measured 2026-09-13: those
# steps take 0.308 s against a plain step's 0.175 — the partial inverse transform in
# `OU_forcing_step!` is O(N_f^3 N^3) and costs most of a timestep on its own. Classifying them as
# plain left them out of the per-class projection, which then came in 1.5 h under the block mean
# over 400,000 steps.
isstore(n) = n % plotfreq == 0
isqoi(n) = n % savefreq == 0 && !isstore(n)
isou(n) = (n - 1) % freeze == 0
isckpt(n) = n == ckpt_n
keep(n) = !isckpt(n)
t_store = [d for (n, d) in zip(ns, dts) if isstore(n) && !isou(n) && keep(n)]
t_qoi = [d for (n, d) in zip(ns, dts) if isqoi(n) && !isou(n) && keep(n)]
t_ou = [d for (n, d) in zip(ns, dts) if isou(n) && keep(n)]
t_plain = [d for (n, d) in zip(ns, dts) if !isstore(n) && !isqoi(n) && !isou(n) && keep(n)]

med(v) = isempty(v) ? NaN : median(v)

# ---------------------------------------------------------------------------------------------
# 3. Checkpoint cost. Measured, not guessed: it is a 3 GiB device-to-host copy plus a JLD2 write,
#    and it is the only part of the production run that is not per-step.
# ---------------------------------------------------------------------------------------------

# The write already happened, inside the solve. What is left is to price it and to prove the file
# is real — a checkpoint that serialises but cannot be loaded is worth nothing, and the production
# run would not find that out until it needed one.
ckpt_bytes = isfile(ckpt_file) ? filesize(ckpt_file) : 0
ckpt_step = findfirst(==(ckpt_n), ns)
# The write, net of the step it rode along with.
ckpt_time =
    isnothing(ckpt_step) || isempty(t_plain) ? NaN : dts[ckpt_step] - med(t_plain)

if ckpt_bytes == 0
    @warn "no checkpoint was written" ckpt_file ckpt_n
    @warn "`filtersaver`'s `n in checkpoints` branch did not fire — the production run would " *
          "also write nothing. This is a real failure, not a probe artefact."
else
    @printf("checkpoint: %s at n = %d, write cost %.1f s\n",
        humanbytes(ckpt_bytes), ckpt_n, ckpt_time)
    try
        d = load(ckpt_file)
        ku = size(d["u_cpu"])
        nf = length(d["results"].data[1].u)
        nq = length(d["results"].data[1].qoi_hist)
        @printf("  reads back: u_cpu %s %s, %d stored fields, %d QoI samples\n",
            string(ku), string(eltype(d["u_cpu"])), nf, nq)
        ku == (dns.N..., 3) || @warn "checkpoint u_cpu is not the DNS field shape" got = ku expected =
            (dns.N..., 3)
    catch err
        @warn "checkpoint was written but does not load" ckpt_file err
    end
end

# Several GiB on a shared filesystem, and not data. `HFT_KEEP_CHECKPOINT=1` keeps it for
# inspection.
if get(ENV, "HFT_KEEP_CHECKPOINT", "0") == "1"
    println("  kept (HFT_KEEP_CHECKPOINT=1): $ckpt_file")
else
    isfile(ckpt_file) && rm(ckpt_file; force = true)
    isdir(ckpt_dir) && isempty(readdir(ckpt_dir)) && rm(ckpt_dir)
end

# ---------------------------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------------------------

nt = store.nt
# How production distributes its steps across the classes. `nfields - 1` because the field stored
# at n = 0 is not a step. The OU class is counted first and removed from the others, matching the
# classification above: a step that both advances the chain and samples the QoIs is charged once,
# to the OU class, because that is where its time was measured.
n_ou = length(1:freeze:nt)
n_store = count(n -> n % plotfreq == 0 && !((n - 1) % freeze == 0), 1:nt)
n_qoi = count(n -> n % savefreq == 0 && n % plotfreq != 0 && !((n - 1) % freeze == 0), 1:nt)
n_plain = nt - n_ou - n_store - n_qoi

# 🔴 Scale the checkpoint cost by size; do not reuse the measured seconds. The probe writes a
# checkpoint holding the DNS field plus the handful of LES fields *it* accumulated; production's
# holds hundreds, so on the 512^3 case the real file is ~4.3 GiB against the probe's ~3.1 — and at
# a smaller `n_dns` or a shorter block the gap is larger still.
ckpt_rate = isfinite(ckpt_time) && ckpt_time > 0 ? ckpt_bytes / ckpt_time : NaN   # bytes/s
ckpt_total = isfinite(ckpt_rate) ? sum(store.checkpoints; init = 0.0) / ckpt_rate : 0.0

# 🔴 Extrapolate from the summed step times, not from the wall around the solve.
#
# `block_wall` also carries what `solve_unsteady` pays once: building the processors, the QoI
# computation that `filtersaver`'s `state[] = state[]` triggers at n = 0, and — the expensive one —
# the OU priming, whose partial inverse transform is O(N_f^3 N^3). On the smoke that was 4.6 s
# against 5.8 s of actual stepping, and dividing it across 40 steps then multiplying by 400,000
# projected 52% high. Production pays it once for 400,000 steps, so it must not be per-step.
step_wall = sum(dts)
setup_wall = block_wall - step_wall          # one-off: processors, priming, and step 1
per_step = (step_wall - (isfinite(ckpt_time) ? ckpt_time : 0.0)) / length(dts)
proj_block = per_step * nt + ckpt_total
proj_class =
    n_plain * med(t_plain) + n_qoi * med(t_qoi) + n_store * med(t_store) +
    n_ou * med(t_ou) + ckpt_total
spread = abs(proj_block - proj_class) / max(proj_block, proj_class)

hrs(s) = s / 3600
"Seconds as hours, or as seconds when that would print `0.0 h` — smoke runs project to minutes."
dur(s) = s >= 600 ? @sprintf("%.1f h", hrs(s)) : @sprintf("%.0f s", s)

println()
println("="^88)
@printf("measured block         %d steps, %.1f s of stepping\n", measure_steps, step_wall)
@printf("  per step (mean)      %.4f s   (checkpoint removed)\n", per_step)
@printf("  one-off setup        %.1f s   (processors, OU priming, step 1; paid once)\n",
    setup_wall)
@printf("  plain step           %.4f s   (median of %d)\n", med(t_plain), length(t_plain))
@printf("  + QoI step           %.4f s   (median of %d, every %d steps)\n",
    med(t_qoi), length(t_qoi), savefreq)
@printf("  + field-store step   %.4f s   (median of %d, every %d steps)\n",
    med(t_store), length(t_store), plotfreq)
@printf("  + OU forcing step    %.4f s   (median of %d, every %d steps: n = 1, %d, ...)\n",
    med(t_ou), length(t_ou), freeze, freeze + 1)
if isfinite(ckpt_time) && ckpt_bytes > 0
    @printf("  checkpoint write     %.1f s for %s measured  ->  %s/s\n",
        ckpt_time, humanbytes(ckpt_bytes), humanbytes(ckpt_rate))
    @printf("                       %.1f s for production's %d checkpoint(s), %s total\n",
        ckpt_total, n_checkpoints, humanbytes(sum(store.checkpoints; init = 0)))
else
    println("  checkpoint write     NOT MEASURED — no checkpoint file was produced")
end
println()
@printf("PROJECTION to tsim = %g  (%d steps: %d plain, %d QoI, %d store, %d OU)\n",
    target_tsim, nt, n_plain, n_qoi, n_store, n_ou)
@printf("  from the block mean   %s\n", dur(proj_block))
@printf("  from the step classes %s\n", dur(proj_class))
@printf("  the two differ by     %.1f%%\n", 100 * spread)
println()

worst = max(proj_block, proj_class)
if spread > 0.05
    println("⚠️  The two projections differ by more than 5%. The measured block is too short or the")
    println("   node is noisy; re-run with a larger HFT_BUDGET_S before trusting either number.")
end
@printf("Take %s as the estimate, %s with a 20%% margin.\n", dur(worst), dur(1.2 * worst))
if 1.2 * hrs(worst) > wall_hours
    @printf("🔴 That does not fit a %.0f h wall limit. The run has to be split, and\n", wall_hours)
    println("   `create_ref_data` has no resume path — its checkpoints are written but never read")
    println("   back, so a job that hits the limit loses everything. Splitting the run means")
    println("   writing that restart first.")
elseif hrs(worst) > 0.7 * wall_hours
    @printf("⚠️  %.1f h against a %.0f h wall limit leaves little room. One slow node and the job\n",
        hrs(worst), wall_hours)
    println("   is killed with nothing recoverable — `create_ref_data` cannot resume from its own")
    println("   checkpoints.")
else
    @printf("Fits a %.0f h wall limit with room.\n", wall_hours)
end
println("="^88)

filename = "$outdir/hf_timing_probe_$(n_dns)_f64_lmwray3.jld2"
jldsave(
    filename;
    step_n = ns,
    step_dt = dts,
    warmup_dt = dtsw,
    qoi_hist,
    qoi_warm,
    qois,
    fields,
    fields_warm,
    params = (;
        n_dns,
        n_les,
        Re,
        Δt,
        savefreq,
        plotfreq,
        freeze,
        target_tsim,
        warmup_steps,
        measure_steps,
        n_checkpoints,
        stepper = "LMWray3",
        precision = string(T),
        oncpu = ONCPU,
        synthetic = SYNTHETIC,
    ),
    timing = (;
        block_wall,
        step_wall,
        setup_wall,
        per_step,
        t_plain = med(t_plain),
        t_qoi = med(t_qoi),
        t_store = med(t_store),
        t_ou = med(t_ou),
        n_ou,
        ckpt_time,
        ckpt_bytes,
        ckpt_rate,
        ckpt_total,
        proj_block,
        proj_class,
        proj_hours = hrs(worst),
    ),
    storage = store,
)
@printf("\nWritten: %s  (%s)\n", filename, humanbytes(filesize(filename)))
