if false                                               #src
    include("../src/RikFlow.jl")                  #src
    include("../../../src/IncompressibleNavierStokes.jl") #src
end

# Low-fidelity tracking run: nudge the LF solver onto the HF reference's QoI trajectory and record
# the subgrid corrections `dQ` it takes to get there. This is the training data for every time
# series model in the project, and the source of D6's initial conditions.
#
# ---------------------------------------------------------------------------------------------
# 🔴 REBASELINED 2026-09-14. This run is on the REGENERATED HF reference, not the archive.
#
# Every archived record sits on the pre-`09954be1` Nyquist convention. That commit changed `∂`,
# `∂` feeds `get_vi_functions`, and the direction vectors feed `tau` — so the archive's tracking run
# is a *different dynamical system*, not a less accurate measurement of this one
# (claude_memory.md #45, #46). Nothing downstream of the reference can be inherited.
#
# 🔑 `tsim = 100`, not 10 (Rik, 2026-09-14). Three reasons:
#   1. One run yields both records. `train_range = (400, 4000)` in `5_train_LinReg.jl` selects
#      t ∈ [1, 10] TU out of whatever record it is given, so "fit to 10 TU" needs no edit anywhere.
#   2. D6 needs the 401 fields at 0.25 TU that only 40 000 steps at `savefreq = 100` produce. A
#      10 TU run gives 41 fields and D6 has nowhere to draw 180 initial conditions from.
#   3. It removes gotcha #39 by construction. The archive has *two* tracked records, 10 TU and
#      100 TU, which start from the same field and whose `dQ` decorrelates completely — 0.7-1.1 sd
#      apart by step 1000. That is why `LinReg1` reproduces the archive from one record and not the
#      other. One run cannot have that problem.
#
# ⚠️ Accepted consequence: the held-out 10-100 TU window is a *continuation* of the training window
# rather than a second realisation. That is what paper 2's own partition always was; the archive's
# two-record split was an accident of how the files were made, not a design.
# ---------------------------------------------------------------------------------------------

using JLD2
using Printf
using RikFlow
using IncompressibleNavierStokes
using CUDA

# For running on a CUDA compatible GPU
T = Float64
ArrayType = CuArray
backend = CUDABackend()

# parameters
n_dns = Int(512)
n_les = Int(64)
Re = T(2_000)
Δt = T(2.5e-3)
tsim = T(100)

# 🔴 The regenerated reference, not `output/paper_data_HIT/`. Overridable so a Snellius run can
# point at a staged copy without editing the file.
ref_file = get(ENV, "RIKFLOW_HF_REF",
    @__DIR__()*"/output/data_train_dns$(n_dns)_les$(n_les)_Re$(Re)_freeze_10_tsim100.0_f64_lmwray3.jld2")
outdir = @__DIR__()*"/output"
ispath(outdir) || mkpath(outdir)

# forcing
T_L = 0.01  # correlation time of the forcing
e_star = 0.1 # energy injection rate
k_f = sqrt(2) # forcing wavenumber
# 🔴 `freeze = 1` here is load-bearing and must not be "tidied" to match the reference's 10. The LES
# at Δt = 2.5e-3 with freeze = 1 advances the OU chain on exactly the schedule the HF reference's
# Δt = 2.5e-4 with freeze = 10 does: 40 001 advances of 2.5e-3 either way, same seed, each forcing
# field covering the same physical interval. That correspondence is what lets a tracking run and the
# reference see the same forcing realisation (gotcha #33), and every downstream online driver
# asserts `freeze == 1` after inheriting `ou_bodyforce` from this file.
freeze = 1

seeds = (;
    dns = 123, # DNS initial condition
    ou = 333, # OU process
    to = 234, # TO method online sampling
)

# QoI samples are recorded every step (`qoisaver` runs at `nupdate = 1`); `savefreq` is the LF
# *field* interval. 🔴 100 is not cosmetic: at Δt = 2.5e-3 over 40 000 steps it gives 401 fields at
# exactly 0.25 TU, which is the spacing `analysis/build_d6_ics.jl` asserts (`FIELD_STRIDE`,
# `FIELD_DT`, `N_FIELDS`). Changing it silently removes D6's IC pool.
savefreq = 100

nt = round(Int, tsim / Δt)
n_fields = nt ÷ savefreq + 1   # +1 for the t = 0 field: fieldsaver is registered before qoisaver,
                               # so the initial `state[] = state[]` poke reaches it (gotcha #41)

# ---------------------------------------------------------------------------------------------
# Disk preflight, before the reference is loaded and long before the first step.
#
# 🔴 `track_ref` accumulates every stored field in host memory and writes the whole record at the
# end, exactly as `create_ref_data` does — so running out of space is discovered after the compute
# is spent, with nothing written. `2_HF_ref.jl` has had this check since #50; this path never did.
# ---------------------------------------------------------------------------------------------
field_bytes = n_fields * prod((n_les, n_les, n_les)) * 3 * sizeof(T)
series_bytes = 4 * 6 * (nt + 1) * sizeof(T)          # q, q_star, dQ, tau
needed = field_bytes + series_bytes
@printf("tracking output estimate: %d fields at %.2f MiB = %.2f GiB, series %.2f MiB, total %.2f GiB\n",
        n_fields, prod((n_les, n_les, n_les)) * 3 * sizeof(T) / 2^20, field_bytes / 2^30,
        series_bytes / 2^20, needed / 2^30)
check_output_space(outdir, needed; hard = true)

# load reference data
@info "loading the HF reference" ref_file
data_train = load(ref_file, "data_train");
params_train = load(ref_file, "params_train");
# get initial condition
if data_train.data[1].u[1] isa Tuple
    ustart = stack(ArrayType{T}.(data_train.data[1].u[1]));
elseif data_train.data[1].u[1] isa Array{<:Number,4}
    ustart = ArrayType{T}(data_train.data[1].u[1]);
end
@info "initial condition" eltype(ustart) size(ustart)

# get ref trajectories
qoi_ref = stack(data_train.data[1].qoi_hist[1:Int(tsim/Δt)+1]);
size(qoi_ref, 2) == nt + 1 || error(
    "the reference carries $(size(qoi_ref, 2)) QoI samples over this window, expected $(nt + 1); " *
    "its sample spacing is not this run's Δt")
ref_reader = Reference_reader(qoi_ref);

params_track = (;
    params_train...,
    # 🔴 Override the archived Re. The splat above carries the archive's Float32 parameters, and a
    # later key wins — without this the setup is built at Float32 while the script declares
    # Float64, and `typeof(setup.Re)` silently drives every QoI buffer back to single precision.
    # `rf_setup` now refuses that mismatch outright, so this is what keeps the script runnable.
    Re = T(2_000),
    tsim,
    Δt,
    ArrayType,
    backend,
    ou_bodyforce = (;T_L, e_star, k_f, freeze, rng_seed = seeds.ou),
    savefreq);

# 🔴 LMWray3 stated, never inherited. It is already the library default since 2026-09-11, but a
# defaulting site is exactly where the merge's worst near-miss lived: upstream silently moved
# `solve_unsteady`'s default from RK44 to LMWray3 and `create_ref_data` inherited the change.
# Reproducing an archived run means passing `RKMethods.RK44(; T)` here instead.
data_track = track_ref(; params_track..., ref_reader, ustart, rk_method = LMWray3(; T));

# ---------------------------------------------------------------------------------------------
# Save FIRST, then check.
#
# 🔑 The order matters and it used to be the other way round. A tracking gate that throws before
# `jldsave` discards the whole run over a number a human could have looked at — the same failure
# mode as a job that writes only at the end (#50). Write the record, then report, then assert.
# ---------------------------------------------------------------------------------------------
filename = "$outdir/data_track_dns$(n_dns)_les$(n_les)_Re$(Re)_tsim$(tsim)_f64_lmwray3.jld2"
jldsave(filename; data_track, params_track);
@printf("wrote %s (%.2f GiB)\n", basename(filename), filesize(filename) / 2^30)

# check tracking
n_steps = size(data_track.q, 2)
erel = abs.((qoi_ref[:, 1:n_steps] .- data_track.q) ./ qoi_ref[:, 1:n_steps]);
qoi_labels = ["Z[0,6]", "E[0,6]", "Z[7,15]", "E[7,15]", "Z[16,32]", "E[16,32]"]
println()
println("tracking error |q_ref - q| / |q_ref|, per QoI over $n_steps samples:")
println("  band            max         mean        p99")
for i in axes(erel, 1)
    e = sort(view(erel, i, :))
    @printf("  %-12s  %.3e   %.3e   %.3e\n", qoi_labels[i], e[end], sum(e) / length(e),
            e[max(1, round(Int, 0.99 * length(e)))])
end
@printf("overall max %.3e\n", maximum(erel))

# 🔴 PROVISIONAL THRESHOLD, and it is deliberately loose. This is a blow-up detector, not a
# tracking-quality check: 1e-1 is an order above anything a working tracking run should produce, and
# it exists so a diverged run cannot be mistaken for a good one. **A threshold in the same range as
# the error it must detect is not a check** (gotcha #45's second lesson, which cost a day) — so the
# tight bound is set from the table above, on the first real run, and recorded in
# `analysis/results.md`. Do not tighten it by guessing.
const TRACK_GATE = parse(Float64, get(ENV, "RIKFLOW_TRACK_GATE", "1e-1"))
if maximum(erel) >= TRACK_GATE
    @error "tracking error exceeds the gate; the record is written but should not be used" maximum(erel) TRACK_GATE
    error("tracking gate failed")
end
println("tracking gate passed (max $(maximum(erel)) < $TRACK_GATE)")
