# M4's end-to-end smoke test: does a fit run, on the selected device, and come back readable?
#
#     julia --startup-file=no --project=lib/RikFlow/training \
#           lib/RikFlow/exp_square_HIT/tools/m4_smoke.jl
#
#     M4_DEVICE=cuda  julia ... m4_smoke.jl        # the same, on a GPU
#
# ---------------------------------------------------------------------------------------------
# What this is for, and what it deliberately is NOT
# ---------------------------------------------------------------------------------------------
#
# 🔴 **The first job on a new device should be one that FINISHES.** The GPU training path has never
# run on a GPU (`results_LSTMS.md` §5), and the way that was discovered was a 20-minute scan that
# printed nothing and had to be cancelled -- a scan is a bad first job because a single point is
# thousands of updates with no output until it ends. This runs a few seconds of arithmetic, prints
# a timestamped line at every phase, and exits 0 with a PASS line or non-zero with the failure.
#
# 🔑 **It depends on nothing that has to be set up first.** No `inputs_lstm.jld2` -- the spec is
# built here, so `10_setup_lstm.jl` need not have been run -- and it writes its fit to a temporary
# directory, so it cannot overwrite a real one or race fifteen grid jobs.
#
# ⚠️ **It is not a measurement.** `RIKFLOW_M4_SMOKE_EPOCHS` epochs of a deliberately small model
# says nothing about cost or quality; the first epoch is compilation (gotcha #44), and on a GPU it
# also pays PTX compilation. The numbers are printed to localise a stall, not to be quoted.
#
# What it does cover, in order, so a failure names its own phase:
#   1. the Lux extension is loaded at all
#   2. `M4_DEVICE` resolves, and `cuda` finds a functional device
#   3. the QoIs resolve (cache or record) -- the same `m4_data.jl` path the real drivers use
#   4. `build_history` produces the regressor the model expects
#   5. a fit runs on the device and returns finite losses
#   6. the parameters come back on the HOST and convert to `LSTMWeights`
#   7. `save_stochlstm` / `load_stochlstm` round-trip, including `emission` (which was once lost)
#   8. the deployed closure takes one step and returns `dQ` at the SOLVER's precision
#   9. how long the real scan would take ON THIS DEVICE -- the number a walltime is set from

using RikFlow
using Lux, Optimisers, Zygote
using JLD2
using Random
using Printf
using LinearAlgebra

const RF = RikFlow

include(joinpath(@__DIR__, "m4_data.jl"))

const EPOCHS = parse(Int, get(ENV, "RIKFLOW_M4_SMOKE_EPOCHS", "3"))
const DEV_NAME = get(ENV, "M4_DEVICE", "cpu")

failures = String[]
check(ok, what) = ok ? m4_phase("  ok: $what") : (push!(failures, what); m4_phase("  🔴 FAIL: $what"))

m4_phase("M4 smoke: device=$DEV_NAME epochs=$EPOCHS")

# --- 1. the extension ---------------------------------------------------------------------------
Base.get_extension(RikFlow, :RikFlowLuxExt) === nothing &&
    error("the Lux extension is not loaded -- run under --project=lib/RikFlow/training")
m4_phase("1. Lux extension loaded")

# --- 2. the device ------------------------------------------------------------------------------
# 🔑 Resolved BEFORE anything expensive: `m4_device` throws when `cuda` has no functional device,
# and a job that is going to fail for that reason should fail in seconds, not after a data load.
# 🔑 `jl` is an extra the scans do not have: `JLArray` is host-backed but **refuses scalar
# indexing exactly as `CuArray` does** and its `similar` returns its own type, so it exercises the
# device code path -- and the two failure modes a port actually has -- on a machine with no GPU.
# It lives here rather than in `m4_device` because `JLArrays` is a dependency of the training
# environment, not of `RikFlow`, and putting it in the library would be gotcha #53's class.
device = if lowercase(DEV_NAME) in ("jl", "jlarray")
    @eval using JLArrays
    Base.invokelatest(getfield, Main, :JLArray)
else
    RF.m4_device(DEV_NAME)
end
m4_phase("2. device resolved: $DEV_NAME -> $(device === identity ? "host Array" : string(device))")

# --- 3. the QoIs --------------------------------------------------------------------------------
track_file = get(ENV, "RIKFLOW_TRACK_FILE",
                 normpath(joinpath(@__DIR__, "..", "output",
                                   "data_track_dns512_les64_Re2000.0_tsim100.0_f64_lmwray3.jld2")))
rec = load_m4_qois(; track_file)
m4_phase("3. QoIs: q $(size(rec.q)) from $(basename(rec.source))")
check(size(rec.q, 1) == 6, "six QoIs")

# --- 4. the regressor ---------------------------------------------------------------------------
# A short window on purpose: enough rows for a handful of segments, not the production fit.
a, b = 400, 1400
spec = RF.LSTMSpec(; hist = RF.HistorySpec(; h = 1, n_qoi = size(rec.q, 1)),
                   n_hidden = 16, n_latent = 4, n_encoder = 0, arch = :storn, emission = :none)
_, in_scaling = RF._normalise(rec.q[:, a:(b - 1)]; normalization = :normal)
scaling = (in_scaling = in_scaling, out_scaling = in_scaling)
qs = RF.scale_input(rec.q[:, a:b], in_scaling)
qss = RF.scale_input(rec.q_star[:, a:(b - 1)], in_scaling)
X, Y, steps = RF.build_history(spec.hist, qss, qs)
m4_phase("4. regressor: $(size(X, 1)) rows x $(size(X, 2)) features")
check(size(X, 2) == RF.n_input(spec), "regressor width matches the spec")

# --- 5. the fit ---------------------------------------------------------------------------------
m4_phase("5. fitting $EPOCHS epochs on $DEV_NAME (the first includes compilation) ...")
t0 = time()
ps, hist = RF.train_stochlstm(spec, permutedims(X), permutedims(Y), steps;
                              L = 200, burn = 50, epochs = EPOCHS, batch = 8, lr = 1e-2,
                              beta = 1e-4, seed = 1, verbose = true, device)
m4_phase(@sprintf("5. fit done in %.1f s; val %s", time() - t0,
                  join((@sprintf("%.4g", v) for v in hist.val), " -> ")))
check(length(hist.val) == EPOCHS, "one validation value per epoch")
check(all(isfinite, hist.val) && all(isfinite, hist.train), "losses are finite")

# --- 6. parameters come back on the host --------------------------------------------------------
check(all(v -> v === nothing || v isa Array, values(ps)),
      "parameters returned on the host, not on the device")
w = RF.LSTMWeights(ps, spec)
m4_phase("6. LSTMWeights built")

# --- 7. save / load round-trip ------------------------------------------------------------------
# ⚠️ Into a temp dir: this must not overwrite a real fit, and fifteen grid jobs must not race it.
dir = mktempdir()
path = joinpath(dir, "smoke.jld2")
RF.save_stochlstm(path, spec, w, scaling; smoke = true)
fit = RF.load_stochlstm(path)
check(fit.spec.arch === spec.arch, "arch survives the round-trip")
check(fit.spec.emission === spec.emission, "emission survives the round-trip")
check(RF.n_input(fit.spec) == RF.n_input(spec), "shapes survive the round-trip")
m4_phase("7. save/load round-trip via $(basename(path))")

# --- 8. one deployed step -----------------------------------------------------------------------
# 🔑 The deployed closure is stdlib-only and CPU by design; this is the boundary the solver sees.
nwarm = 8
spinnup = Float64.(rec.q[:, 1:nwarm]) .* 0 .+ 1e-3
m = RF.StochLSTM(fit.spec, fit.weights, fit.scaling; spinnup_data = spinnup, rng = Xoshiro(1))
q_star = Float64.(rec.q[:, b])
# 🔴 A `let`, not `local dQ` + a top-level `for`. At top level a `for` body is SOFT scope, so an
# assignment inside it creates a loop-local binding and the value never escapes -- `UndefVarError`
# after the loop, which is what the first version of this file did. `let` is hard scope, so the
# loop assigns the binding declared here. Soft scope has bitten this repository repeatedly, which
# is why V36 exists (`claude_memory.md` #47, #55).
dQ = let out = nothing
    for _ in 1:(nwarm + 1)
        out = RF.get_next_item_timeseries(m, q_star)
    end
    out
end
check(eltype(dQ) === Float64, "dQ returns at the solver's precision (Float64), not the model's")
check(length(dQ) == 6 && all(isfinite, dQ), "dQ is six finite numbers")
m4_phase("8. deployed step: dQ eltype $(eltype(dQ))")

# --- 9. how long the real scan would take, ON THIS DEVICE -------------------------------------
#
# 🔴 **This is the stage that sets the walltime, and it is a measurement, not a projection.** The
# 2 h in the batch scripts was a guess; the only honest source for it is the per-epoch cost at the
# geometry the scan will actually run, on the device it will actually run on. Each point is probed
# with one compile epoch (discarded -- gotcha #44) and a couple of timed ones, then projected over
# the epochs that point plans.
#
# ⚠️ **Bounded on purpose.** `RIKFLOW_M4_TIMING_BUDGET` seconds (default 600) caps the whole stage;
# if a device is slow enough to blow that, the estimate stops early and says which points were not
# measured — because a smoke test whose timing stage does not finish is not a smoke test.
# Set `RIKFLOW_M4_TIMING=0` to skip it entirely.
# 🔴 `let`, not a bare `if`. An `if` at top level does NOT introduce a scope, so `total` below
# would be a GLOBAL, and `total += proj` inside the `for` is then soft scope -- Julia makes a new
# loop-local and the global stays undefined. That is the second time this file hit it in one
# sitting (the first was `dQ`, stage 8), and #47/#55 say it is the repository's most repeated
# defect. A `let` makes every binding here lexical and the loop assigns the one declared.
if get(ENV, "RIKFLOW_M4_TIMING", "1") == "1"
  let
    m4_phase("9. timing the real scan geometry on $DEV_NAME ...")
    # The production window and segmentation, not the short one stages 4-8 used.
    pa, pb = 400, 4000
    pqs = RF.scale_input(rec.q[:, pa:pb], in_scaling)
    pqss = RF.scale_input(rec.q_star[:, pa:(pb - 1)], in_scaling)
    pX, pY, psteps = RF.build_history(spec.hist, pqss, pqs)
    pXc, pYc = permutedims(pX), permutedims(pY)
    L, burn, val_frac = 500, 100, 0.2
    ntrain = floor(Int, (1 - val_frac) * length(psteps))
    updates = parse(Int, get(ENV, "RIKFLOW_M4_UPDATES", "3000"))
    budget = parse(Float64, get(ENV, "RIKFLOW_M4_TIMING_BUDGET", "600"))

    points = [(500 - 100, 32), (500 - 100, 2), (100, 32), (50, 32), (20, 32)]
    t_stage = time()
    total = 0.0
    unmeasured = Tuple{Int,Int}[]
    @printf("%-8s %-6s %6s %6s %8s %12s %12s
",
            "stride", "batch", "segs", "u/ep", "epochs", "s/epoch", "point (min)")
    for (stride, batch) in points
        if time() - t_stage > budget
            push!(unmeasured, (stride, batch)); continue
        end
        g = m4_geometry(psteps, ntrain; L, burn, stride, batch)
        epochs = max(1, cld(updates, g.upd))
        # one compile epoch, then two timed -- the same shape as `m4_probe_step`, inline so the
        # per-point row can be printed as a table rather than as prose.
        RF.train_stochlstm(spec, pXc, pYc, psteps; L, burn, stride, epochs = 1, batch,
                           lr = 1e-2, beta = 1e-4, val_frac, seed = 1, verbose = false, device)
        t0 = time()
        RF.train_stochlstm(spec, pXc, pYc, psteps; L, burn, stride, epochs = 2, batch,
                           lr = 1e-2, beta = 1e-4, val_frac, seed = 1, verbose = false, device)
        sec = (time() - t0) / 2
        proj = sec * epochs / 60
        total += proj
        @printf("%-8d %-6d %6d %6d %8d %10.3f s %10.1f
",
                stride, batch, g.nseg, g.upd, epochs, sec, proj)
        flush(stdout)
    end
    println()
    if isempty(unmeasured)
        m4_phase(@sprintf("9. scan projection on %s: %.0f min total at %d updates/point",
                          DEV_NAME, total, updates))
        # 🔑 A walltime, with the margin a projection from two timed epochs deserves, plus the
        # fixed cost every job pays before the first point (package load + compilation).
        m4_phase(@sprintf("9. suggested walltime: -t %02d:00:00  (1.5x the %.0f min, +15 min startup)",
                          max(1, ceil(Int, (1.5 * total + 15) / 60)), total))
    else
        m4_phase(@sprintf("9. PARTIAL: %.0f min over %d of %d points; %d not measured (budget %.0f s)",
                          total, length(points) - length(unmeasured), length(points),
                          length(unmeasured), budget))
        m4_phase("9. unmeasured: $(join(("stride $s batch $b" for (s, b) in unmeasured), ", "))")
        m4_phase("9. 🔴 the total above is a LOWER BOUND -- do not set a walltime from it")
    end
  end
end

rm(dir; recursive = true, force = true)

println()
if isempty(failures)
    m4_phase("M4 SMOKE PASS on device=$DEV_NAME")
else
    m4_phase("M4 SMOKE FAILED on device=$DEV_NAME: $(length(failures)) check(s)")
    foreach(f -> println("  - ", f), failures)
    exit(1)
end
