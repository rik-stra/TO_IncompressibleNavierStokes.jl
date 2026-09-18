# Does a higher starting learning rate reach convergence in fewer steps?
#
#     julia --startup-file=no --project=lib/RikFlow/training \
#           lib/RikFlow/exp_square_HIT/tools/m4_lr_scan.jl [cell] [seed]
#
# Environment: `RIKFLOW_QOI_CACHE` (as the training driver), `RIKFLOW_M4_LR_EPOCHS` (budget per
# point, default 1500), `RIKFLOW_M4_LRS` (comma-separated list, default the decade ladder below).
#
# ---------------------------------------------------------------------------------------------
# Why this is a separate script and not a driver flag
# ---------------------------------------------------------------------------------------------
#
# 🔑 **Every point runs inside ONE process, sequentially, on one prepared dataset.** The scan is a
# comparison between learning rates, so everything else has to be held still: the same segments,
# the same weight initialisation (one `seed` for all points -- `init_lstm_params` is called with
# it inside `train_stochlstm`), the same validation split and the same fixed validation noise
# draws. Sweeping through the driver instead would write each point over the last one's output
# file and put the points on different processes, where BLAS thread contention alone moves the
# wall time by more than the effect being measured.
#
# ⚠️ **The x axis is epochs, and an epoch here is ~one optimiser step.** At `L = 500` on a 3599-row
# record there are 9 segments, 7 of them training, and `batch = 8`, so one epoch is one update.
# "Faster" therefore means "in fewer gradient steps", not "in less wall time" -- the per-epoch cost
# does not depend on `lr` at all, so the two orderings are the same here.
#
# 🔴 **A higher `lr` is not automatically better even when it descends faster.** Two failure modes
# the table below is built to expose: a rate that dives and then plateaus far above what a slower
# one reaches (`n_decay` fires early and `best_val` is poor), and a rate that diverges outright
# (`best_val` non-finite). Both look like "fast" in the first fifty epochs.

using RikFlow
using Lux, Optimisers, Zygote
using JLD2
using Random
using Statistics
using Printf

const RF = RikFlow

cell_index = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 2
seed = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 1

Base.get_extension(RikFlow, :RikFlowLuxExt) === nothing &&
    error("the Lux extension is not loaded -- run under --project=lib/RikFlow/training")

TO_folder = normpath(joinpath(@__DIR__, "..", "output", "TO_LSTM"))
track_file = get(ENV, "RIKFLOW_TRACK_FILE",
                 joinpath(TO_folder, "..",
                          "data_track_dns512_les64_Re2000.0_tsim100.0_f64_lmwray3.jld2"))
const EPOCHS = parse(Int, get(ENV, "RIKFLOW_M4_LR_EPOCHS", "1500"))
const LRS = [parse(Float64, x) for x in
             split(get(ENV, "RIKFLOW_M4_LRS", "1e-3,3e-3,1e-2,3e-2,1e-1,3e-1"), ",")]

# 🔑 One QoI resolver for all three M4 drivers -- each used to carry its own.
include(joinpath(@__DIR__, "m4_data.jl"))

inputs = load(joinpath(TO_folder, "inputs_lstm.jld2"), "inputs")
cfg = inputs[cell_index]
rec = load_m4_qois(; track_file)
a, b = cfg.train_range

# --- exactly the driver's data preparation, so the scan is about `lr` and nothing else ----------
_, in_scaling = RF._normalise(rec.q[:, a:(b - 1)]; normalization = cfg.normalization)
hist = RF.HistorySpec(; h = cfg.h, n_qoi = size(rec.q, 1), cfg.hist_var, cfg.include_predictor)
qs = RF.scale_input(rec.q[:, a:b], in_scaling)
qss = RF.scale_input(rec.q_star[:, a:(b - 1)], in_scaling)
X, Y, steps = RF.build_history(hist, qss, qs)
spec = RF.LSTMSpec(; hist, cfg.n_hidden, cfg.n_latent, cfg.n_encoder, cfg.arch, cfg.uclip,
                   emission = get(cfg, :emission, :none))
Xc, Yc = permutedims(X), permutedims(Y)

segs = RF.segment_indices(steps; cfg.L, cfg.burn)
nscored = sum(length(s.score) for s in segs)
@info "M4 lr scan" cell=cfg.name arch=cfg.arch beta=cfg.beta emission=spec.emission seed epochs=EPOCHS
@info "data" rows=size(Xc, 2) features=size(Xc, 1) segments=length(segs) scored_steps=nscored L=cfg.L burn=cfg.burn

"First epoch at which the validation loss is at or below `thr`, or 0 if it never is."
first_below(val, thr) = (i = findfirst(<=(thr), val); i === nothing ? 0 : i)


# 🔴 Device placement. `M4_DEVICE=cpu` (the default) or `cuda`. `m4_device` refuses `cuda` when no
# device is functional rather than falling back to the host, because a silent fallback would report
# a GPU run that took CPU time. ⚠️ **The GPU path has never been run on a GPU** -- it is verified
# only against `JLArrays`, which enforces the same no-scalar-indexing semantics on the host (V54).
# And expect it to be SLOWER at the current geometry: `results_LSTMS.md` §5.
const DEVICE = RF.m4_device(get(ENV, "M4_DEVICE", "cpu"))
@info "device" M4_DEVICE=get(ENV, "M4_DEVICE", "cpu")

results = NamedTuple[]
for lr in LRS
    @info "lr = $lr"
    t0 = time()
    _, h = RF.train_stochlstm(spec, Xc, Yc, steps;
                              cfg.L, cfg.burn, epochs = EPOCHS, cfg.batch, lr,
                              cfg.beta, cfg.val_frac, seed, verbose = false, device = DEVICE)
    push!(results, (; lr, wall = time() - t0, train = h.train, val = h.val, lrhist = h.lr,
                    best_val = h.best_val, best_epoch = h.best_epoch))
    @printf("    best val %.5g at epoch %d; final %.5g; lr %g -> %g; %.1f s\n",
            h.best_val, h.best_epoch, h.val[end], h.lr[1], h.lr[end], time() - t0)
end

# 🔑 The thresholds are read off the SCAN, not chosen in advance: the best validation loss any
# point reached, and two decades above it. "Epochs to reach what the best run reached" is the
# question "is a higher lr faster?" in the only form that has an answer.
best_overall = minimum(r.best_val for r in results if isfinite(r.best_val))
THRESHOLDS = (100 * best_overall, 10 * best_overall, 2 * best_overall)

println()
@printf("%-8s %11s %9s %11s %8s | %s\n", "lr", "best val", "best ep", "final val", "lr end",
        "epochs to reach " * join((@sprintf("%.3g", t) for t in THRESHOLDS), " / "))
println("-"^104)
for r in results
    reach = join((@sprintf("%7d", first_below(r.val, t)) for t in THRESHOLDS), " ")
    @printf("%-8g %11.5g %9d %11.5g %8.2g | %s\n",
            r.lr, r.best_val, r.best_epoch, r.val[end], r.lrhist[end], reach)
end
println("\n(0 in a reach column = that threshold was never reached inside $EPOCHS epochs)")

out = joinpath(TO_folder, "lr_scan_$(cfg.name)_seed$(seed).jld2")
jldsave(out; results, cell = cfg.name, cfg, seed, epochs = EPOCHS, thresholds = THRESHOLDS)
println("\nwrote $out")
